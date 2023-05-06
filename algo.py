'''To store learning algorithms'''
import abc
import logging
import random
import timeit

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical

import utils
from model import CriticNetworkCNN, CriticNetworkLSTM


class AbstractAlgorithm(abc.ABC):
    """
    Learning algorithm abstract class, serves as the template for all learning algorithms.

    Parameters
    ----------
    args
        arguments to pass into, from command line args in train.py
    model : nn.Module
        neural network model to use for the learning algorithm
    """
    def __init__(self, args, model) -> None:
        super().__init__()
        self.args=args
        self.model=model
        
    @abc.abstractmethod
    def algo_step(self, stepidx: int, model: nn.Module, optimizer, scheduler, envs: list, observations: list, prev_state, bsz: int):
        """
        Perform a step for the learning algorithm (in an abstract sense since some algorithms may have actually do more than one step per call)

        Parameters
        ----------
        stepidx : int
            stepidx, usually the frame number
        model : nn.module
            model to pass into the learning algorithm
        optimizer
            optimizer to use for the learning algorithm
        scheduler
            scheduler to use for the learning algorithm
        envs : list or None
            list of environments
        observations : list or None
            list of observations
        prev_state : None or object
            previous state, useful for recurrent models
        bsz : int
            batch size
        """
        pass
    
    @abc.abstractmethod
    def train(self):
        """
        Define training steps for the learning algorithm
        """
        pass

class VanillaPolicyGradient(AbstractAlgorithm):
    """
    Baseline implementation using vanilla policy gradient.
    """
    def __init__(self, args, model) -> None:
        super().__init__(args, model)
    
    def algo_step(self, stepidx, model, optimizer, scheduler, envs, observations, prev_state, bsz):
        args = self.args
        if envs is None:
            envs = [gym.make(args.env) for _ in range(bsz)]
            observations = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
            observations = torch.stack( # bsz x ic x iH x iW -> bsz x 1 x ic x iH x iW
                [utils.preprocess_observation(obs) for obs in observations]).unsqueeze(1)
            prev_state = None

        logits, rewards, actions = [], [], []
        not_terminated = torch.ones(bsz) # agent is still alive
        for t in range(args.unroll_length):
            logits_t, prev_state = model(observations, prev_state) # logits are bsz x 1 x naction
            logits.append(logits_t)
            # sample actions
            actions_t = Categorical(logits=logits_t.squeeze(1)).sample()
            actions.append(actions_t.view(-1, 1)) # bsz x 1
            # get outputs for each env, which are (observation, reward, terminated, truncated, info) tuples
            env_outputs = [env.step(actions_t[b].item()) for b, env in enumerate(envs)]
            rewards_t = torch.tensor([eo[1] for eo in env_outputs])
            # if we lose a life, zero out all subsequent rewards
            still_alive = torch.tensor([env.ale.lives() == args.start_nlives for env in envs])
            not_terminated.mul_(still_alive.float())
            rewards.append(rewards_t*not_terminated)
            observations = torch.stack([utils.preprocess_observation(eo[0]) for eo in env_outputs]).unsqueeze(1)

        # calculate reward-to-go
        r2g = torch.zeros(bsz, args.unroll_length)
        curr_r = 0
        for r in range(args.unroll_length-1, -1, -1):
            curr_r = rewards[r] + args.discounting * curr_r
            r2g[:, r].copy_(curr_r)

        adv = (r2g - r2g.mean()) / (r2g.std() + 1e-7) # biased, but people do it
        logits = torch.cat(logits, dim=1) # bsz x T x naction
        actions = torch.cat(actions, dim=1) # bsz x T 
        cross_entropy = F.cross_entropy(
            logits.view(-1, logits.size(2)), actions.view(-1), reduction='none')
        pg_loss = (cross_entropy.view_as(actions) * adv).mean()
        total_loss = pg_loss

        stats = {"mean_return": sum(r.mean() for r in rewards)/len(rewards),
                "pg_loss": pg_loss.item()}
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        # reset any environments that have ended
        for b in range(bsz):
            if not_terminated[b].item() == 0:
                obs = envs[b].reset(seed=stepidx+b)[0]
                observations[b].copy_(utils.preprocess_observation(obs))

        return stats, envs, observations, prev_state

    def train(self):
        args = self.args
        T = args.unroll_length
        B = args.batch_size

        args.device = torch.device("cpu")
        env = gym.make(args.env)
        naction = env.action_space.n
        args.start_nlives = env.ale.lives()
        del env

        model = self.model(naction, args)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        def lr_lambda(epoch): # multiplies learning rate by value returned; can be used to decay lr
            return 1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        def checkpoint():
            if args.save_path is None:
                return
            logging.info(f"\n{'*'*3} Saving checkpoint to {args.save_path} {'*'*3}\n")
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "args": args}, args.save_path)


        timer = timeit.default_timer
        train_start_time = timer()
        last_checkpoint_time = timer()
        envs, observations, prev_state = None, None, None
        frame = 0
        while frame < args.total_frames:
            start_time = timer()
            start_frame = frame
            stats, envs, observations, prev_state = self.algo_step(
                frame, model, optimizer, scheduler, envs, observations, prev_state, bsz=B
            )
            frame += T*B # here steps means number of observations
            
            if timer() - last_checkpoint_time > args.min_to_save * 60:      # checkpoint to store model
                checkpoint()
                last_checkpoint_time = timer()

            sps = (frame - start_frame) / (timer() - start_time)            # calculate frame per second
            logging.info("Frame {:d} @ {:.1f} FPS: loss {:.3f} | mean_return {:.3f}".format(
            frame, sps, stats['pg_loss'], stats["mean_return"]))
            
            if frame > 0 and frame % (args.eval_every*T*B) == 0:        # perform validation step after some number of steps
                utils.validate(model, args.render, nepisodes=5)
                model.train()
        
        print(f"\n{'='*30} TRAINING FINISHED {'='*30}")
        print(f"Time taken: {(timer() - train_start_time)/60:.1f} minutes")


class DeepQLearning(AbstractAlgorithm):
    """
    Deep Q Learning implementation
    
    Parameters
    ----------
        - args (Namespace): A namespace object containing the following arguments:
            - episodes (int): The number of episodes to train for.
            - replay_buffer_size (int): The size of the replay buffer.
            - replay_start_size (int): The number of transitions to store in the replay buffer before learning starts.
            - epsilon_start (float): The initial value of epsilon for the epsilon-greedy exploration strategy.
            - epsilon_end (float): The final value of epsilon after decay.
            - frame_skipping_interval (int): The number of frames to skip between consecutive actions.
            - epsilon_decay_frames (int): The number of frames over which epsilon will decay.
            - frames_per_state (int): The number of frames to combine to form a single state.
            - target_update_frequency (int): number of learning network updates between target network updates (C in original paper, see link below).
                
        - model (nn.Module): A PyTorch model to be used as the Q-network and target network.
    
    Pre-preprocessing
    -----------------
        - Convert RGB image to grayscale, normalize value to [0, 1]
        - Resize image to 84x84
        - Stack 4 frames together to form a state
    
    Training
    --------
        - Use frame skipping to speed up learning
        - Use experience replay to break correlation between consecutive frames
        - Use target network to stabilize training
        - Use epsilon greedy as behavior policy, with a linearly decaying epsilon
    
    Validation
    ----------
        - Done on original environments (no frame skipping, with epsilon of 0.05)

    Reference
    ---------
        - https://www.nature.com/articles/nature14236
    """
    def __init__(self, args, model) -> None:
        super().__init__(args, model)
        self.args = args
        self.replay_buffer = utils.ReplayBuffer(size=args.replay_buffer_size)
        self.epsilon = args.epsilon_start
        self.model = model
        self.frame_skipping_interval = args.frame_skipping_interval
        self.epsilon_decay_frames = args.epsilon_decay_frames
        self.frames_per_state = args.frames_per_state
        self.target_update_frequency = args.target_update_frequency
        self.epsilon_decay = (args.epsilon_start - args.epsilon_end) / args.epsilon_decay_frames
    
    def preprocess(self, obs, size=(84, 84), mode='resize'):
        if mode == 'simple':
            return torch.from_numpy(obs).permute(2, 0, 1)/255.0     # just do normalize
        elif mode == 'resize':      # resize, to grey scale, then normalize
            image_tensor = torch.tensor(obs).float()
            image_tensor = image_tensor.permute(2, 0, 1)
            # Resize image using torch.nn.functional.interpolate
            image_tensor = image_tensor.unsqueeze(0) # Add a batch dimension
            transform = T.Compose([
                T.Grayscale(),              # Convert to grayscale to save memory
                T.Resize(size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)  # resize smaller to save memory
            ])
            resized_image_tensor = transform(image_tensor)
            resized_image_tensor = resized_image_tensor.squeeze() # Remove the extra dimension
            resized_image_tensor /= 255.0   # Normalize the pixel values to [0, 1] range

            return resized_image_tensor
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def epsilon_greedy(self, state: torch.Tensor):
        '''perform epsilon greedy action selection'''
        if torch.rand(1).item() < self.epsilon:     # with prob. epsilon, take random action
            return random.randint(0,8)
        else:
            with torch.no_grad():       # with 1-epsilon, taken greedy action
                return self.Q_network(state.unsqueeze(0)).argmax(1).item()
    
    def update_most_recent_observation(self, frame: torch.Tensor):
        '''store frames that are most recently observed'''
        if not hasattr(self, 'most_recent_observation'):
            self.most_recent_observation = [frame]      # if not initialized, initialize it
        else:
            if len(self.most_recent_observation) >= self.frames_per_state:   # if full, pop the oldest state, then update
                self.most_recent_observation.pop(0)
            self.most_recent_observation.append(frame)
    
    def algo_step(self, stepidx: int, optimizer, scheduler, envs: list, observations: list, prev_state, bsz: int):
        raise NotImplementedError("algo_step() not implemented")
    
    def initialize_networks(self, naction):
        self.Q_network = self.model(naction, self.args)
        self.Q_target = self.model(naction, self.args)
        self.Q_target.load_state_dict(self.Q_network.state_dict())      # same initial weights
        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr=self.args.learning_rate)
    
    def checkpoint(self):
        if self.args.save_path is None:
            return
        logging.info(f"\n{'*'*3} Saving checkpoint to {self.args.save_path} {'*'*3}\n")
        torch.save({"model_state_dict": self.Q_network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "args": self.args}, self.args.save_path)
    
    def train(self):
        timer = timeit.default_timer
        last_checkpoint_time = timer()
        # NOTE: frame skipping only applies to training, not evaluation/validation
        # validation/evaluation is still one frame per step
        env = utils.SkipFrameWrapper(gym.make(self.args.env), skip=self.frame_skipping_interval)
        self.initialize_networks(env.action_space.n)
        
        network_updates, target_updates = 0, 0
        for episode in range(self.args.episodes):
            # initialize first state using multiple frames
            frame = self.preprocess(env.reset(seed=590060+episode)[0])
            self.update_most_recent_observation(frame)
            for _ in range(self.frames_per_state-1):
                action = random.randint(0,8)     # random action to produce more frames to generate initial state
                env_output = env.step(action)
                self.update_most_recent_observation(self.preprocess(env_output[0]))
            state = torch.stack(self.most_recent_observation, dim=0).unsqueeze(0)    # stack the frames to form a single state
            
            done, timestep, reward_total, loss_total = False, 1, 0, 0
            while not done:
                # perform action, get reward, stack frames to produce state
                action = self.epsilon_greedy(state)
                env_output = env.step(action)
                reward, next_frame, done = env_output[1], self.preprocess(env_output[0]), env_output[2]
                reward_total += reward
                timestep += 1
                self.epsilon = max(self.epsilon - self.epsilon_decay, 0.1)   # decay epsilon
                self.update_most_recent_observation(next_frame)
                next_state = torch.stack(self.most_recent_observation, dim=0).unsqueeze(0)
                
                # store experience in replay buffer
                experience = utils.Transition(state=state, action=action, next_state=next_state, reward=reward, done=done)
                self.replay_buffer.push(experience)
                state = next_state          # update state
                
                # use experience replay mechanism
                if len(self.replay_buffer) < self.args.replay_start_size:   # not enoguh experience in replay buffer, skip training
                    not_filled = True
                    continue        
                else:               # if enough, sample a minibatch uniformly randomly
                    minibatch = self.replay_buffer.sample(self.args.batch_size)
                    not_filled = False
                
                # batchify the experineces
                replay_states = torch.stack([experience.state for experience in minibatch])
                replay_next_states = torch.stack([experience.next_state for experience in minibatch])
                replay_rewards = torch.tensor([experience.reward for experience in minibatch], dtype=torch.float)
                replay_dones = torch.tensor([experience.done for experience in minibatch], dtype=torch.float)
                replay_actions = torch.tensor([experience.action for experience in minibatch], dtype=torch.long)
                
                # calculate loss
                q_next_values = self.Q_target(replay_next_states).max(dim=1).values.detach()
                y = replay_rewards + self.args.discounting * q_next_values * (1 - replay_dones)
                q_values = self.Q_network(replay_states).gather(1, replay_actions.unsqueeze(1)).squeeze()
                loss = F.mse_loss(y, q_values)
                loss_total += loss.item()
                
                # Update Q network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.Q_network.parameters(), self.args.grad_norm_clipping)
                self.optimizer.step()
                network_updates += 1
                
                if network_updates % self.target_update_frequency == 0:   # update target network for every target_update_frequency steps
                    self.Q_target.load_state_dict(self.Q_network.state_dict())
                    target_updates += 1
                
                if timer() - last_checkpoint_time > self.args.min_to_save * 60:      # checkpoint to store model
                    self.checkpoint()
                    last_checkpoint_time = timer()
                
            if not not_filled and episode > 0 and episode % self.args.eval_every == 0:        # perform validation step after some number of episodes
                utils.validate(self.Q_network, self.args.render, nepisodes=5, mode='resize')
                self.Q_network.train()
            
            logging.info(f"Episode: {episode+1} | Timesteps Played: {timestep} | Mean Loss: {loss_total/timestep:.3f} | Mean Reward: {reward_total/timestep:.3f} | Target Updates: {target_updates}")
