'''To store learning algorithms'''
import abc
import logging
import random
import timeit

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
from torch.utils.data import DataLoader

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


class ActorCritic(AbstractAlgorithm):
    """
    Implementation for Actor-Critic, some ideas are referenced from algorithm S3 in https://arxiv.org/pdf/1602.01783.pdf
    """
    def __init__(self, args, model) -> None:
        super().__init__(args, model)
        self.CriticNet = CriticNetworkCNN()            # this is critic network, self.model is the actor network
        self.critic_optimizer = torch.optim.Adam(self.CriticNet.parameters(), lr=0.0001)
        self.previous_frame = False
        
    def algo_step(self, stepidx, model, optimizer, scheduler, envs, observations, prev_state, prev_state_value, bsz):
        args = self.args
        if envs is None:
            envs = [utils.SkipFrameWrapper(gym.make(args.env), skip=4) for _ in range(bsz)]
            observations = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
            # NOTE: preprocess_shape() can be switched to other preprocess functions
            observations = torch.stack( # bsz x ic x iH x iW -> bsz x 1 x ic x iH x iW
                [utils.preprocess_observation(obs, mode='resize') for obs in observations]).unsqueeze(1)
            prev_state, prev_state_value = None, None
        
        log_probs, rewards, actions, cur_values = [], [], [], []
        not_terminated = torch.ones(bsz) # agent is still alive
        
        if self.previous_frame:     # reset hidden state to zero at start of each episode (unroll_length of steps)
            prev_state, prev_state_value = None, None
        
        for t in range(args.unroll_length):     # collect samples for unroll_length steps
            # get forward values from neural networks
            prob_t, prev_state = model(observations, prev_state)    # prob_t is bsz x 1 x naction
            if self.previous_frame:
                values_t, prev_state_value = self.CriticNet(observations, prev_state_value)
            else:
                values_t = self.CriticNet(observations) # values are bsz x 1 x 1
            cur_values.append(values_t.squeeze())
            
            # sample actions
            actions_t = Categorical(probs=prob_t.squeeze(1)).sample()
            log_prob = torch.log(prob_t)        # turn into log probability
            selected_log_probs = log_prob.squeeze(1).gather(-1, actions_t.unsqueeze(1))     # get log probability of selected actions
            actions.append(actions_t.view(-1, 1)) # bsz x 1
            log_probs.append(selected_log_probs)
            
            # get outputs for each env, which are (observation, reward, terminated, truncated, info) tuples
            env_outputs = [env.step(actions_t[b].item()) for b, env in enumerate(envs)]
            rewards_t = torch.tensor([eo[1] for eo in env_outputs])
            
            # if we lose a life, zero out all subsequent rewards
            still_alive = torch.tensor([env.ale.lives() == args.start_nlives for env in envs])
            not_terminated.mul_(still_alive.float())
            rewards.append(rewards_t*not_terminated)
            observations = torch.stack([utils.preprocess_observation(eo[0], mode='resize') for eo in env_outputs]).unsqueeze(1)

        curr_return = cur_values[-1]    # bsz x 1 x 1, using the last value
        policy_total_loss, value_total_loss = 0, 0
        for t in range(args.unroll_length-2, -1, -1):
            curr_return = rewards[t] + args.discounting * curr_return   # bsz x 1
            log_prob, value = log_probs[t], cur_values[t]
            value_loss = torch.sum((curr_return - value)**2)
            advantage = curr_return - value    # bsz x 1
            policy_loss = torch.sum(log_prob * advantage.detach()) 
            
            # accumulate loss
            policy_total_loss += policy_loss
            value_total_loss += value_loss
        policy_total_loss = -policy_total_loss /((args.unroll_length-1) * args.batch_size)
        value_total_loss = value_total_loss / ((args.unroll_length-1) * args.batch_size)
            
        # update networks
        self.critic_optimizer.zero_grad()
        optimizer.zero_grad()
        value_total_loss.backward()
        policy_total_loss.backward()
        nn.utils.clip_grad_norm_(self.CriticNet.parameters(), args.grad_norm_clipping)
        self.critic_optimizer.step()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        stats = {
            "mean_return": sum(r.mean() for r in rewards)/len(rewards),
            "pg_loss": policy_total_loss,
            "value_loss": value_total_loss,
        }

        # reset any environments that have ended
        for b in range(bsz):
            if not_terminated[b].item() == 0:
                obs = envs[b].reset(seed=stepidx+b)[0]
                observations[b].copy_(utils.preprocess_observation(obs, mode='resize'))

        return stats, envs, observations, prev_state, prev_state_value
    
    def train(self):
        args = self.args
        T = args.unroll_length
        B = args.batch_size
        wandb = self.args.wandb
        del self.args.wandb

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
        envs, observations, prev_state, prev_state_value = None, None, None, None
        frame, update = 0, 0
        while frame < args.total_frames:
            start_time = timer()
            start_frame = frame
            stats, envs, observations, prev_state, prev_state_value = self.algo_step(
                frame, model, optimizer, scheduler, envs, observations, prev_state, prev_state_value, bsz=B
            )
            frame += T*B # here steps means number of observations
            update += 1
            
            if timer() - last_checkpoint_time > args.min_to_save * 60:      # checkpoint to store model
                checkpoint()
                last_checkpoint_time = timer()

            sps = (frame - start_frame) / (timer() - start_time)            # calculate frame per second
            if wandb:
                wandb.log({"pg_loss": stats['pg_loss'], "value_loss": stats['value_loss'], "mean_return": stats["mean_return"], 'Updates': update})
                
            logging.info("Frame {:d} @ {:.1f} FPS | Updates {}: pg_loss {:.3f} | value_loss {:.3f} | mean_return {:.3f}".format(
            frame, sps, update, stats['pg_loss'], stats['value_loss'], stats["mean_return"]))
            
            if frame > 0 and frame % (args.eval_every*T*B) == 0:        # perform validation step after some number of steps
                utils.validate(model, args.render, nepisodes=5, wandb=wandb, mode='resize')
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
            return random.randint(0,self.naction-1)
        else:
            with torch.no_grad():       # with 1-epsilon, taken greedy action
                state = state.to(self.args.device)
                return self.Q_network(state.unsqueeze(0)).max(dim = 1).indices.item()
    
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
        self.Q_network.to(self.args.device)
        self.Q_target.to(self.args.device)
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
        wandb = self.args.wandb
        del self.args.wandb
        env = utils.SkipFrameWrapper(gym.make(self.args.env), skip=self.frame_skipping_interval)
        self.naction = env.action_space.n
        self.initialize_networks(env.action_space.n)
        
        network_updates, target_updates = 0, 0
        for episode in range(self.args.episodes):
            # initialize first state using multiple frames
            frame = self.preprocess(env.reset(seed=utils.SEED+episode)[0])
            self.update_most_recent_observation(frame)
            for _ in range(self.frames_per_state-1):
                action = random.randint(0,self.naction-1)     # random action to produce more frames to generate initial state
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
                replay_states = torch.stack([experience.state for experience in minibatch]).to(self.args.device)
                replay_next_states = torch.stack([experience.next_state for experience in minibatch]).to(self.args.device)
                replay_rewards = torch.tensor([experience.reward for experience in minibatch], dtype=torch.float).to(self.args.device)
                replay_dones = torch.tensor([experience.done for experience in minibatch], dtype=torch.float).to(self.args.device)
                replay_actions = torch.tensor([experience.action for experience in minibatch], dtype=torch.long).to(self.args.device)
                
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
                utils.validate(self.Q_network, self.args, self.args.render, nepisodes=5, wandb=wandb, mode='resize')
                self.Q_network.train()
            
            logging.info(f"Episode: {episode+1} | Timesteps Played: {timestep} | Mean Loss: {loss_total/timestep:.3f} | Mean Reward: {reward_total/timestep:.3f} | Target Updates: {target_updates}")
            if wandb:
                if not not_filled:
                    wandb.log({"Episode": episode+1, "Timesteps Played": timestep, "Mean Loss": loss_total/timestep, "Mean Reward": reward_total/timestep, "Target Updates": target_updates, "Epsilon": self.epsilon, "Network Updates": network_updates})

    
class ProximalPolicyOptimization(AbstractAlgorithm):
    def __init__(self, args, model) -> None:
        super().__init__(args, model)
        self.model = model
        self.lam = self.args.lam        # lambda for GAE
        self.gamma = self.args.discounting
        self.bsz = self.args.batch_size
        self.T = self.args.unroll_length
        self.initial_clip_epsilon = self.args.clip_epsilon
        self.epochs = self.args.epochs
        self.entropy_coef = self.args.entropy_coef
        self.value_coef = self.args.value_coef
        self.nactors = self.args.nactors
    
    def checkpoint(self):
        if self.args.save_path is None:
            return
        logging.info(f"\n{'*'*3} Saving checkpoint to {self.args.save_path} {'*'*3}\n")
        torch.save({"model_state_dict": self.ac_network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "args": self.args}, self.args.save_path)
    
    def initialize_networks(self, naction):
        def lr_lambda(epoch):
            return 1 - epoch/self.args.total_epochs
        self.ac_network = self.model(naction, self.args).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.ac_network.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.clip_epsilon_decay_rate = 1
        self.global_step = 0
    
    def collect_samples(self, envs, observations) -> tuple:
        """
        collect samples using current policy for self.T steps with self.nactors actors
        """
        with torch.no_grad():       # no need to keep track of computation graph since using old policy
            not_terminated = torch.ones(self.nactors).to(self.args.device) # agent is still alive
            # old_log_probs, rewards, old_values, not_ends, actions, obs_arr = [], [], [], [], [], []
            old_log_probs = torch.zeros((self.T, self.nactors)).to(self.args.device)
            rewards = torch.zeros((self.T, self.nactors)).to(self.args.device)
            old_values = torch.zeros((self.T, self.nactors)).to(self.args.device)
            not_ends = torch.zeros((self.T, self.nactors)).to(self.args.device)
            actions = torch.zeros((self.T, self.nactors)).to('cpu')
            obs_tensor = torch.zeros((self.T, self.nactors, 1, 4, 84, 84)).to(self.args.device)
            for t in range(self.T):     # collect samples for unroll_length steps
                actions_t, log_probs_t, values_t, _ = self.ac_network.get_action_and_value(observations)         # use old parameter as policy
                
                # obs_arr.append(observations)
                # old_values.append(values_t.squeeze())
                # old_log_probs.append(log_probs_t.squeeze().to(self.args.device))
                # actions.append(actions_t.view(-1, 1).squeeze())
                obs_tensor[t] = observations
                old_values[t] = values_t.squeeze()
                old_log_probs[t] = log_probs_t.squeeze()
                actions[t] = actions_t.view(-1, 1).squeeze()
                
                envs_outputs = tuple(env.step(actions_t[b].item()) for b, env in enumerate(envs))
                self.global_step += self.nactors
                rewards_t = torch.tensor([eo[1] for eo in envs_outputs]).to(self.args.device)
                
                # if we lose a life, zero out all subsequent rewards
                still_alive = (~torch.tensor([eo[2] for eo in envs_outputs])).float().to(self.args.device)
                # not_ends.append(still_alive)
                not_ends[t] = still_alive
                not_terminated.mul_(still_alive)        # if dead, record as dead
                # rewards.append(rewards_t*not_terminated)            # record reward if still alive
                rewards[t] = rewards_t*not_terminated
                observations = torch.stack([eo[0] for eo in envs_outputs], dim=0).unsqueeze(1).to(self.args.device)          # transitin to next state
        
        return old_log_probs, rewards, old_values, not_ends, actions, not_terminated, obs_tensor, observations, envs
    
    def calculate_GAE_and_vtarget(self, rewards: torch.Tensor, old_values: torch.Tensor, not_ends: torch.Tensor) -> tuple:
        with torch.no_grad():   # no need to keep track of computation graph since using values collected from old policy
            advantage_t = torch.zeros((self.nactors, )).to(self.args.device)
            advantages = torch.zeros((self.T, self.nactors)).to(self.args.device)
            v_target = old_values[-1]*(not_ends[-1])
            v_target_tensor = torch.zeros((self.T, self.nactors)).to(self.args.device)
            v_target_tensor[-1] = v_target
            for t in reversed(range(self.T-1)):
                delta_t = rewards[t] + self.gamma*old_values[t+1]*(not_ends[t+1]) - old_values[t]       # calculate GAE (generalized advantage estimation: https://arxiv.org/pdf/1506.02438.pdf)
                advantage_t = delta_t + self.gamma*self.lam*advantage_t*(not_ends[t+1])
                # advantages.insert(0, advantage_t)
                advantages[t] = advantage_t
                
                # v_target = advantage_t + old_values[t]                  # this is equivalent to TD(lambda), see P297 in Sutton's book
                # # v_target_arr.insert(0, v_target)
                # v_target_tensor[t] = v_target
            v_target_tensor = advantages + old_values
            
        return advantages, v_target_tensor
    
    def train_epochs(self, data_loader: DataLoader) -> tuple[float, float, float, float, float]:
        mean_total_loss, mean_policy_loss, mean_value_loss, mean_entropy, mean_kl = [], [], [], [], []
        for _ in range(self.epochs):
            for _, (bsz_old_log_probs, bsz_obs, bsz_advantages, bsz_vtargets, bsz_actions) in enumerate(data_loader):
                self.clip_epsilon = self.initial_clip_epsilon*self.clip_epsilon_decay_rate      # anneal clip epsilon
                self.clip_epsilon_decay_rate = max(self.clip_epsilon_decay_rate - 1/self.args.total_epochs, 0)
                
                # flatten in order to calculate in batches
                bsz_actions, bsz_old_log_probs, bsz_advantages, bsz_vtargets = bsz_actions.flatten(), bsz_old_log_probs.flatten(), bsz_advantages.flatten(), bsz_vtargets.flatten()
                _, new_log_probs, new_values, new_entropys = self.ac_network.get_action_and_value(bsz_obs, bsz_actions)       # flatten action to 1D
                new_log_probs, new_values = new_log_probs.to(self.args.device), new_values.squeeze()                          # some sanity check
                ratio = torch.exp(new_log_probs - bsz_old_log_probs)                                        # calculate ratio
                bsz_advantages = (bsz_advantages - bsz_advantages.mean())/(bsz_advantages.std() + 1e-8)      # normalize advantages
                
                # calculate loss
                policy_loss = torch.min(ratio*bsz_advantages, torch.clip(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)*bsz_advantages)     # clipped surrogate objective
                policy_loss = -policy_loss.mean()
                value_loss = F.mse_loss(bsz_vtargets, new_values)
                entropy_bonus = new_entropys.mean()
                total_loss = policy_loss + self.value_coef*value_loss - self.entropy_coef*entropy_bonus
                
                mean_total_loss.append(total_loss.item())
                mean_policy_loss.append(policy_loss.item())
                mean_value_loss.append(value_loss.item())
                mean_entropy.append(entropy_bonus.item())
                mean_kl.append((bsz_old_log_probs - new_log_probs).mean().item())          # use Monte Carlo estimate of KL divergence
                
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.args.grad_norm_clipping)
                self.optimizer.step()
                self.scheduler.step()           # anneal learning rate
        
        return mean_total_loss, mean_entropy, mean_value_loss, mean_policy_loss, mean_kl
        
    def algo_step(self, stepidx: int, envs: list, observations: list):
        if envs is None:
            envs = [utils.AtariGameEnv(env_name=self.args.env) for _ in range(self.nactors)]
            observations = [env.reset(seed=stepidx)[0] for env in envs]
            observations = torch.stack([obs for obs in observations], dim=0).unsqueeze(1).to(self.args.device)
        
        old_log_probs, rewards, old_values, not_ends, actions, not_terminated, obs_tensor, observations, envs = self.collect_samples(envs, observations)     # collect samples for unroll_length steps
        advantages, v_target = self.calculate_GAE_and_vtarget(rewards, old_values, not_ends)
        
        samples = utils.TrajectorySamples(old_probs=old_log_probs, obs=obs_tensor, advantages=advantages, vtargets=v_target, actions=actions)
        data_loader = DataLoader(samples, batch_size=self.bsz, shuffle=True)
        
        mean_total_loss, mean_entropy, mean_value_loss, mean_policy_loss, kl_mean = self.train_epochs(data_loader)
        
        stats = {
            'surrogate_loss': np.mean(mean_total_loss),
            'entropy_bonus': np.mean(mean_entropy),
            'value_loss': np.mean(mean_value_loss),
            'policy_loss': np.mean(mean_policy_loss),
            'kl_divergence': np.mean(kl_mean),
            'mean_return': sum(r.mean().item() for r in rewards)/len(rewards),
        }
        
        # reset any environments that have ended
        for i in range(self.nactors):
            if not_terminated[i].item() == 0:
                obs = envs[i].reset(seed=stepidx+i)[0]
                observations[i].copy_(obs)
        
        return stats, envs, observations
    
    def train(self):
        timer = timeit.default_timer
        last_checkpoint_time = timer()
        wandb = self.args.wandb
        del self.args.wandb
        self.naction = gym.make(self.args.env).action_space.n
        self.initialize_networks(self.naction)
        
        epochs, envs, observations = 0, None, None
        while epochs < self.args.total_epochs:
            stats, envs, observations = self.algo_step(epochs, envs, observations)
            epochs += self.epochs
            
            if timer() - last_checkpoint_time > self.args.min_to_save * 60:      # checkpoint to store model
                self.checkpoint()
                last_checkpoint_time = timer()
                
            if wandb:
                wandb.log({
                    "Policy Gradient Loss": stats['policy_loss'], "Value Loss": stats['value_loss'], "Mean Return": stats["mean_return"], 
                    'Epochs': epochs, "Entropy Bonus": stats['entropy_bonus'], "Surrogate Loss": stats['surrogate_loss'], 'KL Divergence': stats['kl_divergence'],
                    "Clip Epsilon": self.clip_epsilon, "Learning Rate": self.optimizer.param_groups[0]['lr'], 'Global Step': self.global_step,
                })
                
            logging.info(f"Epoch {epochs:d} | sur_loss {stats['surrogate_loss']:.3f} | value_loss {stats['value_loss']:.3f} | pg_loss {stats['policy_loss']:.3f} | KL: {stats['kl_divergence']:.3f}| mean_return {stats['mean_return']:.3f}")
            
            if epochs > 0 and epochs % self.args.eval_every == 0:        # perform validation step after some number of steps
                utils.validate_atari(self.ac_network, self.args.env, self.args.render, nepisodes=5, wandb=wandb, device=self.args.device)
                self.ac_network.train()