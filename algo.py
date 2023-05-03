'''To store learning algorithms'''
import abc
import logging
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

try:        # use wandb to log stuff if we have it, else don't
    import wandb

    # wandb = False
    project_name = "RL-implementation"
except:
    wandb = False


class AbstractAlgorithm(abc.ABC):
    """
    Learning algorithm abstract class, serves as the template for all learning algorithms.

    Parameters
    ----------
    args
        arguments to pass into
    model : nn.Module
        model to run the learning algorithm
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
    Implementation for Actor-Critic algorithm
    
    Reference
    ---------
        https://arxiv.org/pdf/1602.01783.pdf (see Algorithm S3 in Appendix)
    """
    def __init__(self, args, model) -> None:
        super().__init__(args, model)
        self.CriticNet = CriticNetworkCNN()            # this is critic network, self.model is the actor network
        self.critic_optimizer = torch.optim.Adam(self.CriticNet.parameters(), lr=0.0001)
        self.previous_frame = False
        
    def algo_step(self, stepidx, model, optimizer, scheduler, envs, observations, prev_state, prev_state_value, bsz):
        args = self.args
        if envs is None:
            envs = [gym.make(args.env) for _ in range(bsz)]
            observations = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
            # NOTE: preprocess_shape() can be switched to other preprocess functions
            observations = torch.stack( # bsz x ic x iH x iW -> bsz x 1 x ic x iH x iW
                [utils.preprocess_observation(obs) for obs in observations]).unsqueeze(1)
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
            observations = torch.stack([utils.preprocess_observation(eo[0]) for eo in env_outputs]).unsqueeze(1)

        curr_return = cur_values[-1]    # bsz x 1 x 1, using the last value
        policy_total_loss, value_total_loss = 0, 0
        for t in range(args.unroll_length-2, -1, -1):
            curr_return = rewards[t] + args.discounting * curr_return   # bsz x 1
            log_prob, value = log_probs[t], cur_values[t]
            value_loss = F.mse_loss(curr_return, value)
            advantage = curr_return - value    # bsz x 1
            policy_loss = (log_prob * advantage.detach()).mean()       
            
            # accumulate loss
            policy_total_loss += policy_loss
            value_total_loss += value_loss
            
        # update networks
        self.critic_optimizer.zero_grad()
        optimizer.zero_grad()
        value_total_loss.backward()         # perform gradient descent
        policy_total_loss = -1*policy_total_loss    # perform gradient ascent
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
                observations[b].copy_(utils.preprocess_observation(obs))

        return stats, envs, observations, prev_state, prev_state_value
    
    def train(self):
        if wandb:
            wandb.init(project=project_name)
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
                utils.validate(model, args.render, nepisodes=5, wandb=wandb)
                model.train()
        
        print(f"\n{'='*30} TRAINING FINISHED {'='*30}")
        print(f"Time taken: {(timer() - train_start_time)/60:.1f} minutes")


class DeepQLearning(AbstractAlgorithm):
    def __init__(self, args, model) -> None:
        super().__init__(args, model)
        self.args = args
        self.replay_buffer = utils.ReplayBuffer(size=args.replay_buffer_size)
        self.epsilon = 1.0
        self.model = model
        self.frame_skipping_interval = 4
        self.epsilon_decay_frames = 1_000_000
        self.frames_per_state = 4
        self.target_update_frequency = 10_000
        self.epsilon_decay = (self.epsilon - 0.1) / self.epsilon_decay_frames
    
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
            return torch.randint(low=0, high=9, size=(1,)).item()
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
                assert len(self.most_recent_observation) == self.frames_per_state-1
            self.most_recent_observation.append(frame)
    
    def algo_step(self, stepidx: int, optimizer, scheduler, envs: list, observations: list, prev_state, bsz: int):
        raise NotImplementedError("algo_step() not implemented")
    
    def initialize_networks(self, naction):
        self.Q_network = self.model(naction, self.args)
        self.Q_target = self.model(naction, self.args)
        self.Q_target.load_state_dict(self.Q_network.state_dict())
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
        if wandb:
            wandb.init(project=project_name)
        env = gym.make(self.args.env)
        self.initialize_networks(env.action_space.n)
        
        network_updates, target_updates = 0, 0
        for episode in range(self.args.episodes):
            # initialize first state using multiple frames
            frame = self.preprocess(env.reset(seed=590060+episode)[0])
            self.update_most_recent_observation(frame)
            for _ in range(self.frames_per_state-1):
                action = torch.randint(low=0, high=9, size=(1,)).item()
                env_output = env.step(action)
                self.update_most_recent_observation(self.preprocess(env_output[0]))
            state = torch.stack(self.most_recent_observation, dim=0).unsqueeze(0)    # stack the frames to form a single state
            
            done, timestep, reward_total, loss_total = False, 1, 0, 0
            while not done:
                # perform action, get reward
                action = self.epsilon_greedy(state)
                env_output = env.step(action)
                reward, next_frame, done = env_output[1], self.preprocess(env_output[0]), utils.get_done(env)
                reward_total += reward
                timestep += 1
                self.epsilon = max(self.epsilon - self.epsilon_decay, 0.1)   # decay epsilon
                self.update_most_recent_observation(next_frame)
                next_state = torch.stack(self.most_recent_observation, dim=0).unsqueeze(0)
                
                # store experience in replay buffer
                experience = utils.Transition(state=state, action=action, next_state=next_state, reward=reward, done=done)
                self.replay_buffer.push(experience)
                state = next_state
                
                # sample random minibatch from replay buffer
                if len(self.replay_buffer) < self.args.replay_start_size:
                    not_filled = True
                    continue        # not enoguh experience in replay buffer, skip training
                else:
                    minibatch = self.replay_buffer.sample(self.args.batch_size)
                    not_filled = False
                    
                replay_states, replay_next_states, replay_rewards, replay_dones, replay_actions = [], [], [], [], []
                for experience in minibatch:
                    replay_states.append(experience.state)
                    replay_next_states.append(experience.next_state)
                    replay_rewards.append(experience.reward)
                    replay_dones.append(experience.done)
                    replay_actions.append(experience.action)
                replay_states = torch.stack(replay_states)
                replay_next_states = torch.stack(replay_next_states)
                replay_rewards = torch.tensor(replay_rewards, dtype=torch.float)
                replay_dones = torch.tensor(replay_dones, dtype=torch.float)
                replay_actions = torch.tensor(replay_actions, dtype=torch.long)
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
                
                if network_updates > 0 and network_updates % self.args.eval_every == 0:        # perform validation step after some number of steps
                    utils.validate(self.Q_network, self.args.render, nepisodes=5, wandb=wandb, mode='resize')
                    self.Q_network.train()
            
            logging.info(f"Episode: {episode+1} | Timesteps Played: {timestep} | Mean Loss: {loss_total/timestep:.3f} | Mean Reward: {reward_total/timestep:.3f} | Target Updates: {target_updates}")
            if wandb:
                if not not_filled:
                    wandb.log({"Episode": episode+1, "Timesteps Played": timestep, "Mean Loss": loss_total/timestep, "Mean Reward": reward_total/timestep, "Target Updates": target_updates, "Epsilon": self.epsilon, "Network Updates": network_updates})

                

class PPO(AbstractAlgorithm):
    def __init__(self, args, model) -> None:
        super().__init__(args, model)
    
    def algo_step(self, stepidx: int, model: nn.Module, optimizer, scheduler, envs: list, observations: list, prev_state, bsz: int):
        pass
    
    def train(self):
        pass
