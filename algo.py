'''To store learning algorithms'''
import abc
import logging
import timeit

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import utils
from model import ValueNetwork


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
    """Implementation for Actor-Critic algorithm"""
    def __init__(self, args, model) -> None:
        super().__init__(args, model)
        self.ValueNet = ValueNetwork()            # this is critic network, self.model is the actor network
        self.value_optimizer = torch.optim.Adam(self.ValueNet.parameters(), lr=0.0001)
        self.old_values = None
        
    def algo_step(self, stepidx, model, optimizer, scheduler, envs, observations, prev_state, bsz):
        args = self.args
        if envs is None:
            envs = [gym.make(args.env) for _ in range(bsz)]
            observations = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
            observations = torch.stack( # bsz x ic x iH x iW -> bsz x 1 x ic x iH x iW
                [utils.preprocess_observation(obs) for obs in observations]).unsqueeze(1)
            prev_state = None

        if stepidx == 0:        # initialize old values for the very first step
            self.old_values = torch.full((bsz, args.unroll_length), 0)  # bsz x T
        
        logits, rewards, actions, cur_values = [], [], [], []
        not_terminated = torch.ones(bsz) # agent is still alive
        for t in range(args.unroll_length):     # collect samples for unroll_length steps
            logits_t, prev_state = model(observations, prev_state) # logits are bsz x 1 x naction
            values_t = self.ValueNet(observations) # values are bsz x 1
            cur_values.append(values_t)
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

        # implement fitting value network with MSE
        cur_values = torch.cat(cur_values, dim=1).squeeze()  # bsz x T
        rewards = torch.stack(rewards).t()  # bsz x T
        next_values = torch.cat([self.old_values[:, 1:], torch.zeros(bsz, 1)], dim=1)
        td_targets = rewards + args.discounting * next_values
        value_loss = F.mse_loss(cur_values, td_targets)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.ValueNet.parameters(), args.grad_norm_clipping)
        self.value_optimizer.step()
        self.old_values = cur_values.detach().clone()         # update old values

        # compute advantages (using critic network)
        next_values = torch.cat([cur_values[:, 1:], torch.zeros(bsz, 1)], dim=1)
        td_targets = rewards + args.discounting * next_values
        advantages = td_targets - cur_values
        adv = advantages.detach()
        
        # calculate loss and do back propagation for policy network
        logits = torch.cat(logits, dim=1) # bsz x T x naction
        # TODO: check advantage function implementation, it should have same shape as logit (log probabilities returned by policy network)
        # use the advantages to weight the loss
        pg_loss = (logits * adv).mean()
        total_loss = pg_loss

        stats = {
            "mean_return": sum(r.mean() for r in rewards)/len(rewards),
            "pg_loss": pg_loss.item(),
            "value_loss": value_loss.item(),
        }
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
            logging.info("Frame {:d} @ {:.1f} FPS: pg_loss {:.3f} | value_loss {:.3f} | mean_return {:.3f}".format(
            frame, sps, stats['pg_loss'], stats['value_loss'], stats["mean_return"]))
            
            if frame > 0 and frame % (args.eval_every*T*B) == 0:        # perform validation step after some number of steps
                utils.validate(model, args.render, nepisodes=5)
                model.train()
        
        print(f"\n{'='*30} TRAINING FINISHED {'='*30}")
        print(f"Time taken: {(timer() - train_start_time)/60:.1f} minutes")


class DQN(AbstractAlgorithm):
    def __init__(self, args, model) -> None:
        super().__init__(args, model)
    
    def algo_step(self, stepidx: int, model: nn.Module, optimizer, scheduler, envs: list, observations: list, prev_state, bsz: int):
        pass
    
    def train(self):
        pass


class PPO(AbstractAlgorithm):
    def __init__(self, args, model) -> None:
        super().__init__(args, model)
    
    def algo_step(self, stepidx: int, model: nn.Module, optimizer, scheduler, envs: list, observations: list, prev_state, bsz: int):
        pass
    
    def train(self):
        pass
