import logging
import timeit

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import utils.utils as utils
from algo.AbstractAlgorithm import AbstractAlgorithm
from model import CriticNetworkCNN


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