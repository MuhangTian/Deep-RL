import logging
import timeit

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils.utils as utils
from algo.AbstractAlgorithm import AbstractAlgorithm


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
    
    def __preprocess_output(self, env_output: tuple) -> tuple:
        obs = torch.tensor(np.asarray(env_output[0]))
        env_output = list(env_output)
        env_output[0] = obs
        
        return tuple(env_output)
    
    def initialize_ppo(self, naction):
        self.ac_network = self.model(naction, self.args).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.ac_network.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.clip_epsilon_decay_rate = 1
        self.global_step = 0
    
    def collect_samples(self, envs, observations, still_alive) -> tuple:
        """
        collect samples using current policy for self.T steps with self.nactors actors
        """
        with torch.no_grad():       # no need to keep track of computation graph since using old policy
            old_log_probs = torch.zeros((self.T, self.nactors)).to(self.args.device)
            rewards = torch.zeros((self.T, self.nactors)).to(self.args.device)
            old_values = torch.zeros((self.T, self.nactors)).to(self.args.device)
            not_ends = torch.zeros((self.T, self.nactors)).to(self.args.device)
            actions = torch.zeros((self.T, self.nactors)).to(self.args.device)
            obs_tensor = torch.zeros((self.T, self.nactors, 4, 84, 84)).to(self.args.device)
            for t in range(self.T):     # collect samples for unroll_length steps
                actions_t, log_probs_t, values_t, _ = self.ac_network.get_action_and_value(observations)         # use old parameter as policy
                obs_tensor[t] = observations
                not_ends[t] = still_alive
                old_values[t] = values_t.squeeze()
                old_log_probs[t] = log_probs_t.squeeze()
                actions[t] = actions_t.view(-1, 1).squeeze()
                
                envs_outputs = self.__preprocess_output(envs.step(actions_t))
                self.global_step += self.nactors
                rewards_t = torch.tensor(envs_outputs[1]).to(self.args.device)
                
                rewards[t] = rewards_t
                observations = envs_outputs[0].to(self.args.device)          # transitin to next state
                still_alive = (~torch.tensor(envs_outputs[2])).float().to(self.args.device)
        
        return old_log_probs, rewards, old_values, not_ends, actions, obs_tensor, observations, still_alive, envs
    
    def calculate_GAE_and_vtarget(self, rewards: torch.Tensor, old_values: torch.Tensor, not_ends: torch.Tensor, observations: torch.Tensor, still_alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        calculate GAE (generalized advantage estimation: https://arxiv.org/pdf/1506.02438.pdf) and TD target, which is TD(\lambda))
        """
        with torch.no_grad():   # no need to keep track of computation graph since using values collected from old policy
            _, _, next_state_value, _ = self.ac_network.get_action_and_value(observations)              # bootstrap from last state
            advantage_t = 0
            advantages = torch.zeros((self.T, self.nactors)).to(self.args.device)
            v_target_tensor = torch.zeros((self.T, self.nactors)).to(self.args.device)
            for t in reversed(range(self.T)):
                if t == self.T-1:
                    next_value = next_state_value.squeeze()
                    next_not_ends = still_alive
                else:
                    next_value = old_values[t+1]
                    next_not_ends = not_ends[t+1]
                delta_t = rewards[t] + self.gamma*next_value*(next_not_ends) - old_values[t]       
                advantage_t = delta_t + self.gamma*self.lam*advantage_t*(next_not_ends)
                advantages[t] = advantage_t
            v_target_tensor = advantages + old_values           # TD(\lambda)
            
        return advantages, v_target_tensor
    
    def train_epochs(self, data_loader: DataLoader) -> tuple[float, float, float, float, float]:
        """perform training for some epochs based on data sampled using the old policy"""
        for _ in range(self.epochs):
            for _, (bsz_old_log_probs, bsz_old_values, bsz_obs, bsz_advantages, bsz_vtargets, bsz_actions) in enumerate(data_loader):
                self.clip_epsilon = self.initial_clip_epsilon*self.clip_epsilon_decay_rate      # anneal clip epsilon
                # self.clip_epsilon_decay_rate = max(self.clip_epsilon_decay_rate - 1/self.args.total_epochs, 0)
                
                # flatten in order to calculate in batches
                bsz_actions, bsz_old_log_probs, bsz_advantages, bsz_vtargets, bsz_old_values = bsz_actions.flatten(), bsz_old_log_probs.flatten(), bsz_advantages.flatten(), bsz_vtargets.flatten(), bsz_old_values.flatten()
                _, new_log_probs, new_values, new_entropys = self.ac_network.get_action_and_value(bsz_obs, bsz_actions)       # flatten action to 1D
                new_values = new_values.squeeze()
                ratio = torch.exp(new_log_probs - bsz_old_log_probs)                                        # calculate ratio
                bsz_advantages = (bsz_advantages - bsz_advantages.mean())/(bsz_advantages.std() + 1e-8)      # normalize advantages
                
                # calculate loss
                policy_loss = torch.min(ratio*bsz_advantages, torch.clip(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)*bsz_advantages)     # clipped surrogate objective
                policy_loss = -policy_loss.mean()
                
                v_unclipped = (bsz_vtargets - new_values)**2      # clip value loss, intuition is to ensure value network is not overfitting to bsz_vtargets in each epoch
                v_clipped = bsz_old_values + torch.clip(new_values - bsz_old_values, -self.clip_epsilon, self.clip_epsilon)
                v_clipped = (bsz_vtargets - v_clipped)**2
                value_loss = 0.5*torch.max(v_unclipped, v_clipped).mean()
                approx_kl = ((ratio - 1) - (new_log_probs - bsz_old_log_probs)).mean()
                
                entropy_bonus = new_entropys.mean()
                total_loss = policy_loss + self.value_coef*value_loss - self.entropy_coef*entropy_bonus
                    
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.args.grad_norm_clipping)
                self.optimizer.step()
        
        return total_loss, entropy_bonus, value_loss, policy_loss, approx_kl
        
    def algo_step(self, stepidx: int, envs: list, observations: torch.Tensor, still_alive: torch.Tensor):
        """subroutine for self.train(), implements the "inner loop" based on PPO algorithm in paper: https://arxiv.org/abs/1707.06347"""
        if envs is None:
            envs = gym.vector.SyncVectorEnv([utils.make_atari_env(self.args.env, utils.SEED+i) for i in range(self.nactors)])
            observations = self.__preprocess_output(envs.reset(seed=stepidx))[0].to(self.args.device)
            still_alive = torch.ones(self.nactors).to(self.args.device)
        
        # sample data and calculate TD target and GAE
        old_log_probs, rewards, old_values, not_ends, actions, obs_tensor, observations, still_alive, envs = self.collect_samples(envs, observations, still_alive)     # collect samples for unroll_length steps
        advantages, v_target = self.calculate_GAE_and_vtarget(rewards, old_values, not_ends, observations, still_alive)
        
        # prepare loader based on sampled data
        samples = utils.TrajectorySamples(old_probs=old_log_probs, old_values=old_values, obs=obs_tensor, advantages=advantages, vtargets=v_target, actions=actions)
        data_loader = DataLoader(samples, batch_size=self.bsz, shuffle=True)
        
        # perform training using minibatches
        surrogate_loss, entropy, value_loss, policy_loss, kl_divergence = self.train_epochs(data_loader)
        stats = {
            'surrogate_loss': surrogate_loss.item(),
            'entropy_bonus': entropy.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_divergence': kl_divergence.item(),
            'mean_return': rewards.mean().item(),
        }
        
        return stats, envs, observations, still_alive
    
    def train(self):
        """Function to start training PPO while logging data to wandb (if needed)"""
        timer = timeit.default_timer
        last_checkpoint_time = timer()
        wandb = self.args.wandb
        del self.args.wandb
        self.naction = gym.make(self.args.env).action_space.n
        self.initialize_ppo(self.naction)
        
        epochs, envs, observations, still_alive = 0, None, None, None
        total_steps = self.args.total_frames // (self.T*self.nactors)
        for i in range(1, total_steps+1):
            frac = 1.0 - (i - 1.0) / total_steps            # anneal learning rate linearly
            self.optimizer.param_groups[0]["lr"] = frac * self.args.learning_rate
            start_time = timer()
            stats, envs, observations, still_alive = self.algo_step(epochs, envs, observations, still_alive)
            sps = (self.T*self.nactors) / (timer()-start_time)
            epochs += self.epochs
            
            if timer() - last_checkpoint_time > self.args.min_to_save * 60:      # checkpoint to store model
                self.checkpoint()
                last_checkpoint_time = timer()
            
            if wandb:
                wandb.log({
                    "PPO/Policy Gradient Loss": stats['policy_loss'], "PPO/Value Loss": stats['value_loss'], "PPO/Mean Return": stats["mean_return"], 
                    'PPO/Epochs': epochs, "PPO/Entropy Bonus": stats['entropy_bonus'], "PPO/Surrogate Loss": stats['surrogate_loss'], 'PPO/Approx KL Divergence': stats['kl_divergence'],
                    "PPO/Clip Epsilon": self.clip_epsilon, "PPO/Learning Rate": self.optimizer.param_groups[0]['lr'], 'PPO/Global Step': self.global_step, "PPO/Samples Per Second": sps,
                }, step=self.global_step,
                )
                
            logging.info(f"Epoch {epochs:d} @ SPS {sps:.1f} | sur_loss {stats['surrogate_loss']:.3f} | value_loss {stats['value_loss']:.3f} | pg_loss {stats['policy_loss']:.3f} | KL: {stats['kl_divergence']:.3f}| mean_return {stats['mean_return']:.3f}")
            
            if epochs > 0 and epochs % self.args.eval_every == 0:        # perform validation step after some number of steps
                utils.validate_atari(
                    self.ac_network, self.args.env, self.args.render, nepisodes=1, wandb=wandb, 
                    device=self.args.device, video_folder=f'trained/videos/{self.args.run_id}'
                )
                self.ac_network.train()