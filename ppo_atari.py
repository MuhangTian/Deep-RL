import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from torch.distributions.categorical import Categorical

import utils
import wandb

# from torch.utils.tensorboard import SummaryWriter



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="ALE/Breakout-v5",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="RL-implementation",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="muhang-tian",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


# def make_env(gym_id, seed, idx, capture_video, run_name):
#     def thunk():
#         env = gym.make(gym_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         env = NoopResetEnv(env, noop_max=30)
#         env = MaxAndSkipEnv(env, skip=4)
#         env = EpisodicLifeEnv(env)
#         if "FIRE" in env.unwrapped.get_action_meanings():
#             env = FireResetEnv(env)
#         env = ClipRewardEnv(env)
#         env = gym.wrappers.ResizeObservation(env, (84, 84))
#         env = gym.wrappers.GrayScaleObservation(env)
#         env = gym.wrappers.FrameStack(env, 4)
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk

def make_env(env_name, seed):
    assert 'v5' in env_name
    def make():
        env = AtariGameEnv(env_name)
        env.seed(seed)
        return env
    return make

class AtariGameEnv(gym.Wrapper):
    def __init__(self, env_name: str, num_stack_frame: int=4, terminal_on_life_loss: bool=True, render_mode: str=None, rescale_reward=True) -> None:
        assert 'v5' in env_name, 'Please use envs with v5!'
        env = gym.make(env_name, frameskip=1, repeat_action_probability=0.0, render_mode=render_mode)
        if env.unwrapped.get_action_meanings()[1] == 'FIRE' and len(env.unwrapped.get_action_meanings()) >= 3:
            env = FireResetEnv(env)
        if rescale_reward:
            env = ClipRewardEnv(env)        # clip reward to {-1, 0, 1}
        env = AtariPreprocessing(env, scale_obs=True, terminal_on_life_loss=terminal_on_life_loss)           # normalize and rescale, end episode if life lost
        env = FrameStack(env, num_stack_frame)
        super().__init__(env)
    
    def reset(self, **kwargs) -> tuple:
        env_output = self.env.reset(**kwargs)
        
        return env_output
    
    def step(self, action: int) -> tuple:
        env_output = self.env.step(action)
        
        return env_output

class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reward(self, reward: float) -> float:
        return np.sign(reward)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE', 'Please use envs with FIRE action!'
        assert len(env.unwrapped.get_action_meanings()) >= 3, 'Please use envs with at least 3 actions!'
    
    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        env_output = self.env.step(1)
        if env_output[2]:
            self.env.reset(**kwargs)
        env_output = self.env.step(2)
        if env_output[2]:
            self.env.reset(**kwargs)
            
        return env_output[0], env_output[-1]
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def validate_atari(model, env_name, render, nepisodes, wandb=None, device='cpu', rescale_reward=False):
        assert hasattr(model, "get_action")
        torch.manual_seed(1)
        np.random.seed(1)
        model.eval()        # turn into eval mode
        if render:
            nepisodes = 1       # only render one episode if render is True

        steps_alive, reward_arr = [], []        # to store each episode's reward and steps taken
        for i in range(nepisodes):
            print(f"Validating episode {i+1}...")
            render_mode = "human"  if render else None
            env = AtariGameEnv(env_name, render_mode=render_mode, rescale_reward=rescale_reward)
            env_output = env.reset(seed=1+i)
            obs = torch.tensor(env_output[0])
            env_output = list(env_output)
            env_output[0] = obs
            env_output = tuple(env_output)
            observation = env_output[0].to(device)       # use a different seed for each separate episode
            prev_state = None
            step, ep_total_reward, done = 0, 0, False
            # play until the agent dies or we exceed 50000 observations
            while not done and step < 50000:
                action, prev_state = model.get_action(observation.reshape(-1, 4, 84, 84), prev_state)
                env_output = env.step(action)
                obs = torch.tensor(env_output[0])
                env_output = list(env_output)
                env_output[0] = obs
                env_output = tuple(env_output)
                ep_total_reward += env_output[1]
                done = env_output[2]
                observation = env_output[0].to(device)
                step += 1
            steps_alive.append(step)
            reward_arr.append(ep_total_reward)
        
        if wandb:           # log into wandb if using it
            wandb.log({"Validation/Mean Reward": np.mean(reward_arr), 'Validation/std Reward': np.std(reward_arr)})
        
        print(f"{'-'*10} BEGIN VALIDATION {'-'*10}")
        print("Steps taken over each of {:d} episodes: {}".format(
            nepisodes, ", ".join(str(step) for step in steps_alive)))
        print("Total return after {:d} episodes: {:.3f}".format(nepisodes, np.sum(reward_arr)))
        print(f"Mean return for each episode: {np.mean(reward_arr):.3f}, (std: {np.std(reward_arr):.3f})")
        print(f"{'-'*10} END VALIDATION {'-'*10}")


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
    def get_action(self, x, prev_state):
        '''get action, since policy is pi(a|s), it's stochastic, so we call sample() based on parameterized distribution'''
        with torch.no_grad():
            action, _, _, _ = self.get_action_and_value(x)
        
        return action, prev_state
    
def preprocess_output(env_output):
    obs = torch.tensor(env_output[0])
    env_output = list(env_output)
    env_output[0] = obs
        
    return tuple(env_output)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            save_code=True,
        )
    else:
        wandb = None
    # writer = SummaryWriter(f"runs/{run_name}")
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    env_output = envs.reset()
    next_obs = preprocess_output(env_output)[0].to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            env_outputs = envs.step(action)
            rewards[step] = torch.tensor(env_outputs[1]).to(device).view(-1)
            next_obs, next_done = torch.Tensor(env_outputs[0]).to(device), torch.Tensor(env_outputs[2]).to(device)

            # for item in info:
                # if "episode" in item.keys():
                #     print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    # writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    # writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    # break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if wandb:
            wandb.log({
                'charts/lr': optimizer.param_groups[0]['lr'],
                'losses/value_loss': v_loss.item(),
                'losses/policy_loss': pg_loss.item(),
                'losses/entropy': entropy_loss.item(),
                'losses/approx_kl': approx_kl.item(),
                'losses/old_approx_kl': old_approx_kl.item(),
                'losses/clipfrac': np.mean(clipfracs),
                'losses/explained_variance': explained_var,
                'sps': int(global_step / (time.time() - start_time)) 
            })
        validate_atari(agent, args.gym_id, None, nepisodes=1, wandb=wandb, device=device)
        agent.train()
        # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    # writer.close()
