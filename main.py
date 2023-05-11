import argparse
import logging
import os
import random

import gymnasium as gym
import numpy as np
import torch

import utils
from algo import (ActorCritic, DeepQLearning, ProximalPolicyOptimization,
                  VanillaPolicyGradient)
from model import ActorNetworkCNN, ActorNetworkLSTM, PolicyNetwork, QNetwork, ActorCriticNetwork

num_threads = os.cpu_count()
ALGO = {
    'vpg': VanillaPolicyGradient,
    'a2c': ActorCritic,
    'dqn': DeepQLearning,
    'ppo': ProximalPolicyOptimization,
}
MODEL = {
    'pn': PolicyNetwork,
    'an_cnn': ActorNetworkCNN,
    'an_lstm': ActorNetworkLSTM,
    'qn': QNetwork,
    'ac': ActorCriticNetwork,
}
torch.set_num_threads(num_threads)
logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ALE/MsPacman-v5", help="gym environment")
    parser.add_argument("--model", type=str, default="ac", help="model to use")
    parser.add_argument("--algo", type=str, default="ppo", help="algorithm to use")
    parser.add_argument("--mode", default="train", choices=["train", "valid",], help="training or validation mode")
    parser.add_argument("--total_frames", default=1_000_000_000, type=int, help="total environment frames to train for")
    parser.add_argument("--batch_size", default=32, type=int, help="learner batch size.")
    parser.add_argument("--unroll_length", default=128, type=int, help="unroll length (time dimension)")
    parser.add_argument("--hidden_dim", default=512, type=int, help="policy net hidden dim")
    parser.add_argument("--discounting", default=0.99, type=float, help="discounting factor")
    parser.add_argument("--learning_rate", default=0.00025, type=float, help="Learning rate")
    parser.add_argument("--grad_norm_clipping", default=0.5, type=float, help="Global gradient norm clip.")
    parser.add_argument("--save_path", type=str, default='trained/dql.pt', help="save model here")
    parser.add_argument("--load_path", type=str, default='trained/dql.pt', help="load model from here")
    parser.add_argument("--min_to_save", default=5, type=int, help="save every this many minutes")
    parser.add_argument("--eval_every", default=90, type=int, help="eval every this many updates")
    parser.add_argument("--render", action="store_true", default=False, help="render game-play at validation time")
    # -------------------------------------------- Deep Q Learning Specific --------------------------------------------
    parser.add_argument("--episodes", type=int, default=250_000, help="episodes to train for")
    parser.add_argument("--replay_buffer_size", type=int, default=1_000_000, help="size for replay buffer")
    parser.add_argument("--replay_start_size", type=int, default=50_000, help="number of experiences to store before training")
    parser.add_argument("--frame_skipping_interval", type=int, default=4, help="number of frames to skip")
    parser.add_argument("--frames_per_state", type=int, default=4, help="number of consecutive frames that are treated as a state")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="exploration rate at the beginning")
    parser.add_argument("--epsilon_end", type=float, default=0.1, help="minimum exploration rate")
    parser.add_argument("--epsilon_decay_frames", type=int, default=1_000_000, help="number of frames to decay epsilon from epsilon_start to epsilon_end (linearly)")
    parser.add_argument("--target_update_frequency", type=int, default=10_000, help="number of learning network updates between target network updates")
    # ------------------------------------------------- PPO Specific -------------------------------------------------
    parser.add_argument("--lam", type=float, default=0.95, help="lambda to calculate GAE")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="epsilon to clip ratio for surrogate loss function")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="entropy coefficient for loss function")
    parser.add_argument("--value_coef", type=float, default=1, help="value coefficient for loss function")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs to train for")
    parser.add_argument("--total_epochs", type=int, default=500_000, help='total epochs to train for')
    parser.add_argument("--nactors", type=int, default=8, help="number of actors to use")
    parser.add_argument("--target_kl", type=float, default=0.015, help="number of actors to use")
    
    parser.add_argument("--nolog", action="store_true", default=True, help="disable wandb")
    
    torch.manual_seed(utils.SEED)
    np.random.seed(utils.SEED)
    random.seed(utils.SEED)   # python's random library, control randomness used in experience replay (only matter for DQL)
    args = parser.parse_args()
    try:        # use wandb to log stuff if we have it, else don't
        import wandb
        if args.nolog or args.render:
            wandb = False    
        else:
            wandb.init(project="RL-implementation", entity='muhang-tian')
    except:
        wandb = False
    args.device = utils.get_device()
    args.start_nlives = gym.make(args.env).ale.lives()
    args.wandb = wandb 
    logging.info(args)
    logging.info(f"Using {num_threads} threads for torch")
    
    if args.mode == "train":
        model = MODEL[args.model]
        algo = ALGO[args.algo](args, model)
        algo.train()
    else:
        assert args.load_path is not None, "must load saved model!"
        checkpoint = torch.load(args.load_path, map_location=torch.device('cpu'))
        saved_args = checkpoint["args"]
        env = gym.make(args.env)
        naction = env.action_space.n
        saved_args.start_nlives = env.ale.lives()
        del env        
        model = MODEL[args.model](naction, saved_args)              # initialize model with saved args
        model.load_state_dict(checkpoint["model_state_dict"])       # load model weights
        model = model

        utils.validate(model, args, render=args.render, nepisodes=100, mode='resize')      # evaluate with 100 episodes
