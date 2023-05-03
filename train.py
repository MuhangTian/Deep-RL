import argparse
import logging

import gymnasium as gym
import numpy as np
import torch

import utils
from algo import VanillaPolicyGradient, ActorCritic, DeepQLearning
from model import PolicyNetwork, ActorNetworkCNN, ActorNetworkLSTM, QNetwork

num_threads = torch.get_num_threads()
ALGO = {
    'vpg': VanillaPolicyGradient,
    'a2c': ActorCritic,
    'dql': DeepQLearning,
}
MODEL = {
    'pn': PolicyNetwork,
    'an_cnn': ActorNetworkCNN,
    'an_lstm': ActorNetworkLSTM,
    'qn': QNetwork,
}
torch.set_num_threads(num_threads)
logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)
logging.info(f"Using {num_threads} threads for torch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ALE/MsPacman-v5", help="gym environment")
    parser.add_argument("--model", type=str, default="qn", help="model to use")
    parser.add_argument("--algo", type=str, default="dql", help="algorithm to use")
    parser.add_argument("--mode", default="train", choices=["train", "valid",], help="training or validation mode")
    parser.add_argument("--total_frames", default=1000000, type=int, help="total environment frames to train for")
    parser.add_argument("--batch_size", default=32, type=int, help="learner batch size.")
    parser.add_argument("--unroll_length", default=150, type=int, help="unroll length (time dimension)")
    parser.add_argument("--hidden_dim", default=256, type=int, help="policy net hidden dim")
    parser.add_argument("--discounting", default=0.99, type=float, help="discounting factor")
    parser.add_argument("--learning_rate", default=0.00025, type=float, help="Learning rate")
    parser.add_argument("--grad_norm_clipping", default=10.0, type=float, help="Global gradient norm clip.")
    parser.add_argument("--save_path", type=str, default='trained/dql.pt', help="save model here")
    parser.add_argument("--load_path", type=str, default='trained/dql.pt', help="load model from here")
    parser.add_argument("--min_to_save", default=5, type=int, help="save every this many minutes")
    parser.add_argument("--eval_every", default=2000, type=int, help="eval every this many updates")
    parser.add_argument("--render", action="store_true", default=False, help="render game-play at validation time")
    # ---------------------------------- DQN specific ----------------------------------
    parser.add_argument("--episodes", type=int, default=10000, help="episodes to train for")
    parser.add_argument("--replay_buffer_size", type=int, default=500000, help="size for replay buffer")
    parser.add_argument("--replay_start_size", type=int, default=50000, help="number of experiences to store before training")
    
    torch.manual_seed(59006)
    np.random.seed(59006)
    args = parser.parse_args()
    logging.info(args)
    
    if args.mode == "train":
        model = MODEL[args.model]
        algo = ALGO[args.algo](args, model)
        algo.train()
    else:
        assert args.load_path is not None, "must load saved model!"
        checkpoint = torch.load(args.load_path)
        saved_args = checkpoint["args"]
        env = gym.make(args.env)
        naction = env.action_space.n
        saved_args.start_nlives = env.ale.lives()
        del env        
        model = MODEL[args.model](naction, saved_args)              # initialize model with saved args
        model.load_state_dict(checkpoint["model_state_dict"])       # load model weights
        model = model

        utils.validate(model, render=args.render, nepisodes=100, mode='resize')      # evaluate with 100 episodes
