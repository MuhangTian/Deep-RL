import argparse
import logging
import timeit

import gymnasium as gym
import numpy as np
import torch

import utils
from algo import pg_step
from model import PolicyNetwork

num_threads = torch.get_num_threads()
torch.set_num_threads(num_threads)
logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)
logging.info(f"Using {num_threads} threads for torch")

def train(args):
    T = args.unroll_length
    B = args.batch_size

    args.device = torch.device("cpu")
    env = gym.make(args.env)
    naction = env.action_space.n
    args.start_nlives = env.ale.lives()
    del env

    model = PolicyNetwork(naction, args)

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
    last_checkpoint_time = timer()
    envs, observations, prev_state = None, None, None
    frame = 0
    while frame < args.total_frames:
        start_time = timer()
        start_frame = frame
        stats, envs, observations, prev_state = pg_step(
            args, frame, model, optimizer, scheduler, envs, observations, prev_state, bsz=B)
        frame += T*B # here steps means number of observations
        if timer() - last_checkpoint_time > args.min_to_save * 60:
            checkpoint()
            last_checkpoint_time = timer()

        sps = (frame - start_frame) / (timer() - start_time)
        logging.info("Frame {:d} @ {:.1f} FPS: loss {:.3f} | mean_return {:.3f}".format(
          frame, sps, stats['pg_loss'], stats["mean_return"]))
        
        if frame > 0 and frame % (args.eval_every*T*B) == 0:
            utils.validate(model, render=args.render)
            model.train()
    
    print(f"{'='*10} TRAINING FINISHED {'='*10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="ALE/MsPacman-v5", help="gym environment")
    parser.add_argument("--mode", default="train", choices=["train", "valid",], 
                        help="training or validation mode")
    parser.add_argument("--total_frames", default=1000000, type=int, 
                        help="total environment frames to train for")
    parser.add_argument("--batch_size", default=8, type=int, help="learner batch size.")
    parser.add_argument("--unroll_length", default=80, type=int, 
                        help="unroll length (time dimension)")
    parser.add_argument("--hidden_dim", default=256, type=int, help="policy net hidden dim")
    parser.add_argument("--discounting", default=0.99, type=float, help="discounting factor")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--grad_norm_clipping", default=10.0, type=float,
                        help="Global gradient norm clip.")
    parser.add_argument("--save_path", type=str, default='trained/baseline_default_params.pt', help="save model here")
    parser.add_argument("--load_path", type=str, default='trained/baseline_default_params.pt', help="load model from here")
    parser.add_argument("--min_to_save", default=2, type=int, help="save every this many minutes")
    parser.add_argument("--eval_every", default=50, type=int, help="eval every this many updates")
    parser.add_argument("--render", action="store_true", default=False, help="render game-play at validation time")
    
    torch.manual_seed(59006)
    np.random.seed(59006)
    args = parser.parse_args()
    logging.info(args)
    if args.mode == "train":
        train(args)
    else:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path)
        saved_args = checkpoint["args"]
        env = gym.make(args.env)
        naction = env.action_space.n
        saved_args.start_nlives = env.ale.lives()
        del env        
        model = PolicyNetwork(naction, saved_args)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model
        args = saved_args

        utils.validate(model, args, nepisodes=100)
