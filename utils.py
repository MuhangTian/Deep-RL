import logging
import random
import time
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T
from gymnasium.core import Env

logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


def get_done(env, start_lives):
    '''to determine whether an episode is done'''
    return env.ale.lives() != start_lives

def get_device():
    if torch.cuda.is_available():
        print('************* Using CUDA *************')
        return torch.device('cuda')
    elif torch.has_mps:
        print('************* Using MPS *************')
        return torch.device('mps')
    else:
        print('************* Using CPU *************')
        return torch.device('cpu')

def preprocess_observation(obs, mode='simple', new_size=(84, 84)):
    """
    obs - a 210 x 160 x 3 ndarray representing an atari frame
    returns:
      a 3 x 210 x 160 normalized pytorch tensor
    """
    if mode == 'simple':
        return torch.from_numpy(obs).permute(2, 0, 1)/255.0
    elif mode == 'resize':
        image_tensor = torch.tensor(obs).float()
        image_tensor = image_tensor.permute(2, 0, 1)
        # Resize image using torch.nn.functional.interpolate
        image_tensor = image_tensor.unsqueeze(0) # Add a batch dimension
        transform = T.Compose([
            T.Grayscale(),              # Convert to grayscale to save memory
            T.Resize(new_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True)  # resize smaller to save memory
        ])
        resized_image_tensor = transform(image_tensor)
        resized_image_tensor = resized_image_tensor.squeeze() # Remove the extra dimension
        resized_image_tensor /= 255.0   # Normalize the pixel values to [0, 1] range

        return resized_image_tensor.unsqueeze(0)

def validate(model, args, render:bool=False, nepisodes=5, wandb=False, mode='simple'):
    """
    Evaluates the performance of the given agent on the Ms. Pac-Man Atari game using a specified number of episodes, and returns the average reward and number of steps taken per episode.

    Parameters
    ----------
    model : object
        A reinforcement learning model object with a `get_action` method.
    args : argparse.Namespace
        argumetns passed in from command line
    render : bool, optional
        Whether to render the game during validation. Defaults to False.
    nepisodes : int, optional
        The number of episodes to play for evaluation. Defaults to 5.
    wandb : bool, optional
        Whether to log results to Weights & Biases. Defaults to False.
    mode : str, optional
        The image preprocessing mode to use before feeding image to the model. Can be 'simple' or 'resize'. Defaults to 'simple'.
    """
    assert hasattr(model, "get_action")
    torch.manual_seed(590060)
    np.random.seed(590060)
    # NOTE: seed for python's random library is set in train.py, it's not set in here since this function will be called
    # during training, and I want to avoid to setting seeds since that may create patterns in samples generated in experience replay
    
    model.eval()        # turn into eval mode
    if render:
        nepisodes = 1       # only render one episode if render is True

    steps_alive, reward_arr = [], []        # to store each episode's reward and steps taken
    for i in range(nepisodes):
        logging.info(f"Validating episode {i+1}...")
        render_mode = "human"  if render else None
        env = gym.make(args.env, render_mode=render_mode)      # NOTE: modify render functionality for better graphics
        obs = env.reset(seed=590060+i)[0]       # use a different seed for each separate episode
        
        observation = preprocess_observation(obs, mode=mode).unsqueeze(0).unsqueeze(0).to(args.device)      # 1 x 1 x ic x iH x iW
        prev_state = None
        step, ep_total_reward = 0, 0
        # play until the agent dies or we exceed 50000 observations
        while env.ale.lives() == 3 and step < 50000:
            action, prev_state = model.get_action(observation, prev_state)
            env_output = env.step(action)
            ep_total_reward += env_output[1]
            observation = preprocess_observation(env_output[0], mode=mode).unsqueeze(0).unsqueeze(0).to(args.device)
            step += 1
            if render:
                time.sleep(0.02)        # sleep for 0.02 seconds to slow down the rendering
        steps_alive.append(step)
        reward_arr.append(ep_total_reward)
    
    if wandb:           # log into wandb if using it
        wandb.log({"Mean Reward (Validation)": np.mean(reward_arr),
                   'std Reward (Validation)': np.std(reward_arr)})
    
    logging.info(f"{'-'*10} BEGIN VALIDATION {'-'*10}")
    logging.info("Steps taken over each of {:d} episodes: {}".format(
        nepisodes, ", ".join(str(step) for step in steps_alive)))
    logging.info("Total return after {:d} episodes: {:.3f}".format(nepisodes, np.sum(reward_arr)))
    logging.info(f"Mean return for each episode: {np.mean(reward_arr):.3f}, (std: {np.std(reward_arr):.3f})")
    logging.info(f"{'-'*10} END VALIDATION {'-'*10}")


class ReplayBuffer:
    '''a simple replay buffer implemented using deque'''
    def __init__(self, size: int) -> None:
        self.memory = deque([], maxlen=size)
    
    def push(self, transition: Transition):
        '''add transitions to the replay buffer'''
        self.memory.append(Transition(*transition))
    
    def sample(self, bsz: int):
        '''sample bsz number of transitions from the replay buffer'''
        return random.sample(self.memory, bsz)
    
    def __len__(self):
        return len(self.memory)


class SkipFrameWrapper(gym.Wrapper):
    '''
    Wrapper to skip frames (which is same as performing the same action for multiple frames)
    This has two advantages:
    - make more efficient use of frames, since most frames are redundant
    - reduce computation need for forward pass, since we only need one forward pass (to decide an action) to play four times
      (whereas doing four forward passes to decide separate actions need more computation)
    '''
    def __init__(self, env: Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        self.start_lives = env.ale.lives()
    
    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            env_output = self.env.step(action)
            total_reward += env_output[1]
            done = get_done(self.env, self.start_lives)
            obs = env_output[0]
            if done:
                break
        return obs, total_reward, done
        