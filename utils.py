import logging
import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T

logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

try:
    import matplotlib.pyplot as plt
    can_render = True
except:
    logging.warning("Cannot import matplotlib; will not attempt to render")
    can_render = False

def get_done(env):
    '''to determine whether an episode is done'''
    return env.ale.lives() != 3

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
        

def validate(model, render:bool=False, nepisodes=5, wandb=False, mode='simple'):
    assert hasattr(model, "get_action")
    torch.manual_seed(590060)
    np.random.seed(590060)
    
    model.eval()        # turn into eval mode
    render = render and can_render

    if render:
        nepisodes = 1
        fig, ax = plt.subplots(1, 1)

    steps_alive, reward_arr = [], []        # to store each episode's reward and steps taken
    for i in range(nepisodes):
        env = gym.make("ALE/MsPacman-v5")
        obs = env.reset(seed=590060+i)[0]       # use a different seed for each separate episode
        if render:
            im = ax.imshow(obs)
        # NOTE: this can be changed into preprocess_observation() which keeps the original dimensionality
        observation = preprocess_observation( # 1 x 1 x ic x iH x iW
            obs, mode=mode).unsqueeze(0).unsqueeze(0)
        prev_state = None
        step, ep_total_reward = 0, 0
        # play until the agent dies or we exceed 50000 observations
        while env.ale.lives() == 3 and step < 50000:
            action, prev_state = model.get_action(observation, prev_state)
            env_output = env.step(action)
            ep_total_reward += env_output[1]
            if render:
                img = env_output[0]
                im.set_data(img)
                fig.canvas.draw_idle()
                plt.pause(0.1)
            observation = preprocess_observation(
                env_output[0], mode=mode).unsqueeze(0).unsqueeze(0)
            step += 1
        steps_alive.append(step)
        reward_arr.append(ep_total_reward)
    
    if wandb:           # log into wandb if using it
        wandb.log({"validation/mean_return": np.mean(reward_arr),
                   'validation/std_return': np.std(reward_arr)})
    
    logging.info(f"{'-'*10} BEGIN VALIDATION {'-'*10}")
    logging.info("Steps taken over each of {:d} episodes: {}".format(
        nepisodes, ", ".join(str(step) for step in steps_alive)))
    logging.info("Total return after {:d} episodes: {:.3f}".format(nepisodes, np.sum(reward_arr)))
    logging.info(f"Mean return for each episode: {np.mean(reward_arr):.3f}, (std: {np.std(reward_arr):.3f})")
    logging.info(f"{'-'*10} END VALIDATION {'-'*10}")
    

class ReplayBuffer:
    def __init__(self, size) -> None:
        self.memory = deque([], maxlen=size)
    
    def push(self, transition):
        self.memory.append(Transition(*transition))
    
    def sample(self, bsz):
        return random.sample(self.memory, bsz)
    
    def __len__(self):
        return len(self.memory)