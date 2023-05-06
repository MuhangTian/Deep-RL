import random
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import utils
import time


fig, ax = plt.subplots(1, 1)
# env = utils.SkipFrameWrapper(gym.make("ALE/MsPacman-v5"), skip=2)
env = utils.SkipFrameWrapper(gym.make("ALE/Assault-v5", render_mode="human"))
im = ax.imshow(env.reset()[0])
nactions = env.action_space.n
reward = 0
for _ in range(10000):
    action = random.randint(0,nactions-1)
    env_outputs = env.step(action)
    # if env_outputs[2]:
    #     break
    reward += env_outputs[1]
    env.render()
    if env_outputs[2]:
        break
    # img = env_outputs[0]
    # im.set_data(img)
    # fig.canvas.draw_idle()
    # plt.pause(0.2)
    time.sleep(0.05)
print(f"Total reward: {reward}")

# epsilon = 1
# size = 1_000_000
# decay = (epsilon - 0.1)/size
# for i in range(size+1_000_000):
#     epsilon = max(epsilon - decay, 0.1)
#     # epsilon = epsilon - decay
# print(epsilon)