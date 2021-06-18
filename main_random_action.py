import torch
import random
import numpy as np
import torch.optim as optim

from config import *
from env import Env
from copy import deepcopy


def test(test_env):
    obs = test_env.reset()
    done = False
    episode_reward = 0.
    while not done:
        action = random.randint(0, 15)
        obs, r, done, _ = test_env.step(action)
        episode_reward += r
        if done:
            # print("test episode reward :", episode_reward)
            return episode_reward


# Make Env
env, test_env = Env(), Env()

# for _ in range(100):
#     obs = env.reset()
#     done = False
#     while not done:
#         obs, r, done, _ = env.step(random.randint(0, n_rows**2 - 1))
#         if done:
#             print("done, reward :", r.item())


episode_reward = 0
best_test_reward = -999999
obs = env.reset()
for t in range(total_step):
    test_rewards = []
    for _ in range(test_batch_size):
        test_rewards.append(test(test_env=test_env))
    avg_test_reward = sum(test_rewards) / len(test_rewards)
    print("average test reward :", avg_test_reward.item())
    with open('results/test_reward_random_action.txt', 'a') as f:
        f.write(str(avg_test_reward.item()))
        f.write('\n')

    
    


