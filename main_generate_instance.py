import torch
import random
import numpy as np
import torch.optim as optim

from reinforce_nnets import Policy
from replay_buffer import ReplayBuffer
from config import *
from env import Env
from copy import deepcopy
import matplotlib.pyplot as plt


# Make Env
env = Env()

# for _ in range(100):
#     obs = env.reset()
#     done = False
#     while not done:
#         obs, r, done, _ = env.step(random.randint(0, n_rows**2 - 1))
#         if done:
#             print("done, reward :", r.item())


# shape
obs_shape = tuple([n_nodes, 2])
act_shape = tuple([n_rows**2 - 1])
obs_dim = obs_shape[0]
act_dim = act_shape[0]

policy_model = Policy(obs_dim, act_dim, device).to(device)
policy_model.load_state_dict(torch.load("results/REINFORCE_policy_model.pt"))


obs = env.reset()
i = 0
while i <= 11:
    action, _ = policy_model.get_action_logprob(obs)
    next_obs, rew, done, _ = env.step(action)
    obs = next_obs

    if done:
        print("generating ...", i)
        xs = obs[0, :, 0]
        ys = obs[0, :, 1]
        plt.scatter(xs, ys, color='b')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.savefig("generated_instance/instance_" + str(i) + ".png")
        # plt.savefig("results/instance_" + str(upto) + ".png")
        plt.cla()
        obs = env.reset()
        i += 1
        continue


    





