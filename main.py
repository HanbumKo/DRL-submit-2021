import torch
import random
import numpy as np
import torch.optim as optim

from reinforce_nnets import Policy
from replay_buffer import ReplayBuffer
from config import *
from env import Env
from copy import deepcopy


def train(logprobs, returns, optim):
    optim.zero_grad()
    # Cumulate gradients
    for ret, logprob in zip(returns, logprobs):
        j = -1 * logprob * ret
        j.backward()
    optim.step()

def test(test_env, test_policy_model):
    obs = test_env.reset()
    done = False
    episode_reward = 0.
    while not done:
        # action = test_policy_model.get_best_action(obs)
        action, _ = test_policy_model.get_action_logprob(obs)
        obs, r, done, _ = test_env.step(action)
        episode_reward += r
        if done:
            # print("test episode reward :", episode_reward)
            return episode_reward


# Make Env
env, test_env = Env(), Env()

# shape
obs_shape = tuple([n_nodes, 2])
act_shape = tuple([n_rows**2 - 1])
obs_dim = obs_shape[0]
act_dim = act_shape[0]

policy_model = Policy(obs_dim, act_dim, device).to(device)

# Optimizer
optimizer = optim.Adam(policy_model.parameters(), lr=0.01)

# list for saving results
rewards = []

# list for saving return & logprobs
logprobs = []
returns = []


episode_reward = 0
best_test_reward = -999999
obs = env.reset()
for t in range(total_step):
    if t % 1000 == 0:
        test_rewards = []
        for _ in range(test_batch_size):
            test_rewards.append(test(test_env=test_env, test_policy_model=policy_model))
        avg_test_reward = sum(test_rewards) / len(test_rewards)
        print("average test reward :", avg_test_reward.item())
        with open('results/test_reward_reinforce_2.txt', 'a') as f:
            f.write(str(avg_test_reward.item()))
            f.write('\n')

    action, logprob = policy_model.get_action_logprob(obs)
    next_obs, rew, done, _ = env.step(action)
    logprobs.append(logprob)
    returns.append(rew)
    episode_reward += rew
    obs = next_obs

    if done:
        # print(episode_reward.item())
        # Save the best model so far
        if best_test_reward <= episode_reward:
            best_test_reward = episode_reward
            torch.save(policy_model.state_dict(), "results/REINFORCE_policy_model_2.pt")

        # complete returns
        for i in range(len(returns)-2, -1, -1):
            returns[i] += returns[i+1]*gamma
        rewards.append(episode_reward)
        train(logprobs, returns, optimizer)
        logprobs = []
        returns = []
        obs = env.reset()
        episode_reward = 0
    


