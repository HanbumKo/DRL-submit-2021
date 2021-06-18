import matplotlib.pyplot as plt
import numpy as np


with open("results/test_reward_reinforce_2.txt") as f:
    content = f.readlines()
rewards_ddpg = [float(x.strip()) for x in content]

with open("results/test_reward_random_action.txt") as f:
    content = f.readlines()
rewards_random = [float(x.strip()) for x in content]


plt.plot(rewards_ddpg[:800], color='b', label='REINFORCE')
plt.plot(rewards_random[:800], color='k', label='random')
plt.xlabel("time step")
plt.ylabel("Reward")
plt.legend()

# plt.xlim(0.0, 1.0)
# plt.ylim(-4, 2)

# plt.title("[0, 1] Uniformly generated instance")

plt.savefig("results/REINFORCE_test_reward.png")

print("done")