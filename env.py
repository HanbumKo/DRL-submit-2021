import copy
import torch
import numpy as np

from config import *
from nnets import Actor, Critic
from gym import spaces


class Env(object):
    def __init__(self):
        self.observation_space = spaces.Box(low=0., high=1., shape=(40, ))
        self.action_space = spaces.Discrete(n_rows ** 2)
        self.n_row_zone = n_rows # assuming squre
        self.n_total_zone = self.n_row_zone ** 2
        self.zone_len = 1 / self.n_row_zone

        # Load trained TSP RL agent
        self.actor = Actor(hidden_dim, mode='drone')
        self.critic = Critic(hidden_dim)
        self.actor.load_state_dict(torch.load(actor_weight_path, map_location=device))
        self.critic.load_state_dict(torch.load(critic_weight_path, map_location=device))
        self.actor.eval()
        self.critic.eval()

        self.dummy_data = [torch.FloatTensor(n_nodes, 2).uniform_(0, 1).to(device) for i in range(1)]
        self.dummy_data = torch.cat(self.dummy_data, 0).view(-1, n_nodes, 2)
        self.tspEnv = TSPEnv(self.dummy_data)
    
    def reset(self):
        self.data = torch.zeros_like(self.dummy_data, dtype=torch.float32)
        self.data[0, 0, :] = torch.rand(2) # select first tensor randomly
        self.node_i = 1

        return self._get_state()
    
    def step(self, zone):
        row = zone // self.n_row_zone
        col = zone % self.n_row_zone

        row_low, row_high = self.zone_len * row, self.zone_len * (row+1)
        col_low, col_high = self.zone_len * col, self.zone_len * (col+1)

        coordinate_x = torch.FloatTensor(1).uniform_(col_low, col_high)
        coordinate_y = torch.FloatTensor(1).uniform_(row_low, row_high)
        coordinates = torch.Tensor([coordinate_x[0], coordinate_y[0]]).to(device)

        # if isinstance(coordinates, np.ndarray):
        #     coordinates = torch.as_tensor(coordinates, dtype=torch.float32).to(device)
        assert coordinates[0] >= 0. and coordinates[0] <= 1.
        assert coordinates[1] >= 0. and coordinates[1] <= 1.
        self.data[0, self.node_i, :] = coordinates
        self.node_i += 1
        done = False
        cost = 0.
        info = {}

        if self.node_i >= n_nodes:
            assert self.node_i == n_nodes
            done = True
            cost = self.solve_and_get_cost()

        return self._get_state(), cost, done, info
    
    def solve_and_get_cost(self):
        self.actor.eval()
        self.tspEnv.input_data = self.data

        state, mask = self.tspEnv.reset()

        with torch.no_grad():
            static_hidden = self.actor.emd_stat(self.data).permute(0, 2, 1)
            hx = torch.zeros(1, self.tspEnv.batch_size, hidden_dim).to(device)
            cx = torch.zeros(1, self.tspEnv.batch_size, hidden_dim).to(device)
            last_hh = (hx, cx)

            terminated = torch.zeros(self.tspEnv.batch_size).to(device)
            decoder_input = static_hidden[:, :, self.tspEnv.n_nodes-1].unsqueeze(2)

            for _ in range(n_nodes):
                idx, prob, logp, last_hh = self.actor.forward(static_hidden, state, 
                                                              decoder_input, last_hh, 
                                                              terminated, mask)
                state, mask, terminated = self.tspEnv.step(idx)
                decoder_input =  torch.gather(static_hidden, 2, idx.view(-1, 1, 1).expand(self.tspEnv.batch_size, hidden_dim, 1)).detach()

        cost = copy.copy(self.tspEnv.R)

        return cost.mean() - 3.8

    def _get_state(self):
        # return self.data[:, :self.node_i, :].cpu().numpy()
        return self.data[:, :, :].cpu().numpy()

