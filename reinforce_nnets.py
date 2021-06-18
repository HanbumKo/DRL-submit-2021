import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical
# from pygcn.models import GCN
from graph_encoder import GraphAttentionEncoder


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, device):
        super(Policy, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device

        self.embedder = GraphAttentionEncoder(
            n_heads=8,
            embed_dim=128,
            n_layers=3,
            node_dim=2,
            normalization='batch'
        ).to(device)
        # self.gcn = GCN(nfeat=2,
        #     nhid=128,
        #     nclass=128,
        #     dropout=0.5)

        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, act_dim)

    def forward(self, x):

        _, x = self.embedder(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = F.softmax(x, dim=-1)
        x = x.squeeze(0)

        return x
    
    def get_action_logprob(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device)
        output = self.forward(obs)
        categorical = Categorical(output)
        action = categorical.sample()
        logprob = categorical.log_prob(action)

        return action.item(), logprob
    
    def get_best_action(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device)
        output = self.forward(obs)
        action = torch.argmax(output)
        return action.item()

