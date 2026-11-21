import torch, torch.nn as nn
from torch.distributions import Normal
class SACActor(nn.Module):
    def __init__(self, s_dim=26, a_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, a_dim)
        self.log_std = nn.Linear(256, a_dim)
    def forward(self, s):
        x = torch.relu(self.fc2(torch.relu(self.fc1(s))))
        std = torch.exp(torch.clamp(self.log_std(x), -20, 2))
        return torch.tanh(Normal(self.mean(x), std).rsample())

class SACCritic(nn.Module):
    def __init__(self, s_dim=26, a_dim=2):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(s_dim + a_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(s_dim + a_dim, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.q1(sa), self.q2(sa)
