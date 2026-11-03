import torch, torch.nn as nn
class DDPGActor(nn.Module):
    def __init__(self, s_dim=26, a_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, a_dim), nn.Tanh())
    def forward(self, s): return self.net(s)

class DDPGCritic(nn.Module):
    def __init__(self, s_dim=26, a_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim + a_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, s, a): return self.net(torch.cat([s, a], dim=1))
