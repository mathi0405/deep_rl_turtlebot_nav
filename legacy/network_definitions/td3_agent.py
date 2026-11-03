import torch, torch.nn as nn
class TD3Actor(nn.Module):
    def __init__(self, s_dim=26, a_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, a_dim), nn.Tanh())
    def forward(self, s): return self.net(s)

class TD3Critic(nn.Module):
    def __init__(self, s_dim=26, a_dim=2):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(s_dim + a_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(s_dim + a_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.q1(sa), self.q2(sa)
