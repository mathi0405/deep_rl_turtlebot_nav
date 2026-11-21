import torch, torch.nn as nn
class CrossQActor(nn.Module):
    def __init__(self, s_dim=26, a_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, a_dim), nn.Tanh())
    def forward(self, s): return self.net(s)

class CrossQCritic(nn.Module):
    def __init__(self, s_dim=26, a_dim=2):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(s_dim + a_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(s_dim + a_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        return self.q1(sa), self.q2(sa)
