import torch.nn as nn
class TurtleBotDQN(nn.Module):
    def __init__(self, state_dim=26, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))
    def forward(self, state): return self.net(state)
