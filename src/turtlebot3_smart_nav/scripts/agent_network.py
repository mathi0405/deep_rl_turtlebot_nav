#!/usr/bin/env python3
import torch
import torch.nn as nn

class TurtleBotDQN(nn.Module):
    def __init__(self, state_dimension=26, action_dimension=5):
        super(TurtleBotDQN, self).__init__()
        
        self.feature_network = nn.Sequential(
            nn.Linear(state_dimension, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.action_head = nn.Linear(64, action_dimension)

    def forward(self, state_tensor):
        features = self.feature_network(state_tensor)
        return self.action_head(features)
