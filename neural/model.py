# neural/model.py
import torch
import torch.nn as nn

class SimpleSplitNet(nn.Module):
    def __init__(self, input_dim=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # output: normalized split position along an axis
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (batch, input_dim) aggregated scene descriptor
        return self.net(x).squeeze(-1)  # (batch,)
