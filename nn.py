import torch
import torch.nn as nn
from torch.nn.functional import relu


class BackBone(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)


class CartPoleNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.value = nn.Sequential(
            BackBone(input_dim, hidden_dim), nn.Linear(hidden_dim, 1)
        )
        self.policy = nn.Sequential(
            BackBone(input_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.policy(x), self.value(x)


class CheetahNet(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=64, output_dim=6):
        super().__init__()

        # Shared base
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Policy head
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(
            torch.zeros(output_dim)
        )  # learnable, state-independent

        # Value head
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        mean = self.mean_layer(h)
        value = self.value_layer(h).squeeze(-1)
        return mean, self.log_std, value
