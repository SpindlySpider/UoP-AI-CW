import torch
import torch.nn as nn
from typing import List


class TorchNet(nn.Module):
    def __init__(self, input_size: int = 24, hidden_sizes: List[int] = [128, 64, 32], output_size: int = 24, activation: str = 'sigmoid'):
        super().__init__()
        layers = []
        in_size = input_size

        act_map = {
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'linear': nn.Identity,
        }

        Act = act_map.get(activation, nn.Sigmoid)

        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(Act())
            in_size = h

        layers.append(nn.Linear(in_size, output_size))

        # final activation is left linear since original network predicts continuous angles
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expect input shape (batch_size, features)
        return self.net(x)
