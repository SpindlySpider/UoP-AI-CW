import torch
import torch.nn as nn
from typing import List


class TorchNet(nn.Module):
    def __init__(self, input_size: int = 24, hidden_sizes: List[int] = [128, 64, 32], output_size: int = 24, activation: str = 'sigmoid'):
        """
        PyTorch implementation matching the original Neural_network class.
        
        Architecture matches original nn:
        - self.layers = [num_inputs] + hidden_layers + [num_outputs]
        - For [24, 128, 64, 32, 24], we have 4 weight matrices:
          * W0: 24→128, W1: 128→64, W2: 64→32, W3: 32→24
        - Each weight matrix is followed by an activation function (sigmoid)
        - Total: 4 activations (one per layer transition, including output)
        """
        super().__init__()
        
        # Store layer configuration (matches original self.layers)
        self.layers = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(self.layers)
        
        act_map = {
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'linear': nn.Identity,
        }

        Act = act_map.get(activation, nn.Sigmoid)

        # Build network: for each layer transition, add Linear + Activation
        # This creates len(self.layers)-1 weight matrices with activations
        layers = []
        for i in range(len(self.layers) - 1):
            # Add linear transformation from layer i to layer i+1
            layers.append(nn.Linear(self.layers[i], self.layers[i+1]))
            # Add activation function (applied to all layers including output)
            layers.append(Act())

        self.net = nn.Sequential(*layers)
        
        # Initialize weights and biases to match original nn implementation
        # Original: weights = np.random.rand() - 0.5 (range: [-0.5, 0.5))
        # Original: bias = np.zeros() - 0.5 (value: -0.5)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights and biases to match original Neural_network.
        
        Original initialization:
        - weights[layer_idx] = np.random.rand(layers[i], layers[i+1]) - 0.5
        - bias[layer_idx] = np.zeros((1, layers[i+1])) - 0.5
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Initialize weights: uniform [0, 1) - 0.5 = uniform [-0.5, 0.5)
                nn.init.uniform_(module.weight, -0.5, 0.5)
                # Initialize bias: constant -0.5
                nn.init.constant_(module.bias, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        Matches original feed_forward:
        - Input goes through each weight matrix followed by activation
        - Returns final activated output
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        return self.net(x)
