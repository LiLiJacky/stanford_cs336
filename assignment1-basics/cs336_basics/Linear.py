import torch
import torch.nn as nn
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                  device:torch.device | None = None, dtype: torch.dtype | None=None):
        """
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        w_init = self.initialize_weights(in_features, out_features, factory_kwargs)
        self.weight = nn.Parameter(w_init)
    
    def initialize_weights(self, input_dim: int, output_dim: int, factory_kwargs: dict) -> torch.Tensor:
        # Initialize weights with Kaiming uniform initialization
        W = torch.empty(output_dim, input_dim, **factory_kwargs)
        mean = 0
        std = np.sqrt(2 / (input_dim + output_dim))

        nn.init.trunc_normal_(W, mean, std, -3*std, 3*std)
        return W
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T