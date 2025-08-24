import torch
import torch.nn as nn
import numpy as np

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Args:
            theta (float): Theta value for the RoPE
            d_k (int): Dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted 
            device (torch.device): | None = None Device to store the buffer on
        """
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.rotation_matrix_table = self.generate_rotation_matrix(theta, d_k, max_seq_len)
        self.register_buffer("rotation_matrix", self.rotation_matrix_table, persistent=False)

    def generate_rotation_matrix(self, theta: float, d_k: int, max_seq_len: int):
        rotation_matrix_table = torch.zeros(max_seq_len, d_k, d_k)
        for i in range(max_seq_len):
            blocks = [self.generate_rotation_block(theta, k, i, d_k) for k in range(d_k // 2)]
            rotation_matrix_table[i, :, :] = torch.block_diag(*blocks)
        
        return rotation_matrix_table

    def generate_rotation_block(self, theta: float, block_index: int, seq_pos: int, d_k:int) -> torch.Tensor:
        angle = torch.tensor(seq_pos / (theta ** (2 * block_index / d_k)))
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        r_matrix = torch.Tensor([[cos, -sin], [sin, cos]])

        return r_matrix
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Run RoPE for a given input tensor.
        Args:
            x (Float[Tensor, "... sequence_length d_k"]): Input tensor(Query or Key) to run RoPE on.
            token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
        Returns:
            Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
        """
        *prefix_dims, seq_len, d_k = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        rotation_matrix = self.rotation_matrix_table[token_positions]
        x_rotated = rotation_matrix @ x.unsqueeze(-1)
        x_rotated = x_rotated.squeeze(-1)

        return x_rotated
         