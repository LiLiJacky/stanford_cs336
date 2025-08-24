import torch
import torch.nn as nn 
import numpy as np

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: int Size of the vocabulary
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        w_init = self.initialize_weights(num_embeddings, embedding_dim, factory_kwargs)
        self.weight = nn.Parameter(w_init)
    
    def initialize_weights(self, vocab_size: int, d_model: int, factory_kwargs: dict) -> torch.Tensor:
        # Initialize weights W using truncated normal method
        W = torch.empty(vocab_size, d_model, **factory_kwargs)
        mean = 0
        std = 1

        nn.init.trunc_normal_(W, mean, std, -3, 3)
        return W

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        Args:
            token_ids: (batch_size, sequence_length)
            output: (batch_size, sequence_length, embedding_dim)
        Returns:
            embeddings for given token IDs
        """
        batch_size, sequence_length = token_ids.shape
        output = torch.empty(batch_size, sequence_length, self.embedding_dim)

        for i, seq in enumerate(token_ids):
            for j, token_id in enumerate(seq):
                output[i][j] = self.weight[token_id]
        
        return output
    

