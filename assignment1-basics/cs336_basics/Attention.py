import torch
import torch.nn as nn
import math
from einops import einsum, rearrange
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding
from .Linear import Linear

def softmax(x: torch.Tensor, dim: int):
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        x (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `x` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `x` with the output of
        softmax normalizing the specified `dim`.
    """
    x_max = torch.max(x, dim, keepdim=True).values
    x_stable = x - x_max
    x_exp = torch.exp(x_stable)
    output = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

    return output

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of scaled dot product attention.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = K.shape[-1]
    attention_score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)

    if mask is not None:
        attention_score = attention_score.masked_fill(~mask, float('-inf'))
    
    output = softmax(attention_score, -1) @ V
    
    return output

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False, max_seq_len: int | None = None,
                 theta: float | None = None, token_positions: torch.Tensor | None = None):
        """
        Given the key, query, and value projection weights of a naive unbatched
        implementation of multi-head attention, return the output of an optimized batched
        implementation. This implementation should handle the key, query, and value projections
        for all heads in a single matrix multiply.
        This function should not use RoPE.
        See section 3.2.2 of Vaswani et al., 2017.

        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
            k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
            v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
            o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
            in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
            implementation with the given QKV projection weights and input features.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len) if use_rope else None
        self.token_positions = token_positions
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(self, in_features: torch.Tensor):
        """
        Args:
            in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
        """
        seq_len = in_features.shape[-2]
        qkv_proj = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight])
        qkv = in_features @ qkv_proj.T
        q, k, v = qkv.chunk(3, -1)

        q = rearrange(
            q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        k = rearrange(
            k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )
        v = rearrange(
            v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads
        )

        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)
        
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask[None, None, :, :]
        output = scaled_dot_product_attention(q, k, v, ~causal_mask)
        output = rearrange(
            output, "... h seq_len d_head -> ... seq_len (h d_head)"
        )

        return self.o_proj(output)


