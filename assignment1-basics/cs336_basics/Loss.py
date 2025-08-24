import torch
import einx
from .Attention import softmax

def cross_entropy_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # 没有经过归一化，不能直接使用softmax，会产生上下溢出的问题（这里主要是下溢出
    max_x = torch.max(x, dim=-1, keepdim=True).values
    denom = einx.sum("... [vocab_size]", (x - max_x).exp())
    output = -(einx.get_at("... [vocab_size], ... -> ...", x, y) - max_x - torch.log(torch.clamp_min(denom, eps))) 


    return einx.mean("[...]", output)

def perplexity_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return cross_entropy_loss(x, y, eps).exp()
    