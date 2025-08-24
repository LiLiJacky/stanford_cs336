import torch
from typing import Iterable

def clip_gradients(params: Iterable[torch.nn.Parameter], max_l2: float, eps: float=1e-6) -> None:
    grads = [param.grad for param in params if param.grad is not None]

    if not grads:
        return
    
    flattened_grads = torch.cat([g.reshape(-1) for g in grads])
    l2_norm = flattened_grads.norm(p=2)
    if l2_norm > (max_l2 + eps):
        scale_factor = max_l2 / (l2_norm + eps)  # 统一缩放因子
        
        for param in params:
            if param.grad is not None:
                param.grad.mul_(scale_factor)