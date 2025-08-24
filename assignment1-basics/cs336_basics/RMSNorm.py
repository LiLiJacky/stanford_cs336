import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, 
                 device: torch.device | None = None, dtype = None):
        """
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.eps = eps
        
        w_init = self.initialize_weights(d_model, factory_kwargs)
        self.weight = nn.Parameter(w_init)

    def initialize_weights(self, d_mode: int, factory_kwargs: dict):
        w = torch.ones(d_mode, **factory_kwargs)
        return w
    
    def RMS(self, x: torch.tensor, d_model: int, eps: float):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + eps)

        return rms
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = self.RMS(x, self.d_model, self.eps)
        result = (x / rms) * self.weight

        return result.to(in_dtype)
