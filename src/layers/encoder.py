from typing import Optional, Tuple
import torch.nn as nn
import torch


from .flash import RTTransformerLayer

class RTEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1,) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            RTTransformerLayer( d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **_) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer in self.layers:
            h = layer(h, attention_mask)

        return self.norm(h), h  # post_norm, pre_norm

