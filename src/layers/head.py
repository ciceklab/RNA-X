from typing import Optional, Tuple
from dataclasses import dataclass
import torch.nn as nn
import torch

@dataclass
class RTOutput:
    sequence_logits: torch.Tensor
    embeddings: torch.Tensor

class RegressionHead(nn.Module):
    def __init__(self, d_model: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.SiLU()
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B L D) → (B L out)
        """Forward pass through the regression head."""
        return self.fc2(self.ln(self.act(self.fc1(x))))
    
class OutputHeads(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.sequence_head = RegressionHead(d_model, vocab_size)
        # other heads (structure, sasa …) could live here

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> RTOutput:
        seq_logits = self.sequence_head(x)
        return RTOutput(sequence_logits=seq_logits, embeddings=embed)
