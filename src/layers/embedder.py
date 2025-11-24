import torch
import torch.nn as nn


class RTEmbedder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, max_position: int = 4096, num_residue_types: int = 5):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.type_embed = nn.Embedding(num_residue_types, d_model) 
        self.pos_embed = nn.Embedding(max_position, d_model)
        self.max_position = max_position
        
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, sequence_tokens: torch.Tensor, residue_type_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = sequence_tokens.shape
        
        if seq_len > self.max_position:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum supported position {self.max_position}")

        positions = torch.arange(seq_len, device=sequence_tokens.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        token_emb = self.token_embed(sequence_tokens)
        pos_emb = self.pos_embed(positions) 
        type_emb = self.type_embed(residue_type_ids)
        return token_emb + pos_emb + type_emb 
