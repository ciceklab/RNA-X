from typing import Optional, Tuple
import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func

import torch.nn.functional as F
import warnings

class RTFlashAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.dropout_p = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:        
        attention_strategy = "varlen_training"
        
        if attention_strategy == "varlen_training":
            return self._flash_varlen_attention(x, attention_mask)
        elif attention_strategy == "qkvpacked_training":
            return self._flash_qkvpacked_attention(x)
        elif attention_strategy == "func_inference":
            return self._flash_func_attention(x)
        else:
            return False
    
    def _has_significant_padding(self, attention_mask: torch.Tensor, threshold: float = 0.15) -> bool:
        total_tokens = attention_mask.numel()
        valid_tokens = attention_mask.sum().item()
        padding_ratio = 1.0 - (valid_tokens / total_tokens)
        
        return padding_ratio > threshold
    
    def _flash_varlen_attention(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_seq_len, d_model = x.shape
        
        seqlens = attention_mask.sum(dim=1, dtype=torch.int32)
        
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device) #cu_seqlens: [B+1] - [0, len1, len1+len2, len1+len2+len3, ...]
        cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
        
        indices = attention_mask.nonzero(as_tuple=False) # indices shape: [total_valid_tokens, 2] where each row is (batch_idx, seq_idx)
        
        x_packed = x[indices[:, 0], indices[:, 1]] # x_packed: [total_valid_tokens, D] instead of [B, S, D]
        
        qkv_packed = self.qkv_proj(x_packed) # [total_valid_tokens, D] -> [total_valid_tokens, 3*D]
        
        total_valid_tokens = x_packed.size(0) # [total_valid_tokens, 3*D] -> [total_valid_tokens, 3, H, D_head]
        qkv_packed = qkv_packed.reshape(total_valid_tokens, 3, self.n_heads, self.head_dim)

        attn_output_packed = flash_attn_varlen_qkvpacked_func(
            qkv_packed,                                   
            cu_seqlens,                                   
            max_seqlen=max_seq_len,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False                                   # Bidirectional attention for BERT
        )
        
        attn_output_packed = attn_output_packed.reshape(total_valid_tokens,  self.n_heads * self.head_dim)   # [total_valid_tokens, H*D_head] -> [total_valid_tokens, D]

        attn_output_packed = self.out_proj(attn_output_packed)
        # print("attn_output_packed shape:", attn_output_packed.shape)
        attn_output_packed = attn_output_packed.to(dtype=x.dtype, device=x.device)


        attn_output = torch.zeros_like(x)  # [B, S, D] - zeros for padding positions
        attn_output[indices[:, 0], indices[:, 1]] = attn_output_packed

        # print("target shape:", attn_output[indices[:, 0], indices[:, 1]].shape)

        
        return attn_output
    
    def _flash_qkvpacked_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x) # [B, S, D] -> [B, S, 3*D]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim) # [B, S, 3*D] -> [B, S, 3, H, D_head]
        
        attn_output = flash_attn_qkvpacked_func(
            qkv,                                          # [B, S, 3, H, D_head]
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False 
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)  # [B, S, D]
        output = self.out_proj(attn_output)  # [B, S, D] -> [B, S, D]
        
        return output
    
    def _flash_func_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x) # [B, S, D] -> [B, S, 3*D]
        
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim) # [B, S, 3*D] -> [B, S, 3, H, D_head] -> [3, B, S, H, D_head]
        qkv = qkv.permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, S, H, D_head]
        
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)  # [B, S, D]
        output = self.out_proj(attn_output)  # [B, S, D] -> [B, S, D]
        
        return output

class RTFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        dim_feedforward = 4 * d_model 
        
        self.gate_up_proj = nn.Linear(d_model, 2 * dim_feedforward, bias=False) #[B, S, D] -> [B, S, 2*4*D] = [B, S, 8*D]
        
        self.down_proj = nn.Linear(dim_feedforward, d_model, bias=False) # [B, S, 4*D] -> [B, S, D]
        
        self.dropout = nn.Dropout(dropout)
    
        nn.init.xavier_uniform_(self.gate_up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x) # [B, S, D] -> [B, S, 8*D]
        gate, up = gate_up.chunk(2, dim=-1) # [B, S, 8*D] -> 2 tensors of [B, S, 4*D]
        
        hidden = F.silu(up) * gate # [B, S, 4*D]
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden) # [B, S, 4*D] -> [B, S, D]
        
        return output


class RTTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = RTFlashAttention(d_model, n_heads, dropout) 
        self.feed_forward = RTFeedForward(d_model, dropout)
    
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed_x = self.norm1(x)  # [B, S, D]
        attn_output = self.self_attn(normed_x, attention_mask)  # [B, S, D]
        x = x + self.dropout1(attn_output)  # [B, S, D]

        normed_x = self.norm2(x)  # [B, S, D]
        ff_output = self.feed_forward(normed_x)  # [B, S, D]
        x = x + self.dropout2(ff_output)  #  [B, S, D]
        
        return x
