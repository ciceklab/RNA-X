
from src.utils.tokenizer import ResidueTokenizer
import torch.nn as nn
import torch


def _pad_stack(seqs, pad_value=0.0):
    if len(seqs) == 0:
        raise ValueError("Empty sequence list passed to _pad_stack")
    device = seqs[0].device
    dtype = seqs[0].dtype
    lengths = torch.tensor([s.size(0) for s in seqs], device=device, dtype=torch.long)
    T_max = int(lengths.max().item())
    D = seqs[0].size(1)
    out = torch.full((len(seqs), T_max, D), fill_value=pad_value, device=device, dtype=dtype)
    for i, s in enumerate(seqs):
        t = s.size(0)
        if t > 0:
            out[i, :t] = s
    return out, lengths


def _ensure_bt_tensor(list, device, dtype=None):
    seqs = [t.to(device=device) for t in list]
    x, lengths = _pad_stack(seqs, pad_value=0.0)  # x: [B, T_max, H]
    if dtype is not None:
        x = x.to(dtype=dtype)
    return x

class MaskedSequenceCNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, cnn_channels: int = 512,
                 kernels=(3,5,7), dropout=0.1, pad_token_id: int = 0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_token_id)
        ch = emb_dim
        blocks = []
        for k in kernels:
            p = k // 2
            blocks += [
                nn.Conv1d(ch, cnn_channels, kernel_size=k, padding=p),
                nn.BatchNorm1d(cnn_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            ch = cnn_channels
        self.cnn = nn.Sequential(*blocks)
        self.out_ch = ch

    def forward(self, ids: torch.Tensor, attn_mask: torch.Tensor):
        x = self.embed(ids)         # (B,T,E)
        x = x.transpose(1, 2)       # (B,E,T)
        x = self.cnn(x)             # (B,C,T)
        x = x.transpose(1, 2)       # (B,T,C)
        return x
    
def masked_mean_max_pool(x: torch.Tensor, mask: torch.Tensor):
    B, T, C = x.shape
    m = mask.unsqueeze(-1).to(dtype=x.dtype)
    lens = m.sum(dim=1).clamp(min=1)

    mean = (x * m).sum(dim=1) / lens  # broadcast works: (B, C) / (B, 1)

    neginf = torch.finfo(x.dtype).min
    mx = x.masked_fill(m == 0, neginf).max(dim=1).values
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))

    return torch.cat([mean, mx], dim=-1)

def off_diagonal(m: torch.Tensor) -> torch.Tensor:
    n, _ = m.shape
    return m.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

class GatedAddFuse(nn.Module):
    def __init__(self, rt_dim: int, cnn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.rt2cnn = nn.Linear(rt_dim, cnn_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(cnn_dim),
            nn.Linear(cnn_dim, 4*cnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*cnn_dim, cnn_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(cnn_dim),
            nn.Linear(cnn_dim, cnn_dim),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))  # small residual init
        self.ln = nn.LayerNorm(cnn_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, rt_tok: torch.Tensor, cnn_tok: torch.Tensor):
        r = self.rt2cnn(rt_tok)          # (B,T,C_cnn)
        delta = self.ff(r)               # (B,T,C_cnn)
        g = self.gate(r)                 # (B,T,C_cnn)
        fused = cnn_tok + self.alpha * g * delta
        return self.ln(self.drop(fused) + fused)



class RL(nn.Module):
    def __init__(
        self,
        tokenizer: ResidueTokenizer,
        proj_dim: int = 512,
        dropout: float = 0.2,
        cnn_emb_dim: int = 256,
        cnn_channels: int = 512,
        cnn_kernels=(3,5,7),
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.num_features = proj_dim
        self._mask_id = tokenizer.mask_token_id
        self._pad_id  = tokenizer.pad_token_id

        self.prot_cnn = MaskedSequenceCNN(tokenizer.vocab_size, cnn_emb_dim, cnn_channels, cnn_kernels, dropout, tokenizer.pad_token_id)
        self.rna_cnn  = MaskedSequenceCNN(tokenizer.vocab_size, cnn_emb_dim, cnn_channels, cnn_kernels, dropout, tokenizer.pad_token_id)

        self.prot_fuse = GatedAddFuse(768, self.prot_cnn.out_ch, dropout)
        self.rna_fuse  = GatedAddFuse(768, self.rna_cnn.out_ch, dropout)

        self.fuse_dim_prot = self.prot_cnn.out_ch
        self.fuse_dim_rna  = self.rna_cnn.out_ch

        self.proj_prot = nn.Sequential(
            nn.LayerNorm(2*self.fuse_dim_prot),
            nn.Linear(2*self.fuse_dim_prot, proj_dim),
            nn.GELU()
        )
        self.proj_rna  = nn.Sequential(
            nn.LayerNorm(2*self.fuse_dim_rna),
            nn.Linear(2*self.fuse_dim_rna, proj_dim),
            nn.GELU()
        )

        self.predict_prot = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim*4),
            nn.GELU(),
            nn.Linear(proj_dim*4, proj_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim*2, proj_dim*2),
        )

        self.predict_rna  = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim*4),
            nn.GELU(),
            nn.Linear(proj_dim*4, proj_dim*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim*2, proj_dim*2),
        )

        self.cls = nn.Sequential(
            nn.Linear(2*proj_dim, 512),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 2),
        )


    def _align_time(self, A_tok, B_tok, B_mask):
        T = min(A_tok.size(1), B_tok.size(1))
        return A_tok[:, :T], B_tok[:, :T], B_mask[:, :T]
    
    def pair_logits(self, p, r):
        return self.cls(torch.cat([p, r], dim=-1))

    def forward(self, target_seqs, rna_seqs, target_embeddings, rna_embeddings, device="cuda"):
        ids_pack = self.tokenizer.batch_encode_no_special(target_seqs, rna_seqs, device=device, pad_to_multiple_of=8)
        prot_ids, prot_mask = ids_pack["prot_ids"], ids_pack["prot_mask"]
        rna_ids,  rna_mask  = ids_pack["rna_ids"],  ids_pack["rna_mask"]
        prot_cnn = self.prot_cnn(prot_ids, prot_mask)   # (B,Tp,Cc)
        rna_cnn  = self.rna_cnn (rna_ids,  rna_mask)    # (B,Tr,Cc)

        want_dtype = prot_cnn.dtype
        prot_rt = _ensure_bt_tensor(target_embeddings, device=device, dtype=want_dtype)
        rna_rt  = _ensure_bt_tensor(rna_embeddings,  device=device, dtype=want_dtype)

        prot_rt, prot_cnn, prot_mask = self._align_time(prot_rt, prot_cnn, prot_mask)
        rna_rt,  rna_cnn,  rna_mask  = self._align_time(rna_rt,  rna_cnn,  rna_mask)

        prot_fused = self.prot_fuse(prot_rt, prot_cnn)  # (B,Tp,Cc)
        rna_fused  = self.rna_fuse (rna_rt,  rna_cnn)   # (B,Tr,Cc)

        prot_pool = masked_mean_max_pool(prot_fused, prot_mask)   # (B, 2*Cc)
        rna_pool  = masked_mean_max_pool(rna_fused,  rna_mask)    # (B, 2*Cc)

        z_prot = self.proj_prot(prot_pool)
        z_rna  = self.proj_rna (rna_pool)

        # p_prot = self.predict_prot(z_prot)
        # p_rna  = self.predict_rna(z_rna)
        p_prot = None
        p_rna = None

        return p_prot, p_rna, z_prot, z_rna
