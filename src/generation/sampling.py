import torch
import torch.nn.functional as F

BASES = ["A", "C", "G", "U"]

import torch
import torch.nn.functional as F

def calculate_token_confidence(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values

def calculate_token_entropy(logits: torch.Tensor, temperature: float = 1.0, normalize: bool = True) -> torch.Tensor:
    mask = torch.isfinite(logits)
    scaled = (logits / temperature).to(torch.float32)
    very_neg = torch.tensor(-1e9, dtype=scaled.dtype, device=scaled.device)
    masked_logits = torch.where(mask, scaled, very_neg)

    probs = F.softmax(masked_logits, dim=-1)
    probs = probs * mask.to(probs.dtype)
    Z = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probs = probs / Z

    if hasattr(torch.special, "xlogy"):
        ent = -(torch.special.xlogy(probs, probs)).sum(dim=-1)
    else:
        ent = -(probs * probs.clamp_min(1e-45).log()).sum(dim=-1)

    if normalize:
        vocab_eff = mask.to(torch.float32).sum(dim=-1).clamp_min(1)
        ent = ent / torch.log(vocab_eff)  # -> [0,1]
    return ent.to(logits.dtype)

def choose_positions_to_fill(mask_positions: torch.Tensor,
                             confidences: torch.Tensor,
                             entropies: torch.Tensor,
                             num_select: int,
                             prefer: str = "confidence",
                             tradeoff_alpha: float = 0.0) -> torch.Tensor:
    device = mask_positions.device
    idx = torch.where(mask_positions)[0]
    if idx.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if idx.numel() <= num_select:
        return idx

    c = confidences[idx]
    e = entropies[idx]

    if prefer == "confidence":
        score = c - float(tradeoff_alpha) * e
        k_idx = torch.topk(score, num_select, largest=True).indices
        return idx[k_idx]
    elif prefer == "entropy":
        # pick lowest entropy
        k_idx = torch.topk(e, num_select, largest=False).indices
        return idx[k_idx]
    else:
        raise ValueError(f"Unknown prefer='{prefer}' for choose_positions_to_fill")

def choose_positions_to_mask(middle_len: int,
                             confidences_mid: torch.Tensor,
                             entropies_mid: torch.Tensor,
                             num_select: int,
                             metric: str = "entropy",        # 'entropy' or 'low_confidence'
                             random_strategy: str = "epsilon",# 'epsilon'|'gumbel'|'random'|'greedy'
                             epsilon: float = 0.2,
                             gumbel_tau: float = 1.0) -> torch.Tensor:
    device = confidences_mid.device
    if middle_len <= 0:
        return torch.empty(0, dtype=torch.long, device=device)

    idx = torch.arange(middle_len, device=device)
    if idx.numel() <= num_select:
        return idx

    if metric == "entropy":
        u = entropies_mid  # higher -> more uncertain
    elif metric == "low_confidence":
        u = 1.0 - confidences_mid
    else:
        raise ValueError(f"Unknown metric='{metric}' for choose_positions_to_mask")

    n = idx.numel()

    if random_strategy == "greedy":
        k_idx = torch.topk(u, num_select, largest=True).indices
        return idx[k_idx]

    if random_strategy == "random":
        perm = torch.randperm(n, device=device)
        return idx[perm[:num_select]]

    if random_strategy == "epsilon":
        epsilon = float(epsilon)
        epsilon = max(0.0, min(1.0, epsilon))

        if num_select == 1:
            if torch.rand((), device=device) < epsilon:
                j = torch.randint(0, n, (1,), device=device)
                return idx[j]
            else:
                k_idx = torch.topk(u, 1, largest=True).indices
                return idx[k_idx]

        n_rand = int(torch.ceil(torch.tensor(epsilon * num_select)).item())
        n_rand = min(n_rand, num_select)
        n_greedy = num_select - n_rand

        greedy = (torch.topk(u, n_greedy, largest=True).indices
                if n_greedy > 0 else torch.empty(0, dtype=torch.long, device=device))

        mask_remaining = torch.ones(n, dtype=torch.bool, device=device)
        mask_remaining[greedy] = False
        remaining = idx[mask_remaining]
        if remaining.numel() > 0 and n_rand > 0:
            perm = torch.randperm(remaining.numel(), device=device)
            rnd = remaining[perm[:n_rand]]
        else:
            rnd = torch.empty(0, dtype=torch.long, device=device)

        chosen = torch.cat([greedy, rnd], dim=0)
        return chosen

    if random_strategy == "gumbel":
        g = -torch.empty_like(u).exponential_().log()   # Gumbel(0,1)
        noisy = u + g * gumbel_tau
        k_idx = torch.topk(noisy, num_select, largest=True).indices
        return idx[k_idx]

    raise ValueError(f"Unknown random_strategy='{random_strategy}'")


def sample_tokens_from_logits(logits: torch.Tensor, method: str, temperature: float, top_k: int=None, top_p: float=None) -> torch.Tensor:
    if method == 'greedy':
        return torch.argmax(logits, dim=-1)
    
    elif method == 'sampling':
        scaled_logits = logits / max(temperature, 0.01)
        probs = F.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    elif method == 'top_k':
        scaled_logits = logits / max(temperature, 0.01)
        top_k_actual = min(top_k, scaled_logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(scaled_logits, k=top_k_actual)
        probs = F.softmax(top_k_logits, dim=-1)
        sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    
    elif method == 'top_p':
        scaled_logits = logits / max(temperature, 0.01)
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0  # Keep at least one token
        
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        filtered_logits = scaled_logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")