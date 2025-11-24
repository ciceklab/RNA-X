
from collections import defaultdict
import math
import logging
import time
from typing import Optional, Iterable, Tuple, List, Set

import torch
import torch.nn.functional as F

from .sampling import calculate_token_entropy
from .configs import RTGenerationConfig, GenerationResult
from .utils import clean_rna_sequence, BASES, _rna_base_token_ids
from .logits import _logits_for_sequence
from .score import value_of_seq



class Node:
    __slots__ = ("state","depth","parent","children","P","N_edge","W_edge","Q_edge",
                 "N_total","expanded")
    def __init__(self, state: str, depth: int = 0, parent=None):
        self.state   = state                    
        self.depth   = depth
        self.parent  = parent
        self.children = {}                       # action(tuple)-> child Node
        self.P = {}                              # action -> prior P(s,a)
        self.N_edge = defaultdict(int)           # count per action
        self.W_edge = defaultdict(float)         # total value per action
        self.Q_edge = defaultdict(float)         # mean value per action
        self.N_total = 0
        self.expanded = False

    def terminal(self, max_depth: int):
        return self.depth >= max_depth

    def fully_expanded(self):
        return self.expanded and len(self.children) > 0 and \
               all(a in self.children for a in self.P.keys())

    def link_child(self, a, child):
        self.children[a] = child

    def pick_unexpanded_action(self):
        for a in self.P.keys():
            if a not in self.children:
                return a
        return None
    

def _ensure_rna_prefix(seq: str) -> str:
    seq_clean = clean_rna_sequence(seq)
    return seq_clean if seq_clean.startswith("RNA:") else ("RNA:" + seq_clean)

def _puct(node: Node, a, c_puct: float):
    N_s = max(1, node.N_total)
    Q   = node.Q_edge[a] if a in node.Q_edge else 0.0
    N_sa = node.N_edge[a] if a in node.N_edge else 0
    P   = node.P.get(a, 0.0)
    U   = c_puct * P * (math.sqrt(N_s) / (1 + N_sa))
    return Q + U


def _policy_priors_from_logits(logits: torch.Tensor,
                               tokenizer,
                               rna_seq: str,
                               top_pos_by: str,
                               K_pos: int,
                               K_base: int,
                               allowed_positions: Optional[Iterable[int]] = None):
    tidx = _rna_base_token_ids(tokenizer)
    L = logits.size(0)
    base_logits = logits[:, tidx]               
    probs = F.softmax(base_logits, dim=-1)      

    conf = probs.max(dim=-1).values             
    ent  = calculate_token_entropy(base_logits) 

    if top_pos_by == "entropy":
        scores = ent; largest = True
    elif top_pos_by == "low_conf":
        scores = -conf; largest = True
    else:
        scores = ent; largest = True

    if allowed_positions is not None:
        allowed_set: Set[int] = set(int(i) for i in allowed_positions if 0 <= int(i) < L)
        mask = torch.zeros(L, dtype=torch.bool, device=scores.device)
        if allowed_set:
            idx = torch.tensor(sorted(list(allowed_set)), device=scores.device, dtype=torch.long)
            mask[idx] = True
        scores = scores.clone()
        scores[~mask] = float("-inf")
    else:
        mask = torch.ones(L, dtype=torch.bool, device=scores.device)

    if mask.any():
        K_pos = min(int(K_pos), int(mask.sum().item()))
        _, pos_idx = torch.topk(scores, k=K_pos, largest=largest)
        pos_idx = [int(i) for i in pos_idx.tolist() if mask[int(i)]]
    else:
        pos_idx = []

    current = list(clean_rna_sequence(rna_seq))
    base_to_col = {"A":0, "C":1, "G":2, "U":3}

    actions = []
    priors  = {}
    for i in pos_idx:
        p_i = probs[i].clone()
        cur_b = current[i] if i < len(current) else None
        if cur_b in base_to_col:
            p_i[base_to_col[cur_b]] = 0.0
        if p_i.sum() <= 0:
            p_i = probs[i].clone()

        p_i = p_i / (p_i.sum() + 1e-12)
        Kb = min(int(K_base), p_i.numel())
        vals, idxs = torch.topk(p_i, k=Kb, largest=True)

        for v, j in zip(vals.tolist(), idxs.tolist()):
            b = BASES[j]
            if b == cur_b:
                continue
            a = (i, b)
            actions.append(a)
            priors[a] = float(v)

    if not actions:
        best_score, best_a = -1.0, None
        for i in range(L):
            if allowed_positions is not None and i not in allowed_set:
                continue
            p_i = probs[i]
            cur_b = current[i] if i < len(current) else None
            for j, b in enumerate(BASES):
                if b == cur_b: 
                    continue
                s = float(p_i[j])
                if s > best_score:
                    best_score, best_a = s, (i, b)
        if best_a is not None:
            actions = [best_a]
            priors  = {best_a: 1.0}

    Z = sum(priors.values()) + 1e-12
    for a in list(priors.keys()):
        priors[a] /= Z
    return actions, priors


def _apply_edit(rna_seq: str, pos: int, base: str, allowed_positions: Optional[Iterable[int]] = None) -> str:
    s = clean_rna_sequence(rna_seq)
    arr = list(s)
    if pos < 0 or pos >= len(arr):
        raise IndexError(f"Edit position {pos} out of range for length {len(arr)}")
    if allowed_positions is not None:
        allowed_set = set(int(i) for i in allowed_positions)
        if pos not in allowed_set:
            raise ValueError(f"Edit at position {pos} is outside the editable window")
    arr[pos] = base
    return "RNA:" + "".join(arr)


def _run_mcts(
    model,
    tokenizer,
    prediction_model,
    protein_seq: str,
    init_rna: str,
    use_scoring:bool,
    cfg: RTGenerationConfig,
    logger: logging.Logger,
    *,
    edit_window: Optional[Tuple[int, int]] = None,      
    allowed_positions: Optional[Iterable[int]] = None,
) -> GenerationResult:
    device = next(model.parameters()).device

    rna_init = _ensure_rna_prefix(init_rna)
    root = Node(state=rna_init, depth=0, parent=None)

    if allowed_positions is None:
        if edit_window is not None:
            a, b = int(edit_window[0]), int(edit_window[1])
            allowed_positions = range(max(0, a), max(0, b))
        else:
            allowed_positions = None

    logits_cache = {}
    value_cache = {}

    logger.info("[tree] Expanding root")
    logits = _logits_for_sequence(model, tokenizer, protein_seq, root.state, device=device)
    actions, priors = _policy_priors_from_logits(
        logits, tokenizer, root.state, cfg.top_pos_by, cfg.K_pos, cfg.K_base,
        allowed_positions=allowed_positions,                                  # NEW
    )
    root.P.update(priors)

    score, best_scores = value_of_seq(prediction_model, protein_seq, root.state, use_scoring)
    best_seq, best_val = root.state, score
    candidates: List[str] = []
    scores_history: List[Tuple[int, float]] = []

    t0 = time.time()
    for it in range(int(cfg.iterations)):
        node = root
        path = [node]
        edge_path = []

        while node.fully_expanded() and not node.terminal(cfg.max_depth):
            a = max(node.P.keys(), key=lambda x: _puct(node, x, cfg.c_puct))
            node = node.children[a]
            path.append(node)
            edge_path.append(a)
            if cfg.tree_debug:
                logger.debug(f"[select] step={it} -> action={a} depth={node.depth}")

        if not node.terminal(cfg.max_depth):
            if not node.expanded:
                logits = logits_cache.get(node.state)
                if logits is None:
                    logits = _logits_for_sequence(model, tokenizer, protein_seq, node.state, device=device)
                    logits_cache[node.state] = logits
                actions, priors = _policy_priors_from_logits(
                    logits, tokenizer, node.state, cfg.top_pos_by, cfg.K_pos, cfg.K_base,
                    allowed_positions=allowed_positions,                        
                )
                node.P.update(priors)
                node.expanded = True
                if cfg.tree_debug:
                    logger.debug(f"[expand] node at depth {node.depth} expanded with {len(priors)} priors")

            a = node.pick_unexpanded_action()
            if a is not None:
                i, base = a
                child_state = _apply_edit(node.state, i, base, allowed_positions=allowed_positions)
                child = Node(state=child_state, depth=node.depth + 1, parent=node)
                node.link_child(a, child)
                node = child
                path.append(node)
                edge_path.append(a)
                if cfg.tree_debug:
                    logger.debug(f"[expand->child] action={a} -> depth={node.depth}")

        if node.state in value_cache:
            V = value_cache[node.state]
        else:
            V, scores = value_of_seq(prediction_model, protein_seq, node.state, use_scoring)
            value_cache[node.state] = V

        if V > best_val:
            best_val, best_seq = V, node.state
            best_scores = scores
            logger.info(f"[eval] iter={it} NEW BEST val={best_val:.4f}")

        candidates.append(node.state)

        for idx in range(len(path) - 1):
            parent = path[idx]
            a = edge_path[idx]
            parent.N_total += 1
            parent.N_edge[a] += 1
            parent.W_edge[a] += V
            parent.Q_edge[a] = parent.W_edge[a] / parent.N_edge[a]

    elapsed = time.time() - t0
    logger.info("\n=== MCTS RESULT ===")
    logger.info(f"Best RNA: {best_seq}")
    logger.info(f"Best val: {best_val:.4f}")
    logger.info(f"Tree sims: {cfg.iterations} in {elapsed:.2f}s")

    if len(root.N_edge) > 0:
        best_root_action = max(root.N_edge.items(), key=lambda kv: kv[1])[0]
        pos, base = best_root_action
        logger.info(f"Suggested first edit at root -> pos={pos}, base={base}")

    return GenerationResult(
        best_sequence=best_seq,
        best_value=best_val,
        all_candidates=candidates,
        init_sequence=root.state,
        scores_history=scores_history,
        best_scores=best_scores,
    )