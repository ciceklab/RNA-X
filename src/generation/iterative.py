
import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F

from .configs import RTGenerationConfig
from .sampling import calculate_token_entropy, calculate_token_confidence, sample_tokens_from_logits, choose_positions_to_fill, choose_positions_to_mask
from .logits import _logits_for_sequence
from .score import value_of_seq


def generate_single_rna_iterative(
    model,
    tokenizer,
    inputs: Dict[str, Any],
    config,
    logger: Optional[logging.Logger] = None,
    *,
    allow_unknown_base: bool = False,
) -> Dict[str, Any]:
    logger.debug("=" * 20)

    required = ["input_ids", "attention_mask", "residue_type_ids", "rna_region", "rna_middle_region"]
    for k in required:
        if k not in inputs:
            raise ValueError(f"generate_single_rna_iterative: missing '{k}' in inputs")

    input_ids = inputs["input_ids"].clone()
    attention_mask = inputs["attention_mask"]
    residue_type_ids = inputs["residue_type_ids"]
    rna_start, rna_end = inputs["rna_region"]
    mid_start, mid_end = inputs["rna_middle_region"]
    logger.debug("Initial Input Ids: ", input_ids)

    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("input_ids must be shape [1, T]")

    device = input_ids.device
    B, T = input_ids.shape
    L_mid = int(mid_end - mid_start)
    if L_mid <= 0:
        raise ValueError(f"Invalid rna_middle_region {inputs['rna_middle_region']}")

    middle_slice = input_ids[0, mid_start:mid_end]
    initial_mask = (middle_slice == tokenizer.mask_token_id)
    if not torch.any(initial_mask):
        raise ValueError("No [MASK] tokens found in rna_middle_region")

    allowed_tokens = [
        tokenizer.convert_tokens_to_ids("RNA_A"),
        tokenizer.convert_tokens_to_ids("RNA_C"),
        tokenizer.convert_tokens_to_ids("RNA_G"),
        tokenizer.convert_tokens_to_ids("RNA_U"),
    ]
    if allow_unknown_base:
        tid_x = tokenizer.convert_tokens_to_ids("RNA_X")
        if tid_x is not None and tid_x >= 0:
            allowed_tokens.append(tid_x)

    allowed_tokens = [t for t in allowed_tokens if t is not None and t >= 0]
    allowed_tokens_t = torch.tensor(allowed_tokens, device=device, dtype=torch.long)
    assert allowed_tokens_t.numel() > 0, "No allowed RNA tokens available for sampling."

    iteration = 0
    positions_filled = 0

    last_confidences = torch.zeros(L_mid, device=device, dtype=torch.float32)
    last_entropies   = torch.zeros(L_mid, device=device, dtype=torch.float32)

    with torch.no_grad():
        while True:
            middle_slice = input_ids[0, mid_start:mid_end]
            mask_positions = (middle_slice == tokenizer.mask_token_id)
            remaining = int(mask_positions.sum().item())
            if remaining == 0:
                break

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                residue_type_ids=residue_type_ids,
            )
            logits = outputs.logits[0]
            mid_logits = logits[mid_start:mid_end]

            filtered = mid_logits.new_full(mid_logits.shape, float("-inf"))
            filtered[:, allowed_tokens_t] = mid_logits[:, allowed_tokens_t]

            confidences = calculate_token_confidence(filtered)
            entropies   = calculate_token_entropy(filtered, normalize=True)
            last_confidences = confidences
            last_entropies   = entropies

            to_fill_cfg = config.positions_per_iteration
            to_fill = max(1, min(to_fill_cfg, remaining))

            prefer = config.position_selection  # 'confidence' or 'entropy'
            alpha  = config.unmask_tradeoff_alpha 

            logger.debug("Confidences before choosing the unmask positions: ", confidences)
            selected_local_idx = choose_positions_to_fill(
                mask_positions=mask_positions,
                confidences=confidences,
                entropies=entropies,
                num_select=to_fill,
                prefer=prefer,
                tradeoff_alpha=alpha,
            )
            logger.debug("Positions to fill (local): ", selected_local_idx)
            if selected_local_idx.numel() == 0:
                if logger:
                    logger.info("No positions selected to unmask. In single RNA iterative generation.")
                break

            chosen_logits = filtered[selected_local_idx]
            new_tokens = sample_tokens_from_logits(
                logits=chosen_logits,
                method=config.sampling_method,
                temperature=config.sampling_temperature,
                top_k=config.top_k,
                top_p=config.top_p,
            )

            absolute_positions = mid_start + selected_local_idx
            logger.debug("Positions to fill (local): ", selected_local_idx)
            logger.debug("Input Ids before filling: ", input_ids)
            input_ids[0, absolute_positions] = new_tokens
            positions_filled += int(new_tokens.numel())
            logger.debug("Input Ids after filling: ", input_ids)
            iteration += 1

    final_rna_tokens = []
    full_rna_ids = input_ids[0, rna_start:rna_end]

    for tok_id in full_rna_ids.tolist():
        if tok_id in (tokenizer.mask_token_id, tokenizer.pad_token_id, getattr(tokenizer, "eos_token_id", -1)):
            continue
        tok = tokenizer.convert_ids_to_tokens(tok_id)
        if tok and tok.startswith("RNA_"):
            final_rna_tokens.append(tok.split("_", 1)[1])

    rna_sequence = "RNA:" + "".join(final_rna_tokens)

    return {
        "rna_sequence": rna_sequence,
        "iterations": iteration,
        "positions_filled": positions_filled,
        "final_input_ids": input_ids.detach().to("cpu"),
        "rna_region": (int(rna_start), int(rna_end)),
        "rna_middle_region": (int(mid_start), int(mid_end)),
        "confidences": last_confidences.detach().to("cpu"),
        "entropies": last_entropies.detach().to("cpu"),
        # "ended_with_eos": bool(ended_with_eos),
    }



def _iterative_gen_ref(
    model,
    tokenizer,
    prediction_model,
    protein_seq: str,
    cfg: RTGenerationConfig,
    logger: logging.Logger,
    *,
    rna_prefix: Optional[str] = None,
    rna_suffix: Optional[str] = None,
    use_scoring: bool = True
) -> Tuple[str, float]:
    
    device = next(model.parameters()).device
    logger.info("Starting initial mask predict initialization")
    def _map_selection(sel: Optional[str]) -> str:
        s = str(sel or "").lower()
        if s =="confidence":  # “confidence” means pick LOWEST conf
            return "confidence"
        if s  == "entropy":
            return "entropy"
        if s == "random":
            return "random"
        return "confidence"

    def _allowed_base_indices():
        tids = [
            tokenizer.convert_tokens_to_ids("RNA_A"),
            tokenizer.convert_tokens_to_ids("RNA_C"),
            tokenizer.convert_tokens_to_ids("RNA_G"),
            tokenizer.convert_tokens_to_ids("RNA_U")
        ]
        return torch.tensor([t for t in tids if t is not None and t >= 0], device=device, dtype=torch.long)
    
    def _choose_positions_from_full_logits(
        logits_full_rna: torch.Tensor,
        tokenizer,
        middle_window: Tuple[int, int],
        position_selection: str,
        mask_ratio: float,
        *,
        allow_unknown_base: bool,
        device: torch.device,
    ) -> torch.Tensor:
        tidx = _allowed_base_indices()
        base_logits = logits_full_rna[:, tidx]  # [L_rna, |allowed|]
        confidences = calculate_token_confidence(base_logits)
        entropies   = calculate_token_entropy(base_logits, normalize=True)

        a, b = int(middle_window[0]), int(middle_window[1])
        Lm = max(0, b - a)
        if Lm <= 0:
            return torch.empty(0, dtype=torch.long, device=device)

        conf_m = confidences[a:b]
        ent_m  = entropies[a:b]

        k = max(1, int(round(float(mask_ratio) * Lm)))

        metric = "entropy" if _map_selection(position_selection) == "entropy" else "low_confidence"
        random_strategy = cfg.refine.random_strategy
        epsilon = cfg.refine.epsilon
        gumbel_tau = cfg.refine.gumbel_tau

        logger.debug("Confidences before choosing the mask positions: ", conf_m)
        selected_local_idx = choose_positions_to_mask(
            middle_len=Lm,
            confidences_mid=conf_m,
            entropies_mid=ent_m,
            num_select=k,
            metric=metric,                # uncertain = high entropy or low confidence
            random_strategy=random_strategy,  # 'epsilon'|'gumbel'|'random'|'greedy'
            epsilon=epsilon,
            gumbel_tau=gumbel_tau,
        )
        logger.debug("Positions to mask (local): ", selected_local_idx)
        return selected_local_idx


    def _build_partial_mask_inputs(
        current_rna: str,
        mid_window_local: Tuple[int, int],
        mask_lidx: torch.Tensor,
        prefix_len: int,
        suffix_len: int
    ):
        emb = tokenizer.get_embedder_inputs(protein_seq, current_rna, device=device)
        input_ids = emb["input_ids"].clone()
        attention_mask = emb["attention_mask"]
        rna_abs_start, rna_abs_end = emb["region_spans"]["rna"]
        mid_a_local, mid_b_local = mid_window_local
        mid_abs_start = rna_abs_start + int(mid_a_local)
        mid_abs_end   = rna_abs_start + int(mid_b_local)

        # apply masks
        if mask_lidx.numel() > 0:
            abs_pos = mid_abs_start + mask_lidx
            input_ids[0, abs_pos] = tokenizer.mask_token_id
        residue_type_ids = tokenizer._make_residue_type_ids(input_ids, mask_as_rna=True)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "residue_type_ids": residue_type_ids,
            "rna_region": (rna_abs_start, rna_abs_end),
            "rna_middle_region": (mid_abs_start, mid_abs_end),
            "rna_prefix_len": int(prefix_len),
            "rna_suffix_len": int(suffix_len),
            "rna_middle_len": int(mid_b_local - mid_a_local),
        }

    tok_inputs = tokenizer.get_generator_inputs(
        target_seq=protein_seq,
        middle_length=cfg.rna_length,
        rna_prefix=rna_prefix,
        rna_suffix=rna_suffix,
        device=device,
    )
    mid_abs_start, mid_abs_end = tok_inputs["rna_middle_region"]
    rna_abs_start, rna_abs_end = tok_inputs["rna_region"]
    mid_start_local = int(mid_abs_start - rna_abs_start)
    mid_end_local   = int(mid_abs_end   - rna_abs_start)
    mid_window_local = (mid_start_local, mid_end_local)

    prefix_len = int(tok_inputs["rna_prefix_len"])
    suffix_len = int(tok_inputs["rna_suffix_len"])

    generated = generate_single_rna_iterative(
        model,
        tokenizer,
        tok_inputs,
        cfg.iterate,
        logger=logger,
    )
    current_rna = generated["rna_sequence"]

    score, _ = value_of_seq(prediction_model, protein_seq, current_rna, use_scoring)
    best_rna = current_rna
    best_score = float(score)
    logger.info(f"Seed RNA: {best_rna}  | score={best_score:.4f}")

    confidences_min_threshold = cfg.refine.confidences_min_threshold
    mask_ratio = cfg.refine.initial_mask_ratio
    mask_ratio_rate = cfg.refine.mask_ratio_rate
    min_mask_ratio = cfg.refine.min_mask_ratio
    refine_rounds_max = cfg.refine.max_refine_rounds

    try:
        need_more = (generated["confidences"] < confidences_min_threshold).any().item()
    except Exception:
        need_more = True

    rounds = 0
    
    while need_more and rounds < refine_rounds_max and mask_ratio >= min_mask_ratio and best_score < cfg.refine.max_score:
        rounds += 1
        logits_full = _logits_for_sequence(model, tokenizer, protein_seq, best_rna, device=device) 
        mask_lidx = _choose_positions_from_full_logits(
            logits_full_rna=logits_full,
            tokenizer=tokenizer,
            middle_window=mid_window_local,
            position_selection=cfg.refine.position_selection,
            mask_ratio=mask_ratio,
            allow_unknown_base=getattr(cfg.iterate, "allow_unknown_base", False),
            device=device,
        )

        if mask_lidx.numel() == 0:
            logger.info("[init-refine] no positions selected to mask; stopping.")
            break

        partial_inputs = _build_partial_mask_inputs(
            best_rna,
            mid_window_local,
            mask_lidx,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
        )

        regen = generate_single_rna_iterative(
            model,
            tokenizer,
            partial_inputs,
            cfg.refine,
            logger=logger,
        )
        candidate_rna = regen["rna_sequence"]
        # logger.debug("candidate_ran: ", candidate_rna)
        cand_score, _ = value_of_seq(prediction_model, protein_seq, candidate_rna, use_scoring)

        if cand_score >= best_score:
            best_score = float(cand_score)
            best_rna = candidate_rna
            logger.debug(f"[init-refine] round={rounds} NEW BEST score={best_score:.4f}")
            
        try:
            need_more = (regen["confidences"] < confidences_min_threshold).any().item()
        except Exception:
            need_more = False

        logger.debug("mask_ratio before update: ", mask_ratio, mask_ratio * mask_ratio_rate, mask_ratio_rate)
        mask_ratio = float(mask_ratio - mask_ratio * mask_ratio_rate)
        logger.debug("mask_ratio after update: ", mask_ratio, mask_ratio * mask_ratio_rate, mask_ratio_rate)

    logger.info(f"Mask ratio: {mask_ratio}, Rounds: {rounds}, Best score: {best_score}, Best RNA: {best_rna}, high confidence: {need_more}")
    return best_rna, float(best_score)