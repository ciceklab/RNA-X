import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import os

from .setup import load_models, _setup_logger
from .utils import read_protein_from_csv, save_to_fasta
from .iterative import _iterative_gen_ref
from .mcts import _run_mcts
from .configs import RTGenerationConfig, GenerationResult

def generate_rna(
    protein_name: Optional[str] = None,
    protein_csv_path: Optional[str] = None,
    protein_seq: Optional[str] = None,
    apply_mcts: bool = True,
    use_scoring:bool = True,
    model_path: str = None,
    prediction_model_path: str = None,
    config: Optional[RTGenerationConfig] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    fasta_name: Optional[str] = None,
    *,
    rna_prefix: Optional[str] = None,
    rna_suffix: Optional[str] = None,
) -> GenerationResult:
    print("Starting RNA generation for protein:", protein_name or "provided sequence")
    logger = _setup_logger(log_level=log_level, log_file=log_file)
    cfg = config or RTGenerationConfig()

    if protein_seq is None:
        if not protein_name:
            raise ValueError("Either protein_seq must be provided OR protein_name must be set.")
        if not protein_csv_path:
            raise ValueError("protein_csv_path is required when using protein_name.")
        logger.info(f"Reading protein sequence for '{protein_name}' from CSV: {protein_csv_path}")
        seq = read_protein_from_csv(protein_name.lower(), protein_csv_path)
        if not seq:
            raise ValueError(f"Protein '{protein_name}' not found or empty sequence in CSV '{protein_csv_path}'.")
        protein_seq = "AA:" + seq
    else:
        logger.info("Using provided protein sequence directly.")

    # Load models
    model, tokenizer, prediction, device = load_models(
        model_path=model_path,
        fusion_weights_path=prediction_model_path,
        device_str=cfg.device,
        logger=logger,
        use_scoring=use_scoring
    )

    # TOkennizing the inputs to get the edit window
    tok_inputs_for_window = tokenizer.get_generator_inputs(
        target_seq=protein_seq,
        middle_length=cfg.rna_length,
        rna_prefix=rna_prefix,
        rna_suffix=rna_suffix,
        device=device,
    )
    rna_abs_start, rna_abs_end = tok_inputs_for_window["rna_region"]
    mid_abs_start,  mid_abs_end = tok_inputs_for_window["rna_middle_region"]
    mid_start_local = int(mid_abs_start - rna_abs_start)
    mid_end_local   = int(mid_abs_end   - rna_abs_start)
    edit_window = (mid_start_local, mid_end_local)

    # Iterative generation and refinement
    init_rna, init_score = _iterative_gen_ref(
        model,
        tokenizer,
        prediction,
        protein_seq,
        cfg,
        logger,
        rna_prefix=rna_prefix,
        rna_suffix=rna_suffix,
        use_scoring=use_scoring,
    )
    logger.info("-------------------------------------")
    logger.info(f"Iterative Genreation completed: score={init_score:.4f}, rna sequence={init_rna}")

    if not apply_mcts:
        return GenerationResult(
            best_sequence=init_rna,
            best_value=init_score,
            all_candidates=None,
            init_sequence=None,
            scores_history=None,
            best_scores=None,
        )

    result = _run_mcts(
        model,
        tokenizer,
        prediction,
        protein_seq,
        init_rna,
        use_scoring,
        cfg.mcts,
        logger,
        edit_window=edit_window,
    )

    if output_dir:
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        try:
            save_to_fasta([result.best_sequence], filename=output_dir+(fasta_name or "generated.fasta"))
            logger.info(f"Saved best RNA to FASTA: {output_dir}")
        except Exception as e:
            logger.exception(f"Failed to save FASTA to '{output_dir}': {e}")

    return result