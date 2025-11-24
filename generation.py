#!/usr/bin/env python
import os
import json
import argparse

from src.generation.generate import generate_rna
from src.generation.configs import RTGenerationConfig
from src.generation.utils import save_to_fasta, read_fasta_sequences


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate RNA sequences for protein / DNA / RNA targets using RNA-X."
    )

    parser.add_argument(
        "--targets-fasta",
        required=True,
        help="Path to FASTA file with target sequences (protein/DNA/RNA).",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Directory where FASTA and JSON outputs will be written (default: outputs).",
    )
    parser.add_argument(
        "--n-per-target",
        type=int,
        default=20,
        help="Number of RNA sequences to generate per target (default: 20).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.6,
        help="Minimum score for accepting a generated RNA (default: 0.6).",
    )
    parser.add_argument(
        "--rna-length",
        type=int,
        default=50,
        help="Length of generated RNA sequences (default: 50).",
    )
    parser.add_argument(
        "--target-type",
        choices=["protein", "dna", "rna"],
        default="protein",
        help="Type of target sequences provided in the FASTA file (default: protein).",
    )
    parser.add_argument(
        "--model-path",
        default="weights/checkpoint",
        help="Path to the main RNA-X model checkpoint (default: weights/checkpoint).",
    )
    parser.add_argument(
        "--prediction-model-dir",
        default="weights/prediction",
        help="Directory containing prediction models, e.g. weights/prediction/protein.pth.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to a log file. If not set, logs are written to OUTPUT_DIR/log.",
    )
    parser.add_argument(
        "--no-mcts",
        action="store_true",
        help="Disable MCTS during generation (apply_mcts=False).",
    )
    parser.add_argument(
        "--no-scoring",
        action="store_true",
        help="Disable scoring model (use_scoring=False).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    out_fasta = os.path.join(args.output_dir, "generated_rnas.fasta")
    out_json = os.path.join(args.output_dir, "generated_rnas.json")
    gen_work_dir = args.output_dir

    log_file = (
        args.log_file
        if args.log_file is not None
        else os.path.join(gen_work_dir, "log")
    )

    targets = read_fasta_sequences(args.targets_fasta)
    if not targets:
        raise ValueError(f"No targets found in FASTA file: {args.targets_fasta}")

    all_sequences = []
    all_headers = []
    all_meta = []
    total_attempts = 0

    prediction_model_path = os.path.join(
        args.prediction_model_dir, f"{args.target_type}.pth"
    )

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at: {args.model_path}"
        )
    if not os.path.exists(prediction_model_path) and not args.no_scoring:
        raise FileNotFoundError(
            f"Prediction model not found at: {prediction_model_path}"
        )

    print(f"[info] Loaded {len(targets)} targets from {args.targets_fasta}")
    print(f"[info] Output directory: {args.output_dir}")
    print(f"[info] Generating {args.n_per_target} RNAs per target")
    print(f"[info] Score threshold: {args.score_threshold}")
    print(f"[info] RNA length: {args.rna_length}")
    print(f"[info] Target type: {args.target_type}")
    print(f"[info] Model path: {args.model_path}")
    print(f"[info] Prediction model path: {prediction_model_path}")
    print(f"[info] Log file: {log_file}")
    print()

    for target_idx, (target_name, target_seq) in enumerate(targets, start=1):
        print(
            f"[target {target_idx}/{len(targets)}] {target_name} "
            f"(sequence length = {len(target_seq)})"
        )

        # Prepare the target string expected by generate_rna
        if args.target_type == "protein":
            target_for_model = "AA:" + target_seq
        elif args.target_type == "dna":
            target_for_model = "DNA:" + target_seq
        elif args.target_type == "rna":
            target_for_model = "RNA:" + target_seq
        else:
            raise ValueError(f"Unsupported target type: {args.target_type}")

        accepted_for_this_target = 0

        # Keep generating until we have n_per_target for THIS target
        while accepted_for_this_target < args.n_per_target:
            total_attempts += 1

            cfg = RTGenerationConfig()
            cfg.rna_length = args.rna_length

            result = generate_rna(
                protein_seq=target_for_model,
                config=cfg,
                log_file=log_file,
                model_path=args.model_path,
                prediction_model_path=prediction_model_path,
                output_dir=None,
                fasta_name=None,
                use_scoring=not args.no_scoring,
                apply_mcts=not args.no_mcts,
            )

            score = getattr(result, "best_value", None)
            seq = getattr(result, "best_sequence", None)
            scores_all = getattr(result, "best_scores", None)

            if seq is None or score is None:
                print(
                    f"[warn] Attempt {total_attempts}: generation failed; "
                    f"retrying this target."
                )
                continue

            # Strip "RNA:" prefix if present
            if seq.startswith("RNA:"):
                seq = seq[4:]

            if score >= args.score_threshold:
                global_idx = len(all_sequences) + 1
                header = (
                    f"RNA-X_{global_idx}|target={target_name}|"
                    f"len={len(seq)}|score={score:.4f}"
                )

                all_sequences.append(seq)
                all_headers.append(header)
                all_meta.append(
                    {
                        "global_index": global_idx,
                        "target_index": target_idx,
                        "target_name": target_name,
                        "header": header,
                        "score": float(score),
                        "scores": scores_all,
                        "length": len(seq),
                        "target_type": args.target_type,
                    }
                )
                accepted_for_this_target += 1

                print(
                    f"[ok] Accepted RNA_{global_idx} for target "
                    f"{target_name} (len={len(seq)}, score={score:.3f}) "
                    f"[{accepted_for_this_target}/{args.n_per_target}]"
                )
            else:
                print(
                    f"[low] Attempt {total_attempts}: score={score:.3f} "
                    f"< threshold {args.score_threshold}; retrying."
                )

        print(f"[target {target_idx}] Completed {args.n_per_target} sequences.\n")

    # Save results once at the end
    save_to_fasta(all_sequences, filename=out_fasta, headers=all_headers)

    summary = {
        "targets_fasta": os.path.abspath(args.targets_fasta),
        "output_fasta": os.path.abspath(out_fasta),
        "n_targets": len(targets),
        "n_generated_total": len(all_sequences),
        "n_per_target": args.n_per_target,
        "rna_length": args.rna_length,
        "score_threshold": args.score_threshold,
        "target_type": args.target_type,
        "model_path": os.path.abspath(args.model_path),
        "prediction_model_path": os.path.abspath(prediction_model_path),
        "attempts": total_attempts,
        "details": all_meta,
    }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] Wrote {len(all_sequences)} RNAs to {out_fasta}")
    print(f"[done] Wrote metadata to {out_json}")


if __name__ == "__main__":
    main()



