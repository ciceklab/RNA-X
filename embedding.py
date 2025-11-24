import argparse
import os
from pathlib import Path
import torch

from src.representation.embedder import setup_embedder

def read_sequences(path: str):
    seqs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seqs.append(line)
    if not seqs:
        raise ValueError(f"No sequences found in {path}")
    return seqs


def main():
    parser = argparse.ArgumentParser(
        description="Compute RT representations for (target, RNA) sequence pairs."
    )

    parser.add_argument(
        "--targets-file",
        default="examples/dna.fasta",
        help="Text file with one target sequence per line (AA/DNA/RNA).",
    )
    parser.add_argument(
        "--rnas-file",
        default="examples/generated_rnas.fasta",
        help="Text file with one RNA sequence per line (paired with targets-file).",
    )
    parser.add_argument(
        "--base-model-path",
        default="weights/checkpoint",
        help="Path to the base RT model directory (used by create_embedder).",
    )
    parser.add_argument(
        "--rt-model-path",
        default="weights/representation/dna.pt",
        help=(
            "Path to RT head checkpoint (e.g. weights/representation/dna.pt). "
            "Required if --rt-type head."
        ),
    )
    parser.add_argument(
        "--rt-type",
        choices=["head", "base"],
        default="head",
        help=(
            "Representation mode: "
            "'head' = base RT + fusion head, "
            "'base' = base RT only (no head)."
        ),
    )
    parser.add_argument(
        "--output-path",
        default="outputs/rt_embeddings.pt",
        help="Where to save embeddings (PyTorch .pt file).",
    )

    args = parser.parse_args()

    targets = read_sequences(args.targets_file)
    rnas = read_sequences(args.rnas_file)
    if len(targets) != len(rnas):
        raise ValueError(
            f"targets-file has {len(targets)} lines but rnas-file has {len(rnas)} lines"
        )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    embedder = setup_embedder(
        embedder_type="rt",
        base_model_path=args.base_model_path,
        rt_model_path=args.rt_model_path if args.rt_type == "head" else None,
        rt_type=args.rt_type,
    )

    print(f"Embedding {len(targets)} pairs with rt_type={args.rt_type} ...")
    target_emb, rna_emb = embedder.embed_sequences(targets, rnas)

    out = {
        "targets": targets,
        "rnas": rnas,
        "target_embeddings": target_emb.cpu(),
        "rna_embeddings": rna_emb.cpu(),
        "embed_dim": target_emb.shape[-1],
    }
    torch.save(out, args.output_path)

    print(f"[done] Saved embeddings to {args.output_path}")

if __name__ == "__main__":
    main()
