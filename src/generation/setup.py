import logging
from typing import Optional

import torch
import torch.nn as nn

from src.models.RT import RTHFWrapper, RTConfig
from embeddings import create_embedder, EmbeddingConfig
from src.models.RL import RL
from src.utils.tokenizer import ResidueTokenizer

def _setup_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("rna_generator")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if not logger.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def _select_device(pref: str) -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def load_models(
    model_path: str,
    fusion_weights_path: str,
    device_str: str = "cuda",
    logger: Optional[logging.Logger] = None,
    use_scoring: bool = True
):
    logger = logger or _setup_logger()
    device = _select_device(device_str)

    logger.info(f"Loading tokenizer and RT model from: {model_path}")
    tokenizer = ResidueTokenizer()
    if model_path is None:
        raise ValueError("Model path must be provided.")
    config = RTConfig.from_pretrained(model_path)

    model = RTHFWrapper.from_pretrained(model_path, config=config).to(device)
    if device.type == "cuda":
        model = model.half()
    model.eval()

    if use_scoring and fusion_weights_path is None:
        raise ValueError("Fusion weights path must be provided when use_scoring is True.")

    if use_scoring:
        logger.info("Loading fusion scoring model")
        config = EmbeddingConfig(pool="all", batch_size=32, normalize_embeddings=True)
        base_embedder = create_embedder(
            model_path=model_path,
            tokenizer_class=ResidueTokenizer,
            device=device,
            config=config
        )
        tokenizer = ResidueTokenizer()
        rt_model = RL(
            tokenizer=tokenizer,
            proj_dim=1024,
            dropout=0.1,
            cnn_emb_dim=512,
            cnn_channels=512,
            cnn_kernels=(3,5,11)
        ).to(device)

        class RTSequenceEmbedder(nn.Module):
            def __init__(self, base_embedder, rt_model, embed_dim=1024, device="cuda"):
                super().__init__()
                self.base_embedder = base_embedder
                self.rt_model = rt_model
                self.embed_dim = embed_dim
                self.device = device

            @torch.no_grad()
            def _base_encode(self, target_seqs, rna_seqs):
                embedding_results = self.base_embedder.embed_batch(
                    targets=target_seqs,
                    rnas=rna_seqs,
                    show_progress=False,
                    return_pooled=True
                )
                tgt_base = embedding_results['target_embeddings']
                rna_base = embedding_results['rna_embeddings']
                return tgt_base, rna_base

            def forward(self, target_seqs, rna_seqs):
                tgt_base, rna_base = self._base_encode(target_seqs, rna_seqs)
                _, _, target_embeddings, rna_embeddings = self.rt_model(
                    target_seqs=target_seqs,
                    rna_seqs=rna_seqs,
                    target_embeddings=tgt_base,
                    rna_embeddings=rna_base,
                    device=self.device
                )
                return target_embeddings.float(), rna_embeddings.float()

        class Head(nn.Module):
            def __init__(self, embed_dim, hidden_sizes=(1024, 512, 256), dropout=0.4):
                super().__init__()
                input_dim = embed_dim * 2
                layers = []
                prev_dim = input_dim
                for hd in hidden_sizes:
                    layers += [nn.Linear(prev_dim, hd), nn.ReLU(inplace=True), nn.Dropout(dropout)]
                    prev_dim = hd
                layers += [nn.Linear(prev_dim, 1)]
                self.mlp = nn.Sequential(*layers)

            def forward(self, target_embeddings, rna_embeddings):
                x = torch.cat([target_embeddings, rna_embeddings], dim=-1)
                return self.mlp(x)

        class SingleModel(nn.Module):
            def __init__(self, base_embedder, rt_model, embed_dim=1024, device="cuda"):
                super().__init__()
                self.sequence_embedder = RTSequenceEmbedder(base_embedder, rt_model, embed_dim, device)
                self.head = Head(embed_dim=embed_dim)

            def forward(self, target_seqs, rna_seqs):
                tgt_emb, rna_emb = self.sequence_embedder(target_seqs, rna_seqs)
                logits = self.head(tgt_emb, rna_emb)
                return logits.squeeze(-1)


        prediction = SingleModel(base_embedder, rt_model, embed_dim=1024, device=device).to(device)

        ckpt = torch.load(fusion_weights_path, map_location=device)
        missing, unexpected = prediction.load_state_dict(ckpt["model"], strict=False)
        print("Loaded prediction checkpoint.")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        prediction.eval()
        return model, tokenizer, prediction, device
    else:
        return model, tokenizer, None, device