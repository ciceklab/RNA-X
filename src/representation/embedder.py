import os
import gc
import json
import pickle
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
import warnings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class EmbeddingConfig:
    pool: str = "mean"       # "mean", "max", "cls", "all"
    batch_size: int = 8
    max_length: int = 4096
    device: str = "auto"
    precision: str = "float32"  # "float32", "float16", "bfloat16"
    return_attention_weights: bool = False
    normalize_embeddings: bool = True


class RTEmbedder:
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "auto",
        config: Optional[EmbeddingConfig] = None,
    ):
        self.config = config or EmbeddingConfig()
        self._validate_config()

        self.device = self._setup_device(device)
        self.tokenizer = tokenizer

        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None")
        if model is None:
            raise ValueError("Model cannot be None")

        self.model = model.to(self.device)
        self.model.eval()

        if self.config.precision == "float16":
            if not torch.cuda.is_available() and self.device.type == "cuda":
                logger.warning("float16 requested but CUDA not available, using float32")
                self.config.precision = "float32"
            else:
                self.model = self.model.half()
        elif self.config.precision == "bfloat16":
            if not torch.cuda.is_available():
                logger.warning("bfloat16 requested but CUDA not available, using float32")
                self.config.precision = "float32"
            else:
                self.model = self.model.bfloat16()

        self._validate_model_state()
        print(f"RTEmbedder on {self.device} ({self.config.precision})")

    def _validate_config(self):
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        pools = ["mean", "max", "cls", "all"]
        if self.config.pool not in pools:
            raise ValueError(f"pool must be one of {pools}")

        precisions = ["float32", "float16", "bfloat16"]
        if self.config.precision not in precisions:
            raise ValueError(f"precision must be one of {precisions}")

        if not isinstance(self.config.normalize_embeddings, bool):
            raise ValueError("normalize_embeddings must be bool")

        if not isinstance(self.config.return_attention_weights, bool):
            raise ValueError("return_attention_weights must be bool")

    def _validate_model_state(self):
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("Model not initialized")

        if self.model.training:
            logger.warning("Model was in train mode, switching to eval()")
            self.model.eval()

        if not hasattr(self.model, "__call__"):
            raise RuntimeError("Model is not callable")

    def _setup_device(self, device: Union[str, torch.device]) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"Using CUDA: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                print("CUDA not available, using CPU")

        device = torch.device(device)

        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")

        if device.type == "cuda":
            try:
                torch.cuda.get_device_properties(device.index or 0)
            except Exception as e:
                logger.warning(f"Cannot access CUDA device {device}, falling back to CPU: {e}")
                device = torch.device("cpu")

        return device

    def _get_memory_info(self) -> Dict[str, float]:
        info = {}
        try:
            mem = psutil.virtual_memory()
            info["system_used_gb"] = mem.used / 1024**3
            info["system_total_gb"] = mem.total / 1024**3
            info["system_percent"] = mem.percent
            info["system_avail_gb"] = mem.available / 1024**3
        except Exception as e:
            logger.warning(f"Could not get system memory info: {e}")

        if torch.cuda.is_available() and self.device.type == "cuda":
            try:
                idx = self.device.index or 0
                info["gpu_alloc_gb"] = torch.cuda.memory_allocated(idx) / 1024**3
                info["gpu_reserved_gb"] = torch.cuda.memory_reserved(idx) / 1024**3
                info["gpu_total_gb"] = torch.cuda.get_device_properties(idx).total_memory / 1024**3
                info["gpu_free_gb"] = info["gpu_total_gb"] - info["gpu_alloc_gb"]
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")

        return info

    def _check_memory_before_batch(self, batch_size: int, seq_lens: List[int]) -> int:
        if not seq_lens:
            return batch_size

        if self.device.type == "cuda":
            try:
                idx = self.device.index or 0
                total_mem = torch.cuda.get_device_properties(idx).total_memory
                alloc_mem = torch.cuda.memory_allocated(idx)
                avail_mem = total_mem - alloc_mem

                max_len = max(seq_lens) if seq_lens else self.config.max_length
                hidden_dim = 768
                est_mem = batch_size * max_len * hidden_dim * 4 * 2

                if est_mem > avail_mem * 0.7:
                    new_bs = max(1, int(batch_size * avail_mem * 0.7 / est_mem))
                    logger.warning(f"Reducing batch size {batch_size} -> {new_bs} due to memory")
                    return new_bs

            except Exception as e:
                logger.warning(f"Could not check GPU memory: {e}")

        return batch_size

    def _check_gradients(self):
        if torch.is_grad_enabled():
            logger.warning("Gradients enabled during inference")

    def _check_for_anomalies(self, t: torch.Tensor, name: str):
        if t is None:
            raise ValueError(f"{name} is None")
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} is not a torch.Tensor")
        if t.numel() == 0:
            raise ValueError(f"{name} is empty")
        if torch.isnan(t).any():
            raise RuntimeError(f"NaN in {name}")
        if torch.isinf(t).any():
            raise RuntimeError(f"Inf in {name}")

    def _validate_tensor_shapes(
        self, hidden: torch.Tensor, spans: Union[Dict, List[Dict]]
    ):
        if hidden is None:
            raise ValueError("hidden_states is None")

        if hidden.dim() != 3:
            raise ValueError(f"Expected (B, L, H), got {hidden.shape}")

        bsz, seqlen, hdim = hidden.shape
        if bsz == 0 or seqlen == 0 or hdim == 0:
            raise ValueError(f"Invalid tensor shape: {hidden.shape}")

        if hdim < 64:
            logger.warning(f"Suspiciously small hidden dim: {hdim}")

        spans_list = spans if isinstance(spans, list) else [spans]

        for i, s in enumerate(spans_list):
            if not isinstance(s, dict):
                raise TypeError(f"region_spans[{i}] must be dict")

            for name, span in s.items():
                if not isinstance(span, (list, tuple)) or len(span) != 2:
                    raise ValueError(f"Bad span for {name}: {span}")

                start, end = span
                if not isinstance(start, int) or not isinstance(end, int):
                    raise TypeError(f"Span indices must be int for {name}: {span}")

                if start < 0 or end > seqlen or start >= end:
                    raise ValueError(f"Invalid span {name} [{start}:{end}] for seq_len {seqlen}")

                if end - start == 0:
                    raise ValueError(f"Empty span for {name}: [{start}:{end}]")

    def _validate_tokenizer_output(self, inputs: Dict):
        if not isinstance(inputs, dict):
            raise TypeError("Tokenizer output must be dict")

        for key in ["input_ids", "region_spans"]:
            if key not in inputs:
                raise ValueError(f"Tokenizer output missing key: {key}")

        input_ids = inputs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be tensor")

        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D, got {input_ids.shape}")

        if input_ids.size(1) > self.config.max_length:
            logger.warning(
                f"seq len {input_ids.size(1)} > max_length {self.config.max_length}"
            )

        if input_ids.size(1) == 0:
            raise ValueError("Tokenizer produced empty sequence")

    def _cleanup_and_validate(self):
        if self.device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error during CUDA cleanup: {e}")

    def _validate_sequences(
        self, targets: List[str], rnas: List[str]
    ) -> Tuple[List[str], List[str], List[int]]:
        if not isinstance(targets, list) or not isinstance(rnas, list):
            raise TypeError("targets and rnas must be lists")

        if len(targets) != len(rnas):
            raise ValueError(f"Length mismatch: {len(targets)} targets vs {len(rnas)} RNAs")

        if len(targets) == 0:
            raise ValueError("Empty input lists")

        good_t = []
        good_r = []
        good_idx = []

        for i, (t, r) in enumerate(zip(targets, rnas)):
            try:
                if not isinstance(t, str) or not isinstance(r, str):
                    logger.warning(f"skip {i}: non-string")
                    continue

                if not t.strip() or not r.strip():
                    logger.warning(f"skip {i}: empty string")
                    continue

                if not t.startswith(("DNA:", "AA:", "RNA:")):
                    if self._is_dna_sequence(t):
                        t = "DNA:" + t.upper().replace("U", "T")
                    elif self._is_protein_sequence(t):
                        t = "AA:" + t.upper()
                    elif self._is_rna_sequence(t):
                        t = "RNA:" + t.upper()
                    else:
                        logger.warning(f"skip {i}: unknown target type")
                        continue

                if not r.startswith("RNA:"):
                    if self._is_rna_sequence(r):
                        r = "RNA:" + r.upper().replace("T", "U")
                    else:
                        logger.warning(f"skip {i}: bad RNA")
                        continue

                if len(t) < 4 or len(r) < 4:
                    logger.warning(f"skip {i}: too short")
                    continue

                good_t.append(t)
                good_r.append(r)
                good_idx.append(i)

            except Exception as e:
                logger.warning(f"skip {i}: {e}")
                continue

        if not good_t:
            raise ValueError("No valid sequences after validation")

        return good_t, good_r, good_idx

    def _is_dna_sequence(self, seq: str) -> bool:
        if not seq:
            return False
        s = seq.strip().upper()
        return len(s) > 0 and all(c in "ATCGXN-" for c in s)

    def _is_rna_sequence(self, seq: str) -> bool:
        if not seq:
            return False
        s = seq.strip().upper()
        return len(s) > 0 and all(c in "AUCGXN-" for c in s)

    def _is_protein_sequence(self, seq: str) -> bool:
        if not seq:
            return False
        allowed = set("ACDEFGHIKLMNPQRSTVWYXZ-")
        s = seq.strip().upper()
        return len(s) > 0 and all(c in allowed for c in s)

    def _extract_embeddings_from_output(self, outputs) -> torch.Tensor:
        if outputs is None:
            raise ValueError("Model outputs is None")

        emb = None

        if hasattr(outputs, "embeddings") and outputs.embeddings is not None:
            emb = outputs.embeddings
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            emb = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            emb = outputs.last_hidden_state
        else:
            raise RuntimeError("Cannot extract embeddings from model output")

        if emb is None:
            raise RuntimeError("Extracted embeddings is None")

        self._check_for_anomalies(emb, "embeddings")
        return emb

    def _pool_embeddings(self, emb: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        if emb is None:
            raise ValueError("embeddings is None")
        if not isinstance(emb, torch.Tensor):
            raise TypeError("embeddings must be tensor")
        if emb.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {emb.shape}")
        if emb.size(0) == 0:
            raise ValueError("Cannot pool empty sequence")

        self._check_for_anomalies(emb, "pool_input")

        if pool == "mean":
            out = emb.mean(dim=0)
        elif pool == "max":
            out = emb.max(dim=0).values
        elif pool == "cls":
            out = emb[0]
        elif pool == "all":
            out = emb
        else:
            raise ValueError(f"Unknown pooling: {pool}")

        self._check_for_anomalies(out, f"pooled_{pool}")
        return out

    @torch.no_grad()
    def embed_single(
        self,
        target: str,
        rna: str,
        pool: Optional[str] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if not isinstance(target, str) or not isinstance(rna, str):
            raise TypeError("target and rna must be strings")
        if not target.strip() or not rna.strip():
            raise ValueError("target and rna cannot be empty")

        self._validate_model_state()
        self._check_gradients()

        pool = pool or self.config.pool
        targets, rnas, _ = self._validate_sequences([target], [rna])
        if not targets:
            raise ValueError("Invalid sequences")

        target, rna = targets[0], rnas[0]

        try:
            inputs = self.tokenizer.get_embedder_inputs(
                target, rna, device=self.device, max_length=self.config.max_length
            )

            self._validate_tokenizer_output(inputs)

            t0 = time.time()
            outputs = self.model(**inputs)
            infer_time = time.time() - t0

            if outputs is None:
                raise RuntimeError("Model returned None")

            hidden = self._extract_embeddings_from_output(outputs)
            spans = inputs["region_spans"]

            self._validate_tensor_shapes(hidden, spans)

            t_start, t_end = spans["target"]
            r_start, r_end = spans["rna"]

            t_tokens = hidden[0, t_start:t_end]
            r_tokens = hidden[0, r_start:r_end]
            cls_emb = hidden[0, 0]

            self._check_for_anomalies(t_tokens, "target_tokens")
            self._check_for_anomalies(r_tokens, "rna_tokens")
            self._check_for_anomalies(cls_emb, "cls")

            if pool == "cls":
                t_emb = cls_emb
                r_emb = cls_emb
            else:
                t_emb = self._pool_embeddings(t_tokens, pool)
                r_emb = self._pool_embeddings(r_tokens, pool)

            if self.config.normalize_embeddings:
                t_emb = torch.nn.functional.normalize(t_emb, dim=-1)
                r_emb = torch.nn.functional.normalize(r_emb, dim=-1)
                self._check_for_anomalies(t_emb, "norm_target")
                self._check_for_anomalies(r_emb, "norm_rna")

            if return_dict:
                out = {
                    "target_embedding": t_emb,
                    "rna_embedding": r_emb,
                    "cls_embedding": cls_emb,
                    "target_token_embeddings": t_tokens,
                    "rna_token_embeddings": r_tokens,
                    "all_embeddings": hidden[0],
                    "region_spans": spans,
                    "input_ids": inputs["input_ids"][0],
                    "inference_time": infer_time,
                }

                if self.config.return_attention_weights and hasattr(outputs, "attentions"):
                    if outputs.attentions is not None:
                        out["attention_weights"] = outputs.attentions

                return out
            else:
                return t_emb, r_emb

        except Exception as e:
            logger.error(f"embed_single error: {e}")
            raise
        finally:
            self._cleanup_and_validate()

    def _pool_embeddings_masked(
        self, emb: torch.Tensor, mask: Optional[torch.Tensor], pool: str
    ):
        if pool == "mean":
            if mask is None:
                return emb.mean(0)
            m = emb * mask.unsqueeze(-1)
            denom = mask.sum().clamp_min(1.0)
            return m.sum(0) / denom
        if pool == "max":
            if mask is None:
                return emb.max(0).values
            m = emb.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            return m.max(0).values
        if pool == "cls":
            return emb[0]
        if pool == "all":
            return emb
        raise ValueError(pool)

    @torch.no_grad()
    def embed_batch(
        self,
        targets: List[str],
        rnas: List[str],
        pool: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
        return_failed_indices: bool = False,
        return_pooled: bool = False,
    ) -> Dict[str, List[torch.Tensor]]:
        if not isinstance(targets, list) or not isinstance(rnas, list):
            raise TypeError("targets and rnas must be lists")
        if not targets or not rnas:
            raise ValueError("Empty input lists")
        if len(targets) != len(rnas):
            raise ValueError("targets and rnas length mismatch")

        self._validate_model_state()
        self._check_gradients()

        pool = pool or self.config.pool
        batch_size = batch_size or self.config.batch_size

        good_t, good_r, good_idx = self._validate_sequences(targets, rnas)
        if not good_t:
            raise ValueError("No valid sequences found")

        seq_lens = [len(t) + len(r) for t, r in zip(good_t, good_r)]
        adapt_bs = self._check_memory_before_batch(batch_size, seq_lens)

        all_t = []
        all_r = []
        all_t_pooled = []
        all_r_pooled = []
        all_cls = []
        failed: List[int] = []

        total_batches = (len(good_t) + adapt_bs - 1) // adapt_bs
        pbar = None
        if show_progress:
            pbar = tqdm(total=total_batches, desc="batches")

        for i in range(0, len(good_t), adapt_bs):
            retries = 0
            max_retries = 2
            cur_bs = adapt_bs
            done = False

            while retries <= max_retries and not done:
                try:
                    j = min(i + cur_bs, len(good_t))
                    batch_t = good_t[i:j]
                    batch_r = good_r[i:j]

                    if not batch_t:
                        break

                    inputs = self.tokenizer.get_batch_embedder_inputs(
                        batch_t,
                        batch_r,
                        device=self.device,
                        max_length=self.config.max_length,
                    )

                    self._validate_tokenizer_output(inputs)

                    outputs = self.model(**inputs)
                    if outputs is None:
                        raise RuntimeError("Model returned None")

                    hidden = self._extract_embeddings_from_output(outputs)
                    span_batch = inputs["region_spans"]

                    self._validate_tensor_shapes(hidden, span_batch)

                    for k, spans in enumerate(span_batch):
                        try:
                            t_start, t_end = spans["target"]
                            r_start, r_end = spans["rna"]

                            t_tokens = hidden[k, t_start:t_end]
                            r_tokens = hidden[k, r_start:r_end]
                            cls_emb = hidden[k, 0]

                            self._check_for_anomalies(
                                t_tokens, f"target_tokens_batch_{i}_seq_{k}"
                            )
                            self._check_for_anomalies(
                                r_tokens, f"rna_tokens_batch_{i}_seq_{k}"
                            )
                            self._check_for_anomalies(
                                cls_emb, f"cls_batch_{i}_seq_{k}"
                            )

                            if pool == "cls":
                                t_emb = cls_emb
                                r_emb = cls_emb
                            else:
                                attn = inputs["attention_mask"][k]
                                t_mask = attn[t_start:t_end]
                                r_mask = attn[r_start:r_end]

                                t_emb = self._pool_embeddings_masked(
                                    hidden[k, t_start:t_end], t_mask, pool
                                )
                                r_emb = self._pool_embeddings_masked(
                                    hidden[k, r_start:r_end], r_mask, pool
                                )

                                if return_pooled:
                                    t_emb_p = self._pool_embeddings_masked(
                                        hidden[k, t_start:t_end], t_mask, "mean"
                                    )
                                    r_emb_p = self._pool_embeddings_masked(
                                        hidden[k, r_start:r_end], r_mask, "mean"
                                    )

                            if self.config.normalize_embeddings:
                                t_emb = torch.nn.functional.normalize(t_emb, dim=-1)
                                r_emb = torch.nn.functional.normalize(r_emb, dim=-1)
                                if return_pooled:
                                    t_emb_p = torch.nn.functional.normalize(t_emb_p, dim=-1)
                                    r_emb_p = torch.nn.functional.normalize(r_emb_p, dim=-1)

                                self._check_for_anomalies(
                                    t_emb,
                                    f"norm_target_batch_{i}_seq_{k}",
                                )
                                self._check_for_anomalies(
                                    r_emb, f"norm_rna_batch_{i}_seq_{k}"
                                )

                            all_t.append(t_emb.cpu())
                            all_r.append(r_emb.cpu())
                            all_cls.append(cls_emb.cpu())
                            all_t_pooled.append(t_emb_p.cpu() if return_pooled else None)
                            all_r_pooled.append(r_emb_p.cpu() if return_pooled else None)

                        except Exception as e:
                            logger.error(
                                f"Failed seq {k} in batch {i//adapt_bs + 1}: {e}"
                            )
                            all_t.append(None)
                            all_r.append(None)
                            all_cls.append(None)
                            failed.append(good_idx[i + k])

                    done = True

                    del hidden, outputs, inputs
                    if "t_tokens" in locals():
                        del t_tokens, r_tokens, cls_emb

                    self._cleanup_and_validate()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and retries < max_retries:
                        retries += 1
                        cur_bs = max(1, cur_bs // 2)
                        logger.warning(
                            f"OOM in batch {i//adapt_bs + 1}, retry with batch_size={cur_bs}"
                        )
                        self._cleanup_and_validate()
                        time.sleep(1)
                        continue
                    else:
                        logger.error(
                            f"Failed batch {i//adapt_bs + 1}: {e}"
                        )
                        actual_bs = min(cur_bs, len(good_t) - i)
                        idxs = list(range(i, i + actual_bs))
                        failed.extend([good_idx[idx] for idx in idxs])

                        for _ in range(actual_bs):
                            all_t.append(None)
                            all_r.append(None)
                            all_cls.append(None)

                        done = True
                        self._cleanup_and_validate()

                except Exception as e:
                    logger.error(
                        f"Unexpected error in batch {i//adapt_bs + 1}: {e}"
                    )
                    actual_bs = min(cur_bs, len(good_t) - i)
                    idxs = list(range(i, i + actual_bs))
                    failed.extend([good_idx[idx] for idx in idxs])

                    for _ in range(actual_bs):
                        all_t.append(None)
                        all_r.append(None)
                        all_cls.append(None)

                    done = True
                    self._cleanup_and_validate()

            if show_progress and pbar is not None:
                pbar.update(1)

        if show_progress and pbar is not None:
            pbar.close()

        self._cleanup_and_validate()

        expected = len(good_t)
        actual = len(all_t)
        if actual != expected:
            logger.warning(f"Result count mismatch: expected {expected}, got {actual}")

        if return_pooled:
            res = {
                "target_embeddings": all_t,
                "rna_embeddings": all_r,
                "cls_embeddings": all_cls,
                "valid_indices": good_idx,
                "target_pooled_embeddings": all_t_pooled,
                "rna_pooled_embeddings": all_r_pooled,
            }
        else:
            res = {
                "target_embeddings": all_t,
                "rna_embeddings": all_r,
                "cls_embeddings": all_cls,
                "valid_indices": good_idx,
            }

        if return_failed_indices:
            res["failed_indices"] = failed

        return res


def create_embedder(
    model_path: str,
    tokenizer_class,
    device: Union[str, torch.device] = "auto",
    config: Optional[EmbeddingConfig] = None,
) -> RTEmbedder:
    if not model_path:
        raise ValueError("model_path cannot be empty")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if tokenizer_class is None:
        raise ValueError("tokenizer_class cannot be None")

    try:
        from src.models.RT import RTHFWrapper, RTConfig

        tokenizer = tokenizer_class()
        if tokenizer is None:
            raise RuntimeError("Failed to create tokenizer")

        if not os.path.exists(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(f"config.json not found in {model_path}")

        cfg = RTConfig.from_pretrained(model_path)
        if cfg is None:
            raise RuntimeError("Failed to load RT config")

        model = RTHFWrapper.from_pretrained(model_path, config=cfg)
        model.eval().half()

        if model is None:
            raise RuntimeError("Failed to load RT model")

        emb = RTEmbedder(model, tokenizer, device=device, config=config)
        print(f"Loaded RT model from {model_path}")

        try:
            test = emb.embed_single("DNA:ATCG", "RNA:AUCG", return_dict=False)
            if test is None or len(test) != 2:
                raise RuntimeError("Embedder test failed")
            logger.info("Embedder test passed")
        except Exception as e:
            logger.warning(f"Embedder test failed: {e}")

        return emb

    except Exception as e:
        logger.error(f"create_embedder failed: {e}")
        raise


def setup_rt_embedder(rt_model_path, base_model_path, rt_type: str = "head", config=None):
    from src.models.RL import RL2
    from src.utils.tokenizer import ResidueTokenizer

    print("Setting up RT embedder...")
    if config is None:
        print("rt_type:", rt_type)
        if rt_type == "head":
            config = EmbeddingConfig(pool="all", batch_size=64, normalize_embeddings=True)
        else:
            config = EmbeddingConfig(pool="mean", batch_size=64, normalize_embeddings=True)

    base_emb = create_embedder(
        model_path=base_model_path,
        tokenizer_class=ResidueTokenizer,
        device=DEVICE,
        config=config,
    )

    rt_model = None
    if rt_type == "head":
        tok = ResidueTokenizer()
        rt_model = RL2(
            tokenizer=tok,
            proj_dim=1024,
            dropout=0.1,
            cnn_emb_dim=512,
            cnn_channels=512,
            cnn_kernels=(3, 5, 11),
        ).to(DEVICE)
        # rt_model = RL2(
        #     tokenizer=tok,
        #     proj_dim=1024,
        #     dropout=0.1,
        #     cnn_emb_dim=1024,
        #     cnn_channels=1024,
        #     cnn_kernels=(3,5,7,9,11)
        # ).to(DEVICE)

        ckpt = torch.load(rt_model_path, map_location=DEVICE)
        rt_model.load_state_dict(ckpt["model_state_dict"])
        rt_model.eval()
        print("RT head params:", sum(p.numel() for p in rt_model.parameters() if p.requires_grad))

        for p in rt_model.parameters():
            p.requires_grad = False

        print("RT head ready")

    return base_emb, rt_model


class RTSequenceEmbedder:
    def __init__(self, base_embedder: RTEmbedder, rt_model: Optional[nn.Module]):
        self.base_embedder = base_embedder
        self.rt_model = rt_model
        self.embed_dim = 1024 if self.rt_model else 768

    def embed_sequences(self, target_seqs, rna_seqs):
        with torch.no_grad():
            base_res = self.base_embedder.embed_batch(
                targets=target_seqs,
                rnas=rna_seqs,
                show_progress=False,
                return_pooled=True,
            )
            if self.rt_model:
                _, _, t_emb, r_emb = self.rt_model(
                    target_seqs=target_seqs,
                    rna_seqs=rna_seqs,
                    target_embeddings=base_res["target_embeddings"],
                    rna_embeddings=base_res["rna_embeddings"],
                    device=DEVICE,
                )
            else:
                t_emb = torch.stack(base_res["target_embeddings"])
                r_emb = torch.stack(base_res["rna_embeddings"])

        return t_emb, r_emb


def setup_embedder(embedder_type: str = "rt", **kwargs):
    if embedder_type != "rt":
        raise ValueError(f"Only 'rt' embedder_type is supported, got: {embedder_type}")

    rt_ckpt = kwargs.get("rt_model_path")
    base_ckpt = kwargs.get("base_model_path")
    cfg = kwargs.get("config", None)
    rt_type = kwargs.get("rt_type", "head")

    if not base_ckpt:
        raise ValueError("RT embedder needs base_model_path")
    if rt_type == "head" and not rt_ckpt:
        raise ValueError("RT head needs rt_model_path")

    base_emb, rt_model = setup_rt_embedder(
        rt_model_path=rt_ckpt,
        base_model_path=base_ckpt,
        rt_type=rt_type,
        config=cfg,
    )
    return RTSequenceEmbedder(base_emb, rt_model)
