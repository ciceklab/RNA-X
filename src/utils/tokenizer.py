import os
import torch
from transformers import PreTrainedTokenizer
from typing import Dict, Tuple, Optional, Union, List

VOCAB = [
    "[PAD]", "[CLS]", "[EOS]", "$", "[MASK]",
    *[f"AA_{aa}" for aa in list("ACDEFGHIKLMNPQRSTVWY")], "AA_X",
    *[f"RNA_{b}" for b in list("ACGU")], "RNA_X",
    *[f"DNA_{b}" for b in list("ACGT")], "DNA_X",
]

TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}

class ResidueTokenizer(PreTrainedTokenizer):
    vocab_files_names = {}
    max_model_input_sizes = {}

    def __init__(self):
        self.token_to_id = TOKEN_TO_ID
        self.id_to_token = ID_TO_TOKEN
        
        self._residue_type_mapping = {}
        for token_type, idx in {'AA_': 1, 'RNA_': 2, 'DNA_': 3}.items():
            for token_str, token_id in self.token_to_id.items():
                if token_str.startswith(token_type):
                    self._residue_type_mapping[token_id] = idx

        super().__init__(
            cls_token="[CLS]",
            sep_token="$",
            eos_token="[EOS]",
            mask_token="[MASK]",
            pad_token="[PAD]",
        )

    def _tokenize(self, text: str):
        seq_type, seq = text.split(":", 1)
        if seq_type not in ("AA", "RNA", "DNA"):
            raise ValueError(f"Unknown sequence type {seq_type!r}")
        prefix = f"{seq_type}_"

        tokens = []
        for c in seq:
            tok = prefix + c.upper()
            if tok not in self.token_to_id:
                tok = prefix + "X"
            tokens.append(tok)
        return tokens

    def _convert_token_to_id(self, token: str):
        return self.token_to_id[token]

    def _convert_id_to_token(self, index: int):
        return self.id_to_token[index]

    def get_vocab(self):
        return dict(self.token_to_id)

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def build_inputs_with_special_tokens(self, target_tokens_id, rna_token_ids=None):
        cls_id = self.convert_tokens_to_ids(self.cls_token)
        sep_id = self.convert_tokens_to_ids(self.sep_token)
        eos_id = self.convert_tokens_to_ids(self.eos_token)

        if rna_token_ids is None:
            return [cls_id] + target_tokens_id + [eos_id]
        return [cls_id] + target_tokens_id + [sep_id] + rna_token_ids + [eos_id]
    

    def build_inputs_with_special_tokens_for_generation(self, target_tokens_id, rna_token_ids=None):
        cls_id = self.convert_tokens_to_ids(self.cls_token)
        sep_id = self.convert_tokens_to_ids(self.sep_token)

        if rna_token_ids is None:
            return [cls_id] + target_tokens_id
        return [cls_id] + target_tokens_id + [sep_id] + rna_token_ids 

    def get_special_tokens_mask(self, target_tokens_id, rna_token_ids=None):
        seq = self.build_inputs_with_special_tokens(target_tokens_id, rna_token_ids)
        return [1 if tid in (
            self.cls_token_id,
            self.sep_token_id,
            self.eos_token_id,
            self.mask_token_id,
            self.pad_token_id
        ) else 0 for tid in seq]

    def _make_residue_type_ids(self, input_ids: torch.Tensor, mask_as_rna: bool = False) -> torch.Tensor:
        residue_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        input_ids_flat = input_ids.flatten()
        residue_type_flat = residue_type_ids.flatten()
        
        for token_id, type_id in self._residue_type_mapping.items():
            mask = input_ids_flat == token_id
            residue_type_flat[mask] = type_id
        
        # Handle special tokens
        special_tokens = torch.tensor([
            self.cls_token_id, self.sep_token_id, self.eos_token_id, self.pad_token_id
        ], device=input_ids.device)
        special_mask = torch.isin(input_ids, special_tokens)
        residue_type_ids[special_mask] = 4
        
        if mask_as_rna:
            mask_token_mask = input_ids == self.mask_token_id
            residue_type_ids[mask_token_mask] = 2
        else:
            mask_token_mask = input_ids == self.mask_token_id
            residue_type_ids[mask_token_mask] = 4
        
        return residue_type_ids

    def _make_residue_type_ids_vectorized(self, input_ids: torch.Tensor, mask_as_rna: bool = False) -> torch.Tensor:
        residue_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        
        if not hasattr(self, '_token_ids_tensor'):
            token_ids = list(self._residue_type_mapping.keys())
            type_ids = list(self._residue_type_mapping.values())
            self._token_ids_tensor = torch.tensor(token_ids, device=input_ids.device)
            self._type_ids_tensor = torch.tensor(type_ids, device=input_ids.device)
        
        if self._token_ids_tensor.device != input_ids.device:
            self._token_ids_tensor = self._token_ids_tensor.to(input_ids.device)
            self._type_ids_tensor = self._type_ids_tensor.to(input_ids.device)
        
        for token_id, type_id in zip(self._token_ids_tensor, self._type_ids_tensor):
            mask = input_ids == token_id
            residue_type_ids[mask] = type_id
        
        special_tokens = torch.tensor([
            self.cls_token_id, self.sep_token_id, self.eos_token_id, self.pad_token_id
        ], device=input_ids.device)
        
        for special_token in special_tokens:
            special_mask = input_ids == special_token
            residue_type_ids[special_mask] = 4
        
        mask_token_mask = input_ids == self.mask_token_id
        if mask_as_rna:
            residue_type_ids[mask_token_mask] = 2
        else:
            residue_type_ids[mask_token_mask] = 4
        
        return residue_type_ids

    def get_embedder_inputs(self, target_seq: str, rna_seq: str, device: Union[str, torch.device] = "cpu", max_length: Optional[int] = 4096, return_tensors: bool = True) -> Dict[str, Union[torch.Tensor, Tuple[int, int]]]:
        if not target_seq.startswith(("DNA:", "AA:", "RNA:")):
            raise ValueError("Target sequence must start with ")
        if not rna_seq.startswith("RNA:"):
            raise ValueError("RNA sequence must start with")
        
        target_tokens = self._tokenize(target_seq)
        rna_tokens = self._tokenize(rna_seq)
        
        target_ids = self.convert_tokens_to_ids(target_tokens)
        rna_ids = self.convert_tokens_to_ids(rna_tokens)
        
        input_ids = self.build_inputs_with_special_tokens(target_ids, rna_ids)
        
        if max_length and len(input_ids) > max_length:
            cls_id = input_ids[0]
            sep_idx = input_ids.index(self.sep_token_id)
            eos_id = input_ids[-1]
            
            special_tokens_count = 3  # CLS + SEP + EOS
            available_space = max_length - special_tokens_count
            
            target_len = len(target_ids)
            rna_len = len(rna_ids)
            total_len = target_len + rna_len
            
            if total_len > available_space:
                target_ratio = target_len / total_len
                target_keep = min(target_len, int(available_space * target_ratio))
                rna_keep = available_space - target_keep
                
                target_ids = target_ids[:target_keep]
                rna_ids = rna_ids[:rna_keep]
                
                input_ids = self.build_inputs_with_special_tokens(target_ids, rna_ids)
        
        # Find region spans
        sep_pos = input_ids.index(self.sep_token_id)
        eos_pos = len(input_ids) - 1
        
        region_spans = {
            "target": (1, sep_pos),
            "rna": (sep_pos + 1, eos_pos),
            "cls": (0, 1),
        }
        
        if return_tensors:
            device = torch.device(device)
            input_ids = torch.tensor([input_ids], device=device)
            attention_mask = torch.ones_like(input_ids)
            residue_type_ids = self._make_residue_type_ids(input_ids, mask_as_rna=False)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "residue_type_ids": residue_type_ids,
                "region_spans": region_spans,
            }
        else:
            return {
                "input_ids": input_ids,
                "region_spans": region_spans,
            }

    def get_batch_embedder_inputs(self, target_seqs: list, rna_seqs: list, device: Union[str, torch.device] = "cpu", max_length: Optional[int] = 4096, pad_to_max: bool = True) -> Dict[str, torch.Tensor]:
        device = torch.device(device)
        batch_size = len(target_seqs)
        
        all_target_ids = []
        all_rna_ids = []
        
        for target_seq, rna_seq in zip(target_seqs, rna_seqs):
            if not target_seq.startswith(("DNA:", "AA:" , "RNA:")):
                raise ValueError(f"Target sequence must start with got: {target_seq[:10]}")
            if not rna_seq.startswith("RNA:"):
                raise ValueError(f"RNA sequence must start with got: {rna_seq[:10]}")
            
            target_tokens = self._tokenize(target_seq)
            rna_tokens = self._tokenize(rna_seq)
            
            target_ids = self.convert_tokens_to_ids(target_tokens)
            rna_ids = self.convert_tokens_to_ids(rna_tokens)
            
            all_target_ids.append(target_ids)
            all_rna_ids.append(rna_ids)
        
        all_input_ids = []
        region_spans = []
        
        for target_ids, rna_ids in zip(all_target_ids, all_rna_ids):
            if max_length:
                special_tokens_count = 3
                available_space = max_length - special_tokens_count
                total_len = len(target_ids) + len(rna_ids)
                
                if total_len > available_space:
                    target_ratio = len(target_ids) / total_len
                    target_keep = min(len(target_ids), int(available_space * target_ratio))
                    rna_keep = available_space - target_keep
                    
                    target_ids = target_ids[:target_keep]
                    rna_ids = rna_ids[:rna_keep]
            
            input_ids = self.build_inputs_with_special_tokens(target_ids, rna_ids)
            all_input_ids.append(input_ids)
            
            sep_pos = input_ids.index(self.sep_token_id)
            eos_pos = len(input_ids) - 1
            
            region_spans.append({
                "target": (1, sep_pos),
                "rna": (sep_pos + 1, eos_pos),
                "cls": (0, 1),
            })
        
        max_seq_len = max(len(seq) for seq in all_input_ids)
        
        batch_input_ids = torch.full(
            (batch_size, max_seq_len), 
            self.pad_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        for i, input_ids in enumerate(all_input_ids):
            seq_len = len(input_ids)
            batch_input_ids[i, :seq_len] = torch.tensor(input_ids, device=device)
        
        attention_mask = (batch_input_ids != self.pad_token_id).long()
        residue_type_ids = self._make_residue_type_ids_vectorized(batch_input_ids, mask_as_rna=False)
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask,
            "residue_type_ids": residue_type_ids,
            "region_spans": region_spans,
        }


    def get_generator_inputs(
        self,
        target_seq: str,
        middle_length: int,
        *,
        rna_prefix: Optional[str] = None,
        rna_suffix: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        max_length: Optional[int] = 4096,
        return_tensors: bool = True,
        force_eos_if_prefix: bool = True,
    ) -> Dict[str, Union[torch.Tensor, Tuple[int, int], int]]:
        if not target_seq.startswith(("DNA:", "AA:", "RNA:")):
            raise ValueError("Target sequence must start wit")

        target_tokens = self._tokenize(target_seq)
        target_ids = self.convert_tokens_to_ids(target_tokens)

        def _prep_rna_piece(piece: Optional[str]) -> List[int]:
            if not piece:
                return []
            s = piece[4:] if piece.startswith("RNA:") else piece
            s = s.upper().replace("T", "U")
            toks = self._tokenize("RNA:" + s)
            return self.convert_tokens_to_ids(toks)

        prefix_ids = _prep_rna_piece(rna_prefix)
        suffix_ids = _prep_rna_piece(rna_suffix)

        include_eos = True
        special_tokens_count = 2 + (1 if include_eos else 0)
        if max_length is not None:
            available_for_rna = max_length - special_tokens_count - len(target_ids)
            if available_for_rna < 1:
                raise ValueError("max_length is too small to fit")
            min_needed = len(prefix_ids) + middle_length + len(suffix_ids)
            if min_needed > available_for_rna:
                middle_length = max(1, available_for_rna - len(prefix_ids) - len(suffix_ids))

        masked_middle = [self.mask_token_id] * int(middle_length)
        rna_ids = prefix_ids + masked_middle + suffix_ids

        if include_eos:
            input_ids = [self.cls_token_id] + target_ids + [self.sep_token_id] + rna_ids + [self.eos_token_id]
        else:
            input_ids = [self.cls_token_id] + target_ids + [self.sep_token_id] + rna_ids

        sep_pos = input_ids.index(self.sep_token_id)
        rna_start = sep_pos + 1
        rna_end = len(input_ids) - (1 if include_eos else 0) 

        mid_start = rna_start + len(prefix_ids)
        mid_end = mid_start + len(masked_middle)

        mid_start = max(mid_start, rna_start)
        mid_end = min(mid_end, rna_end)

        if return_tensors:
            device = torch.device(device)
            input_ids_t = torch.tensor([input_ids], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids_t)
            residue_type_ids = self._make_residue_type_ids(input_ids_t, mask_as_rna=True)

            return {
                "input_ids": input_ids_t,
                "attention_mask": attention_mask,
                "residue_type_ids": residue_type_ids,
                "rna_region": (rna_start, rna_end),
                "rna_middle_region": (mid_start, mid_end),
                "rna_prefix_len": len(prefix_ids),
                "rna_suffix_len": len(suffix_ids),
                "rna_middle_len": int(middle_length),
            }
        else:
            return {
                "input_ids": input_ids,
                "rna_region": (rna_start, rna_end),
                "rna_middle_region": (mid_start, mid_end),
                "rna_prefix_len": len(prefix_ids),
                "rna_suffix_len": len(suffix_ids),
                "rna_middle_len": int(middle_length),
            }


    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        return ()

    def encode_no_special(self, seq: str) -> List[int]:
        tokens = self._tokenize(seq)
        return self.convert_tokens_to_ids(tokens)

    def encode_protein_no_special(self, seq: str) -> List[int]:
        if not seq.startswith(("AA:", "DNA:", "RNA:")):
            raise ValueError("target must start with")
        return self.encode_no_special(seq)

    def encode_rna_no_special(self, seq: str) -> List[int]:
        if not seq.startswith("RNA:"):
            raise ValueError("RNA must start with")
        return self.encode_no_special(seq)

    def batch_encode_no_special(
        self,
        proteins: List[str],
        rnas: List[str],
        device: str = "cpu",
        pad_to_multiple_of: int = 1,
    ) -> Dict[str, torch.Tensor]:
        if len(proteins) != len(rnas):
            raise ValueError("proteins and rnas must have same length")

        prot_ids = [torch.tensor(self.encode_protein_no_special(p), dtype=torch.long)
                    for p in proteins]
        rna_ids  = [torch.tensor(self.encode_rna_no_special(r), dtype=torch.long)
                    for r in rnas]

        def pad_stack(seqs: List[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
            lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
            T = int(lengths.max().item())
            if pad_to_multiple_of > 1:
                rem = T % pad_to_multiple_of
                if rem != 0:
                    T = T + (pad_to_multiple_of - rem)
            B = len(seqs)
            out = torch.full((B, T), fill_value=self.pad_token_id, dtype=torch.long)
            for i, s in enumerate(seqs):
                if s.numel() > 0:
                    out[i, :s.numel()] = s
            mask = (out != self.pad_token_id).long()
            return out.to(device), mask.to(device), lengths.to(device)

        prot_ids, prot_mask, prot_len = pad_stack(prot_ids)
        rna_ids,  rna_mask,  rna_len  = pad_stack(rna_ids)

        return {
            "prot_ids": prot_ids, "prot_mask": prot_mask, "prot_len": prot_len,
            "rna_ids":  rna_ids,  "rna_mask":  rna_mask,  "rna_len":  rna_len,
        }


    def _tokenize_typed_seq(self, seq_type: str, seq: str):
            s = f"{seq_type.upper()}:{seq}"
            return self._tokenize(s)

    def build_inputs_with_multi_targets(self, targets_token_ids: List[List[int]], rna_token_ids: Optional[List[int]] = None, include_eos: bool = True):
        cls_id = self.convert_tokens_to_ids(self.cls_token)
        sep_id = self.convert_tokens_to_ids(self.sep_token)
        eos_id = self.convert_tokens_to_ids(self.eos_token)

        ids = [cls_id]
        for i, tid in enumerate(targets_token_ids):
            ids += tid
            ids += [sep_id]
        if rna_token_ids is not None:
            ids += rna_token_ids
            if include_eos:
                ids += [eos_id]
        elif include_eos:
            ids += [eos_id]
        return ids

    def get_generator_inputs_multi(
        self,
        targets: List,
        middle_length: int,
        *,
        rna_prefix: Optional[str] = None,
        rna_suffix: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        max_length: Optional[int] = 4096,
        return_tensors: bool = True,
        force_eos_if_prefix: bool = True,
    ):
        tgt_token_lists = []
        for t in targets:
            toks = self._tokenize_typed_seq(t.seq_type, t.seq)
            tgt_token_lists.append(self.convert_tokens_to_ids(toks))

        def _prep_rna_piece(piece: Optional[str]) -> List[int]:
            if not piece:
                return []
            s = piece[4:] if piece.startswith("RNA:") else piece
            toks = self._tokenize("RNA:" + s.upper().replace("T", "U"))
            return self.convert_tokens_to_ids(toks)

        prefix_ids = _prep_rna_piece(rna_prefix)
        suffix_ids = _prep_rna_piece(rna_suffix)

        include_eos = True

        special_tokens_count = 1 + len(targets) + (1 if include_eos else 0)
        total_targets_len = sum(len(t) for t in tgt_token_lists)
        if max_length is not None:
            available_for_rna = max_length - special_tokens_count - total_targets_len
            if available_for_rna < 1:
                raise ValueError("max_length too small for multi target input")
            min_needed = len(prefix_ids) + middle_length + len(suffix_ids)
            if min_needed > available_for_rna:
                middle_length = max(1, available_for_rna - len(prefix_ids) - len(suffix_ids))

        masked_middle = [self.mask_token_id] * int(middle_length)
        rna_ids = prefix_ids + masked_middle + suffix_ids

        input_ids = self.build_inputs_with_multi_targets(tgt_token_lists, rna_ids, include_eos=True)

        sep_id = self.sep_token_id
        sep_positions = [i for i, x in enumerate(input_ids) if x == sep_id]
        if len(sep_positions) != len(targets):
            raise RuntimeError("Separator count mismatch in multi target build")

        rna_start = sep_positions[-1] + 1
        rna_end   = len(input_ids) - 1

        mid_start = rna_start + len(prefix_ids)
        mid_end   = mid_start + len(masked_middle)

        if return_tensors:
            device = torch.device(device)
            input_ids_t = torch.tensor([input_ids], device=device, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids_t)
            residue_type_ids = self._make_residue_type_ids(input_ids_t, mask_as_rna=True)
            return {
                "input_ids": input_ids_t,
                "attention_mask": attention_mask,
                "residue_type_ids": residue_type_ids,
                "rna_region": (rna_start, rna_end),
                "rna_middle_region": (mid_start, mid_end),
                "rna_prefix_len": len(prefix_ids),
                "rna_suffix_len": len(suffix_ids),
                "rna_middle_len": int(middle_length),
                "target_spans": self._compute_target_spans_multi(tgt_token_lists)
            }
        else:
            return {
                "input_ids": input_ids,
                "rna_region": (rna_start, rna_end),
                "rna_middle_region": (mid_start, mid_end),
                "rna_prefix_len": len(prefix_ids),
                "rna_suffix_len": len(suffix_ids),
                "rna_middle_len": int(middle_length),
            }

    def _compute_target_spans_multi(self, targets_token_ids: List[List[int]]):
        spans = []
        pos = 1
        sep = self.sep_token_id
        for t in targets_token_ids:
            start = pos
            end   = pos + len(t)
            spans.append((start, end))
            pos = end + 1
        return spans