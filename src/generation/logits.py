import torch
from typing import Optional


@torch.no_grad()
def _logits_for_sequence(model, tokenizer, protein_seq: str, rna_seq: str, device: Optional[torch.device] = None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
        
    inputs = tokenizer.get_embedder_inputs(protein_seq, rna_seq, device=device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    residue_type_ids = inputs["residue_type_ids"]
    rna_start, rna_end = inputs["region_spans"]["rna"]

    out = model(input_ids=input_ids, attention_mask=attention_mask, residue_type_ids=residue_type_ids)
    logits = out.logits[0, rna_start:rna_end]  # [L, vocab]
    return logits