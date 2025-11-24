from typing import Optional, Tuple
import torch
import torch.nn as nn

from transformers.modeling_outputs import MaskedLMOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


from src.layers.head import RTOutput, OutputHeads
from src.layers.encoder import RTEncoder
from src.layers.embedder import RTEmbedder



class RT(nn.Module):
    def __init__(self, vocab_size: int = 35, d_model: int = 768, n_heads: int = 12, n_layers: int = 12, max_position: int = 4096,) -> None:
        super().__init__()
        self.embedder = RTEmbedder(d_model, vocab_size, max_position)

        self.encoder = RTEncoder(d_model, n_heads, n_layers)
        self.output_heads = OutputHeads(d_model, vocab_size)

    def forward(self, sequence_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, residue_type_ids: Optional[torch.Tensor] = None, **unused) -> RTOutput:
        x = self.embedder(sequence_tokens, residue_type_ids)

        x, embedding = self.encoder(x, attention_mask=attention_mask)
        return self.output_heads(x, embedding)


class RTConfig(PretrainedConfig):
    model_type = "RT"

    def __init__(self, vocab_size: int = 35, d_model: int = 512, n_heads: int = 8, n_layers: int = 8, max_position: int = 4096, pad_token_id: int = 0, **kwargs,) -> None:
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_position = max_position


class RTHFWrapper(PreTrainedModel):
    config_class = RTConfig

    def __init__(self, config: RTConfig):
        super().__init__(config)
        self.model = RT(config.vocab_size, config.d_model, config.n_heads, config.n_layers, getattr(config, "max_position", 4096))

    def forward(self, input_ids, attention_mask, labels=None, residue_type_ids=None, **kwargs):
        outputs = self.model(sequence_tokens=input_ids, attention_mask=attention_mask, residue_type_ids=residue_type_ids)
        logits = outputs.sequence_logits
        embeddings = outputs.embeddings
        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits, hidden_states=(embeddings,))