from pathlib import Path

import torch
from transformers import DebertaV2ForSequenceClassification


def load_mask(path):
    mask = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("Expected a Boolean mask tensor")
    if mask.ndim == 3:
        if torch.equal(mask, mask[:1].expand_as(mask)):
            mask = mask[0]
    if mask.ndim not in (2, 3) or mask.shape[-2] != mask.shape[-1] or not mask.diagonal(dim1=-2, dim2=-1).all():
        raise ValueError("Expected a square mask with self-loops")
    return mask


def pair_attention_mask(padding, mask):
    if padding.ndim != 2:
        raise ValueError("Expected a [batch, length] padding mask")
    length = padding.shape[1]
    if mask.dtype != torch.bool or mask.ndim not in (2, 3) or mask.shape[-2] < length or mask.shape[-1] < length:
        raise ValueError("Expected a Boolean mask covering the input")
    valid = padding.bool()
    if mask.ndim == 2:
        return mask[:length, :length].to(padding.device) & valid[:, :, None] & valid[:, None, :]
    return mask[:, :length, :length].to(padding.device) & valid[:, None, :, None] & valid[:, None, None, :]


def _embedding_mask(_module, args, kwargs):
    mask = kwargs.get("mask")
    if mask is not None and mask.ndim in (3, 4):
        kwargs = dict(kwargs)
        diagonal = mask.diagonal(dim1=-2, dim2=-1)
        kwargs["mask"] = diagonal if mask.ndim == 3 else diagonal.any(dim=1)
    return args, kwargs


def _attention_mask(module, args, kwargs):
    module.dropout._cayley_mask = args[1] if len(args) > 1 else kwargs["attention_mask"]


def _zero_blocked(module, args):
    mask = module._cayley_mask.bool()
    return (args[0].masked_fill(~mask, 0),)


class CayleyDebertaV2ForSequenceClassification(DebertaV2ForSequenceClassification):
    def __init__(self, config):
        if getattr(config, "position_biased_input", True) is not False:
            raise ValueError("DeBERTa-v3 requires position_biased_input=False")
        super().__init__(config)
        self.register_buffer("structural_mask", None, persistent=False)
        self.deberta.embeddings.register_forward_pre_hook(_embedding_mask, with_kwargs=True)
        for layer in self.deberta.encoder.layer:
            attention = layer.attention.self
            attention.register_forward_pre_hook(_attention_mask, with_kwargs=True)
            attention.dropout.register_forward_pre_hook(_zero_blocked)

    def set_structural_mask(self, mask):
        if mask is not None:
            if mask.dtype != torch.bool or mask.ndim not in (2, 3) or mask.shape[-2] != mask.shape[-1]:
                raise ValueError("Expected a square Boolean mask")
            if mask.ndim == 3 and mask.shape[0] != self.config.num_attention_heads:
                raise ValueError("Wrong number of attention heads")
            if not mask.diagonal(dim1=-2, dim2=-1).all():
                raise ValueError("Expected self-loops at every position")
        self.structural_mask = mask.to(next(self.parameters()).device) if mask is not None else None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        if self.structural_mask is not None:
            if attention_mask is None:
                if input_ids is None:
                    raise ValueError("An attention mask or input IDs are required")
                attention_mask = torch.ones_like(input_ids)
            attention_mask = pair_attention_mask(attention_mask, self.structural_mask)
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, inputs_embeds=inputs_embeds, labels=labels,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, **kwargs,
        )
