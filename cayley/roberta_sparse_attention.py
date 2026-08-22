import math
import os

import numpy as np
import torch


GLOBAL_SPARSE_MASK = None
GLOBAL_VERBOSE_MASK = False


def load_mask(mask_path: str) -> torch.Tensor:
    ext = os.path.splitext(mask_path)[1].lower()
    if ext in {".pt", ".pth"}:
        mask = torch.load(mask_path, map_location="cpu")
    elif ext == ".npy":
        mask = torch.from_numpy(np.load(mask_path))
    else:
        raise ValueError(f"Unsupported mask extension: {ext}")

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    if mask.dtype != torch.bool:
        mask = mask != 0

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    elif mask.dim() != 4:
        raise ValueError(f"Mask must have 2, 3, or 4 dims, got {tuple(mask.shape)}")

    return mask.contiguous()


def get_sliced_mask(mask: torch.Tensor, n_heads: int, q_len: int, k_len: int, device) -> torch.Tensor:
    if mask.size(-2) < q_len or mask.size(-1) < k_len:
        raise ValueError(
            f"Mask too small. Mask shape={tuple(mask.shape)}, needed at least (..., {q_len}, {k_len})"
        )

    sliced = mask[..., :q_len, :k_len].to(device)
    if sliced.size(1) not in (1, n_heads):
        raise ValueError(f"Mask head dim must be 1 or {n_heads}, got {sliced.size(1)}")
    return sliced


def configure_sparse_attention(mask_path: str | None, verbose: bool = False) -> None:
    global GLOBAL_SPARSE_MASK, GLOBAL_VERBOSE_MASK
    GLOBAL_VERBOSE_MASK = verbose
    if mask_path is None:
        GLOBAL_SPARSE_MASK = None
        return
    GLOBAL_SPARSE_MASK = load_mask(mask_path)
    patch_roberta_attention()


def patch_roberta_attention() -> None:
    """Patch RoBERTa self-attention to apply a bool keep-mask before softmax.

    This matches the previous local benchmark semantics:
    True keeps an attention edge; False blocks it. The regular Hugging Face
    padding mask is applied first, then this structural mask is applied.

    This is a quality benchmark patch, not an optimized sparse kernel. It still
    materializes dense attention scores.
    """
    from transformers.models.roberta import modeling_roberta

    if hasattr(modeling_roberta.RobertaSelfAttention, "_cayley_sparse_patch_applied"):
        return

    def patched_forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention is not used by RoBERTa sequence classification.")
        if past_key_value is not None:
            raise NotImplementedError("past_key_value is not supported by this benchmark patch.")

        batch_size, seq_len, _ = hidden_states.size()
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        num_heads = self.num_attention_heads
        head_dim = self.attention_head_size
        query_layer = query_layer.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_layer = key_layer.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value_layer = value_layer.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        global GLOBAL_SPARSE_MASK, GLOBAL_VERBOSE_MASK
        if GLOBAL_SPARSE_MASK is not None:
            sparse_mask = get_sliced_mask(
                GLOBAL_SPARSE_MASK,
                n_heads=attention_scores.size(1),
                q_len=attention_scores.size(-2),
                k_len=attention_scores.size(-1),
                device=attention_scores.device,
            )
            mask_value = torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores.masked_fill(~sparse_mask, mask_value)

            if GLOBAL_VERBOSE_MASK:
                keep_ratio = sparse_mask.float().mean().item()
                print(
                    f"[mask debug] attention_scores={tuple(attention_scores.shape)} "
                    f"mask={tuple(sparse_mask.shape)} keep_ratio={keep_ratio:.4f}"
                )
                GLOBAL_VERBOSE_MASK = False

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, num_heads * head_dim)

        if output_attentions:
            return context_layer, attention_probs
        return context_layer, None

    modeling_roberta.RobertaSelfAttention.forward = patched_forward
    modeling_roberta.RobertaSelfAttention._cayley_sparse_patch_applied = True
