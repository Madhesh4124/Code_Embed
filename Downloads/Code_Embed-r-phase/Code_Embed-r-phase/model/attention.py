"""Multi-Head Self-Attention implemented with raw PyTorch primitives.

This module implements scaled dot-product multi-head self-attention without
using high-level HuggingFace model wrappers.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """Custom Multi-Head Self-Attention layer.

    Computes:
        Attention(Q, K, V) = Softmax( (Q * K^T) / sqrt(d_k) + M ) * V

    Args:
        d_model: Dimensionality of the model / embeddings (default: 256).
        n_heads: Number of attention heads (default: 8).
        dropout: Attention weights dropout probability (default: 0.1).
        bias: Whether to use bias in linear projection layers (default: False).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Query, Key, Value
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for multi-head self-attention.

        Args:
            hidden_states: Input tensor of shape (B, L, D).
            attention_mask: Binary mask of shape (B, L) where 1 indicates
                a valid token and 0 indicates a padding token.
            return_attention_weights: If True, returns attention probabilities.

        Returns:
            Tuple of:
                - output tensor of shape (B, L, D).
                - attention weights tensor of shape (B, H, L, L) or None.
        """
        B, L, D = hidden_states.shape

        # 1. Project Q, K, V and split into heads: (B, L, D) -> (B, H, L, d_k)
        q = (
            self.q_proj(hidden_states)
            .view(B, L, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(B, L, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, L, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Fast path: leverage PyTorch F.scaled_dot_product_attention (FlashAttention /
        # fused memory-efficient attention) when explicit attention weights are not requested.
        # This avoids materializing massive (B, H, L, L) attention tensors in VRAM, eliminating
        # GPU out-of-memory errors and Windows WDDM host-RAM paging thrashing.
        if not return_attention_weights:
            attn_mask = (
                attention_mask.unsqueeze(1).unsqueeze(2).bool()
                if attention_mask is not None
                else None
            )
            dropout_p = self.dropout.p if self.training else 0.0
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )
            context = context.transpose(1, 2).contiguous().view(B, L, D)
            return self.out_proj(context), None

        # 2. Scaled dot-product scores: (B, H, L, d_k) @ (B, H, d_k, L) -> (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: (B, L) -> (B, 1, 1, L)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Use -1e4 in float16/bfloat16 to avoid NaN overflows, -1e9 in float32
            fill_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
            scores = scores.masked_fill(mask == 0, fill_val)

        # 4. Softmax over key sequence dimension
        attn_weights = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_weights)

        # 5. Multiply by values: (B, H, L, L) @ (B, H, L, d_k) -> (B, H, L, d_k)
        context = torch.matmul(attn_probs, v)

        # 6. Concatenate heads back: (B, H, L, d_k) -> (B, L, H, d_k) -> (B, L, D)
        context = context.transpose(1, 2).contiguous().view(B, L, D)

        # 7. Final output projection
        output = self.out_proj(context)

        return output, attn_weights
