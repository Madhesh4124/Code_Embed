"""Dual (Separate) Transformer Encoders for Code and Text (Phase 4: Model 3).

This module implements DualEncoder:
- Completely decoupled BaseEncoder instances for code and natural language text.
- Allows each modality to develop specialized representations and attention patterns.
- Keeps parameter budget controlled by configuring encoder depth (default: 3 layers each).
- Normalizes both output embedding spaces to unit hyperspheres for cosine similarity.
"""

from typing import Any

import torch
from torch import nn

from model.encoder import BaseEncoder


class DualEncoder(nn.Module):
    """Dual Encoder architecture with separate code and natural language encoders.

    Pipeline:
        Code Sequence (B, L_c) ──► Code BaseEncoder (3 layers) ──► z_code (B, D) [Unit Norm]
        Text Sequence (B, L_t) ──► Text BaseEncoder (3 layers) ──► z_text (B, D) [Unit Norm]

    Args:
        vocab_size: Vocabulary size for both tokenizers (default: 16,000).
        d_model: Hidden and embedding dimension (default: 256).
        n_layers: Number of transformer layers per encoder (default: 3).
        n_heads: Number of attention heads (default: 8).
        d_ff: FFN intermediate expansion dimension (default: 1024).
        max_seq_len: Maximum sequence length (default: 256).
        dropout: Dropout probability (default: 0.1).
        padding_idx: Padding token index (default: 0).
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        # Independent encoder for code sequences
        self.code_encoder = BaseEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=padding_idx,
        )

        # Independent encoder for natural language docstrings / queries
        self.text_encoder = BaseEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=padding_idx,
        )

    def encode_code(
        self,
        code_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode code sequences into L2-normalized vectors.

        Args:
            code_ids: LongTensor of shape (B, L_c).
            attention_mask: Optional binary mask of shape (B, L_c).

        Returns:
            Unit-norm FloatTensor of shape (B, D).
        """
        return self.code_encoder(code_ids, attention_mask=attention_mask)

    def encode_text(
        self,
        text_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode text queries / docstrings into L2-normalized vectors.

        Args:
            text_ids: LongTensor of shape (B, L_t).
            attention_mask: Optional binary mask of shape (B, L_t).

        Returns:
            Unit-norm FloatTensor of shape (B, D).
        """
        return self.text_encoder(text_ids, attention_mask=attention_mask)

    def forward(
        self,
        code_ids: torch.Tensor | None = None,
        code_mask: torch.Tensor | None = None,
        text_ids: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass for training or single-modality encoding.

        If both code_ids and text_ids are provided, returns (code_emb, text_emb).
        If only code_ids is provided, returns code_emb.
        If only text_ids is provided, returns text_emb.

        Args:
            code_ids: LongTensor of shape (B, L_c) or None.
            code_mask: Attention mask for code of shape (B, L_c) or None.
            text_ids: LongTensor of shape (B, L_t) or None.
            text_mask: Attention mask for text of shape (B, L_t) or None.

        Returns:
            Tuple of (code_emb, text_emb) or single modality embedding tensor.
        """
        if code_ids is not None and text_ids is not None:
            code_emb = self.encode_code(code_ids, attention_mask=code_mask)
            text_emb = self.encode_text(text_ids, attention_mask=text_mask)
            return code_emb, text_emb
        elif code_ids is not None:
            return self.encode_code(code_ids, attention_mask=code_mask)
        elif text_ids is not None:
            return self.encode_text(text_ids, attention_mask=text_mask)
        else:
            raise ValueError("At least one of code_ids or text_ids must be provided.")

    def get_num_params(self) -> tuple[int, int]:
        """Return (total_params, trainable_params) counts across both encoders."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_encoder_params(self) -> dict[str, Any]:
        """Return individual parameter counts for code and text encoders."""
        code_tot, code_train = self.code_encoder.get_num_params()
        text_tot, text_train = self.text_encoder.get_num_params()
        return {
            "code_encoder_total": code_tot,
            "code_encoder_trainable": code_train,
            "text_encoder_total": text_tot,
            "text_encoder_trainable": text_train,
            "total_params": code_tot + text_tot,
        }
