"""Ablation model variants for CodeEmbed (Phase 6: Ablation Studies).

This module provides AblationSharedEncoder with configurable pooling strategies
(MaskedMeanPooling vs CLSPooling) and optional modality embedding toggling,
preserving all core architecture standards (Pre-LN, SDPA attention, L2 normalization)
without modifying any existing Phase 1-5 model files.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from model.embeddings import EmbeddingLayer
from model.pooling import CLSPooling, MaskedMeanPooling
from model.transformer import TransformerEncoder


class AblationSharedEncoder(nn.Module):
    """Shared Transformer Encoder with configurable pooling for ablation studies.

    Pipeline:
        Tokens (B, L) + Modality IDs (B,)
            │
            ▼
        EmbeddingLayer (Tokens + Positional + Optional Modality) (B, L, D)
            │
            ▼
        TransformerEncoder (Pre-LN blocks with SDPA) (B, L, D)
            │
            ▼
        Pooling (MaskedMeanPooling OR CLSPooling) (B, D)
            │
            ▼
        Projection Head: Linear(D -> D, bias=False) + LayerNorm(D)
            │
            ▼
        L2 Normalization: z / ||z||_2 -> (B, D) unit sphere vectors

    Args:
        vocab_size: Vocabulary size (default: 16,000).
        d_model: Hidden and embedding dimension (default: 256).
        n_layers: Number of transformer blocks (default: 4).
        n_heads: Number of attention heads (default: 8).
        d_ff: FFN intermediate expansion dimension (default: 1024).
        max_seq_len: Maximum sequence length (default: 256).
        dropout: Dropout probability (default: 0.1).
        padding_idx: Padding token index (default: 0).
        num_modalities: Number of distinct input modalities (default: 2; 0=code, 1=text).
        pooling: Sequence aggregation strategy ('mean' for MaskedMeanPooling, 'cls' for CLSPooling).
        use_modality_embedding: Whether to include learned modality embeddings (default: True).
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0,
        num_modalities: int = 2,
        pooling: Literal["mean", "cls"] = "mean",
        use_modality_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_modalities = num_modalities
        self.pooling_strategy = pooling
        self.use_modality_embedding = use_modality_embedding

        # 1. Embeddings with optional learned modality table
        self.embeddings = EmbeddingLayer(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=padding_idx,
            use_modality_embedding=use_modality_embedding,
            num_modalities=num_modalities,
        )

        # 2. Transformer Encoder Stack (shared weights, SDPA accelerated)
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # 3. Configurable Pooling Strategy (Ablation 1)
        if pooling == "cls":
            self.pooling: nn.Module = CLSPooling()
        elif pooling == "mean":
            self.pooling = MaskedMeanPooling()
        else:
            raise ValueError(f"Unsupported pooling strategy: '{pooling}'. Choose 'mean' or 'cls'.")

        # 4. Projection Head (Linear + LayerNorm)
        # bias=False because it is immediately followed by LayerNorm
        self.proj_linear = nn.Linear(d_model, d_model, bias=False)
        self.proj_ln = nn.LayerNorm(d_model)

    def _resolve_modality_ids(
        self,
        modality_ids: torch.Tensor | int | str | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Resolve modality identifier into a standard tensor of shape (B,)."""
        if not self.use_modality_embedding or modality_ids is None:
            return None

        if isinstance(modality_ids, str):
            val = 0 if modality_ids.lower() == "code" else 1
            return torch.full((batch_size,), val, dtype=torch.long, device=device)

        if isinstance(modality_ids, int):
            return torch.full((batch_size,), modality_ids, dtype=torch.long, device=device)

        if isinstance(modality_ids, torch.Tensor):
            return modality_ids.to(device=device, dtype=torch.long)

        raise TypeError(f"Unsupported modality_ids type: {type(modality_ids)}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        modality_ids: torch.Tensor | int | str | None = None,
    ) -> torch.Tensor:
        """Encode token sequence into L2-normalized embeddings.

        Args:
            input_ids: LongTensor of token IDs of shape (B, L).
            attention_mask: Optional binary mask of shape (B, L) where 1 indicates
                a valid token and 0 indicates padding.
            modality_ids: Modality specification (tensor of shape (B,),
                integer 0 for code / 1 for text, or string 'code' / 'text').

        Returns:
            Unit-norm FloatTensor of shape (B, D).
        """
        b_size, _ = input_ids.shape
        mod_tensor = self._resolve_modality_ids(modality_ids, batch_size=b_size, device=input_ids.device)

        # 1. Embedding lookup: (B, L) -> (B, L, D)
        h = self.embeddings(input_ids, modality_ids=mod_tensor)

        # 2. Transformer blocks: (B, L, D) -> (B, L, D)
        h = self.encoder(h, attention_mask=attention_mask)

        # 3. Configurable Pooling: (B, L, D) -> (B, D)
        pooled = self.pooling(h, attention_mask=attention_mask)

        # 4. Projection Head: (B, D) -> (B, D)
        proj = self.proj_linear(pooled)
        proj = self.proj_ln(proj)

        # 5. L2 Normalization onto unit hypersphere
        embeddings = F.normalize(proj, p=2, dim=-1)

        return embeddings

    def encode_code(
        self,
        code_ids: torch.Tensor,
        code_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode code tokens using modality 0."""
        return self.forward(code_ids, attention_mask=code_mask, modality_ids=0)

    def encode_text(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode text tokens using modality 1."""
        return self.forward(text_ids, attention_mask=text_mask, modality_ids=1)

    def get_num_params(self) -> tuple[int, int]:
        """Return (total_params, trainable_params) counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

