"""Shared Encoder implementation for CodeEmbed (Phase 3: Model 2).

This module provides SharedEncoder:
1. EmbeddingLayer with learned ModalityEmbedding (0=CODE, 1=TEXT) + Token + Positional embeddings
2. 4-layer Pre-LN TransformerEncoder
3. MaskedMeanPooling
4. Projection Head (Linear + LayerNorm)
5. L2 Normalization onto unit hypersphere
"""

import torch
import torch.nn.functional as F
from torch import nn

from model.embeddings import EmbeddingLayer
from model.pooling import MaskedMeanPooling
from model.transformer import TransformerEncoder


class SharedEncoder(nn.Module):
    """Shared Transformer Encoder distinguishing modalities via learned modality embeddings.

    Pipeline:
        Tokens (B, L) + Modality IDs (B,)
            │
            ▼
        EmbeddingLayer (Tokens + Positional + Modality) (B, L, D=256)
            │
            ▼
        TransformerEncoder (4 Pre-LN blocks) (B, L, D=256)
            │
            ▼
        MaskedMeanPooling (B, D=256)
            │
            ▼
        Projection Head: Linear(D -> D, bias=False) + LayerNorm(D)
            │
            ▼
        L2 Normalization: z / ||z||_2 -> (B, D=256) unit sphere vectors

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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_modalities = num_modalities

        # 1. Embeddings with learned modality table
        self.embeddings = EmbeddingLayer(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=padding_idx,
            use_modality_embedding=True,
            num_modalities=num_modalities,
        )

        # 2. Transformer Encoder Stack (shared weights)
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # 3. Pooling Strategy
        self.pooling = MaskedMeanPooling()

        # 4. Projection Head (Linear + LayerNorm)
        # bias=False because it is immediately followed by LayerNorm (Rules.md)
        self.proj_linear = nn.Linear(d_model, d_model, bias=False)
        self.proj_ln = nn.LayerNorm(d_model)

    def _resolve_modality_ids(
        self,
        modality_ids: torch.Tensor | int | str | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Resolve modality identifier into a standard tensor of shape (B,)."""
        if modality_ids is None:
            return None

        if isinstance(modality_ids, str):
            val = 0 if modality_ids.lower() == "code" else 1
            return torch.full((batch_size,), val, dtype=torch.long, device=device)

        if isinstance(modality_ids, int):
            return torch.full(
                (batch_size,), modality_ids, dtype=torch.long, device=device
            )

        if isinstance(modality_ids, torch.Tensor):
            return modality_ids.to(device=device, dtype=torch.long)

        raise TypeError(f"Unsupported modality_ids type: {type(modality_ids)}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        modality_ids: torch.Tensor | int | str | None = None,
    ) -> torch.Tensor:
        """Encode token sequence into L2-normalized embeddings with modality routing.

        Args:
            input_ids: LongTensor of token IDs of shape (B, L).
            attention_mask: Optional binary mask of shape (B, L) where 1 indicates
                a valid token and 0 indicates padding.
            modality_ids: Modality specification (tensor of shape (B,) or (B, L),
                integer 0 for code / 1 for text, or string 'code' / 'text').

        Returns:
            Unit-norm FloatTensor of shape (B, D).
        """
        B, _ = input_ids.shape
        mod_tensor = self._resolve_modality_ids(
            modality_ids, batch_size=B, device=input_ids.device
        )

        # 1. Embedding lookup with modality: (B, L) -> (B, L, D)
        h = self.embeddings(input_ids, modality_ids=mod_tensor)

        # 2. Transformer blocks: (B, L, D) -> (B, L, D)
        h = self.encoder(h, attention_mask=attention_mask)

        # 3. Pooling: (B, L, D) -> (B, D)
        pooled = self.pooling(h, attention_mask=attention_mask)

        # 4. Projection Head: (B, D) -> (B, D)
        proj = self.proj_linear(pooled)
        proj = self.proj_ln(proj)

        # 5. L2 Normalization onto unit hypersphere
        embeddings = F.normalize(proj, p=2, dim=-1)

        return embeddings

    def get_num_params(self) -> tuple[int, int]:
        """Return (total_params, trainable_params) counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
