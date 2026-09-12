"""Embedding layers for CodeEmbed Transformer architecture.

This module provides:
- TokenEmbedding: Lookup table for subword token IDs.
- PositionalEmbedding: Learned positional embeddings for sequence positions.
- ModalityEmbedding: Learned embeddings distinguishing code vs. text.
- EmbeddingLayer: Unified container combining token, positional, and optional modality embeddings.
"""


import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Learned subword token embedding table.

    Args:
        vocab_size: Size of the vocabulary (default: 16,000).
        d_model: Dimensionality of the embedding vector (default: 256).
        padding_idx: Token index to treat as padding (zeros gradient, default: 0).
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 256,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to embedding vectors.

        Args:
            input_ids: LongTensor of shape (B, L).

        Returns:
            FloatTensor of shape (B, L, D).
        """
        return self.embedding(input_ids)


class PositionalEmbedding(nn.Module):
    """Learned 1D positional embeddings.

    Args:
        max_seq_len: Maximum supported sequence length (default: 256).
        d_model: Embedding dimension (default: 256).
    """

    def __init__(
        self,
        max_seq_len: int = 256,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model,
        )

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate positional embedding vectors for a given sequence length.

        Args:
            seq_len: Current sequence length L (must be <= max_seq_len).
            device: Target device for position index tensor.

        Returns:
            FloatTensor of shape (1, L, D).

        Raises:
            ValueError: If seq_len exceeds max_seq_len.
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        positions = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
        return self.embedding(positions)  # (1, L, D)


class ModalityEmbedding(nn.Module):
    """Learned modality embeddings to differentiate code vs. natural language.

    Modalities:
        0: Code snippet
        1: Natural language query / docstring

    Args:
        num_modalities: Number of distinct input modalities (default: 2).
        d_model: Embedding dimension (default: 256).
    """

    def __init__(
        self,
        num_modalities: int = 2,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_modalities,
            embedding_dim=d_model,
        )

    def forward(self, modality_ids: torch.Tensor) -> torch.Tensor:
        """Map modality IDs to embedding vectors.

        Args:
            modality_ids: LongTensor of shape (B,) or (B, L).

        Returns:
            FloatTensor of shape (B, 1, D) or (B, L, D).
        """
        if modality_ids.ndim == 1:
            return self.embedding(modality_ids).unsqueeze(1)  # (B, 1, D)
        return self.embedding(modality_ids)  # (B, L, D)


class EmbeddingLayer(nn.Module):
    """Composite embedding layer combining token and positional representations.

    Applies:
        x = LayerNorm(TokenEmbedding(ids) + PositionalEmbedding(pos) [+ ModalityEmbedding(mod)])
        x = Dropout(x)

    Args:
        vocab_size: Vocabulary size (default: 16,000).
        d_model: Hidden dimension (default: 256).
        max_seq_len: Maximum sequence length (default: 256).
        dropout: Dropout probability (default: 0.1).
        padding_idx: Padding token index (default: 0).
    """

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 256,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embed = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_embed = PositionalEmbedding(max_seq_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for composite embeddings.

        Args:
            input_ids: Token ID tensor of shape (B, L).
            modality_ids: Optional modality ID tensor of shape (B,) or (B, L).

        Returns:
            Embedded representation tensor of shape (B, L, D).
        """
        _, L = input_ids.shape
        x = self.token_embed(input_ids)  # (B, L, D)
        x = x + self.pos_embed(L, device=input_ids.device)  # (B, L, D)

        x = self.layer_norm(x)
        x = self.dropout(x)
        return x
