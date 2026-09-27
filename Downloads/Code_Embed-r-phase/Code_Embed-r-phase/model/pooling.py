"""Pooling strategies for sequence representation aggregation.

This module implements:
- MaskedMeanPooling: Length-normalized average pooling over non-padded tokens.
- CLSPooling: Extracts the first position token representation (<BOS> / <CLS>).
"""

import torch
from torch import nn


class MaskedMeanPooling(nn.Module):
    """Masked mean pooling over non-padding tokens.

    Computes:
        h = sum(hidden_states * mask, dim=1) / sum(mask, dim=1)

    Pad tokens contribute 0 to the sum and are not counted in the denominator.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate token embeddings into a single sequence vector.

        Args:
            hidden_states: Sequence hidden states of shape (B, L, D).
            attention_mask: Optional binary mask of shape (B, L) where 1 indicates
                a valid token and 0 indicates padding.

        Returns:
            Pooled sequence representations of shape (B, D).
        """
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        # attention_mask: (B, L) -> (B, L, 1)
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        # Sum over token representations, ignoring pad tokens: (B, D)
        sum_embeddings = torch.sum(hidden_states * mask, dim=1)
        # Sum of non-padded tokens: (B, 1), clamped to prevent division by zero
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask


class CLSPooling(nn.Module):
    """Pooling that extracts the representation at sequence position 0.

    Typically corresponds to the special <BOS>, <CODE>, or <TEXT> token.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract first token vector.

        Args:
            hidden_states: Sequence hidden states of shape (B, L, D).
            attention_mask: Optional binary mask (unused, accepted for interface consistency).

        Returns:
            Pooled vector of shape (B, D).
        """
        return hidden_states[:, 0, :]
