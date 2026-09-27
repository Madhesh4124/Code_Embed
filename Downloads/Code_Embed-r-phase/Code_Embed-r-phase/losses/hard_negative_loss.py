"""InfoNCE loss extended with pre-mined hard negatives (Phase 5).

Extends the in-batch InfoNCE loss by concatenating K hard negative code
embeddings to the denominator of the softmax — making the contrastive task
significantly harder and forcing sharper representation boundaries.

Formula (text-to-code direction for sample i):
    positives: code_emb[i]                    (1 positive)
    negatives: all other code_emb[j, j≠i]     (B-1 in-batch negatives)
               + hard_neg_embs[i, 0..K-1]     (K mined hard negatives)

    s_i = [sim(t_i, c_i)] + [sim(t_i, c_j≠i)] + [sim(t_i, hn_i,k)]
    loss_i = -log( exp(s_i[0]/τ) / Σ exp(s_i/τ) )

The code-to-text direction uses only in-batch negatives (asymmetric is fine
since hard negatives are code snippets — not text queries).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class InfoNCEWithHardNegatives(nn.Module):
    """InfoNCE loss augmented with pre-mined hard negatives.

    Args:
        temperature: Softmax temperature τ (default: 0.07).
        hn_weight: Scalar weight applied to hard-negative loss term (default: 1.0).
            Set > 1.0 to emphasise hard negatives more strongly.
    """

    def __init__(self, temperature: float = 0.07, hn_weight: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature
        self.hn_weight   = hn_weight

    def forward(
        self,
        text_emb:       torch.Tensor,   # (B, D) — query embeddings
        code_emb:       torch.Tensor,   # (B, D) — positive code embeddings
        hard_neg_embs:  torch.Tensor | None = None,   # (B, K, D) — hard negative code embeddings
    ) -> torch.Tensor:
        """Compute InfoNCE loss with hard negatives.

        Args:
            text_emb:      Normalized text embeddings  (B, D).
            code_emb:      Normalized code embeddings  (B, D).
            hard_neg_embs: Normalized hard neg embeddings (B, K, D).

        Returns:
            Scalar loss tensor.
        """
        B, D = text_emb.shape

        # ── In-batch similarities ─────────────────────────────────────────────
        # sim_inbatch[i, j] = cosine(text_i, code_j) / τ
        sim_inbatch = torch.matmul(text_emb, code_emb.T) / self.temperature  # (B, B)

        targets = torch.arange(B, device=text_emb.device)          # positives at diagonal

        if hard_neg_embs is None:
            loss_t2c = F.cross_entropy(sim_inbatch, targets)
        else:
            sim_hn = torch.bmm(
                text_emb.unsqueeze(1),
                hard_neg_embs.transpose(1, 2),
            ).squeeze(1) / self.temperature

            extended_logits = torch.cat([sim_inbatch, sim_hn], dim=1)
            loss_t2c = F.cross_entropy(extended_logits, targets)

        # ── Code-to-text loss (standard in-batch, no hard negs needed) ────────
        loss_c2t = F.cross_entropy(sim_inbatch.T, targets)

        return (loss_t2c + loss_c2t) / 2.0

    @torch.no_grad()
    def compute_accuracy(
        self,
        text_emb: torch.Tensor,
        code_emb: torch.Tensor,
    ) -> tuple[float, float]:
        """Standard in-batch top-1 accuracy (for progress monitoring)."""
        B = text_emb.shape[0]
        sim = torch.matmul(text_emb, code_emb.T)
        t   = torch.arange(B, device=text_emb.device)
        t2c = float((sim.argmax(dim=-1) == t).float().mean())
        c2t = float((sim.T.argmax(dim=-1) == t).float().mean())
        return t2c, c2t
