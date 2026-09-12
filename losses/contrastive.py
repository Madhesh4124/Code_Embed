"""InfoNCE Contrastive Loss with in-batch negatives.

This module implements the symmetric Information Noise-Contrastive Estimation (InfoNCE)
objective used to align natural language text queries and code representations
in a shared embedding space.
"""


import torch
import torch.nn.functional as F
from torch import nn


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE loss with in-batch negatives.

    Given a batch of B pairs (text_i, code_i), we treat the matching pair (i, i)
    as positive, and all other (B - 1) code snippets and (B - 1) docstrings
    as negative examples.

    Formula:
        S = (text_emb @ code_emb.T) / temperature
        loss_text_to_code = CrossEntropy(S, arange(B))
        loss_code_to_text = CrossEntropy(S.T, arange(B))
        loss = (loss_text_to_code + loss_code_to_text) / 2.0

    Args:
        temperature: Softmax temperature scaling parameter tau (default: 0.07).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        text_emb: torch.Tensor,
        code_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            text_emb: Normalized text embeddings of shape (B, D).
            code_emb: Normalized code embeddings of shape (B, D).

        Returns:
            Scalar loss tensor.
        """
        B = text_emb.shape[0]
        if code_emb.shape[0] != B:
            raise ValueError(
                f"Batch sizes mismatch: text_emb has {B}, code_emb has {code_emb.shape[0]}"
            )

        # 1. Cosine similarity matrix scaled by temperature: (B, B)
        sim_matrix = torch.matmul(text_emb, code_emb.T) / self.temperature

        # 2. Ground truth targets: diagonal elements (0, 1, ..., B-1)
        targets = torch.arange(B, device=text_emb.device, dtype=torch.long)

        # 3. Symmetric cross entropy
        loss_t2c = F.cross_entropy(sim_matrix, targets)
        loss_c2t = F.cross_entropy(sim_matrix.T, targets)

        return (loss_t2c + loss_c2t) / 2.0

    @torch.no_grad()
    def compute_accuracy(
        self,
        text_emb: torch.Tensor,
        code_emb: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute in-batch Top-1 retrieval accuracy in both directions.

        Args:
            text_emb: Normalized text embeddings of shape (B, D).
            code_emb: Normalized code embeddings of shape (B, D).

        Returns:
            Tuple of (text_to_code_acc, code_to_text_acc) in [0.0, 1.0].
        """
        B = text_emb.shape[0]
        sim_matrix = torch.matmul(text_emb, code_emb.T)
        targets = torch.arange(B, device=text_emb.device, dtype=torch.long)

        t2c_preds = sim_matrix.argmax(dim=-1)
        c2t_preds = sim_matrix.T.argmax(dim=-1)

        t2c_acc = float((t2c_preds == targets).float().mean().item())
        c2t_acc = float((c2t_preds == targets).float().mean().item())

        return t2c_acc, c2t_acc

