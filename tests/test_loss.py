"""Unit tests for InfoNCELoss."""

import math

import pytest
import torch
import torch.nn.functional as F

from losses.contrastive import InfoNCELoss


class TestInfoNCELoss:
    """Tests for symmetric InfoNCE loss."""

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="strictly positive"):
            InfoNCELoss(temperature=0.0)

        with pytest.raises(ValueError, match="strictly positive"):
            InfoNCELoss(temperature=-0.05)

    def test_mismatched_batch_size(self):
        loss_fn = InfoNCELoss(temperature=0.07)
        text_emb = torch.randn(4, 32)
        code_emb = torch.randn(5, 32)
        with pytest.raises(ValueError, match="Batch sizes mismatch"):
            loss_fn(text_emb, code_emb)

    def test_loss_symmetry(self):
        loss_fn = InfoNCELoss(temperature=0.07)
        text_emb = F.normalize(torch.randn(8, 32), p=2, dim=-1)
        code_emb = F.normalize(torch.randn(8, 32), p=2, dim=-1)

        loss1 = loss_fn(text_emb, code_emb)
        loss2 = loss_fn(code_emb, text_emb)
        assert math.isclose(loss1.item(), loss2.item(), rel_tol=1e-5)

    def test_perfect_alignment(self):
        """When query and code embeddings are perfectly aligned and distinct."""
        loss_fn = InfoNCELoss(temperature=0.1)
        # Use orthogonal unit basis vectors as embeddings (B=4, D=4)
        eye = torch.eye(4)
        loss = loss_fn(eye, eye)
        t2c_acc, c2t_acc = loss_fn.compute_accuracy(eye, eye)

        # On identity matrix, similarity is 1.0 on diagonal and 0.0 off-diagonal
        # Scaled by 0.1: diagonal is 10.0, off-diagonals are 0.0
        # Softmax gives ~1.0 on target
        assert loss.item() < 0.01
        assert t2c_acc == 1.0
        assert c2t_acc == 1.0

    def test_gradient_flow(self):
        loss_fn = InfoNCELoss(temperature=0.07)
        text_emb = torch.randn(4, 16, requires_grad=True)
        code_emb = torch.randn(4, 16, requires_grad=True)

        norm_text = F.normalize(text_emb, p=2, dim=-1)
        norm_code = F.normalize(code_emb, p=2, dim=-1)

        loss = loss_fn(norm_text, norm_code)
        loss.backward()

        assert text_emb.grad is not None
        assert code_emb.grad is not None
        assert not torch.isnan(text_emb.grad).any()
        assert not torch.isnan(code_emb.grad).any()

    def test_false_negative_masking(self):
        """Verify that duplicate in-batch pairs are masked out of the contrastive denominator."""
        loss_fn = InfoNCELoss(temperature=0.07)
        # 3 samples, where sample 0 and 1 are identical duplicates
        text_emb = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        code_emb = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        # Mask indicating item 0 and item 1 share identical queries/codes
        mask = torch.tensor(
            [[True, True, False], [True, True, False], [False, False, True]],
            dtype=torch.bool,
        )

        loss_unmasked = loss_fn(text_emb, code_emb)
        loss_masked = loss_fn(text_emb, code_emb, mask=mask)

        # Unmasked loss penalizes pair (0, 1) and (1, 0) as hard negatives even though they are identical
        # Masked loss excludes them from the denominator, resulting in lower loss
        assert loss_masked.item() < loss_unmasked.item()
        assert not torch.isnan(loss_masked)

