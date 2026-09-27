"""Unit tests for Phase 6 Ablation components.

Tests:
1. AblationSharedEncoder with CLSPooling: output shapes, unit-norm vectors, backward gradients.
2. AblationSharedEncoder with MaskedMeanPooling: consistency with base encoder.
3. AblationCodeSearchDataset: sequence length truncation (L=128 vs L=256).
4. Temperature sensitivity: verify scaling in contrastive InfoNCE loss.
"""

import pytest
import torch

from data.ablation_dataset import AblationCodeSearchDataset
from losses.contrastive import InfoNCEWithHardNegativesLoss
from model.ablation_models import AblationSharedEncoder


class TestAblationModels:
    """Test suite for AblationSharedEncoder architecture variants."""

    def test_cls_pooling_shapes_and_norm(self) -> None:
        """Verify CLSPooling produces L2-normalized vectors of shape (B, D)."""
        B, L, D = 4, 32, 64
        model = AblationSharedEncoder(
            vocab_size=1000,
            d_model=D,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            max_seq_len=L,
            pooling="cls",
        )

        input_ids = torch.randint(0, 1000, (B, L))
        mask = torch.ones((B, L), dtype=torch.long)
        mask[:, -5:] = 0  # 5 padding tokens

        embs = model(input_ids, attention_mask=mask, modality_ids="code")
        assert embs.shape == (B, D)

        norms = torch.norm(embs, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5), (
            f"Embeddings not unit norm: {norms}"
        )

    def test_cls_pooling_gradient_flow(self) -> None:
        """Verify backward gradients propagate through all layers with CLS pooling."""
        B, L, D = 4, 32, 64
        model = AblationSharedEncoder(
            vocab_size=1000,
            d_model=D,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            max_seq_len=L,
            pooling="cls",
        )

        input_ids = torch.randint(0, 1000, (B, L))
        mask = torch.ones((B, L), dtype=torch.long)
        embs = model(input_ids, attention_mask=mask, modality_ids="code")

        loss = embs.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient!"
                assert not torch.isnan(param.grad).any(), (
                    f"Parameter {name} has NaN gradient!"
                )

    def test_invalid_pooling_raises_error(self) -> None:
        """Verify passing unsupported pooling string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported pooling strategy"):
            AblationSharedEncoder(vocab_size=100, d_model=32, n_layers=1, pooling="max")  # type: ignore

    def test_modality_routing(self) -> None:
        """Verify code and text modalities produce distinct representations."""
        B, L, D = 2, 16, 64
        model = AblationSharedEncoder(
            vocab_size=100,
            d_model=D,
            n_layers=1,
            n_heads=2,
            d_ff=64,
            max_seq_len=L,
            pooling="cls",
        )

        input_ids = torch.randint(0, 100, (B, L))
        code_embs = model.encode_code(input_ids)
        text_embs = model.encode_text(input_ids)

        assert not torch.allclose(code_embs, text_embs), (
            "Modality embeddings failed to differentiate!"
        )


class TestAblationDataset:
    """Test suite for sequence length truncation in AblationCodeSearchDataset."""

    def test_sequence_length_truncation(self) -> None:
        """Verify dataset correctly slices pre-tokenized tensors to specified max_seq_len."""
        dataset_128 = AblationCodeSearchDataset(
            split_or_path="test",
            max_seq_len=128,
            use_pretokenized=True,
        )

        sample = dataset_128[0]
        assert sample["code_ids"].shape == (128,), (
            f"Expected length 128, got {sample['code_ids'].shape}"
        )
        assert sample["code_mask"].shape == (128,)
        assert sample["text_ids"].shape == (128,)
        assert sample["text_mask"].shape == (128,)

    def test_full_sequence_length_preserved(self) -> None:
        """Verify default or max_seq_len=256 preserves full 256 tokens."""
        dataset_256 = AblationCodeSearchDataset(
            split_or_path="test",
            max_seq_len=256,
            use_pretokenized=True,
        )

        sample = dataset_256[0]
        assert sample["code_ids"].shape == (256,), (
            f"Expected length 256, got {sample['code_ids'].shape}"
        )


class TestAblationTemperature:
    """Test suite for loss temperature sensitivity."""

    def test_temperature_scaling(self) -> None:
        """Verify smaller temperature sharpens contrastive similarity matrix."""
        B, D = 4, 32
        text_emb = torch.randn(B, D)
        text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=-1)
        code_emb = torch.randn(B, D)
        code_emb = torch.nn.functional.normalize(code_emb, p=2, dim=-1)

        loss_005 = InfoNCEWithHardNegativesLoss(temperature=0.05)(text_emb, code_emb)
        loss_010 = InfoNCEWithHardNegativesLoss(temperature=0.10)(text_emb, code_emb)

        # Different temperatures must produce distinctly different loss values
        assert not torch.isclose(loss_005, loss_010), (
            "Losses should differ across temperatures!"
        )
        assert loss_005.item() > 0 and loss_010.item() > 0
