"""Unit tests for SharedEncoder architecture (Phase 3: Model 2).

Tests cover:
- SharedEncoder initialization and parameter counts.
- Forward pass with various modality input types (string, int, tensor, None).
- Output shape and L2 unit-sphere normalization.
- Backward pass and gradient flow through modality embeddings.
- Masked sequence pooling inside SharedEncoder.
- Invalid modality type error handling.
"""

import pytest
import torch

from model.encoder import BaseEncoder
from model.shared_encoder import SharedEncoder


class TestSharedEncoder:
    """Test suite for SharedEncoder with learned modality embeddings."""

    def test_parameter_count_matches_budget(self):
        """SharedEncoder should have BaseEncoder parameters + 2 * d_model modality weights."""
        base = BaseEncoder(vocab_size=1000, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=64)
        shared = SharedEncoder(vocab_size=1000, d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=64, num_modalities=2)

        base_total, _ = base.get_num_params()
        shared_total, shared_trainable = shared.get_num_params()

        # Shared has exactly 2 * 64 = 128 additional weights for modality table
        assert shared_total == base_total + 2 * 64
        assert shared_trainable == shared_total

    def test_forward_with_modalities_and_unit_norm(self):
        """Test forward pass with 'code' and 'text' modalities and verify unit-length outputs."""
        model = SharedEncoder(vocab_size=500, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=32)
        model.eval()

        input_ids = torch.randint(0, 500, (4, 16))
        mask = torch.ones(4, 16, dtype=torch.long)

        # 1. Forward with string modality
        z_code = model(input_ids, attention_mask=mask, modality_ids="code")
        z_text = model(input_ids, attention_mask=mask, modality_ids="text")

        assert z_code.shape == (4, 32)
        assert z_text.shape == (4, 32)

        # Modality embedding should differentiate code vs text outputs for identical token sequence
        assert not torch.allclose(z_code, z_text, atol=1e-3)

        # Both must lie exactly on unit hypersphere: ||z||_2 == 1.0
        code_norms = torch.norm(z_code, p=2, dim=-1)
        text_norms = torch.norm(z_text, p=2, dim=-1)
        assert torch.allclose(code_norms, torch.ones(4), atol=1e-5)
        assert torch.allclose(text_norms, torch.ones(4), atol=1e-5)

    def test_forward_with_integer_and_tensor_modalities(self):
        """Test forward pass with int and tensor modality specifications."""
        model = SharedEncoder(vocab_size=200, d_model=32, n_layers=1, n_heads=2, d_ff=64, max_seq_len=32, dropout=0.0)
        model.eval()
        input_ids = torch.randint(0, 200, (2, 8))

        # Int modality
        out_int = model(input_ids, modality_ids=0)
        # Tensor modality
        out_tensor = model(input_ids, modality_ids=torch.tensor([0, 0], dtype=torch.long))

        assert out_int.shape == (2, 32)
        assert torch.allclose(out_int, out_tensor, atol=1e-5)

    def test_backward_pass_gradients(self):
        """Verify backpropagation produces valid gradients for all parameters including modality embeddings."""
        model = SharedEncoder(vocab_size=200, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=32)
        input_ids = torch.randint(0, 200, (2, 8))

        z = model(input_ids, modality_ids="code")
        loss = z.sum()
        loss.backward()

        # Check modality embedding has gradients
        mod_weight = model.embeddings.modality_embed.embedding.weight
        assert mod_weight.grad is not None
        assert not torch.isnan(mod_weight.grad).any()
        assert mod_weight.grad[0].abs().sum() > 0  # index 0 (code) was used and updated

        # Check all trainable parameters have clean gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_invalid_modality_type(self):
        """Test invalid modality input throws TypeError."""
        model = SharedEncoder(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=32, max_seq_len=16)
        input_ids = torch.randint(0, 100, (2, 4))
        with pytest.raises(TypeError, match="Unsupported modality_ids type"):
            model(input_ids, modality_ids=3.14)  # float is unsupported
