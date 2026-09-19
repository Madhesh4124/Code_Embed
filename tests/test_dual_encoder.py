"""Unit tests for DualEncoder architecture (Phase 4: Model 3).

Tests cover:
- DualEncoder initialization and parameter budget matching.
- Independent forward pass for code and text sequences.
- Unit-sphere L2 normalization (||z||_2 == 1.0).
- Forward pass with code_ids only, text_ids only, or both.
- Decoupled gradient flow: code gradients do not leak to text encoder and vice versa.
- Invalid input error handling.
"""

import pytest
import torch

from model.dual_encoder import DualEncoder
from model.encoder import BaseEncoder


class TestDualEncoder:
    """Test suite for DualEncoder architecture."""

    def test_parameter_counts_and_breakdown(self):
        """DualEncoder with 3 layers should equal 2x BaseEncoder(3 layers)."""
        vocab_size = 1000
        d_model = 64
        n_layers = 3
        n_heads = 4
        d_ff = 128
        max_seq_len = 64

        single_base = BaseEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
        )
        single_total, single_trainable = single_base.get_num_params()

        dual = DualEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
        )
        dual_total, dual_trainable = dual.get_num_params()
        breakdown = dual.get_encoder_params()

        # Both sub-encoders must exactly equal single_total
        assert breakdown["code_encoder_total"] == single_total
        assert breakdown["text_encoder_total"] == single_total
        assert dual_total == 2 * single_total
        assert dual_trainable == 2 * single_trainable
        assert breakdown["total_params"] == dual_total

    def test_forward_paired_and_unit_norm(self):
        """Forward pass with code and text pairs must return unit vectors."""
        model = DualEncoder(vocab_size=500, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=32)
        model.eval()

        code_ids = torch.randint(0, 500, (4, 16))
        text_ids = torch.randint(0, 500, (4, 12))
        code_mask = torch.ones(4, 16, dtype=torch.long)
        text_mask = torch.ones(4, 12, dtype=torch.long)

        code_emb, text_emb = model(
            code_ids=code_ids,
            code_mask=code_mask,
            text_ids=text_ids,
            text_mask=text_mask,
        )

        assert code_emb.shape == (4, 32)
        assert text_emb.shape == (4, 32)

        # Output embeddings must lie strictly on unit hypersphere
        code_norms = torch.norm(code_emb, p=2, dim=-1)
        text_norms = torch.norm(text_emb, p=2, dim=-1)
        assert torch.allclose(code_norms, torch.ones(4), atol=1e-5)
        assert torch.allclose(text_norms, torch.ones(4), atol=1e-5)

    def test_encode_single_modality_methods(self):
        """Test encode_code and encode_text standalone methods."""
        model = DualEncoder(vocab_size=500, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=32)
        model.eval()

        ids = torch.randint(0, 500, (3, 10))
        mask = torch.ones(3, 10, dtype=torch.long)

        # encode_code vs forward(code_ids=...)
        code_out1 = model.encode_code(ids, attention_mask=mask)
        code_out2 = model(code_ids=ids, code_mask=mask)
        assert torch.allclose(code_out1, code_out2, atol=1e-6)

        # encode_text vs forward(text_ids=...)
        text_out1 = model.encode_text(ids, attention_mask=mask)
        text_out2 = model(text_ids=ids, text_mask=mask)
        assert torch.allclose(text_out1, text_out2, atol=1e-6)

        # Code and text encoders have different random weights, so identical inputs give different embeddings
        assert not torch.allclose(code_out1, text_out1, atol=1e-3)

    def test_independent_gradient_isolation(self):
        """Gradients computed on code_emb should only update code_encoder, leaving text_encoder untouched."""
        model = DualEncoder(vocab_size=200, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=32)
        model.train()

        code_ids = torch.randint(0, 200, (2, 8))
        text_ids = torch.randint(0, 200, (2, 8))

        code_emb, text_emb = model(code_ids=code_ids, text_ids=text_ids)

        # Loss only on code_emb
        loss_code = code_emb.sum()
        loss_code.backward(retain_graph=True)

        for name, param in model.code_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Expected grad on code_encoder param {name}"
                assert not torch.isnan(param.grad).any()

        for name, param in model.text_encoder.named_parameters():
            assert param.grad is None, f"Text encoder param {name} should NOT have received gradients"

        # Zero grads and test loss only on text_emb
        model.zero_grad()
        loss_text = text_emb.sum()
        loss_text.backward()

        for name, param in model.text_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Expected grad on text_encoder param {name}"
                assert not torch.isnan(param.grad).any()

        for name, param in model.code_encoder.named_parameters():
            assert param.grad is None, f"Code encoder param {name} should NOT have received gradients"

    def test_invalid_forward_arguments(self):
        """Calling forward with neither code_ids nor text_ids should raise ValueError."""
        model = DualEncoder(vocab_size=100, d_model=32, n_layers=1, n_heads=2, d_ff=32, max_seq_len=16)
        with pytest.raises(ValueError, match="At least one of code_ids or text_ids must be provided"):
            model(code_ids=None, text_ids=None)
