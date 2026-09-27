"""Unit tests for model architecture components.

Tests cover:
- Token, Positional, and Modality embeddings (shapes and additions).
- MultiHeadSelfAttention (shapes, custom projections, attention mask zeroing).
- PreLNTransformerBlock and TransformerEncoder (Pre-LN residual flow).
- MaskedMeanPooling and CLSPooling (valid token aggregation vs padding ignore).
- BaseEncoder (end-to-end forward pass, L2 unit norm, backward pass gradient check).
"""

import pytest
import torch

from model.attention import MultiHeadSelfAttention
from model.embeddings import (
    EmbeddingLayer,
    ModalityEmbedding,
    PositionalEmbedding,
    TokenEmbedding,
)
from model.encoder import BaseEncoder
from model.pooling import CLSPooling, MaskedMeanPooling
from model.transformer import PreLNTransformerBlock, TransformerEncoder


class TestEmbeddings:
    """Tests for embedding layers."""

    def test_token_embedding(self):
        embed = TokenEmbedding(vocab_size=100, d_model=32, padding_idx=0)
        ids = torch.tensor([[1, 5, 0], [2, 0, 0]], dtype=torch.long)
        out = embed(ids)
        assert out.shape == (2, 3, 32)
        # Pad token embedding at idx 0 should be zeros
        assert torch.all(out[1, 1:] == 0.0)

    def test_positional_embedding(self):
        pos_embed = PositionalEmbedding(max_seq_len=64, d_model=32)
        out = pos_embed(seq_len=10, device=torch.device("cpu"))
        assert out.shape == (1, 10, 32)

        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            pos_embed(seq_len=65, device=torch.device("cpu"))

    def test_modality_embedding(self):
        mod_embed = ModalityEmbedding(num_modalities=2, d_model=32)
        mod_1d = torch.tensor([0, 1], dtype=torch.long)
        out_1d = mod_embed(mod_1d)
        assert out_1d.shape == (2, 1, 32)

    def test_composite_embedding_layer(self):
        layer = EmbeddingLayer(vocab_size=100, d_model=32, max_seq_len=64, dropout=0.0)
        input_ids = torch.randint(0, 100, (4, 16))
        out = layer(input_ids)
        assert out.shape == (4, 16, 32)


class TestAttention:
    """Tests for MultiHeadSelfAttention."""

    def test_shape_and_divisibility(self):
        attn = MultiHeadSelfAttention(d_model=64, n_heads=4, dropout=0.0)
        x = torch.randn(2, 8, 64)
        out, _ = attn(x)
        assert out.shape == (2, 8, 64)

        with pytest.raises(ValueError, match="must be divisible by n_heads"):
            MultiHeadSelfAttention(d_model=65, n_heads=4)

    def test_attention_masking(self):
        """Padded tokens must receive zero attention probability."""
        attn = MultiHeadSelfAttention(d_model=32, n_heads=2, dropout=0.0)
        x = torch.randn(1, 4, 32)
        # Sequence has 2 real tokens and 2 pad tokens
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

        _, weights = attn(x, attention_mask=mask, return_attention_weights=True)
        assert weights is not None
        # weights shape: (B, H, L, L)
        # Column 2 and 3 correspond to keys at padded positions
        pad_attention = weights[0, :, :, 2:]
        # Attention to padded positions should be effectively zero (< 1e-6)
        assert torch.all(pad_attention < 1e-6)


class TestTransformer:
    """Tests for Pre-LN Transformer Blocks and Encoder Stack."""

    def test_block_shape(self):
        block = PreLNTransformerBlock(d_model=32, n_heads=4, d_ff=64, dropout=0.0)
        x = torch.randn(3, 10, 32)
        mask = torch.ones(3, 10, dtype=torch.long)
        out = block(x, attention_mask=mask)
        assert out.shape == (3, 10, 32)

    def test_encoder_stack(self):
        encoder = TransformerEncoder(
            n_layers=3, d_model=32, n_heads=4, d_ff=64, dropout=0.0
        )
        x = torch.randn(2, 12, 32)
        out = encoder(x)
        assert out.shape == (2, 12, 32)


class TestPooling:
    """Tests for pooling strategies."""

    def test_masked_mean_pooling(self):
        pooling = MaskedMeanPooling()
        # Create known hidden states: token 0 = [2, 2], token 1 = [4, 4], token 2 = [100, 100] (padded)
        h = torch.tensor([[[2.0, 2.0], [4.0, 4.0], [100.0, 100.0]]])  # (1, 3, 2)
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long)  # (1, 3)

        pooled = pooling(h, attention_mask=mask)
        # Mean of [2, 2] and [4, 4] is [3, 3]
        expected = torch.tensor([[3.0, 3.0]])
        assert torch.allclose(pooled, expected, atol=1e-5)

    def test_cls_pooling(self):
        pooling = CLSPooling()
        h = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        pooled = pooling(h)
        assert torch.allclose(pooled, torch.tensor([[1.0, 2.0]]))


class TestBaseEncoder:
    """Tests for complete BaseEncoder."""

    def test_end_to_end_and_normalization(self):
        encoder = BaseEncoder(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            max_seq_len=32,
            dropout=0.0,
        )
        input_ids = torch.randint(0, 100, (4, 16))
        mask = torch.ones(4, 16, dtype=torch.long)

        embeddings = encoder(input_ids, attention_mask=mask)
        assert embeddings.shape == (4, 32)

        # Output vectors must be normalized to unit length: ||z||_2 == 1.0
        norms = torch.norm(embeddings, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_backward_pass(self):
        encoder = BaseEncoder(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            max_seq_len=32,
        )
        input_ids = torch.randint(0, 100, (2, 8))
        embeddings = encoder(input_ids)
        loss = embeddings.sum()
        loss.backward()

        # Check gradients exist and contain no NaNs
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
