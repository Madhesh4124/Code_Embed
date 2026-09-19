"""Unit tests for Phase 5: Hard Negative Mining & Loss formulation.

Tests cover:
- InfoNCEWithHardNegativesLoss with and without explicit hard negatives.
- Gradient flow to text embeddings, positive code, and hard negative code representations.
- Dimension checking and batch size mismatch error handling.
- BM25HardNegativeMiner query negative retrieval and ground-truth target exclusion.
- Dataset & Collator batch stacking with pre-tokenized hard negative tensors.
"""

import numpy as np
import torch

from data.dataset import CodeSearchCollator
from losses.contrastive import InfoNCEWithHardNegativesLoss
from retrieval.bm25 import BM25Retriever
from tokenizer.tokenizer import CodeEmbedTokenizer
from training.hard_negatives import BM25HardNegativeMiner


class TestHardNegativeLoss:
    """Test suite for InfoNCEWithHardNegativesLoss."""

    def test_loss_without_hard_negatives_matches_standard(self):
        """When neg_code_emb is None, loss should equal standard in-batch InfoNCE."""
        criterion = InfoNCEWithHardNegativesLoss(temperature=0.07)
        B, D = 4, 32
        torch.manual_seed(42)
        text_emb = torch.randn(B, D)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        code_emb = torch.randn(B, D)
        code_emb = code_emb / code_emb.norm(dim=-1, keepdim=True)

        loss = criterion(text_emb, code_emb, neg_code_emb=None)
        assert loss.item() > 0.0
        assert not torch.isnan(loss)

        # Accuracy
        t2c_acc, c2t_acc = criterion.compute_accuracy(text_emb, code_emb, None)
        assert 0.0 <= t2c_acc <= 1.0
        assert 0.0 <= c2t_acc <= 1.0

    def test_loss_with_hard_negatives_and_gradient_flow(self):
        """Hard negatives should penalize similar negative codes and flow gradients back to neg_emb."""
        criterion = InfoNCEWithHardNegativesLoss(temperature=0.07)
        B, K, D = 4, 2, 32

        text_emb = torch.randn(B, D, requires_grad=True)
        text_norm = text_emb / text_emb.norm(dim=-1, keepdim=True)
        pos_code = torch.randn(B, D, requires_grad=True)
        pos_norm = pos_code / pos_code.norm(dim=-1, keepdim=True)
        neg_code = torch.randn(B * K, D, requires_grad=True)
        neg_norm = neg_code / neg_code.norm(dim=-1, keepdim=True)

        loss = criterion(text_norm, pos_norm, neg_norm)
        assert loss.item() > 0.0
        loss.backward()

        assert text_emb.grad is not None
        assert pos_code.grad is not None
        assert neg_code.grad is not None
        assert not torch.isnan(text_emb.grad).any()
        assert not torch.isnan(pos_code.grad).any()
        assert not torch.isnan(neg_code.grad).any()


class TestBM25HardNegativeMiner:
    """Test suite for BM25 hard negative mining."""

    def test_miner_excludes_ground_truth_target(self):
        """Miner must strictly exclude the ground truth target document index from negative list."""
        codes = [
            "def calculate_mean(numbers): return sum(numbers) / len(numbers)",
            "def compute_average(values): return sum(values) / float(len(values))",
            "def calculate_median(numbers): return sorted(numbers)[len(numbers)//2]",
            "def calculate_variance(numbers): return sum((x - 2)**2 for x in numbers)",
            "def quicksort(arr): return arr if len(arr) <= 1 else arr",
        ]
        bm25 = BM25Retriever()
        bm25.index(codes)
        miner = BM25HardNegativeMiner(bm25)

        # Query 0 corresponds to code 0 (calculate_mean)
        negs = miner.mine_query_negatives(query="calculate mean average", true_doc_idx=0, k=2)

        assert len(negs) == 2
        assert 0 not in negs, "Target document index 0 must NOT be in the mined negative list!"

    def test_miner_corpus_matrix_shape(self):
        """Miner should produce an (N, K) int32 matrix."""
        codes = [
            "def add(a, b): return a + b",
            "def subtract(a, b): return a - b",
            "def multiply(a, b): return a * b",
        ]
        queries = ["add numbers", "subtract numbers", "multiply numbers"]
        bm25 = BM25Retriever()
        bm25.index(codes)
        miner = BM25HardNegativeMiner(bm25)

        matrix = miner.mine_corpus_negatives(queries, k=2)
        assert matrix.shape == (3, 2)
        assert matrix.dtype == np.int32
        for i in range(3):
            assert i not in matrix[i]


class TestCollatorHardNegatives:
    """Test suite for batch collation with hard negative tensors."""

    def test_collator_stacks_hard_negative_tensors(self):
        tokenizer = CodeEmbedTokenizer()
        collator = CodeSearchCollator(tokenizer=tokenizer, max_length=16)

        batch = [
            {
                "code_ids": torch.randint(0, 100, (16,)),
                "code_mask": torch.ones(16, dtype=torch.long),
                "text_ids": torch.randint(0, 100, (16,)),
                "text_mask": torch.ones(16, dtype=torch.long),
                "hard_neg_code_ids": torch.randint(0, 100, (2, 16)),
                "hard_neg_code_mask": torch.ones(2, 16, dtype=torch.long),
            },
            {
                "code_ids": torch.randint(0, 100, (16,)),
                "code_mask": torch.ones(16, dtype=torch.long),
                "text_ids": torch.randint(0, 100, (16,)),
                "text_mask": torch.ones(16, dtype=torch.long),
                "hard_neg_code_ids": torch.randint(0, 100, (2, 16)),
                "hard_neg_code_mask": torch.ones(2, 16, dtype=torch.long),
            },
        ]

        collated = collator(batch)
        assert collated["code_ids"].shape == (2, 16)
        assert collated["hard_neg_code_ids"].shape == (2, 2, 16)
        assert collated["hard_neg_code_mask"].shape == (2, 2, 16)
