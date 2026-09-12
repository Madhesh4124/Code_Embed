"""Unit tests for evaluation/metrics.py.

Tests cover MRR, Recall@K, NDCG@K, bootstrap confidence intervals,
and edge cases (unretrieved items, boundaries, empty inputs).
"""

import math

import numpy as np
import pytest

from evaluation.metrics import (
    bootstrap_metric_ci,
    compute_all_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


class TestMRR:
    """Tests for Mean Reciprocal Rank (MRR)."""

    def test_perfect_ranks(self):
        """All queries ranked #1 should yield MRR 1.0."""
        ranks = [1, 1, 1, 1]
        assert mrr(ranks) == 1.0

    def test_known_values(self):
        """MRR with known rank list: [1, 2, 4] -> (1/1 + 1/2 + 1/4) / 3 = 1.75 / 3."""
        ranks = [1, 2, 4]
        expected = (1.0 + 0.5 + 0.25) / 3.0
        assert math.isclose(mrr(ranks), expected, rel_tol=1e-6)

    def test_unretrieved_or_invalid_ranks(self):
        """Unretrieved targets (0, -1, inf, nan) should score 0 reciprocal rank."""
        ranks = [1, 0, -1, float("inf"), float("nan")]
        # Only the first rank (1) is valid: 1.0 / 5 queries = 0.2
        expected = 1.0 / 5.0
        assert math.isclose(mrr(ranks), expected, rel_tol=1e-6)

    def test_empty_ranks(self):
        """Empty input should return 0.0 gracefully."""
        assert mrr([]) == 0.0
        assert mrr(np.array([])) == 0.0

    def test_numpy_and_nd_input(self):
        """MRR handles 2D or flattened numpy arrays."""
        ranks = np.array([[1, 2], [4, 5]])
        expected = (1.0 + 0.5 + 0.25 + 0.2) / 4.0
        assert math.isclose(mrr(ranks), expected, rel_tol=1e-6)


class TestRecallAtK:
    """Tests for Recall@K."""

    def test_exact_boundary(self):
        """Rank equal to k should be a hit; rank equal to k+1 should be a miss."""
        ranks = [1, 5, 6, 10]
        assert recall_at_k(ranks, k=5) == 2.0 / 4.0  # Ranks 1 and 5 are <= 5
        assert recall_at_k(ranks, k=1) == 1.0 / 4.0  # Only rank 1
        assert recall_at_k(ranks, k=10) == 4.0 / 4.0  # All 4 are <= 10

    def test_unretrieved_excluded(self):
        """0, negative, and inf should never count as hits."""
        ranks = [1, 0, -1, float("inf")]
        assert recall_at_k(ranks, k=5) == 1.0 / 4.0

    def test_empty_ranks(self):
        """Empty input should return 0.0."""
        assert recall_at_k([], k=10) == 0.0

    def test_invalid_k(self):
        """k < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            recall_at_k([1, 2], k=0)

        with pytest.raises(ValueError, match="k must be >= 1"):
            recall_at_k([1, 2], k=-5)


class TestNDCGAtK:
    """Tests for NDCG@K."""

    def test_rank_one(self):
        """Rank 1 gives DCG = 1 / log2(2) = 1.0 -> NDCG = 1.0."""
        ranks = [1]
        assert math.isclose(ndcg_at_k(ranks, k=10), 1.0, rel_tol=1e-6)

    def test_known_ndcg_values(self):
        """Verify logarithmic discount at ranks 2, 3, and beyond k."""
        # Rank 2: 1 / log2(3) ~ 0.63092975
        # Rank 3: 1 / log2(4) = 0.5
        # Rank 4: beyond k=3 -> 0.0
        ranks = [2, 3, 4]
        score = ndcg_at_k(ranks, k=3)
        expected = ((1.0 / np.log2(3.0)) + 0.5 + 0.0) / 3.0
        assert math.isclose(score, expected, rel_tol=1e-6)

    def test_unretrieved_excluded(self):
        """Unretrieved targets score 0."""
        ranks = [1, float("inf"), 0]
        expected = 1.0 / 3.0
        assert math.isclose(ndcg_at_k(ranks, k=10), expected, rel_tol=1e-6)

    def test_invalid_k(self):
        """k < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            ndcg_at_k([1, 2], k=0)

    def test_empty_ranks(self):
        """Empty input returns 0.0."""
        assert ndcg_at_k([], k=5) == 0.0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics convenience wrapper."""

    def test_standard_keys(self):
        """Check all expected keys are computed."""
        ranks = [1, 2, 5, 10, 15]
        metrics = compute_all_metrics(ranks, ks=(1, 5, 10))

        expected_keys = {
            "mrr",
            "recall@1",
            "recall@5",
            "recall@10",
            "ndcg@1",
            "ndcg@5",
            "ndcg@10",
        }
        assert set(metrics.keys()) == expected_keys

        # Check values match individual functions
        assert math.isclose(metrics["mrr"], mrr(ranks))
        assert math.isclose(metrics["recall@1"], recall_at_k(ranks, 1))
        assert math.isclose(metrics["recall@5"], recall_at_k(ranks, 5))
        assert math.isclose(metrics["recall@10"], recall_at_k(ranks, 10))
        assert math.isclose(metrics["ndcg@10"], ndcg_at_k(ranks, 10))


class TestBootstrapCI:
    """Tests for bootstrap_metric_ci."""

    def test_constant_ranks(self):
        """For constant ranks, confidence interval lower and upper bounds must match."""
        ranks = [1, 1, 1, 1, 1]
        ci_low, ci_high = bootstrap_metric_ci(ranks, metric_fn=mrr, n_bootstraps=100)
        assert math.isclose(ci_low, 1.0, rel_tol=1e-6)
        assert math.isclose(ci_high, 1.0, rel_tol=1e-6)

    def test_confidence_interval_bounds(self):
        """CI bounds should bracket the sample mean."""
        ranks = [1, 2, 3, 5, 10, 20, 0, 0]
        point_mrr = mrr(ranks)
        ci_low, ci_high = bootstrap_metric_ci(
            ranks, metric_fn=mrr, n_bootstraps=500, ci=0.95, seed=42
        )
        assert ci_low <= point_mrr <= ci_high
        assert 0.0 <= ci_low <= ci_high <= 1.0

    def test_seed_reproducibility(self):
        """Same seed should yield exact same CI intervals."""
        ranks = [1, 2, 4, 8, 16]
        ci1 = bootstrap_metric_ci(ranks, metric_fn=mrr, n_bootstraps=200, seed=123)
        ci2 = bootstrap_metric_ci(ranks, metric_fn=mrr, n_bootstraps=200, seed=123)
        assert ci1 == ci2

    def test_empty_ranks(self):
        """Empty ranks array returns (0.0, 0.0)."""
        ci_low, ci_high = bootstrap_metric_ci([], metric_fn=mrr)
        assert ci_low == 0.0
        assert ci_high == 0.0

