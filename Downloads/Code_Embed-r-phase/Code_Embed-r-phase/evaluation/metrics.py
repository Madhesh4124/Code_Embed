"""Evaluation metrics for code retrieval benchmarks.

This module provides implementations of core Information Retrieval (IR) metrics:
- Mean Reciprocal Rank (MRR)
- Recall@K (R@1, R@5, R@10, etc.)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Non-parametric bootstrap confidence intervals (1,000 resamples)

Rank Convention:
    Ranks are 1-indexed integers (1 = top recommendation, 2 = second, etc.).
    If a query's ground truth target was not found in the retrieved candidates,
    it should be represented as 0, negative (-1), np.nan, or float('inf').
    All such unretrieved targets contribute 0 to the metric score.
"""

from collections.abc import Callable, Sequence

import numpy as np


def _to_valid_rank_array(ranks: Sequence[int | float] | np.ndarray) -> np.ndarray:
    """Convert input ranks into a 1D float numpy array.

    Args:
        ranks: Sequence or array of 1-based ranks.

    Returns:
        1D numpy float array.
    """
    arr = np.asarray(ranks, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.flatten()
    return arr


def mrr(ranks: Sequence[int | float] | np.ndarray) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Formula:
        MRR = (1 / |Q|) * sum_{i=1}^{|Q|} (1 / rank_i)
        where (1 / rank_i) = 0 if rank_i is invalid (<= 0, inf, or NaN).

    Args:
        ranks: 1-based rank of the ground-truth document for each query.

    Returns:
        MRR score in [0.0, 1.0]. Returns 0.0 if ranks is empty.
    """
    arr = _to_valid_rank_array(ranks)
    if len(arr) == 0:
        return 0.0

    # Valid ranks are positive finite numbers >= 1
    valid_mask = (arr >= 1.0) & np.isfinite(arr)
    reciprocal_ranks = np.zeros_like(arr)
    reciprocal_ranks[valid_mask] = 1.0 / arr[valid_mask]

    return float(np.mean(reciprocal_ranks))


def recall_at_k(
    ranks: Sequence[int | float] | np.ndarray,
    k: int,
) -> float:
    """Compute Recall@K (proportion of queries where true document is in top-K).

    Formula:
        Recall@K = (1 / |Q|) * sum_{i=1}^{|Q|} I(1 <= rank_i <= K)

    Args:
        ranks: 1-based rank of the ground-truth document for each query.
        k: Cutoff threshold (must be >= 1).

    Returns:
        Recall@K score in [0.0, 1.0]. Returns 0.0 if ranks is empty.

    Raises:
        ValueError: If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    arr = _to_valid_rank_array(ranks)
    if len(arr) == 0:
        return 0.0

    hits = (arr >= 1.0) & (arr <= float(k))
    return float(np.mean(hits))


def ndcg_at_k(
    ranks: Sequence[int | float] | np.ndarray,
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K) for single-target retrieval.

    Formula:
        For single relevant item per query (binary relevance r in {0, 1}):
            DCG@K = 1 / log2(rank_i + 1)   if 1 <= rank_i <= K else 0
            IDCG@K = 1 / log2(1 + 1) = 1.0
            NDCG@K = DCG@K / IDCG@K = 1 / log2(rank_i + 1)   if 1 <= rank_i <= K else 0

    Args:
        ranks: 1-based rank of the ground-truth document for each query.
        k: Cutoff threshold (must be >= 1).

    Returns:
        NDCG@K score in [0.0, 1.0]. Returns 0.0 if ranks is empty.

    Raises:
        ValueError: If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    arr = _to_valid_rank_array(ranks)
    if len(arr) == 0:
        return 0.0

    valid_hits = (arr >= 1.0) & (arr <= float(k))
    gains = np.zeros_like(arr)
    gains[valid_hits] = 1.0 / np.log2(arr[valid_hits] + 1.0)

    return float(np.mean(gains))


def compute_all_metrics(
    ranks: Sequence[int | float] | np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute standard IR benchmark suite: MRR, Recall@K, and NDCG@K.

    Args:
        ranks: 1-based rank of the ground-truth document for each query.
        ks: Tuple or list of cutoff thresholds K (default: (1, 5, 10)).

    Returns:
        Dictionary containing all computed metric scores.
    """
    arr = _to_valid_rank_array(ranks)
    metrics: dict[str, float] = {
        "mrr": mrr(arr),
    }

    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(arr, k=k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(arr, k=k)

    return metrics


def bootstrap_metric_ci(
    ranks: Sequence[int | float] | np.ndarray,
    metric_fn: Callable[[np.ndarray], float],
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Calculate non-parametric bootstrap confidence interval for a retrieval metric.

    Algorithm:
        1. Draw sample of size |Q| with replacement from ranks array.
        2. Compute metric value on bootstrap sample.
        3. Repeat n_bootstraps times.
        4. Return percentiles: ((1 - ci) / 2 * 100, (1 + ci) / 2 * 100).

    Args:
        ranks: 1-based ranks array.
        metric_fn: Function that maps a 1D numpy array of ranks to a scalar float metric.
        n_bootstraps: Number of bootstrap resamples (default: 1,000).
        ci: Confidence interval fraction (default: 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    arr = _to_valid_rank_array(ranks)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    # Generate all bootstrap sample indices in a single matrix: (n_bootstraps, n)
    indices = rng.integers(low=0, high=n, size=(n_bootstraps, n))

    bootstrap_scores = np.empty(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
        bootstrap_sample = arr[indices[i]]
        bootstrap_scores[i] = metric_fn(bootstrap_sample)

    alpha = 1.0 - ci
    lower_percentile = (alpha / 2.0) * 100.0
    upper_percentile = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(bootstrap_scores, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_scores, upper_percentile))

    return (ci_lower, ci_upper)
