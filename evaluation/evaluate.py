"""Reusable evaluation runner for code retrieval benchmarks.

This module provides standard evaluation utilities to compute retrieval metrics
from either:
1. Precomputed rank arrays (e.g., from BM25 or FAISS).
2. Dense query and corpus embeddings (using batch cosine similarity / inner product).
"""

import os
import sys
from collections.abc import Sequence
from pathlib import Path

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from evaluation.metrics import (
    bootstrap_metric_ci,
    compute_all_metrics,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def evaluate_rankings(
    ranks: Sequence[int | float] | np.ndarray,
    ks: Sequence[int] = (1, 5, 10),
    bootstrap_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | tuple[float, float]]:
    """Compute complete metric suite and bootstrap confidence intervals from ranks.

    Args:
        ranks: 1D sequence or array of 1-based ranks.
        ks: Cutoff values for Recall@K and NDCG@K.
        bootstrap_resamples: Number of bootstrap iterations.
        confidence_level: Desired coverage probability for CIs (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping metric names to their scalar scores and (ci_low, ci_high) tuples.
    """
    arr = np.asarray(ranks, dtype=np.float64)
    results: dict[str, float | tuple[float, float]] = {}

    # 1. Base metrics
    base_metrics = compute_all_metrics(arr, ks=ks)
    results.update(base_metrics)

    # 2. Bootstrap confidence intervals
    if bootstrap_resamples > 0:
        results["mrr_ci"] = bootstrap_metric_ci(
            arr,
            metric_fn=mrr,
            n_bootstraps=bootstrap_resamples,
            ci=confidence_level,
            seed=seed,
        )
        for k in ks:
            results[f"recall@{k}_ci"] = bootstrap_metric_ci(
                arr,
                metric_fn=lambda r, cutoff=k: recall_at_k(r, k=cutoff),
                n_bootstraps=bootstrap_resamples,
                ci=confidence_level,
                seed=seed,
            )
            results[f"ndcg@{k}_ci"] = bootstrap_metric_ci(
                arr,
                metric_fn=lambda r, cutoff=k: ndcg_at_k(r, k=cutoff),
                n_bootstraps=bootstrap_resamples,
                ci=confidence_level,
                seed=seed,
            )

    return results


def compute_similarity_rankings(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    ground_truth_indices: Sequence[int],
    batch_size: int = 256,
) -> np.ndarray:
    """Compute ground-truth ranks from dense query and corpus embeddings.

    Computes cosine similarities in mini-batches to minimize memory footprint.
    Embeddings should be L2-normalized unit vectors.

    Args:
        query_embeddings: 2D array of shape (|queries|, D).
        corpus_embeddings: 2D array of shape (|corpus|, D).
        ground_truth_indices: Target document index for each query.
        batch_size: Mini-batch size for query similarity computation.

    Returns:
        1D array of shape (|queries|,) containing 1-based ranks.
    """
    n_queries = len(query_embeddings)
    total_docs = len(corpus_embeddings)
    ranks = np.zeros(n_queries, dtype=np.float64)

    # Precompute harmonic numbers H_n = sum_{k=1}^n 1/k for exact expected RR under ties (Protocol v1.1)
    harmonic_table = np.zeros(total_docs + 1, dtype=np.float64)
    harmonic_table[1:] = np.cumsum(
        1.0 / np.arange(1, total_docs + 1, dtype=np.float64)
    )

    for start_idx in range(0, n_queries, batch_size):
        end_idx = min(start_idx + batch_size, n_queries)
        batch_q = query_embeddings[start_idx:end_idx]  # (B, D)

        # Compute similarity matrix for batch: (B, |corpus|)
        sim_matrix = np.dot(batch_q, corpus_embeddings.T)

        for i, global_q_idx in enumerate(range(start_idx, end_idx)):
            gt_idx = ground_truth_indices[global_q_idx]
            target_score = sim_matrix[i, gt_idx]

            higher_count = int(np.sum(sim_matrix[i] > target_score))
            eq_count = int(np.sum(sim_matrix[i] == target_score))
            if eq_count <= 0:
                eq_count = 1

            # Protocol v1.1 & Errata §1.5: Generalized Harmonic Expected Reciprocal Rank
            if eq_count > 1:
                expected_rr = (
                    harmonic_table[higher_count + eq_count]
                    - harmonic_table[higher_count]
                ) / eq_count
                rank = 1.0 / expected_rr if expected_rr > 0 else np.inf
            else:
                rank = float(higher_count + 1)

            ranks[global_q_idx] = rank

    return ranks


def format_metrics_summary(
    results: dict[str, float | tuple[float, float]],
) -> str:
    """Format evaluation dictionary into a clean markdown / ASCII summary table.

    Args:
        results: Dictionary returned by compute_all_metrics / evaluate_rankings.

    Returns:
        Formatted multi-line string.
    """
    lines = [
        "| Metric | Score | 95% Confidence Interval |",
        "| :--- | :---: | :---: |",
    ]

    metric_keys = [k for k in results if not k.endswith("_ci")]
    for key in metric_keys:
        score = results[key]
        ci_key = f"{key}_ci"
        if ci_key in results:
            ci_low, ci_high = results[ci_key]  # type: ignore
            lines.append(
                f"| {key.upper():<12} | {score:0.4f} | [{ci_low:0.4f}, {ci_high:0.4f}] |"
            )
        else:
            lines.append(f"| {key.upper():<12} | {score:0.4f} | — |")

    return "\n".join(lines)


def evaluate_checkpoint(
    checkpoint_path: str,
    split: str = "test",
    sample_size: int | None = 1000,
    seed: int = 42,
    batch_size: int = 128,
    experiment_name: str = "CodeEmbed-Evaluation",
    data_dir: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, float | tuple[float, float]]:
    """Evaluate a saved model checkpoint against a dataset split.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        split: Data split ('test' or 'validation').
        sample_size: Number of queries to evaluate against full corpus (or None for all).
        seed: Random seed for query subsampling.
        batch_size: Inference batch size.
        experiment_name: MLflow experiment name.
        data_dir: Optional path to processed data directory (e.g. data/processed_clean_v2).
        tracking_uri: Optional MLflow tracking URI.

    Returns:
        Dictionary of computed retrieval metrics and confidence intervals.
    """
    from data.dataset import create_dataloader
    from model.dual_encoder import DualEncoder
    from model.encoder import BaseEncoder
    from model.shared_encoder import SharedEncoder
    from tokenizer.tokenizer import CodeEmbedTokenizer

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(
        f"[cyan]Evaluating checkpoint on device:[/cyan] {device.type.upper()}"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    m_cfg = cfg.get("model", {})
    model_name = str(m_cfg.get("name", "basic")).lower()

    if model_name == "dual":
        model = DualEncoder(
            vocab_size=int(m_cfg.get("vocab_size", 16000)),
            d_model=int(m_cfg.get("d_model", 256)),
            n_layers=int(m_cfg.get("n_layers", 3)),
            n_heads=int(m_cfg.get("n_heads", 8)),
            d_ff=int(m_cfg.get("d_ff", 1024)),
            max_seq_len=int(m_cfg.get("max_seq_len", 256)),
            dropout=0.0,
        ).to(device)
        model_type = "dual"
    elif model_name == "shared" or "num_modalities" in m_cfg:
        model = SharedEncoder(
            vocab_size=int(m_cfg.get("vocab_size", 16000)),
            d_model=int(m_cfg.get("d_model", 256)),
            n_layers=int(m_cfg.get("n_layers", 4)),
            n_heads=int(m_cfg.get("n_heads", 8)),
            d_ff=int(m_cfg.get("d_ff", 1024)),
            max_seq_len=int(m_cfg.get("max_seq_len", 256)),
            dropout=0.0,
            num_modalities=int(m_cfg.get("num_modalities", 2)),
        ).to(device)
        model_type = "shared"
    else:
        model = BaseEncoder(
            vocab_size=int(m_cfg.get("vocab_size", 16000)),
            d_model=int(m_cfg.get("d_model", 256)),
            n_layers=int(m_cfg.get("n_layers", 4)),
            n_heads=int(m_cfg.get("n_heads", 8)),
            d_ff=int(m_cfg.get("d_ff", 1024)),
            max_seq_len=int(m_cfg.get("max_seq_len", 256)),
            dropout=0.0,
        ).to(device)
        model_type = "basic"

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    console.print(
        f"[green][OK][/green] Loaded {model_type} weights from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})"
    )

    # Build DataLoader for corpus and queries
    tokenizer = CodeEmbedTokenizer()
    loader = create_dataloader(
        split=split,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=int(m_cfg.get("max_seq_len", 256)),
        data_dir=data_dir,
    )

    console.print(f"[cyan]Encoding {split} corpus and queries...[/cyan]")
    all_code_embs: list[np.ndarray] = []
    all_text_embs: list[np.ndarray] = []

    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            code_ids = batch["code_ids"].to(device)
            code_mask = batch["code_mask"].to(device)
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if model_type == "dual":
                    c_emb = model.encode_code(code_ids, attention_mask=code_mask)
                    t_emb = model.encode_text(text_ids, attention_mask=text_mask)
                elif model_type == "shared":
                    c_emb = model(
                        code_ids, attention_mask=code_mask, modality_ids="code"
                    )
                    t_emb = model(
                        text_ids, attention_mask=text_mask, modality_ids="text"
                    )
                else:
                    c_emb = model(code_ids, attention_mask=code_mask)
                    t_emb = model(text_ids, attention_mask=text_mask)

            all_code_embs.append(c_emb.cpu().numpy())
            all_text_embs.append(t_emb.cpu().numpy())

    corpus_embeddings = np.concatenate(all_code_embs, axis=0)
    query_embeddings = np.concatenate(all_text_embs, axis=0)
    total_docs = len(corpus_embeddings)
    console.print(f"[green][OK][/green] Encoded {total_docs:,} code and query vectors.")

    # Subsample queries if specified
    if sample_size and sample_size < total_docs:
        rng = np.random.RandomState(seed)
        sampled_indices = rng.choice(total_docs, size=sample_size, replace=False)
        eval_queries = query_embeddings[sampled_indices]
        ground_truth = sampled_indices.tolist()
        console.print(
            f"[yellow]Evaluating {sample_size:,} sampled queries against full {total_docs:,} corpus (seed={seed}).[/yellow]"
        )
    else:
        eval_queries = query_embeddings
        ground_truth = list(range(total_docs))
        console.print(
            f"[yellow]Evaluating all {total_docs:,} queries against full corpus.[/yellow]"
        )

    console.print("[cyan]Computing similarity rankings...[/cyan]")
    ranks = compute_similarity_rankings(
        query_embeddings=eval_queries,
        corpus_embeddings=corpus_embeddings,
        ground_truth_indices=ground_truth,
        batch_size=256,
    )

    console.print("[cyan]Computing evaluation metrics & 1,000 bootstrap CIs...[/cyan]")
    results = evaluate_rankings(
        ranks, ks=[1, 5, 10], bootstrap_resamples=1000, seed=seed
    )

    # Print table
    table = Table(
        title=f"Neural Evaluation Benchmark ({Path(checkpoint_path).name} on {split})",
        header_style="bold magenta",
    )
    table.add_column("Metric", style="dim", width=12)
    table.add_column("Score", justify="right", style="bold green", width=10)
    table.add_column(
        "95% Confidence Interval", justify="center", style="cyan", width=26
    )

    scalar_metrics = {k: v for k, v in results.items() if not k.endswith("_ci")}
    for metric_name, score in scalar_metrics.items():
        ci = results.get(f"{metric_name}_ci", ("—", "—"))
        ci_str = f"[{ci[0]:0.4f}, {ci[1]:0.4f}]" if isinstance(ci, tuple) else "—"
        table.add_row(metric_name.upper(), f"{score:0.4f}", ci_str)

    console.print()
    console.print(table)
    console.print()

    # Log to MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    uri = tracking_uri or (
        "sqlite:///mlflow.db" if Path("mlflow.db").exists() else "mlruns"
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"eval-{Path(checkpoint_path).stem}-{split}"):
        mlflow.log_params(
            {
                "checkpoint": checkpoint_path,
                "split": split,
                "corpus_size": total_docs,
                "num_evaluated_queries": len(eval_queries),
                "sample_size": str(sample_size),
                "data_dir": str(data_dir) if data_dir else "default",
            }
        )
        for k, v in scalar_metrics.items():
            mlflow.log_metric(k.replace("@", "_at_"), float(v))
        for k in scalar_metrics:
            ci_key = f"{k}_ci"
            if ci_key in results:
                ci_low, ci_high = results[ci_key]  # type: ignore
                clean_k = k.replace("@", "_at_")
                mlflow.log_metric(f"{clean_k}_ci_low", float(ci_low))
                mlflow.log_metric(f"{clean_k}_ci_high", float(ci_high))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model checkpoint on CodeSearchNet."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to evaluate on ('test' or 'validation').",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of queries to evaluate (default: 1000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding generation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory containing splits (default: data/processed_clean_v2).",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: sqlite:///mlflow.db).",
    )
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split=args.split,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        seed=args.seed,
        data_dir=args.data_dir,
        tracking_uri=args.tracking_uri,
    )
