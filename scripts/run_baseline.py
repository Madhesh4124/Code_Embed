"""Phase 1: Run BM25 Lexical Baseline Benchmark.

This script:
1. Loads the test set from data/processed/test.parquet.
2. Indexes the complete code corpus using BM25Okapi with sub-token splitting.
3. Retrieves candidates for natural language docstrings.
4. Computes MRR, Recall@1, Recall@5, Recall@10, NDCG@10, and 95% bootstrap CIs.
5. Logs all hyperparameters, metrics, and summaries to MLflow.
6. Prints a formatted summary table.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure repository root is in sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import mlflow
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evaluation.evaluate import evaluate_rankings, format_metrics_summary
from retrieval.bm25 import BM25Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 baseline evaluation on CodeSearchNet test set.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def run_baseline(config_path: str) -> None:
    console = Console()
    cfg = OmegaConf.load(config_path)

    console.print(Panel.fit("[bold green]CodeEmbed — Phase 1: BM25 Lexical Baseline Benchmark[/bold green]"))
    console.print(f"[cyan]Configuration:[/cyan] {config_path}")

    # 1. Load test data
    test_path = cfg.data.test_path
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data file not found at: {test_path}")

    console.print(f"[yellow]Loading test dataset from {test_path}...[/yellow]")
    df = pd.read_parquet(test_path)
    corpus = df["code"].tolist()
    all_queries = df["docstring"].tolist()
    corpus_size = len(corpus)
    console.print(f"[green][OK][/green] Loaded {corpus_size:,} code snippets and docstring pairs.")

    # 2. Sample queries if specified
    sample_size: int | None = cfg.data.get("sample_size", None)
    if sample_size is not None and sample_size < corpus_size:
        rng = np.random.RandomState(cfg.data.seed)
        sampled_indices = rng.choice(corpus_size, size=sample_size, replace=False)
        queries = [all_queries[idx] for idx in sampled_indices]
        ground_truth = sampled_indices.tolist()
        console.print(f"[yellow]Evaluating on representative sample of {sample_size:,} queries against full {corpus_size:,} code corpus (seed={cfg.data.seed}).[/yellow]")
    else:
        queries = all_queries
        ground_truth = list(range(corpus_size))
        console.print(f"[yellow]Evaluating all {corpus_size:,} queries against full corpus.[/yellow]")

    # 3. Build BM25 Index
    console.print(f"[cyan]Building BM25 index with k1={cfg.bm25.k1}, b={cfg.bm25.b}...[/cyan]")
    start_index_time = time.time()
    retriever = BM25Retriever(k1=float(cfg.bm25.k1), b=float(cfg.bm25.b))
    retriever.index(corpus)
    index_duration = time.time() - start_index_time
    console.print(f"[green][OK][/green] Index built in {index_duration:0.2f}s.")

    # Optional save index
    if cfg.bm25.get("save_index", False):
        index_path = cfg.bm25.index_path
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        retriever.save(index_path)
        console.print(f"[green][OK][/green] Saved index to {index_path}")

    # 4. Compute Ranks
    console.print(f"[cyan]Retrieving rankings for {len(queries):,} queries...[/cyan]")
    start_query_time = time.time()
    ranks = retriever.get_ranks(
        queries=queries,
        ground_truth_indices=ground_truth,
        max_candidates=1000,
    )
    query_duration = time.time() - start_query_time
    qps = len(queries) / max(query_duration, 1e-6)
    console.print(f"[green][OK][/green] Completed retrieval in {query_duration:0.2f}s ({qps:0.1f} queries/sec).")

    # 5. Compute Metrics & Confidence Intervals
    console.print("[cyan]Computing IR metrics and 1,000 bootstrap confidence intervals...[/cyan]")
    results = evaluate_rankings(
        ranks=ranks,
        ks=[1, 5, 10],
        bootstrap_resamples=cfg.evaluation.bootstrap_resamples,
        confidence_level=cfg.evaluation.confidence_level,
        seed=cfg.data.seed,
    )

    # 6. Display Results Table
    table = Table(title="BM25 Lexical Baseline Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim", width=12)
    table.add_column("Score", justify="right", style="bold green", width=10)
    table.add_column("95% Confidence Interval", justify="center", style="cyan", width=26)

    scalar_metrics = {k: v for k, v in results.items() if not k.endswith("_ci")}
    for metric_name, score in scalar_metrics.items():
        ci = results.get(f"{metric_name}_ci", ("—", "—"))
        ci_str = f"[{ci[0]:0.4f}, {ci[1]:0.4f}]" if isinstance(ci, tuple) else "—"
        table.add_row(metric_name.upper(), f"{score:0.4f}", ci_str)

    console.print()
    console.print(table)
    console.print()

    # 7. Log to MLflow
    tracking_uri = cfg.mlflow.get("tracking_uri", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        console.print(f"[cyan]Logging run to MLflow (Run ID: {run.info.run_id})...[/cyan]")

        # Log params
        mlflow.log_params({
            "model_type": "BM25Okapi",
            "k1": cfg.bm25.k1,
            "b": cfg.bm25.b,
            "corpus_size": corpus_size,
            "num_evaluated_queries": len(queries),
            "sample_size": str(sample_size),
            "bootstrap_resamples": cfg.evaluation.bootstrap_resamples,
            "index_duration_sec": round(index_duration, 2),
            "query_duration_sec": round(query_duration, 2),
            "queries_per_sec": round(qps, 1),
        })

        # Log scalar metrics (replace '@' with '_at_' for MLflow compliance)
        for metric_name, score in scalar_metrics.items():
            mlflow_key = metric_name.replace("@", "_at_")
            mlflow.log_metric(mlflow_key, float(score))

        # Log CI bounds as metrics
        for metric_name in scalar_metrics:
            ci_key = f"{metric_name}_ci"
            if ci_key in results:
                ci_low, ci_high = results[ci_key]  # type: ignore
                mlflow_key = metric_name.replace("@", "_at_")
                mlflow.log_metric(f"{mlflow_key}_ci_low", float(ci_low))
                mlflow.log_metric(f"{mlflow_key}_ci_high", float(ci_high))

        # Save summary markdown artifact
        summary_md = format_metrics_summary(results)
        summary_path = Path("reports") / "baseline_results.md"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# BM25 Lexical Baseline Results\n\n")
            f.write(f"- **Corpus Size**: {corpus_size:,}\n")
            f.write(f"- **Evaluated Queries**: {len(queries):,}\n")
            f.write(f"- **Index Time**: {index_duration:0.2f}s\n")
            f.write(f"- **Retrieval Time**: {query_duration:0.2f}s ({qps:0.1f} QPS)\n\n")
            f.write("## Retrieval Metrics\n\n")
            f.write(summary_md + "\n")

        mlflow.log_artifact(str(summary_path))
        console.print(f"[green][OK][/green] Successfully logged metrics and artifacts to MLflow experiment '{cfg.mlflow.experiment_name}'.")


if __name__ == "__main__":
    args = parse_args()
    run_baseline(args.config)

