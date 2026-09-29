"""Pretrained Baseline Benchmark Suite for CodeEmbed.

Evaluates three leading pretrained models on clean CodeSearchNet Python test set (N = 19,632):
1. sentence-transformers/all-MiniLM-L6-v2 (~22.7M params, 384d, 6L) - Direct architecture/capacity peer
2. microsoft/codebert-base (~125M params, 768d, 12L) - Classical GitHub code-domain baseline
3. jinaai/jina-embeddings-v2-base-code (~161M params, 768d) - Modern 2024 code embedding model

Evaluates both overall metrics (MRR, Recall@1/5/10, NDCG@10, 95% CI) and
the pre-registered lexical overlap stratification (Zero, Low, High).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from evaluation.evaluate import compute_similarity_rankings, evaluate_rankings
from evaluation.metrics import compute_all_metrics

console = Console(legacy_windows=False)


def encode_minilm(texts: list[str], device: str, batch_size: int = 128) -> np.ndarray:
    """Encode texts using all-MiniLM-L6-v2."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    model.max_seq_length = 256
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


def encode_codebert(texts: list[str], device: str, batch_size: int = 64) -> np.ndarray:
    """Encode texts using CodeBERT-base (CLS token + L2 normalization)."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="CodeBERT"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs)
            # Standard CLS token pooling for RoBERTa/CodeBERT
            cls_rep = outputs.last_hidden_state[:, 0, :]
            norm_rep = torch.nn.functional.normalize(cls_rep, p=2, dim=1)
            all_embeddings.append(norm_rep.cpu().numpy())

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(all_embeddings)


def encode_jina(texts: list[str], device: str, batch_size: int = 32) -> np.ndarray:
    """Encode texts using jina-embeddings-v2-base-code."""
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-code",
        trust_remote_code=True,
        device=device,
    )
    model.max_seq_length = 256
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


def run_model_benchmark(
    name: str,
    encode_fn,
    queries: list[str],
    codes: list[str],
    strat_df: pd.DataFrame,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Run full benchmark for a single model."""
    console.print("\n[bold magenta]========================================================================[/bold magenta]")
    console.print(f"[bold cyan]Running Pretrained Benchmark for:[/] [bold yellow]{name}[/bold yellow]")
    console.print("[bold magenta]========================================================================[/bold magenta]")

    t0 = time.time()
    console.print("[dim]Encoding code corpus (N=19,632)...[/dim]")
    code_embs = encode_fn(codes, device=device, batch_size=batch_size)

    console.print("[dim]Encoding queries (N=19,632)...[/dim]")
    query_embs = encode_fn(queries, device=device, batch_size=batch_size)
    encode_time = time.time() - t0

    console.print(f"[green]Encoding complete in {encode_time:.2f}s. Computing Protocol v1.1 harmonic ranks...[/green]")
    ground_truth = list(range(len(queries)))
    ranks = compute_similarity_rankings(
        query_embeddings=query_embs,
        corpus_embeddings=code_embs,
        ground_truth_indices=ground_truth,
        batch_size=256,
    )

    # Free embeddings memory
    del query_embs, code_embs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Full metrics
    console.print("[dim]Computing full bootstrap metrics (N=1,000 resamples)...[/dim]")
    full_metrics = evaluate_rankings(ranks, ks=(1, 5, 10), bootstrap_resamples=1000, seed=42)

    # 2. Stratification
    strata_metrics = {}
    for bin_name in ["Zero", "Low", "High"]:
        mask = (strat_df["overlap_bin"] == bin_name).values
        sub_ranks = ranks[mask]
        m = compute_all_metrics(sub_ranks, ks=(1, 5, 10))
        strata_metrics[bin_name] = {
            "count": int(np.sum(mask)),
            "mrr": float(m["mrr"]),
            "recall@1": float(m["recall@1"]),
            "recall@5": float(m["recall@5"]),
            "recall@10": float(m["recall@10"]),
        }

    return {
        "model_name": name,
        "encode_time_sec": encode_time,
        "metrics": {
            "mrr": float(full_metrics["mrr"]),
            "mrr_ci": [float(full_metrics["mrr_ci"][0]), float(full_metrics["mrr_ci"][1])],
            "recall@1": float(full_metrics["recall@1"]),
            "recall@1_ci": [float(full_metrics["recall@1_ci"][0]), float(full_metrics["recall@1_ci"][1])],
            "recall@5": float(full_metrics["recall@5"]),
            "recall@10": float(full_metrics["recall@10"]),
            "recall@10_ci": [float(full_metrics["recall@10_ci"][0]), float(full_metrics["recall@10_ci"][1])],
            "ndcg@10": float(full_metrics["ndcg@10"]),
        },
        "stratification": strata_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Pretrained Baseline Benchmark Suite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", nargs="+", default=["minilm", "codebert", "jina"], help="Models to evaluate")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if cached")
    args = parser.parse_args()

    console.print(f"[bold green]Starting Pretrained Benchmark Suite on Device:[/] [bold cyan]{args.device}[/bold cyan]")

    data_path = Path("data/processed_clean_v2/test_stratified.parquet")
    if not data_path.exists():
        console.print(f"[bold red]Error: {data_path} not found![/bold red]")
        sys.exit(1)

    strat_df = pd.read_parquet(data_path)
    queries = strat_df["docstring"].tolist()
    codes = strat_df["code"].tolist()
    console.print(f"[bold]Loaded clean test set:[/] {len(queries)} pairs.")

    model_registry = {
        "minilm": ("sentence-transformers/all-MiniLM-L6-v2 (22.7M)", encode_minilm, 128),
        "codebert": ("microsoft/codebert-base (125M)", encode_codebert, 64),
        "jina": ("jinaai/jina-embeddings-v2-base-code (161M)", encode_jina, 32),
    }

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    report_file = out_dir / "pretrained_benchmark_suite.json"

    all_results = {}
    if report_file.exists() and not args.force:
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            console.print(f"[cyan]Loaded existing cached results for: {list(all_results.keys())}[/cyan]")
        except (json.JSONDecodeError, OSError) as e:
            console.print(f"[yellow]Could not read existing cache: {e}. Starting fresh.[/yellow]")
            all_results = {}

    for key in args.models:
        if key not in model_registry:
            console.print(f"[yellow]Skipping unknown model: {key}[/yellow]")
            continue
        if key in all_results and not args.force:
            console.print(f"[green]Model '{key}' already evaluated and cached. Skipping (use --force to re-run).[/green]")
            continue

        display_name, encode_fn, bsize = model_registry[key]
        res = run_model_benchmark(
            name=display_name,
            encode_fn=encode_fn,
            queries=queries,
            codes=codes,
            strat_df=strat_df,
            device=args.device,
            batch_size=bsize,
        )
        all_results[key] = res

        # Incrementally persist after each model finishes
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        console.print(f"[bold green]Updated and persisted benchmark report to:[/] {report_file}")

    # Print Summary Table
    table = Table(title="Pretrained vs. From-Scratch Models on Clean Test Set (N=19,632)", header_style="bold magenta")
    table.add_column("Model / Architecture", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Test MRR [95% CI]", justify="center", style="bold green")
    table.add_column("Recall@1", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("Zero-Overlap", justify="right")
    table.add_column("Low-Overlap", justify="right")
    table.add_column("High-Overlap", justify="right")

    # Add existing baseline references
    table.add_row("BM25 (ATIRE)", "0", "0.5108 [0.5047, 0.5166]", "0.4052", "0.6993", "0.0099", "0.2099", "0.6625")
    table.add_row("CodeEmbed 4L Shared (Confirmatory)", "7.38M", "0.4157 [0.4098, 0.4216]", "0.3178", "0.6018", "0.0487", "0.3204", "0.4716")
    table.add_row("CodeEmbed 17M Scaled (Exploratory)*", "17.03M", "0.4699 [0.4636, 0.4757]", "0.3637", "0.6686", "0.0716", "0.3559", "0.5357")

    param_map = {
        "minilm": "22.7M",
        "codebert": "125M",
        "jina": "161M",
    }

    for key, res in all_results.items():
        m = res["metrics"]
        st = res["stratification"]
        ci_str = f"[{m['mrr_ci'][0]:.4f}, {m['mrr_ci'][1]:.4f}]"
        table.add_row(
            res["model_name"],
            param_map.get(key, "—"),
            f"{m['mrr']:.4f} {ci_str}",
            f"{m['recall@1']:.4f}",
            f"{m['recall@10']:.4f}",
            f"{st['Zero']['mrr']:.4f}",
            f"{st['Low']['mrr']:.4f}",
            f"{st['High']['mrr']:.4f}",
        )

    console.print("\n")
    console.print(table)
    console.print("\n")


if __name__ == "__main__":
    main()
