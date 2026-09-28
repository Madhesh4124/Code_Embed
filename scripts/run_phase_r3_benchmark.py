"""Final Official Benchmark & Statistical Significance Battery for Phase R3 (Protocol v1.1).

Executes:
1. Full Test Set Benchmark (19,632 queries vs 19,632 corpus) across all pre-declared model arms:
   - BM25 (ATIRE piecewise floor)
   - Model 1: Basic Encoder (Pre-LN, 4L-256d-8h-1024ff, zero modality embeddings)
   - Model 2: In-Batch Shared Encoder (LR 5e-4, tau 0.05, 2 epochs)
   - Model 3: Hard-Negative Shared Encoder (1 mined BM25 hard negative + in-batch, winning recipe)
2. Pre-Registered Overlap Stratification:
   - Zero-Overlap (535 queries, 2.73%)
   - Low-Overlap (5,808 queries, 29.58%)
   - High-Overlap (13,289 queries, 67.69%)
3. Paired Metric Bootstrap Significance Testing (N = 2,000 resamples, seed=42):
   - Delta RR and Delta R@1 with 95% CIs and two-sided empirical p-values
   - Equivalence test vs BM25 (margin >= -0.030)
   - Dense hard negative lift (Delta_mining) on unseen test distribution
   - Modality embedding lift (Delta_modality) on unseen test distribution
4. Capacity Error Rubric (Protocol Section 3.4):
   - Seeded sample (seed=42) of 100 validation queries where dense rank > 10
   - Automated & rubric-graded classification: Category A (Ambiguous), Category B (Capacity), Category C (Label noise)
   - Scaling gate evaluation for Phase 6.5
5. Unified MLflow Logging to 'codeembed-clean-r3-final'.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

from data.ablation_dataset import get_ablation_dataloader
from data.preprocess import PROCESSED_DIR
from evaluation.evaluate import compute_similarity_rankings
from evaluation.metrics import (
    bootstrap_metric_ci,
    compute_all_metrics,
    mrr,
    recall_at_k,
)
from model.encoder import BaseEncoder
from model.shared_encoder import SharedEncoder

console = Console(legacy_windows=False)


def paired_bootstrap_test(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    n_bootstraps: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute paired metric differences and bootstrap confidence intervals.

    Delta RR_i = 1 / rank_a_i - 1 / rank_b_i
    Delta R@1_i = I(rank_a_i <= 1) - I(rank_b_i <= 1)
    """
    n = len(ranks_a)
    assert len(ranks_b) == n, f"Length mismatch: {len(ranks_a)} vs {len(ranks_b)}"

    rr_a = np.where((ranks_a >= 1.0) & np.isfinite(ranks_a), 1.0 / ranks_a, 0.0)
    rr_b = np.where((ranks_b >= 1.0) & np.isfinite(ranks_b), 1.0 / ranks_b, 0.0)
    diff_rr = rr_a - rr_b
    mean_diff_rr = float(np.mean(diff_rr))

    r1_a = np.where((ranks_a >= 1.0) & (ranks_a <= 1.0), 1.0, 0.0)
    r1_b = np.where((ranks_b >= 1.0) & (ranks_b <= 1.0), 1.0, 0.0)
    diff_r1 = r1_a - r1_b
    mean_diff_r1 = float(np.mean(diff_r1))

    rng = np.random.default_rng(seed)
    indices = rng.integers(low=0, high=n, size=(n_bootstraps, n))

    boot_rr = np.mean(diff_rr[indices], axis=1)
    boot_r1 = np.mean(diff_r1[indices], axis=1)

    alpha = 1.0 - ci
    ci_low_rr = float(np.percentile(boot_rr, (alpha / 2.0) * 100.0))
    ci_high_rr = float(np.percentile(boot_rr, (1.0 - alpha / 2.0) * 100.0))

    ci_low_r1 = float(np.percentile(boot_r1, (alpha / 2.0) * 100.0))
    ci_high_r1 = float(np.percentile(boot_r1, (1.0 - alpha / 2.0) * 100.0))

    p_val_rr = float(2.0 * min(np.mean(boot_rr <= 0.0), np.mean(boot_rr >= 0.0)))
    p_val_r1 = float(2.0 * min(np.mean(boot_r1 <= 0.0), np.mean(boot_r1 >= 0.0)))

    return {
        "mean_diff_rr": mean_diff_rr,
        "ci_rr": (ci_low_rr, ci_high_rr),
        "p_val_rr": min(1.0, p_val_rr),
        "mean_diff_r1": mean_diff_r1,
        "ci_r1": (ci_low_r1, ci_high_r1),
        "p_val_r1": min(1.0, p_val_r1),
    }


def encode_split(
    model: torch.nn.Module,
    split: str = "test",
    batch_size: int = 128,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode an entire dataset split into corpus and query embeddings."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    loader = get_ablation_dataloader(
        split=split,
        batch_size=batch_size,
        max_seq_len=256,
        shuffle=False,
        num_workers=0,
    )

    all_code_embs: list[np.ndarray] = []
    all_text_embs: list[np.ndarray] = []

    use_amp = device.type == "cuda"
    is_shared = isinstance(model, SharedEncoder) or hasattr(model, "num_modalities")

    with torch.no_grad():
        for batch in loader:
            code_ids = batch["code_ids"].to(device, non_blocking=True)
            code_mask = batch["code_mask"].to(device, non_blocking=True)
            text_ids = batch["text_ids"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if is_shared:
                    c_emb = model.encode_code(code_ids, code_mask)
                    t_emb = model.encode_text(text_ids, text_mask)
                else:
                    c_emb = model(code_ids, attention_mask=code_mask)
                    t_emb = model(text_ids, attention_mask=text_mask)

            all_code_embs.append(c_emb.cpu().numpy())
            all_text_embs.append(t_emb.cpu().numpy())

    corpus_embs = np.concatenate(all_code_embs, axis=0)
    query_embs = np.concatenate(all_text_embs, axis=0)
    return corpus_embs, query_embs


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    """Load model architecture and weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    m_cfg = cfg.get("model", {})
    model_name = str(m_cfg.get("name", "")).lower()

    if "shared" in model_name or "num_modalities" in m_cfg:
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
        arch_type = "SharedEncoder"
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
        arch_type = "BaseEncoder"

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, arch_type


def run_capacity_error_rubric(
    val_ranks: np.ndarray,
    seed: int = 42,
    sample_size: int = 100,
) -> dict[str, Any]:
    """Execute pre-registered Capacity Error Rubric (Protocol Section 3.4).

    Samples 100 validation queries where dense model rank > 10.
    Categorizes failures into:
      Category A: Under-specified / ambiguous queries (multiple valid functions in corpus)
      Category B: Capacity / representation errors (concrete functionality present, missed)
      Category C: Label noise / uninformative docstrings / dead code
    """
    val_strat = pd.read_parquet(PROCESSED_DIR / "validation_stratified.parquet")
    assert len(val_strat) == len(val_ranks)

    failure_indices = np.where(val_ranks > 10.0)[0]
    total_failures = len(failure_indices)

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(failure_indices, size=min(sample_size, total_failures), replace=False)
    sample_df = val_strat.iloc[sampled_indices].copy()
    sample_df["dense_rank"] = val_ranks[sampled_indices]

    cat_a_count = 0
    cat_b_count = 0
    cat_c_count = 0
    records = []

    for row in sample_df.itertuples():
        q = str(row.docstring).strip()
        c = str(row.code).strip()
        words = q.split()
        cov = float(row.coverage)
        rank = float(row.dense_rank)

        # Rule-based rubric grounded in Protocol Section 3.4 definitions:
        # Category C (Label Noise): Extremely short (< 3 words), boilerplate comments, or empty stubs
        if len(words) < 4 or "todo" in q.lower() or len(c.splitlines()) <= 2:
            category = "C"
            cat_c_count += 1
            reason = "Label noise / trivial stub / uninformative query"
        # Category A (Ambiguous / Under-specified): Generic action verbs without specific module domain context
        elif len(words) <= 6 and (
            words[0].lower() in ["get", "set", "return", "update", "check", "create", "initialize", "reset", "clear"]
            and cov <= 0.10
        ):
            category = "A"
            cat_a_count += 1
            reason = "Generic / under-specified query (multiple corpus functions satisfy it)"
        # Category B (Representation / Capacity Error): Query is concrete, docstring is descriptive, but model missed
        else:
            category = "B"
            cat_b_count += 1
            reason = "Concrete domain query missed (representation capacity error)"

        records.append({
            "docstring": q,
            "func_name": getattr(row, "func_name", "function"),
            "dense_rank": rank,
            "category": category,
            "reason": reason,
        })

    pct_a = cat_a_count / len(records) * 100.0
    pct_b = cat_b_count / len(records) * 100.0
    pct_c = cat_c_count / len(records) * 100.0

    scaling_gate_pass = pct_b >= 50.0

    return {
        "total_failures": total_failures,
        "sample_size": len(records),
        "cat_a_count": cat_a_count,
        "cat_b_count": cat_b_count,
        "cat_c_count": cat_c_count,
        "pct_a": pct_a,
        "pct_b": pct_b,
        "pct_c": pct_c,
        "scaling_gate_pass": scaling_gate_pass,
        "records": records,
    }


def main() -> None:
    """Run Phase R3 Final Benchmark, Statistical Significance Testing, and Gates."""
    parser = argparse.ArgumentParser(description="Phase R3 Final Test Benchmark & Significance Battery")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding generation")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000, help="Bootstrap resample count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.rule("[bold magenta]Phase R3: Final Clean Test Set Benchmark & Statistical Evaluation[/bold magenta]")
    console.print(f"Device: {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    console.print("Target Dataset: [bold yellow]data/processed_clean_v2/test.parquet (19,632 pairs)[/bold yellow]")
    console.print(f"Bootstrap Resamples: {args.bootstrap_resamples:,} (Seed: {args.seed})\n")

    # 1. Load Pre-stratified Test Data & BM25 Baseline Ranks
    test_strat = pd.read_parquet(PROCESSED_DIR / "test_stratified.parquet")
    n_test = len(test_strat)
    console.print(f"[green][OK][/green] Loaded test split: {n_test:,} queries against full corpus.")
    bm25_ranks = test_strat["rank"].to_numpy()

    # 2. Checkpoint Definitions
    checkpoints = {
        "BM25 (ATIRE)": None,
        "Basic Encoder": "checkpoints/basic_clean/best_basic.pt",
        "In-Batch Shared": "checkpoints/fallback_grid/best_lr_5e-4_tau_0.05.pt",
        "Hard-Negative Shared": "checkpoints/shared_hard_clean/best_shared.pt",
    }

    all_ranks: dict[str, np.ndarray] = {
        "BM25 (ATIRE)": bm25_ranks,
    }

    # 3. Encode & Evaluate Dense Models on Test Split
    ground_truth = list(range(n_test))
    for name, ckpt_path in checkpoints.items():
        if ckpt_path is None:
            continue
        console.print(f"\n[cyan]>>> Evaluating {name} ({ckpt_path})...[/cyan]")
        start_t = time.time()
        model, _ = load_model(ckpt_path, device=device)
        corpus_embs, query_embs = encode_split(model, split="test", batch_size=args.batch_size, device=device)
        enc_time = time.time() - start_t
        console.print(f"[green][OK][/green] Encoded {n_test:,} vectors in {enc_time:.2f}s ({n_test / enc_time:.1f} items/s).")

        console.print(f"Computing exact harmonic rankings for {name}...")
        ranks = compute_similarity_rankings(
            query_embeddings=query_embs,
            corpus_embeddings=corpus_embs,
            ground_truth_indices=ground_truth,
            batch_size=256,
        )
        all_ranks[name] = ranks

    # 4. Compute Benchmark Metrics for All Arms
    console.rule("[bold cyan]1. Test Set Full Retrieval Suite (19,632 Queries)[/bold cyan]")
    bench_table = Table(title="Phase R3 Final Test Benchmark Suite", header_style="bold yellow")
    bench_table.add_column("Model Arm", style="bold")
    bench_table.add_column("MRR", justify="right", style="bold green")
    bench_table.add_column("95% CI (2,000 Bootstraps)", justify="center", style="cyan")
    bench_table.add_column("Recall@1", justify="right")
    bench_table.add_column("Recall@5", justify="right")
    bench_table.add_column("Recall@10", justify="right")
    bench_table.add_column("NDCG@10", justify="right")

    arm_metrics: dict[str, dict[str, Any]] = {}

    for name, r in all_ranks.items():
        m = compute_all_metrics(r, ks=(1, 5, 10))
        ci = bootstrap_metric_ci(r, metric_fn=mrr, n_bootstraps=args.bootstrap_resamples, seed=args.seed)
        arm_metrics[name] = {**m, "mrr_ci": ci}
        bench_table.add_row(
            name,
            f"{m['mrr']:0.4f}",
            f"[{ci[0]:0.4f}, {ci[1]:0.4f}]",
            f"{m['recall@1']:0.4f}",
            f"{m['recall@5']:0.4f}",
            f"{m['recall@10']:0.4f}",
            f"{m['ndcg@10']:0.4f}",
        )
    console.print(bench_table)

    # 5. Overlap Stratification
    console.rule("[bold cyan]2. Pre-Registered Overlap Stratification[/bold cyan]")
    strat_table = Table(title="Test Retrieval by Overlap Stratum", header_style="bold yellow")
    strat_table.add_column("Stratum", style="bold")
    strat_table.add_column("Count", justify="right")
    strat_table.add_column("% Split", justify="right")
    strat_table.add_column("BM25 MRR", justify="right", style="dim")
    strat_table.add_column("Basic MRR", justify="right")
    strat_table.add_column("In-Batch MRR", justify="right")
    strat_table.add_column("Hard-Neg MRR", justify="right", style="bold green")
    strat_table.add_column("Hard-Neg R@1", justify="right")
    strat_table.add_column("Hard-Neg R@10", justify="right")

    bins = ["Zero", "Low", "High"]
    for b in bins:
        sub_mask = (test_strat["overlap_bin"] == b).to_numpy()
        cnt = int(np.sum(sub_mask))
        pct = cnt / n_test * 100.0

        b_bm25 = mrr(all_ranks["BM25 (ATIRE)"][sub_mask])
        b_basic = mrr(all_ranks["Basic Encoder"][sub_mask])
        b_inbatch = mrr(all_ranks["In-Batch Shared"][sub_mask])
        b_hard = mrr(all_ranks["Hard-Negative Shared"][sub_mask])
        r1_hard = recall_at_k(all_ranks["Hard-Negative Shared"][sub_mask], k=1)
        r10_hard = recall_at_k(all_ranks["Hard-Negative Shared"][sub_mask], k=10)

        strat_table.add_row(
            b,
            f"{cnt:,}",
            f"{pct:.2f}%",
            f"{b_bm25:0.4f}",
            f"{b_basic:0.4f}",
            f"{b_inbatch:0.4f}",
            f"{b_hard:0.4f}",
            f"{r1_hard:0.4f}",
            f"{r10_hard:0.4f}",
        )

    # Overall summary row
    strat_table.add_row(
        "OVERALL",
        f"{n_test:,}",
        "100.00%",
        f"{arm_metrics['BM25 (ATIRE)']['mrr']:0.4f}",
        f"{arm_metrics['Basic Encoder']['mrr']:0.4f}",
        f"{arm_metrics['In-Batch Shared']['mrr']:0.4f}",
        f"{arm_metrics['Hard-Negative Shared']['mrr']:0.4f}",
        f"{arm_metrics['Hard-Negative Shared']['recall@1']:0.4f}",
        f"{arm_metrics['Hard-Negative Shared']['recall@10']:0.4f}",
        style="bold",
    )
    console.print(strat_table)

    # 6. Paired Bootstrap Hypothesis Testing
    console.rule("[bold cyan]3. Paired Metric Bootstrap Hypothesis Tests (N = 2,000 Resamples)[/bold cyan]")
    test_pairs = [
        ("Hard-Negative Shared", "In-Batch Shared", "RQ3: Dense Mining Lift (Δ_mining)"),
        ("In-Batch Shared", "Basic Encoder", "RQ2: Modality Embedding Lift (Δ_modality)"),
        ("Hard-Negative Shared", "BM25 (ATIRE)", "Protocol §3.4: Equivalence Margin vs BM25"),
    ]

    p_table = Table(title="Paired Bootstrap Significance Testing", header_style="bold yellow")
    p_table.add_column("Comparison (Model A vs Model B)", style="bold")
    p_table.add_column("Research Scope", style="dim")
    p_table.add_column("Mean ΔRR", justify="right", style="bold")
    p_table.add_column("95% CI on ΔRR", justify="center", style="cyan")
    p_table.add_column("p-value (ΔRR)", justify="right")
    p_table.add_column("Mean ΔR@1", justify="right")
    p_table.add_column("95% CI on ΔR@1", justify="center", style="cyan")
    p_table.add_column("p-value (ΔR@1)", justify="right")
    p_table.add_column("Result / Gate", justify="center")

    paired_results: dict[str, Any] = {}
    for arm_a, arm_b, scope in test_pairs:
        res = paired_bootstrap_test(
            ranks_a=all_ranks[arm_a],
            ranks_b=all_ranks[arm_b],
            n_bootstraps=args.bootstrap_resamples,
            seed=args.seed,
        )
        paired_results[f"{arm_a} vs {arm_b}"] = res

        # Significance or gate evaluation
        if "Equivalence Margin" in scope:
            ci_low = res["ci_rr"][0]
            # Equivalence margin: lower bound >= -0.030
            status = "[bold green]PASS (Eq Marg)[/bold green]" if ci_low >= -0.030 else "[bold red]FAIL (Margin)[/bold red]"
        else:
            is_sig = res["p_val_rr"] < 0.05 and (res["ci_rr"][0] > 0 or res["ci_rr"][1] < 0)
            status = "[bold green]SIGNIFICANT (p<0.05)[/bold green]" if is_sig else "[yellow]NOT SIGNIFICANT[/yellow]"

        p_table.add_row(
            f"{arm_a} vs {arm_b}",
            scope,
            f"{res['mean_diff_rr']:+0.4f}",
            f"[{res['ci_rr'][0]:+0.4f}, {res['ci_rr'][1]:+0.4f}]",
            f"{res['p_val_rr']:0.4f}",
            f"{res['mean_diff_r1']:+0.4f}",
            f"[{res['ci_r1'][0]:+0.4f}, {res['ci_r1'][1]:+0.4f}]",
            f"{res['p_val_r1']:0.4f}",
            status,
        )
    console.print(p_table)

    # 7. Capacity Error Rubric (Gate on Phase 6.5 Scaling)
    console.rule("[bold cyan]4. Capacity Error Rubric & Scaling Gate (Protocol §3.4)[/bold cyan]")
    # Run rubric using validation ranks from the hard negative model
    console.print("Computing validation ranks for Capacity Error Rubric...")
    val_model, _ = load_model("checkpoints/shared_hard_clean/best_shared.pt", device=device)
    val_corpus_embs, val_query_embs = encode_split(val_model, split="validation", batch_size=args.batch_size, device=device)
    val_ranks = compute_similarity_rankings(
        query_embeddings=val_query_embs,
        corpus_embeddings=val_corpus_embs,
        ground_truth_indices=list(range(len(val_query_embs))),
        batch_size=256,
    )

    rubric_res = run_capacity_error_rubric(val_ranks=val_ranks, seed=args.seed, sample_size=100)

    rubric_table = Table(title="Capacity Error Rubric Classification (100 Sampled Validation Failures, rank > 10)", header_style="bold yellow")
    rubric_table.add_column("Category", style="bold")
    rubric_table.add_column("Definition / Root Cause", style="dim")
    rubric_table.add_column("Sample Count", justify="right")
    rubric_table.add_column("% Share", justify="right", style="bold")
    rubric_table.add_column("Scaling Implication")

    rubric_table.add_row(
        "Category A",
        "Under-specified / Ambiguous queries (multiple valid functions satisfy it)",
        f"{rubric_res['cat_a_count']}",
        f"{rubric_res['pct_a']:.1f}%",
        "Data / Label ambiguity; scaling parameters will not resolve",
    )
    rubric_table.add_row(
        "Category B",
        "Capacity / Representation errors (concrete functionality present, missed)",
        f"{rubric_res['cat_b_count']}",
        f"{rubric_res['pct_b']:.1f}%",
        "[bold green]True representation capacity limit; scaling layers/heads helps[/bold green]",
    )
    rubric_table.add_row(
        "Category C",
        "Label noise / dead code / trivial stubs",
        f"{rubric_res['cat_c_count']}",
        f"{rubric_res['pct_c']:.1f}%",
        "Corrupted pair / uninformative docstring",
    )
    console.print(rubric_table)

    console.print("\n" + "=" * 80)
    console.print("[bold yellow]SCALING GATE DECISION (PROTOCOL §3.4 & §3.5)[/bold yellow]")
    console.print("=" * 80)
    console.print("  Requirement: Category B must constitute >= 50.0% of failures to justify Phase 6.5 Capacity Scaling.")
    console.print(f"  Observed:    Category B = {rubric_res['pct_b']:.1f}% | Category A + C = {rubric_res['pct_a'] + rubric_res['pct_c']:.1f}%")
    console.print("-" * 80)
    if rubric_res["scaling_gate_pass"]:
        console.print("[bold green]>> SCALING GATE: PASSED (Category B >= 50%)[/bold green]")
        console.print("[green]Action: Phase 6.5 Parameter & Layer Scaling is scientifically justified.[/green]")
    else:
        console.print("[bold yellow]>> SCALING GATE: BLOCKED BY LABEL-CEILING (Category A + C >= 50%)[/bold yellow]")
        console.print("[yellow]Action: Failures are dominated by query ambiguity and label noise. Parameter scaling is blocked.[/yellow]")
    console.print("=" * 80 + "\n")

    # 8. Log Everything to MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("codeembed-clean-r3-final")

    with mlflow.start_run(run_name="phase_r3_final_test_benchmark") as run:
        mlflow.set_tags({
            "phase": "R3",
            "phase_name": "Final Clean Test Benchmark & Significance",
            "eval_split": "test",
            "total_test_queries": str(n_test),
            "scaling_gate": "PASS" if rubric_res["scaling_gate_pass"] else "BLOCKED",
        })

        for model_name, metrics in arm_metrics.items():
            clean_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            for k, v in metrics.items():
                if k == "mrr_ci":
                    mlflow.log_metric(f"{clean_name}_mrr_ci_low", float(v[0]))
                    mlflow.log_metric(f"{clean_name}_mrr_ci_high", float(v[1]))
                else:
                    mlflow.log_metric(f"{clean_name}_{k.replace('@', '_at_')}", float(v))

        # Log paired test metrics
        for pair_name, p_res in paired_results.items():
            p_clean = pair_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            mlflow.log_metric(f"diff_{p_clean}_rr", float(p_res["mean_diff_rr"]))
            mlflow.log_metric(f"diff_{p_clean}_rr_p_value", float(p_res["p_val_rr"]))
            mlflow.log_metric(f"diff_{p_clean}_r1", float(p_res["mean_diff_r1"]))

        # Log rubric stats
        mlflow.log_metric("rubric_category_a_pct", float(rubric_res["pct_a"]))
        mlflow.log_metric("rubric_category_b_pct", float(rubric_res["pct_b"]))
        mlflow.log_metric("rubric_category_c_pct", float(rubric_res["pct_c"]))

        console.print(f"[bold green][OK] All Phase R3 benchmarks, statistical tests, and rubric results saved to MLflow run {run.info.run_id}![/bold green]")


if __name__ == "__main__":
    main()
