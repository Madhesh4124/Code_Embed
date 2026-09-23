"""Official Pilot Gate Evaluation Script for CodeEmbed (Protocol v1.1 §3.3).

Evaluates a model checkpoint on the FULL validation split (20,115 queries against
the full 20,115 code corpus) and reports overall retrieval metrics and stratified
performance across the pre-registered overlap bins (Zero, Low, High).

Pilot Gate Decision:
- PASS: Validation MRR >= 0.3910 (0.75 x BM25 Val MRR) OR Low-Overlap Val MRR > 0.2489.
- FAIL: Triggers capped fallback grid (3 LRs x 2 temps = 6 validation runs).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import create_dataloader
from data.preprocess import PROCESSED_DIR
from evaluation.evaluate import compute_similarity_rankings
from evaluation.metrics import bootstrap_metric_ci, compute_all_metrics, mrr
from model.encoder import BaseEncoder
from model.shared_encoder import SharedEncoder
from tokenizer.tokenizer import CodeEmbedTokenizer


def evaluate_pilot_gate(
    checkpoint_path: str | Path = "checkpoints/shared_clean/best_shared.pt",
    batch_size: int = 128,
    bootstrap_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print("[bold cyan]>>> PILOT GATE EVALUATION ON FULL VALIDATION SET (20,115 QUERIES)[/bold cyan]")
    console.print(f"[cyan]Device:[/cyan] {device.type.upper()}")
    console.print(f"[cyan]Checkpoint:[/cyan] {checkpoint_path}")

    # 1. Load Model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    m_cfg = cfg.get("model", {})
    model_name = str(m_cfg.get("name", "shared")).lower()

    if model_name == "shared" or "num_modalities" in m_cfg:
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
        is_shared = True
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
        is_shared = False

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    console.print(f"[green][OK][/green] Loaded model ({'SharedEncoder' if is_shared else 'BaseEncoder'}) from {checkpoint_path}")

    # 2. Encode Full Validation Corpus and Queries
    tokenizer = CodeEmbedTokenizer()
    loader = create_dataloader(
        split="validation",
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=int(m_cfg.get("max_seq_len", 256)),
        data_dir=PROCESSED_DIR,
    )

    console.print("[cyan]Encoding all 20,115 validation code snippets and queries...[/cyan]")
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
                if is_shared:
                    cv = model(code_ids, attention_mask=code_mask, modality_ids="code")
                    tv = model(text_ids, attention_mask=text_mask, modality_ids="text")
                else:
                    cv = model(code_ids, attention_mask=code_mask)
                    tv = model(text_ids, attention_mask=text_mask)

            all_code_embs.append(cv.cpu().numpy())
            all_text_embs.append(tv.cpu().numpy())

    corpus_embeddings = np.concatenate(all_code_embs, axis=0)
    query_embeddings = np.concatenate(all_text_embs, axis=0)
    total_docs = len(corpus_embeddings)
    console.print(f"[green][OK][/green] Encoded {total_docs:,} code and query vectors.")

    # 3. Compute Full Ranking Matrix (20,115 x 20,115)
    console.print(f"[cyan]Computing exact harmonic expected reciprocal ranks across all {total_docs:,} queries...[/cyan]")
    ground_truth = list(range(total_docs))
    ranks = compute_similarity_rankings(
        query_embeddings=query_embeddings,
        corpus_embeddings=corpus_embeddings,
        ground_truth_indices=ground_truth,
        batch_size=256,
    )

    # 4. Overall Metrics & Bootstrap CIs
    console.print(f"[cyan]Computing full metric suite & {bootstrap_resamples:,} bootstrap CIs...[/cyan]")
    base_metrics = compute_all_metrics(ranks, ks=(1, 5, 10))
    mrr_ci = bootstrap_metric_ci(ranks, metric_fn=mrr, n_bootstraps=bootstrap_resamples, ci=0.95, seed=seed)

    # 5. Overlap Stratification
    val_strat_path = PROCESSED_DIR / "validation_stratified.parquet"
    if not val_strat_path.exists():
        raise FileNotFoundError(f"Stratified parquet not found at {val_strat_path}. Run scripts/compute_stratification.py first.")

    val_df = pd.read_parquet(val_strat_path)
    val_df["neural_rank"] = ranks

    strat_results = {}
    for b in ["Zero", "Low", "High"]:
        b_df = val_df[val_df["overlap_bin"] == b]
        b_ranks = b_df["neural_rank"].to_numpy()
        b_bm25_ranks = b_df["rank"].to_numpy()
        strat_results[b] = {
            "count": len(b_df),
            "pct": len(b_df) / len(val_df) * 100.0,
            "neural_mrr": float(mrr(b_ranks)),
            "neural_r1": float(np.mean(b_ranks <= 1.0)),
            "neural_r10": float(np.mean(b_ranks <= 10.0)),
            "bm25_mrr": float(mrr(b_bm25_ranks)),
        }

    # 6. Pilot Gate Check
    overall_mrr = float(base_metrics["mrr"])
    bm25_val_overall_mrr = 0.5214
    bm25_val_low_mrr = 0.2489
    primary_target = 0.75 * bm25_val_overall_mrr  # 0.3910
    low_overlap_target = bm25_val_low_mrr  # 0.2489

    low_overlap_mrr = strat_results["Low"]["neural_mrr"]

    pass_primary = overall_mrr >= primary_target
    pass_low_overlap = low_overlap_mrr > low_overlap_target
    gate_passed = pass_primary or pass_low_overlap

    # Print Summary Tables
    overall_table = Table(title=f"Full Validation Retrieval Benchmark ({total_docs:,} queries)")
    overall_table.add_column("Metric", justify="left", style="bold")
    overall_table.add_column("Score", justify="right")
    overall_table.add_column("95% CI", justify="center")

    overall_table.add_row("MRR", f"{overall_mrr:.4f}", f"[{mrr_ci[0]:.4f}, {mrr_ci[1]:.4f}]")
    overall_table.add_row("Recall@1", f"{base_metrics['recall@1']:.4f}", "—")
    overall_table.add_row("Recall@5", f"{base_metrics['recall@5']:.4f}", "—")
    overall_table.add_row("Recall@10", f"{base_metrics['recall@10']:.4f}", "—")
    overall_table.add_row("NDCG@10", f"{base_metrics['ndcg@10']:.4f}", "—")
    console.print(overall_table)

    strat_table = Table(title=f"Pre-Registered Overlap Stratification ({total_docs:,} queries)")
    strat_table.add_column("Stratum", justify="left", style="bold")
    strat_table.add_column("Count", justify="right")
    strat_table.add_column("% Split", justify="right")
    strat_table.add_column("BM25 MRR", justify="right", style="dim")
    strat_table.add_column("Neural MRR", justify="right", style="cyan")
    strat_table.add_column("Neural R@1", justify="right")
    strat_table.add_column("Neural R@10", justify="right")

    for b, data in strat_results.items():
        strat_table.add_row(
            b,
            f"{data['count']:,}",
            f"{data['pct']:.2f}%",
            f"{data['bm25_mrr']:.4f}",
            f"{data['neural_mrr']:.4f}",
            f"{data['neural_r1']:.4f}",
            f"{data['neural_r10']:.4f}",
        )
    strat_table.add_row(
        "OVERALL",
        f"{total_docs:,}",
        "100.00%",
        f"{bm25_val_overall_mrr:.4f}",
        f"{overall_mrr:.4f}",
        f"{base_metrics['recall@1']:.4f}",
        f"{base_metrics['recall@10']:.4f}",
        style="bold",
    )
    console.print(strat_table)

    # Gate Evaluation Banner
    console.print("\n" + "=" * 75)
    console.print("[bold yellow]OFFICIAL PILOT GATE EVALUATION (PROTOCOL v1.1 §3.3)[/bold yellow]")
    console.print("=" * 75)
    console.print(f"  Criterion 1 (Primary):      Validation MRR >= {primary_target:.4f}  --> Actual: {overall_mrr:.4f}  [{'PASS' if pass_primary else 'FAIL'}]")
    console.print(f"  Criterion 2 (Low-Overlap):  Low-Overlap MRR > {low_overlap_target:.4f}  --> Actual: {low_overlap_mrr:.4f}  [{'PASS' if pass_low_overlap else 'FAIL'}]")
    console.print("-" * 75)
    if gate_passed:
        console.print("[bold green]>> PILOT GATE: PASSED[/bold green]")
        console.print("[green]Action: Proceed to 3-seed replication (seeds 42, 123, 456) of this configuration.[/green]")
    else:
        console.print("[bold red]>> PILOT GATE: FAILED[/bold red]")
        console.print("[yellow]Action: Trigger pre-registered capped fallback tuning grid:[/yellow]")
        console.print("        3 Learning Rates {1e-4, 3e-4, 5e-4} x 2 Temperatures {0.05, 0.07} = 6 validation runs.")
        console.print("        Adopt the single best-performing validation configuration across all arms.")
    console.print("=" * 75 + "\n")

    return {
        "overall_mrr": overall_mrr,
        "mrr_ci": mrr_ci,
        "base_metrics": base_metrics,
        "strat_results": strat_results,
        "gate_passed": gate_passed,
        "pass_primary": pass_primary,
        "pass_low_overlap": pass_low_overlap,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pilot Gate on Full Validation Set.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/shared_clean/best_shared.pt", help="Path to checkpoint.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for encoding.")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000, help="Number of bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap random seed.")
    args = parser.parse_args()

    evaluate_pilot_gate(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
