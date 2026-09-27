"""Pre-Registered Capped Fallback Tuning Grid Runner (Protocol v1.1 §3.3).

Executes the 6-point hyperparameter grid on the clean validation set:
  - 3 Learning Rates: {1e-4, 3e-4, 5e-4}
  - 2 Temperatures: {0.05, 0.07}
  - Model: SharedEncoder (~7.38M parameters, learned modality embeddings)
  - Training: 2 epochs on 360,957 clean train samples (in-batch negatives)
  - Evaluation: Full validation split (20,115 queries against 20,115 corpus)

The single highest-MRR configuration on validation is adopted across all arms.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
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
import torch
from rich.console import Console
from rich.table import Table

from data.dataset import create_dataloader
from data.preprocess import PROCESSED_DIR
from evaluation.evaluate import compute_similarity_rankings
from evaluation.metrics import mrr, recall_at_k
from model.shared_encoder import SharedEncoder
from tokenizer.tokenizer import CodeEmbedTokenizer
from training.trainer import ContrastiveTrainer, get_device, set_seed

GRID_CONFIGS = [
    {"lr": 1.0e-4, "temperature": 0.05, "tag": "lr_1e-4_tau_0.05"},
    {"lr": 1.0e-4, "temperature": 0.07, "tag": "lr_1e-4_tau_0.07"},
    {"lr": 3.0e-4, "temperature": 0.05, "tag": "lr_3e-4_tau_0.05"},
    {"lr": 3.0e-4, "temperature": 0.07, "tag": "lr_3e-4_tau_0.07"},  # Pilot run
    {"lr": 5.0e-4, "temperature": 0.05, "tag": "lr_5e-4_tau_0.05"},
    {"lr": 5.0e-4, "temperature": 0.07, "tag": "lr_5e-4_tau_0.07"},
]

RESULTS_DIR = Path("checkpoints/fallback_grid")
RESULTS_FILE = RESULTS_DIR / "results.json"


def evaluate_checkpoint_on_full_val(
    model: SharedEncoder,
    device: torch.device,
    tokenizer: CodeEmbedTokenizer,
    batch_size: int = 128,
) -> dict:
    """Encode full 20,115 validation corpus and compute exact retrieval metrics."""
    val_loader = create_dataloader(
        split="validation",
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=256,
        data_dir=PROCESSED_DIR,
    )

    all_code_embs: list[np.ndarray] = []
    all_text_embs: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            code_ids = batch["code_ids"].to(device)
            code_mask = batch["code_mask"].to(device)
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                cv = model(code_ids, attention_mask=code_mask, modality_ids="code")
                tv = model(text_ids, attention_mask=text_mask, modality_ids="text")

            all_code_embs.append(cv.cpu().numpy())
            all_text_embs.append(tv.cpu().numpy())

    corpus_embeddings = np.concatenate(all_code_embs, axis=0)
    query_embeddings = np.concatenate(all_text_embs, axis=0)
    total_docs = len(corpus_embeddings)

    ranks = compute_similarity_rankings(
        query_embeddings=query_embeddings,
        corpus_embeddings=corpus_embeddings,
        ground_truth_indices=list(range(total_docs)),
        batch_size=256,
    )

    overall_mrr = float(mrr(ranks))
    r1 = float(recall_at_k(ranks, k=1))
    r5 = float(recall_at_k(ranks, k=5))
    r10 = float(recall_at_k(ranks, k=10))

    # Overlap stratification
    val_strat_path = PROCESSED_DIR / "validation_stratified.parquet"
    low_overlap_mrr = 0.0
    if val_strat_path.exists():
        val_df = pd.read_parquet(val_strat_path)
        val_df["neural_rank"] = ranks
        low_df = val_df[val_df["overlap_bin"] == "Low"]
        low_overlap_mrr = float(mrr(low_df["neural_rank"].to_numpy()))

    return {
        "val_mrr": overall_mrr,
        "val_r1": r1,
        "val_r5": r5,
        "val_r10": r10,
        "val_low_overlap_mrr": low_overlap_mrr,
    }


def train_and_eval_cell(
    lr: float,
    temperature: float,
    tag: str,
    device: torch.device,
    tokenizer: CodeEmbedTokenizer,
    console: Console,
) -> dict:
    """Train SharedEncoder for 2 epochs on clean train, then evaluate on full validation split."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = RESULTS_DIR / f"best_{tag}.pt"

    # Seed 42 for all fallback grid runs
    set_seed(42)

    console.print("\n[bold cyan]================================================================[/bold cyan]")
    console.print(f"[bold cyan]STARTING FALLBACK RUN: {tag} (LR={lr:.1e}, Temp={temperature})[/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]")

    # 1. Build DataLoaders
    batch_size = 128
    train_loader = create_dataloader(
        split="train",
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        max_length=256,
        data_dir=PROCESSED_DIR,
    )
    val_loader = create_dataloader(
        split="validation",
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        max_length=256,
        data_dir=PROCESSED_DIR,
    )

    # 2. Instantiate Model
    model = SharedEncoder(
        vocab_size=16000,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        max_seq_len=256,
        dropout=0.1,
        num_modalities=2,
    ).to(device)

    # 3. Setup Config
    trainer_cfg = {
        "training": {
            "lr": lr,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "temperature": temperature,
            "epochs": 2,
            "warmup_ratio": 0.10,
            "log_every_steps": 50,
            "val_every_steps": 500,
            "seed": 42,
        },
        "model": {
            "name": "shared",
            "vocab_size": 16000,
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 8,
            "d_ff": 1024,
            "max_seq_len": 256,
            "dropout": 0.1,
            "num_modalities": 2,
        },
        "checkpoint_dir": str(RESULTS_DIR / tag),
    }

    # 4. MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("codeembed-pilot-fallback")

    with mlflow.start_run(run_name=f"fallback-{tag}"):
        mlflow.set_tag("phase", "phase_r2b_fallback_grid")
        mlflow.set_tag("data_version", "clean_v2")
        mlflow.log_params({"lr": lr, "temperature": temperature, "tag": tag, "epochs": 2})

        trainer = ContrastiveTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=trainer_cfg,
            device=device,
        )

        train_results = trainer.train()

        # Save model checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": trainer_cfg,
                "epoch": 2,
                "train_results": train_results,
            },
            ckpt_path,
        )
        console.print(f"[green][OK][/green] Checkpoint saved to {ckpt_path}")

        # 5. Full Validation Evaluation (all 20,115 queries)
        console.print("[cyan]Evaluating on full 20,115 validation split...[/cyan]")
        eval_metrics = evaluate_checkpoint_on_full_val(model, device, tokenizer, batch_size=batch_size)

        for k, v in eval_metrics.items():
            mlflow.log_metric(k, v)

        console.print(
            f"[bold green]>> Completed {tag}: Full Val MRR = {eval_metrics['val_mrr']:.4f}, "
            f"Low-Overlap MRR = {eval_metrics['val_low_overlap_mrr']:.4f}, R@1 = {eval_metrics['val_r1']:.4f}[/bold green]"
        )

    result_record = {
        "tag": tag,
        "lr": lr,
        "temperature": temperature,
        "checkpoint": str(ckpt_path),
        **eval_metrics,
    }
    return result_record


def load_existing_results() -> dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def print_leaderboard(results: dict, console: Console) -> None:
    table = Table(title="Fallback Tuning Grid Leaderboard (Full Validation Split: 20,115 Queries)")
    table.add_column("Rank", justify="center")
    table.add_column("Tag", justify="left", style="bold")
    table.add_column("LR", justify="right")
    table.add_column("Temp (τ)", justify="right")
    table.add_column("Val MRR", justify="right", style="cyan")
    table.add_column("Low-Overlap MRR", justify="right")
    table.add_column("Recall@1", justify="right")
    table.add_column("Recall@10", justify="right")

    sorted_runs = sorted(results.values(), key=lambda x: x["val_mrr"], reverse=True)
    for idx, r in enumerate(sorted_runs, 1):
        table.add_row(
            str(idx),
            r["tag"],
            f"{r['lr']:.1e}",
            f"{r['temperature']:.2f}",
            f"{r['val_mrr']:.4f}",
            f"{r.get('val_low_overlap_mrr', 0.0):.4f}",
            f"{r.get('val_r1', 0.0):.4f}",
            f"{r.get('val_r10', 0.0):.4f}",
        )
    console.print("\n")
    console.print(table)

    if sorted_runs:
        winner = sorted_runs[0]
        console.print(f"\n[bold green]WINNING CONFIGURATION: {winner['tag']} (LR={winner['lr']:.1e}, τ={winner['temperature']:.2f})[/bold green]")
        console.print(f"[green]Highest Validation MRR: {winner['val_mrr']:.4f}[/green]")
        console.print("[cyan]Action: Adopt this recipe across all comparison arms and proceed to 3-seed replication.[/cyan]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pre-Registered Fallback Tuning Grid.")
    parser.add_argument("--lr", type=float, default=None, help="Specific learning rate to run.")
    parser.add_argument("--temperature", type=float, default=None, help="Specific temperature to run.")
    parser.add_argument("--force", action="store_true", help="Force rerun even if already in results.json.")
    args = parser.parse_args()

    console = Console()
    device = get_device()
    tokenizer = CodeEmbedTokenizer()

    results = load_existing_results()

    # Seed the pilot run results if not already present
    pilot_tag = "lr_3e-4_tau_0.07"
    if pilot_tag not in results:
        results[pilot_tag] = {
            "tag": pilot_tag,
            "lr": 3.0e-4,
            "temperature": 0.07,
            "checkpoint": "checkpoints/shared_clean/best_shared.pt",
            "val_mrr": 0.3423,
            "val_r1": 0.2500,
            "val_r5": 0.4425,
            "val_r10": 0.5194,
            "val_low_overlap_mrr": 0.2433,
        }
        save_results(results)

    # Filter grid configs
    configs_to_run = []
    for cfg in GRID_CONFIGS:
        if args.lr is not None and abs(cfg["lr"] - args.lr) > 1e-7:
            continue
        if args.temperature is not None and abs(cfg["temperature"] - args.temperature) > 1e-4:
            continue
        configs_to_run.append(cfg)

    console.print(f"[bold cyan]Running {len(configs_to_run)} fallback grid configuration(s)...[/bold cyan]")

    for cfg in configs_to_run:
        tag = cfg["tag"]
        if tag in results and not args.force:
            console.print(f"[yellow][SKIP][/yellow] {tag} already completed (Val MRR = {results[tag]['val_mrr']:.4f}). Use --force to rerun.")
            continue

        res = train_and_eval_cell(
            lr=cfg["lr"],
            temperature=cfg["temperature"],
            tag=tag,
            device=device,
            tokenizer=tokenizer,
            console=console,
        )
        results[tag] = res
        save_results(results)

    print_leaderboard(results, console)


if __name__ == "__main__":
    main()
