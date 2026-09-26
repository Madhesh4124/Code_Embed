"""Unified CLI Runner for Phase 6: Ablation Studies.

Executes controlled architectural ablations:
- Pooling strategy: CLSPooling vs MaskedMeanPooling
- Temperature sensitivity: tau in {0.05, 0.07, 0.10}
- Sequence length impact: L=128 vs L=256

Ensures:
1. Dedicated MLflow experiment: 'codeembed-phase6-ablations'
2. Single-run context per experiment: training metrics, step curves, and test IR benchmark
   are logged to the exact same run ID with zero duplicate runs.
3. Clean tags and human-readable run names.
4. Continuous real-time console and milestone line logging (flush=True).
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
import torch
import yaml
from rich.console import Console
from rich.table import Table

from data.ablation_dataset import get_ablation_dataloader
from evaluation.evaluate import compute_similarity_rankings, evaluate_rankings
from model.ablation_models import AblationSharedEncoder
from training.trainer import ContrastiveTrainer, get_device, set_seed

console = Console(legacy_windows=False)


class AblationTrainer(ContrastiveTrainer):
    """Ablation trainer customized for Windows console compatibility."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.console = Console(legacy_windows=False)

    def train(self) -> dict[str, float]:
        """Execute training loop with ASCII milestone line logging for Windows compatibility."""
        self.console.print(
            f"[bold green]Starting Training on {self.device.type.upper()}[/bold green]"
        )
        total_params, trainable_params = self.model.get_num_params()
        self.console.print(
            f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total (~{total_params / 1e6:0.2f}M)"
        )

        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            self.console.print(
                f"\n[bold cyan]Epoch {epoch}/{self.max_epochs}[/bold cyan]"
            )
            epoch_loss = 0.0
            epoch_steps = 0

            for batch in self.train_loader:
                self.global_step += 1
                loss, t2c_acc, c2t_acc = self.train_step(batch)

                epoch_loss += loss
                epoch_steps += 1

                # Step logging
                if self.global_step % self.log_every_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"Step {self.global_step}/{self.total_training_steps} | "
                        f"Loss: {loss:.4f} | T2C Acc: {t2c_acc:.3f} | "
                        f"LR: {current_lr:.2e}",
                        flush=True,
                    )
                    metrics = {
                        "train_loss": loss,
                        "train_t2c_acc": t2c_acc,
                        "train_c2t_acc": c2t_acc,
                        "learning_rate": current_lr,
                    }
                    for k, v in metrics.items():
                        mlflow.log_metric(k, v, step=self.global_step)

                # Mid-epoch validation
                if self.global_step % self.val_every_steps == 0 and self.val_loader:
                    val_metrics = self.evaluate()
                    for k, v in val_metrics.items():
                        mlflow.log_metric(k, v, step=self.global_step)

                    val_mrr = val_metrics.get("val_mrr", 0.0)
                    if val_mrr > self.best_val_score:
                        self.best_val_score = val_mrr
                        self.save_checkpoint(
                            self.checkpoint_dir
                            / f"checkpoint_step_{self.global_step}.pt",
                            epoch=epoch,
                            is_best=True,
                        )
                        self.console.print(
                            f" [OK] New best val MRR: {val_mrr:0.4f} saved"
                        )

                if self.global_step >= self.max_steps:
                    break

            # Epoch end validation
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            self.console.print(
                f"Epoch {epoch} finished - Avg Loss: {avg_epoch_loss:0.4f}"
            )

            if self.val_loader:
                val_metrics = self.evaluate(max_val_batches=None)
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"epoch_{k}", v, step=epoch)

                val_mrr = val_metrics.get("val_mrr", 0.0)
                is_best = val_mrr > self.best_val_score
                if is_best:
                    self.best_val_score = val_mrr
                    self.console.print(f"[*] New Best Model: MRR = {val_mrr:0.4f}")

                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
                    epoch=epoch,
                    is_best=is_best,
                )

            if self.global_step >= self.max_steps:
                self.console.print(
                    "[yellow]Reached max_steps limit. Concluding training.[/yellow]"
                )
                break

        total_duration = time.time() - start_time
        self.console.print(
            f"\n[bold green]Training Completed in {total_duration / 60:0.2f} minutes.[/bold green]"
        )
        return {"best_val_mrr": self.best_val_score}


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_model_benchmark(
    model: torch.nn.Module,
    split: str = "validation",
    max_seq_len: int | None = None,
    batch_size: int = 128,
    sample_size: int | None = None,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate trained model on retrieval benchmark (strictly validation for Phase R4)."""
    if device is None:
        device = get_device()

    model.eval()
    console.print(
        f"[cyan]Building {split} DataLoader (max_seq_len={max_seq_len or 256})...[/cyan]"
    )
    loader = get_ablation_dataloader(
        split=split,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        shuffle=False,
        num_workers=0,
    )

    console.print(
        f"[cyan]Encoding {split} corpus and queries on {device.type.upper()}...[/cyan]"
    )
    all_code_embs: list[np.ndarray] = []
    all_text_embs: list[np.ndarray] = []

    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            code_ids = batch["code_ids"].to(device, non_blocking=True)
            code_mask = batch["code_mask"].to(device, non_blocking=True)
            text_ids = batch["text_ids"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                c_emb = model.encode_code(code_ids, code_mask)
                t_emb = model.encode_text(text_ids, text_mask)

            all_code_embs.append(c_emb.cpu().numpy())
            all_text_embs.append(t_emb.cpu().numpy())

    corpus_embeddings = np.concatenate(all_code_embs, axis=0)
    query_embeddings = np.concatenate(all_text_embs, axis=0)
    total_docs = len(corpus_embeddings)
    console.print(
        f"[green][OK][/green] Encoded {total_docs:,} corpus and query embeddings."
    )

    # Subsample queries if sample_size is explicitly specified and < total_docs
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
            f"[yellow]Evaluating all {total_docs:,} queries against full {total_docs:,} corpus.[/yellow]"
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
    return results


# Backward compatibility alias
evaluate_model_on_test = evaluate_model_benchmark


def run_single_ablation(
    config_path: str | Path,
    epochs_override: int | None = None,
    smoke_test: bool = False,
    sample_size_override: int | None = None,
) -> dict[str, Any]:
    """Execute a single ablation experiment from config."""
    cfg = load_config(config_path)

    # CLI Overrides
    if epochs_override is not None:
        cfg["training"]["epochs"] = epochs_override
    if smoke_test:
        cfg["training"]["epochs"] = 1
        cfg["training"]["max_steps"] = 5
        cfg["training"]["log_every_steps"] = 1
        cfg["training"]["val_every_steps"] = 3

    t_cfg = cfg["training"]
    m_cfg = cfg["model"]
    d_cfg = cfg["data"]
    a_cfg = cfg.get("ablation", {})
    ml_cfg = cfg.get("mlflow", {})

    seed = int(t_cfg.get("seed", 42))
    set_seed(seed)
    device = get_device()

    console.rule(
        f"[bold cyan]Ablation Experiment: {ml_cfg.get('run_name', 'ablation')}[/bold cyan]"
    )
    console.print(
        f"Device: {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )
    console.print(
        f"Ablation Category: [bold yellow]{a_cfg.get('category', 'unknown')}[/bold yellow]"
    )
    console.print(
        f"Ablation Variant:  [bold yellow]{a_cfg.get('variant', 'unknown')}[/bold yellow]"
    )
    console.print(f"Pooling Strategy:  {m_cfg.get('pooling', 'mean')}")
    console.print(f"Sequence Length:   {d_cfg.get('max_seq_len', 256)}")
    console.print(f"Loss Temperature:  {t_cfg.get('temperature', 0.05)}")
    console.print(f"Training Epochs:   {t_cfg.get('epochs', 2)}")

    # 1. Instantiate Ablation Model
    model = AblationSharedEncoder(
        vocab_size=int(m_cfg.get("vocab_size", 16000)),
        d_model=int(m_cfg.get("d_model", 256)),
        n_layers=int(m_cfg.get("n_layers", 4)),
        n_heads=int(m_cfg.get("n_heads", 8)),
        d_ff=int(m_cfg.get("d_ff", 1024)),
        max_seq_len=int(m_cfg.get("max_seq_len", 256)),
        dropout=float(m_cfg.get("dropout", 0.1)),
        num_modalities=int(m_cfg.get("num_modalities", 2)),
        pooling=m_cfg.get("pooling", "mean"),
    ).to(device)

    total_params, trainable_params = model.get_num_params()
    console.print(
        f"Parameters: {trainable_params:,} trainable / {total_params:,} total (~{total_params / 1e6:0.2f}M)"
    )

    # 2. Build DataLoaders
    batch_size = int(d_cfg.get("batch_size", 128))
    max_seq_len = int(d_cfg.get("max_seq_len", 256))
    max_seq_len_arg = max_seq_len if max_seq_len < 256 else None

    train_loader = get_ablation_dataloader(
        split="train",
        batch_size=batch_size,
        max_seq_len=max_seq_len_arg,
        shuffle=True,
        num_workers=0,
    )

    val_loader = get_ablation_dataloader(
        split="validation",
        batch_size=batch_size,
        max_seq_len=max_seq_len_arg,
        shuffle=False,
        num_workers=0,
    )

    # 3. Setup MLflow Tracking (Unified Single Run)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    exp_name = ml_cfg.get("experiment_name", "codeembed-clean-ablations")
    mlflow.set_experiment(exp_name)
    run_name = ml_cfg.get(
        "run_name", f"phaseR4_{a_cfg.get('category')}_{a_cfg.get('variant')}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        console.print(f"[bold green]Active MLflow Run ID:[/bold green] {run_id}")

        # Log parameters & tags
        mlflow.set_tags(
            {
                "phase": "R4",
                "phase_name": "Ablation Studies",
                "ablation_category": str(a_cfg.get("category", "")),
                "ablation_variant": str(a_cfg.get("variant", "")),
                "pooling": str(m_cfg.get("pooling", "mean")),
                "max_seq_len": str(max_seq_len),
                "temperature": str(t_cfg.get("temperature", 0.05)),
                "model_type": "ablation_shared",
            }
        )

        mlflow.log_params(
            {
                "d_model": m_cfg.get("d_model", 256),
                "n_layers": m_cfg.get("n_layers", 4),
                "n_heads": m_cfg.get("n_heads", 8),
                "d_ff": m_cfg.get("d_ff", 1024),
                "pooling": m_cfg.get("pooling", "mean"),
                "max_seq_len": max_seq_len,
                "temperature": t_cfg.get("temperature", 0.05),
                "lr": t_cfg.get("lr", 5e-4),
                "batch_size": batch_size,
                "epochs": t_cfg.get("epochs", 2),
                "total_params": total_params,
            }
        )

        # 4. Train Model
        trainer = AblationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg,
            device=device,
        )

        train_start = time.time()
        train_results = trainer.train()
        train_duration = time.time() - train_start
        mlflow.log_metric("training_duration_seconds", train_duration)

        # 5. Load Best Checkpoint for Evaluation
        best_ckpt = trainer.best_checkpoint_path
        if best_ckpt.exists():
            console.print(
                f"[cyan]Loading best checkpoint for validation evaluation from {best_ckpt}...[/cyan]"
            )
            ckpt_data = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data["model_state_dict"])
        else:
            console.print(
                "[yellow]Best checkpoint file not found, evaluating current model weights...[/yellow]"
            )

        # 6. Evaluate on Validation Benchmark (all 20,115 queries vs 20,115 corpus, or subsampled)
        if smoke_test:
            eval_sample_size = 50
        elif sample_size_override is not None:
            eval_sample_size = sample_size_override
        else:
            eval_sample_size = None  # Full validation set: 20,115 queries vs 20,115 corpus

        val_results = evaluate_model_benchmark(
            model=model,
            split="validation",
            max_seq_len=max_seq_len_arg,
            batch_size=batch_size,
            sample_size=eval_sample_size,
            seed=seed,
            device=device,
        )

        # 7. Log Validation Benchmark Metrics directly into this run
        scalar_metrics = {
            k: v for k, v in val_results.items() if not k.endswith("_ci")
        }
        for k, v in scalar_metrics.items():
            mlflow.log_metric(f"val_{k.replace('@', '_at_')}", float(v))

        # Log CIs
        for k in ["mrr", "recall@1", "recall@5", "recall@10", "ndcg@10"]:
            ci_key = f"{k}_ci"
            if ci_key in val_results and isinstance(val_results[ci_key], tuple):
                ci_low, ci_high = val_results[ci_key]
                clean_k = k.replace("@", "_at_")
                mlflow.log_metric(f"val_{clean_k}_ci_low", float(ci_low))
                mlflow.log_metric(f"val_{clean_k}_ci_high", float(ci_high))

        # 8. Print Results Table
        table = Table(
            title=f"Ablation Results: {ml_cfg.get('run_name')} ({a_cfg.get('category')}={a_cfg.get('variant')})",
            header_style="bold magenta",
        )
        table.add_column("Metric", style="dim", width=14)
        table.add_column("Score", justify="right", style="bold green", width=10)
        table.add_column(
            "95% Confidence Interval", justify="center", style="cyan", width=26
        )

        for metric_name, score in scalar_metrics.items():
            ci = val_results.get(f"{metric_name}_ci", ("—", "—"))
            ci_str = f"[{ci[0]:0.4f}, {ci[1]:0.4f}]" if isinstance(ci, tuple) else "—"
            table.add_row(metric_name.upper(), f"{score:0.4f}", ci_str)

        console.print()
        console.print(table)
        console.print(
            f"[bold green][OK] Ablation '{run_name}' complete! All metrics saved in single MLflow run {run_id}[/bold green]\n"
        )

        return {
            "run_id": run_id,
            "run_name": run_name,
            "category": a_cfg.get("category"),
            "variant": a_cfg.get("variant"),
            "metrics": scalar_metrics,
            "train_results": train_results,
        }


def main() -> None:
    """CLI Entry point for Phase 6 / Phase R4 ablation runner."""
    parser = argparse.ArgumentParser(description="Run Phase R4 Ablation Studies")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to specific ablation YAML config file",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["cls", "temp_005", "temp_010", "seq_128", "all"],
        default=None,
        help="Predefined ablation target to run",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count (default: 2)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for query evaluation (default: None, full validation split)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick 5-step smoke test for pipeline verification",
    )

    args = parser.parse_args()

    ablation_config_map = {
        "cls": "configs/ablation_cls.yaml",
        "temp_005": "configs/ablation_temp_005.yaml",
        "temp_010": "configs/ablation_temp_010.yaml",
        "seq_128": "configs/ablation_seq_128.yaml",
    }

    if args.config:
        targets = [args.config]
    elif args.ablation == "all":
        targets = [ablation_config_map["cls"], ablation_config_map["seq_128"]]
    elif args.ablation in ablation_config_map:
        targets = [ablation_config_map[args.ablation]]
    else:
        console.print(
            "[yellow]No config or ablation specified. Showing available ablations:[/yellow]"
        )
        for name, path in ablation_config_map.items():
            console.print(f"  --ablation {name:<10} -> {path}")
        return

    summary_rows: list[dict[str, Any]] = []
    for cfg_path in targets:
        result = run_single_ablation(
            cfg_path,
            epochs_override=args.epochs,
            smoke_test=args.smoke_test,
            sample_size_override=args.sample_size,
        )
        summary_rows.append(result)

    # Print comparative summary table if multiple ablations ran
    if len(summary_rows) > 1:
        comp_table = Table(
            title="Phase R4 Ablations Comparative Summary", header_style="bold yellow"
        )
        comp_table.add_column("Run Name", style="cyan")
        comp_table.add_column("Category", style="dim")
        comp_table.add_column("Variant", style="bold")
        comp_table.add_column("MRR", justify="right", style="bold green")
        comp_table.add_column("Recall@1", justify="right")
        comp_table.add_column("Recall@5", justify="right")
        comp_table.add_column("Recall@10", justify="right")
        comp_table.add_column("NDCG@10", justify="right")

        for row in summary_rows:
            m = row["metrics"]
            comp_table.add_row(
                row["run_name"],
                row["category"],
                str(row["variant"]),
                f"{m.get('mrr', 0.0):0.4f}",
                f"{m.get('recall@1', 0.0):0.4f}",
                f"{m.get('recall@5', 0.0):0.4f}",
                f"{m.get('recall@10', 0.0):0.4f}",
                f"{m.get('ndcg@10', 0.0):0.4f}",
            )
        console.print()
        console.print(comp_table)


if __name__ == "__main__":
    main()
