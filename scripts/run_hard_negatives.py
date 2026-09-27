"""Phase 5: Train model with explicit hard negatives (Model 4 / Hard Negative Mining).

This script:
1. Loads datasets and initializes DataLoaders with hard negative batch tensors.
2. Constructs the SharedEncoder (or configured architecture).
3. Launches the ContrastiveTrainer with InfoNCEWithHardNegativesLoss, CUDA mixed precision, and MLflow logging.
4. Checkpoints the best model by validation MRR.
"""

import argparse
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
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel

from data.dataset import create_dataloader
from model.dual_encoder import DualEncoder
from model.shared_encoder import SharedEncoder
from tokenizer.tokenizer import CodeEmbedTokenizer
from training.trainer import ContrastiveTrainer, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CodeEmbed model with Hard Negatives."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/shared_hard.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt) to resume/fine-tune from.",
    )
    return parser.parse_args()


def run_training(
    config_path: str,
    max_steps_override: int | None = None,
    epochs_override: int | None = None,
    lr_override: float | None = None,
    resume_checkpoint: str | None = None,
) -> None:
    console = Console()
    cfg = OmegaConf.load(config_path)

    if max_steps_override is not None:
        cfg.training.max_steps = max_steps_override
    if epochs_override is not None:
        cfg.training.epochs = epochs_override
    if lr_override is not None:
        cfg.training.lr = lr_override

    model_name = str(cfg.model.get("name", "shared")).lower()
    console.print(
        Panel.fit(
            f"[bold green]CodeEmbed — Phase 5: Hard Negative Mining ({model_name.capitalize()} Architecture)[/bold green]"
        )
    )
    console.print(f"[cyan]Configuration:[/cyan] {config_path}")

    # 1. Reproducibility & Device
    seed = int(cfg.training.get("seed", 42))
    set_seed(seed)
    device = get_device()
    console.print(
        f"[cyan]Compute Device:[/cyan] {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )

    # 2. Tokenizer & DataLoaders
    tokenizer_path = cfg.data.get("tokenizer_path", "tokenizer/tokenizer.json")
    tokenizer = CodeEmbedTokenizer(tokenizer_path)

    batch_size = int(cfg.data.batch_size)
    num_workers = int(cfg.data.get("num_workers", 0))
    train_path = cfg.data.get("train_path", "train")
    val_path = cfg.data.get("val_path", "validation")
    hn_file = cfg.data.get("hard_negatives_file", None)
    num_hn = int(cfg.data.get("num_hard_negatives", 1))

    console.print(
        f"[cyan]Building DataLoaders (batch_size={batch_size}, num_hard_negatives={num_hn})...[/cyan]"
    )
    train_loader = create_dataloader(
        split=train_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_length=int(cfg.model.max_seq_len),
        hard_negatives_file=hn_file,
        num_hard_negatives=num_hn,
    )

    val_loader = create_dataloader(
        split=val_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_length=int(cfg.model.max_seq_len),
    )
    console.print(
        f"[green][OK][/green] Train batches: {len(train_loader):,}, Val batches: {len(val_loader):,}"
    )

    # 3. Model Architecture Instantiation
    m_cfg = cfg.model
    if model_name == "dual":
        model = DualEncoder(
            vocab_size=int(m_cfg.vocab_size),
            d_model=int(m_cfg.d_model),
            n_layers=int(m_cfg.n_layers),
            n_heads=int(m_cfg.n_heads),
            d_ff=int(m_cfg.d_ff),
            max_seq_len=int(m_cfg.max_seq_len),
            dropout=float(m_cfg.dropout),
        )
    else:
        model = SharedEncoder(
            vocab_size=int(m_cfg.vocab_size),
            d_model=int(m_cfg.d_model),
            n_layers=int(m_cfg.n_layers),
            n_heads=int(m_cfg.n_heads),
            d_ff=int(m_cfg.d_ff),
            max_seq_len=int(m_cfg.max_seq_len),
            dropout=float(m_cfg.dropout),
            num_modalities=int(m_cfg.get("num_modalities", 2)),
        )

    total_p, train_p = model.get_num_params()
    console.print(
        f"[green][OK][/green] Initialized model: {train_p:,} trainable params total (~{total_p / 1e6:0.2f}M)."
    )

    initial_best_score = -float("inf")
    if resume_checkpoint:
        console.print(f"[cyan]Loading weights from {resume_checkpoint}...[/cyan]")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        initial_best_score = float(ckpt.get("best_metric", -float("inf")))
        console.print(
            f"[green][OK][/green] Successfully loaded checkpoint weights "
            f"(previous best MRR: {initial_best_score:.4f})"
        )

    # 4. Setup MLflow
    tracking_uri = cfg.mlflow.get("tracking_uri", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(
        run_name=f"{cfg.mlflow.run_name}-finetune"
        if resume_checkpoint
        else cfg.mlflow.run_name
    ) as run:
        console.print(f"[cyan]MLflow Run ID:[/cyan] {run.info.run_id}")
        mlflow.set_tag("data_version", "clean_v2")
        mlflow.set_tag("phase", "phase_r2d_hard_negatives")
        mlflow.set_tag("architecture", model_name)
        mlflow.set_tag("negatives", "mined_hard_negatives")
        if resume_checkpoint:
            mlflow.set_tag("resumed_from", str(resume_checkpoint))

        mlflow.log_params(
            {
                "model_type": type(model).__name__,
                "d_model": m_cfg.d_model,
                "n_layers": m_cfg.n_layers,
                "num_hard_negatives": num_hn,
                "total_params": total_p,
                "batch_size": batch_size,
                "lr": cfg.training.lr,
                "weight_decay": cfg.training.weight_decay,
                "temperature": cfg.training.temperature,
                "epochs": cfg.training.get("epochs", 2),
                "warmup_ratio": cfg.training.get("warmup_ratio", 0.10),
                "seed": seed,
            }
        )

        # 5. Launch Trainer
        trainer = ContrastiveTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=OmegaConf.to_container(cfg, resolve=True),
            device=device,
        )
        if initial_best_score > 0:
            trainer.best_val_score = initial_best_score

        results = trainer.train()
        console.print(
            f"[bold green]Training complete! Best Validation MRR: {results.get('best_val_mrr', 0.0):0.4f}[/bold green]"
        )

    # 6. Post-training evaluation on validation set (enforce strict test set discipline)
    eval_split = str(cfg.training.get("eval_split", "validation"))
    console.print(
        f"\n[bold cyan]Evaluating best model on {eval_split} benchmark...[/bold cyan]"
    )
    from evaluation.evaluate import evaluate_checkpoint

    best_ckpt = trainer.best_checkpoint_path
    if best_ckpt.exists():
        evaluate_checkpoint(
            checkpoint_path=str(best_ckpt),
            split=eval_split,
            sample_size=1000,
            batch_size=batch_size,
            seed=seed,
        )


if __name__ == "__main__":
    args = parse_args()
    run_training(
        args.config,
        max_steps_override=args.max_steps,
        epochs_override=args.epochs,
        lr_override=args.lr,
        resume_checkpoint=args.resume_checkpoint,
    )
