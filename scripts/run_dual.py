"""Phase 4: Train and evaluate Separate Dual Transformer Encoders (Model 3).

This script:
1. Loads preprocessed datasets and initializes DataLoaders.
2. Constructs the DualEncoder with decoupled code and text BaseEncoders.
3. Launches the ContrastiveTrainer with CUDA mixed precision, cosine LR schedule, and MLflow tracking.
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
from tokenizer.tokenizer import CodeEmbedTokenizer
from training.trainer import ContrastiveTrainer, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Dual Encoder on CodeSearchNet.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dual.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps for smoke tests or custom budgets.",
    )
    return parser.parse_args()


def run_training(config_path: str, max_steps_override: int | None = None) -> None:
    console = Console()
    cfg = OmegaConf.load(config_path)

    if max_steps_override is not None:
        cfg.training.max_steps = max_steps_override

    console.print(Panel.fit("[bold green]CodeEmbed — Phase 4: Model 3 (Separate Dual Encoders)[/bold green]"))
    console.print(f"[cyan]Configuration:[/cyan] {config_path}")

    # 1. Reproducibility & Device
    seed = int(cfg.training.get("seed", 42))
    set_seed(seed)
    device = get_device()
    console.print(f"[cyan]Compute Device:[/cyan] {device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Tokenizer & DataLoaders
    tokenizer_path = cfg.data.get("tokenizer_path", "tokenizer/tokenizer.json")
    console.print(f"[cyan]Loading tokenizer from {tokenizer_path}...[/cyan]")
    tokenizer = CodeEmbedTokenizer(tokenizer_path)

    batch_size = int(cfg.data.batch_size)
    num_workers = int(cfg.data.get("num_workers", 0))

    console.print(f"[cyan]Building DataLoaders (batch_size={batch_size})...[/cyan]")
    train_loader = create_dataloader(
        split="train",
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        max_length=int(cfg.model.max_seq_len),
    )

    val_loader = create_dataloader(
        split="validation",
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        max_length=int(cfg.model.max_seq_len),
    )
    console.print(f"[green][OK][/green] Train batches: {len(train_loader):,}, Val batches: {len(val_loader):,}")

    # 3. Instantiate DualEncoder Model
    m_cfg = cfg.model
    model = DualEncoder(
        vocab_size=int(m_cfg.vocab_size),
        d_model=int(m_cfg.d_model),
        n_layers=int(m_cfg.n_layers),
        n_heads=int(m_cfg.n_heads),
        d_ff=int(m_cfg.d_ff),
        max_seq_len=int(m_cfg.max_seq_len),
        dropout=float(m_cfg.dropout),
    )
    total_p, train_p = model.get_num_params()
    breakdown = model.get_encoder_params()
    console.print(f"[green][OK][/green] Initialized DualEncoder: {train_p:,} trainable params total (~{total_p / 1e6:0.2f}M).")
    console.print(f"  ├─ Code Encoder: {breakdown['code_encoder_trainable']:,} params (~{breakdown['code_encoder_total'] / 1e6:0.2f}M)")
    console.print(f"  └─ Text Encoder: {breakdown['text_encoder_trainable']:,} params (~{breakdown['text_encoder_total'] / 1e6:0.2f}M)")

    # 4. Setup MLflow
    tracking_uri = cfg.mlflow.get("tracking_uri", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        console.print(f"[cyan]MLflow Run ID:[/cyan] {run.info.run_id}")
        mlflow.set_tag("architecture", "dual")
        mlflow.set_tag("phase", "phase4_dual_encoder")
        mlflow.set_tag("tokenizer", "custom_bpe")
        mlflow.set_tag("negatives", "inbatch")

        # Log hyperparameters
        mlflow.log_params({
            "model_type": "DualEncoder",
            "d_model": m_cfg.d_model,
            "n_layers_per_encoder": m_cfg.n_layers,
            "n_heads": m_cfg.n_heads,
            "d_ff": m_cfg.d_ff,
            "max_seq_len": m_cfg.max_seq_len,
            "total_params": total_p,
            "code_encoder_params": breakdown["code_encoder_total"],
            "text_encoder_params": breakdown["text_encoder_total"],
            "batch_size": batch_size,
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay,
            "temperature": cfg.training.temperature,
            "max_steps": cfg.training.max_steps,
            "warmup_steps": cfg.training.warmup_steps,
            "seed": seed,
        })

        # 5. Launch Trainer
        trainer = ContrastiveTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=OmegaConf.to_container(cfg, resolve=True),
            device=device,
        )

        results = trainer.train()
        console.print(f"[bold green]Training complete! Best Validation MRR: {results.get('best_val_mrr', 0.0):0.4f}[/bold green]")


if __name__ == "__main__":
    args = parse_args()
    run_training(args.config, max_steps_override=args.max_steps)
