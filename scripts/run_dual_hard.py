"""Phase 5: Hard Negative Training Script.

Two-stage pipeline:
  Stage 1 (--mine-only): Mine hard negatives from the Phase 4 checkpoint.
                          Saves data/hard_negatives/train_hard_neg_indices.npy
  Stage 2 (--train-only): Fine-tune the Dual Encoder using those hard negatives.

Run both stages in sequence (default behaviour):
    python scripts/run_dual_hard.py --config configs/dual_hard.yaml

Or run stages separately:
    python scripts/run_dual_hard.py --config configs/dual_hard.yaml --mine-only
    python scripts/run_dual_hard.py --config configs/dual_hard.yaml --train-only
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Allow MLflow to use local filesystem backend without crashing
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from data.hard_negative_dataset import create_hard_negative_dataloader
from data.dataset import create_dataloader
from losses.hard_negative_loss import InfoNCEWithHardNegatives
from model.dual_encoder import DualEncoder
from tokenizer.tokenizer import CodeEmbedTokenizer
from training.hard_negatives import mine_and_save
from training.trainer import ContrastiveTrainer, get_device, set_seed

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

console = Console()


# ── Stage 1: Mining ───────────────────────────────────────────────────────────

def run_mining(cfg: OmegaConf) -> None:
    hn_cfg   = cfg.hard_negatives
    index_path = Path(hn_cfg.index_path)

    if index_path.exists():
        console.print(f"[yellow]Hard negative index already exists at {index_path}.[/yellow]")
        console.print("[yellow]Delete it and re-run --mine-only to regenerate.[/yellow]")
        return

    console.print(Panel("[bold cyan]Stage 1: Mining Hard Negatives[/bold cyan]"))
    console.print(f"  Checkpoint : {hn_cfg.init_checkpoint}")
    console.print(f"  K per sample: {hn_cfg.k_mined}")
    console.print(f"  Output     : {index_path}")

    mine_and_save(
        checkpoint_path=str(hn_cfg.init_checkpoint),
        output_path=str(index_path),
        k=int(hn_cfg.k_mined),
        batch_size=int(cfg.data.batch_size),
        max_length=int(cfg.model.max_seq_len),
    )
    console.print(f"[bold green]✓ Hard negatives saved to {index_path}[/bold green]")


# ── Stage 2: Training ─────────────────────────────────────────────────────────

def run_training(cfg: OmegaConf) -> None:
    console.print(Panel("[bold cyan]Stage 2: Fine-tuning with Hard Negatives[/bold cyan]"))

    hn_cfg = cfg.hard_negatives
    t_cfg  = cfg.training
    m_cfg  = cfg.model

    set_seed(int(t_cfg.seed))
    device = get_device()
    console.print(f"Device : {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    torch.cuda.empty_cache()
    gc.collect()

    # ── DataLoaders ───────────────────────────────────────────────────────────
    tokenizer = CodeEmbedTokenizer(str(cfg.data.tokenizer_path))

    train_loader = create_hard_negative_dataloader(
        hard_neg_path  = str(hn_cfg.index_path),
        split          = "train",
        tokenizer      = tokenizer,
        batch_size     = int(cfg.data.batch_size),
        max_length     = int(m_cfg.max_seq_len),
        max_hard_negs  = int(hn_cfg.max_hard_negs),
        shuffle        = True,
        num_workers    = int(cfg.data.num_workers),
    )

    # Validation uses standard in-batch loader (no hard negs needed at eval)
    val_loader = create_dataloader(
        split       = "validation",
        tokenizer   = tokenizer,
        batch_size  = int(cfg.data.batch_size),
        shuffle     = False,
        num_workers = int(cfg.data.num_workers),
        max_length  = int(m_cfg.max_seq_len),
    )

    console.print(f"[OK] Train: {len(train_loader):,} batches | Val: {len(val_loader):,} batches")

    # 3. Instantiate DualEncoder and load Phase 4 weights
    m_cfg = cfg.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoder(
        vocab_size=int(m_cfg.vocab_size),
        d_model=int(m_cfg.d_model),
        n_layers=int(m_cfg.n_layers),
        n_heads=int(m_cfg.n_heads),
        d_ff=int(m_cfg.d_ff),
        max_seq_len=int(m_cfg.max_seq_len),
        dropout=float(m_cfg.dropout),
    ).to(device)

    # Load checkpoint — supports two modes:
    # 1. Warm-start from a Basic Encoder checkpoint (recommended for first run)
    # 2. Resume from an existing Dual Encoder checkpoint
    init_ckpt = hn_cfg.get("init_checkpoint", None)
    if not init_ckpt or not os.path.exists(init_ckpt):
        raise FileNotFoundError(f"Must provide valid init_checkpoint for Phase 5. Got: {init_ckpt}")

    ckpt_probe = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    ckpt_model_name = ckpt_probe.get("config", {}).get("model", {}).get("name", "dual")

    if ckpt_model_name != "dual":
        # Basic/Shared Encoder checkpoint → warm-start BOTH towers from it
        console.print(f"[bold cyan]Warm-starting both encoder towers from Basic Encoder: {init_ckpt}[/bold cyan]")
        model = DualEncoder.warm_start_from(
            checkpoint_path=init_ckpt,
            device=device,
            dropout=float(m_cfg.dropout),
        )
    else:
        # Existing Dual Encoder checkpoint → standard resume
        console.print(f"[bold cyan]Resuming from Dual Encoder checkpoint: {init_ckpt}[/bold cyan]")
        checkpoint = torch.load(init_ckpt, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
    if torch.cuda.device_count() > 1:
        console.print(f"[cyan]Using {torch.cuda.device_count()} GPUs via DataParallel![/cyan]")
        model = torch.nn.DataParallel(model)

    unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
    total_p, train_p = unwrapped.get_num_params()
    console.print(f"[green][OK][/green] Loaded Phase 4 weights from {init_ckpt}")
    console.print(f"[green][OK][/green] DualEncoder: {train_p:,} trainable params (~{total_p / 1e6:0.2f}M)")

    # ── Custom trainer that uses hard-negative loss ───────────────────────────
    # We subclass ContrastiveTrainer and override train_step to inject hard negs.
    from training.trainer import (
        ContrastiveTrainer,
        get_cosine_schedule_with_warmup,
    )

    class HardNegTrainer(ContrastiveTrainer):
        """ContrastiveTrainer with hard-negative loss substituted in."""

        def __init__(self, hn_weight: float = 1.0, **kwargs):
            super().__init__(**kwargs)
            self.criterion = InfoNCEWithHardNegatives(
                temperature=self.temperature,
                hn_weight=hn_weight,
            )

        def train_step(self, batch: dict) -> tuple[float, float, float]:
            """Override: encode hard negatives and pass to HN loss."""
            self.model.train()
            self.optimizer.zero_grad()

            code_ids   = batch["code_ids"].to(self.device)
            code_mask  = batch["code_mask"].to(self.device)
            text_ids   = batch["text_ids"].to(self.device)
            text_mask  = batch["text_mask"].to(self.device)
            hn_ids     = batch["hard_neg_ids"].to(self.device)    # (B, K, L)
            hn_masks   = batch["hard_neg_masks"].to(self.device)  # (B, K, L)

            B, K, L = hn_ids.shape

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                # Positive encodings
                z_code = self.model(code_ids, attention_mask=code_mask, mode="code")
                z_text = self.model(text_ids, attention_mask=text_mask, mode="text")

                # Hard negative encodings: flatten (B*K, L), encode, reshape
                hn_ids_flat   = hn_ids.view(B * K, L)
                hn_masks_flat = hn_masks.view(B * K, L)
                z_hn_flat = self.model(hn_ids_flat, attention_mask=hn_masks_flat, mode="code")
                z_hn = z_hn_flat.view(B, K, -1)               # (B, K, D)

                loss = self.criterion(z_text, z_code, z_hn)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            t2c_acc, c2t_acc = self.criterion.compute_accuracy(z_text, z_code)
            return float(loss.item()), t2c_acc, c2t_acc

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        console.print(f"[cyan]MLflow Run ID:[/cyan] {run.info.run_id}")

        mlflow.set_tags({
            "architecture" : "dual",
            "phase"        : "phase5_hard_negatives",
            "negatives"    : "inbatch+hard",
            "hardware"     : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "batch_size"   : str(cfg.data.batch_size),
            "max_hard_negs": str(hn_cfg.max_hard_negs),
            "init_from"    : str(hn_cfg.init_checkpoint),
        })
        mlflow.log_params({
            "d_model"      : m_cfg.d_model,
            "n_layers"     : m_cfg.n_layers,
            "n_heads"      : m_cfg.n_heads,
            "d_ff"         : m_cfg.d_ff,
            "batch_size"   : cfg.data.batch_size,
            "lr"           : t_cfg.lr,
            "temperature"  : t_cfg.temperature,
            "max_steps"    : t_cfg.max_steps,
            "warmup_steps" : t_cfg.warmup_steps,
            "max_hard_negs": hn_cfg.max_hard_negs,
            "hn_weight"    : hn_cfg.hn_weight,
            "seed"         : t_cfg.seed,
        })

        trainer = HardNegTrainer(
            hn_weight    = float(hn_cfg.hn_weight),
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            config       = OmegaConf.to_container(cfg, resolve=True),
            device       = device,
        )

        results = trainer.train()

        torch.cuda.empty_cache()
        gc.collect()

        console.print(f"\n[bold green]🏆 Training Complete![/bold green]")
        console.print(f"   Best Validation MRR: [bold yellow]{results.get('best_val_mrr', 0.0):.4f}[/bold yellow]")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Hard Negative Training")
    parser.add_argument("--config",     type=str, default="configs/dual_hard.yaml")
    parser.add_argument("--mine-only",  action="store_true", help="Only run mining stage")
    parser.add_argument("--train-only", action="store_true", help="Only run training stage (index must exist)")
    parser.add_argument("--max-steps",  type=int, default=None, help="Override max_steps")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.max_steps:
        cfg.training.max_steps = args.max_steps

    console.print(Panel("[bold magenta]CodeEmbed — Phase 5: Hard Negative Training[/bold magenta]"))
    console.print(OmegaConf.to_yaml(cfg))

    if args.train_only:
        run_training(cfg)
    elif args.mine_only:
        run_mining(cfg)
    else:
        run_mining(cfg)
        run_training(cfg)


if __name__ == "__main__":
    main()
