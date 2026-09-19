"""Contrastive Training Loop with Mixed Precision and MLflow tracking.

This module implements:
- Hardware device detection (CUDA with Ampere/Turing mixed precision, CPU fallback).
- AdamW optimizer with parameter-group weight decay filtering.
- Cosine Annealing learning rate schedule with linear warmup.
- Gradient clipping (max_norm=1.0).
- Checkpoint management saving best models by validation score.
- MLflow metric logging and run tracking.
"""

import math
import random
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.utils.data import DataLoader

from losses.contrastive import InfoNCELoss
from model.shared_encoder import SharedEncoder


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a cosine annealing learning rate scheduler with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ContrastiveTrainer:
    """Trainer for contrastive dual-representation encoders.

    Args:
        model: BaseEncoder or SharedEncoder model instance.
        train_loader: DataLoader yielding training batches.
        val_loader: DataLoader yielding validation batches.
        config: Dict of training and model configurations.
        device: Target compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.console = Console()

        # Training hyperparameters
        t_cfg = self.config.get("training", {})
        m_cfg = self.config.get("model", {})
        self.model_name = m_cfg.get("name", "model")
        self.lr = float(t_cfg.get("lr", 3e-4))
        self.weight_decay = float(t_cfg.get("weight_decay", 0.01))
        self.grad_clip = float(t_cfg.get("grad_clip", 1.0))
        self.temperature = float(t_cfg.get("temperature", 0.07))
        self.max_epochs = int(t_cfg.get("epochs", 5))
        self.max_steps = int(t_cfg.get("max_steps", 100000))
        self.warmup_steps = int(t_cfg.get("warmup_steps", 1000))
        self.log_every_steps = int(t_cfg.get("log_every_steps", 50))
        self.val_every_steps = int(t_cfg.get("val_every_steps", 500))

        # Checkpoint directory
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / f"best_{self.model_name}.pt"

        # Loss function
        self.criterion = InfoNCELoss(temperature=self.temperature)

        # Weight decay filtering: do not decay 1D parameters (biases, LayerNorm)
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        # Compute total training steps
        steps_per_epoch = len(self.train_loader)
        total_training_steps = min(self.max_steps, self.max_epochs * steps_per_epoch)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_training_steps,
        )

        # Mixed precision GradScaler
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        self.global_step = 0
        self.best_val_score = -float("inf")

    def _encode_pair(
        self,
        code_ids: torch.Tensor,
        code_mask: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode code and text representations with modality routing if applicable."""
        from model.dual_encoder import DualEncoder

        if isinstance(self.model, DualEncoder):
            code_emb = self.model.encode_code(code_ids, attention_mask=code_mask)
            text_emb = self.model.encode_text(text_ids, attention_mask=text_mask)
        elif isinstance(self.model, SharedEncoder) or hasattr(self.model, "num_modalities"):
            code_emb = self.model(code_ids, attention_mask=code_mask, modality_ids="code")
            text_emb = self.model(text_ids, attention_mask=text_mask, modality_ids="text")
        else:
            code_emb = self.model(code_ids, attention_mask=code_mask)
            text_emb = self.model(text_ids, attention_mask=text_mask)
        return code_emb, text_emb

    def train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, float, float]:
        """Execute a single forward-backward optimization step.

        Args:
            batch: Dictionary with 'code_ids', 'code_mask', 'text_ids', 'text_mask'.

        Returns:
            Tuple of (loss_value, text_to_code_acc, code_to_text_acc).
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        code_ids = batch["code_ids"].to(self.device, non_blocking=True)
        code_mask = batch["code_mask"].to(self.device, non_blocking=True)
        text_ids = batch["text_ids"].to(self.device, non_blocking=True)
        text_mask = batch["text_mask"].to(self.device, non_blocking=True)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            code_emb, text_emb = self._encode_pair(code_ids, code_mask, text_ids, text_mask)
            loss = self.criterion(text_emb, code_emb)

        # Backward pass with scaled gradients
        self.scaler.scale(loss).backward()

        # Unscale before gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # Accuracy computation
        t2c_acc, c2t_acc = self.criterion.compute_accuracy(text_emb.detach(), code_emb.detach())

        return float(loss.item()), t2c_acc, c2t_acc

    @torch.no_grad()
    def evaluate(self, max_val_batches: int | None = 50) -> dict[str, float]:
        """Evaluate model on validation split.

        Args:
            max_val_batches: Cap number of validation batches for fast evaluation.

        Returns:
            Dictionary of validation metrics (val_loss, val_t2c_acc, val_mrr).
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_t2c_acc = 0.0
        mrr_sum = 0.0
        num_batches = 0

        for i, batch in enumerate(self.val_loader):
            if max_val_batches and i >= max_val_batches:
                break

            code_ids = batch["code_ids"].to(self.device, non_blocking=True)
            code_mask = batch["code_mask"].to(self.device, non_blocking=True)
            text_ids = batch["text_ids"].to(self.device, non_blocking=True)
            text_mask = batch["text_mask"].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                code_emb, text_emb = self._encode_pair(code_ids, code_mask, text_ids, text_mask)
                loss = self.criterion(text_emb, code_emb)


            total_loss += float(loss.item())

            # Compute in-batch MRR: query i ground truth is candidate i
            sim_matrix = torch.matmul(text_emb, code_emb.T)  # (B, B)
            targets = torch.arange(sim_matrix.shape[0], device=self.device)
            # Find rank of target score in each row
            target_scores = sim_matrix[targets, targets].unsqueeze(1)
            ranks = (sim_matrix > target_scores).sum(dim=1) + 1  # 1-based rank
            reciprocal_ranks = 1.0 / ranks.float()
            mrr_sum += float(reciprocal_ranks.mean().item())

            t2c_acc, _ = self.criterion.compute_accuracy(text_emb, code_emb)
            total_t2c_acc += t2c_acc
            num_batches += 1

        if num_batches == 0:
            return {}

        return {
            "val_loss": total_loss / num_batches,
            "val_t2c_acc": total_t2c_acc / num_batches,
            "val_mrr": mrr_sum / num_batches,
        }

    def save_checkpoint(self, path: Path, epoch: int, is_best: bool = False) -> None:
        """Save training checkpoint to disk."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_val_score,
            "config": self.config,
        }
        torch.save(checkpoint, str(path))
        if is_best:
            torch.save(checkpoint, str(self.best_checkpoint_path))

    def train(self) -> dict[str, float]:
        """Execute full training loop across configured epochs / steps."""
        self.console.print(f"[bold green]Starting Training on {self.device.type.upper()}[/bold green]")
        total_params, trainable_params = self.model.get_num_params()
        self.console.print(f"Model Parameters: {trainable_params:,} trainable / {total_params:,} total (~{total_params / 1e6:0.2f}M)")

        start_time = time.time()

        for epoch in range(1, self.max_epochs + 1):
            self.console.print(f"\n[bold cyan]Epoch {epoch}/{self.max_epochs}[/bold cyan]")
            epoch_loss = 0.0
            epoch_steps = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(f"Training Epoch {epoch}", total=len(self.train_loader))

                for batch in self.train_loader:
                    self.global_step += 1
                    loss, t2c_acc, c2t_acc = self.train_step(batch)

                    epoch_loss += loss
                    epoch_steps += 1
                    progress.update(task, advance=1)

                    # Step logging
                    if self.global_step % self.log_every_steps == 0:
                        current_lr = self.scheduler.get_last_lr()[0]
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
                                self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt",
                                epoch=epoch,
                                is_best=True,
                            )
                            self.console.print(f" [bold green]✓ New best val MRR: {val_mrr:0.4f} saved[/bold green]")

                    if self.global_step >= self.max_steps:
                        break

            # Epoch end validation
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            self.console.print(f"Epoch {epoch} finished — Avg Loss: {avg_epoch_loss:0.4f}")

            if self.val_loader:
                val_metrics = self.evaluate(max_val_batches=None)  # full validation
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"epoch_{k}", v, step=epoch)

                val_mrr = val_metrics.get("val_mrr", 0.0)
                is_best = val_mrr > self.best_val_score
                if is_best:
                    self.best_val_score = val_mrr
                    self.console.print(f"[bold green]★ New Best Model: MRR = {val_mrr:0.4f}[/bold green]")

                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
                    epoch=epoch,
                    is_best=is_best,
                )

            if self.global_step >= self.max_steps:
                self.console.print("[yellow]Reached max_steps limit. Concluding training.[/yellow]")
                break

        total_duration = time.time() - start_time
        self.console.print(f"\n[bold green]Training Completed in {total_duration / 60:0.2f} minutes.[/bold green]")
        return {"best_val_mrr": self.best_val_score}

