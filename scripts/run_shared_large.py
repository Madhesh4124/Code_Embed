import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from omegaconf import OmegaConf
from rich.console import Console
import torch

from tokenizer.tokenizer import CodeEmbedTokenizer
from data.dataset import create_dataloader
from model.shared_encoder import SharedEncoder
from training.trainer import ContrastiveTrainer

console = Console()

def run_training(config_path: str):
    console.print(f"[bold cyan]Loading Configuration from {config_path}[/bold cyan]")
    cfg = OmegaConf.load(config_path)

    console.print("[bold cyan]Loading Tokenizer...[/bold cyan]")
    tokenizer = CodeEmbedTokenizer(cfg.data.tokenizer_path)

    console.print("[bold cyan]Building DataLoaders...[/bold cyan]")
    train_loader = create_dataloader(
        split="train",
        tokenizer=tokenizer,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    val_loader = create_dataloader(
        split="validation",
        tokenizer=tokenizer,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    console.print(f"[bold magenta]Initializing Shared Large Encoder (54M parameters)[/bold magenta]")
    model = SharedEncoder(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
        num_modalities=cfg.model.num_modalities
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"[bold green]Total Parameters: {total_params:,}[/bold green]")

    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg
    )

    console.print("[bold green]Starting Training Phase 6: Shared Large Ablation...[/bold green]")
    trainer.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/shared_large.yaml")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
        
    run_training(args.config)

if __name__ == "__main__":
    main()
