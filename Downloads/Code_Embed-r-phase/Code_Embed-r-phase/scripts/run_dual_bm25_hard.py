"""Phase 5 Extension: Train Dual Encoder with BM25 Hard Negatives.

This script coordinates:
1. Mining hard negatives using BM25.
2. Fine-tuning the Dual Encoder using the InfoNCEWithHardNegatives loss.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel
import torch
import mlflow

# We reuse the training logic from run_dual_hard.py
from scripts.run_dual_hard import run_training
from training.bm25_hard_negatives import mine_bm25_hard_negatives

console = Console()

def run_mining(cfg) -> None:
    """Run BM25 hard negative mining."""
    console.print("\n[cyan]Stage 1: Mining BM25 Hard Negatives[/cyan]")
    hn_cfg = cfg.hard_negatives
    
    os.makedirs(os.path.dirname(hn_cfg.index_path), exist_ok=True)
    
    # We only mine if it doesn't already exist to save time
    if os.path.exists(hn_cfg.index_path):
        console.print(f"[yellow]Hard negatives already exist at {hn_cfg.index_path}. Skipping mining.[/yellow]")
        return
        
    mine_bm25_hard_negatives(
        output_path=str(hn_cfg.index_path),
        k=int(hn_cfg.k_mined),
    )
    console.print("[green][OK][/green] BM25 Hard negatives mined successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dual Encoder BM25 Hard Negative Fine-Tuning.")
    parser.add_argument("--config",     type=str, default="configs/dual_bm25_hard.yaml", help="Path to config file")
    parser.add_argument("--mine-only",  action="store_true", help="Only run mining stage")
    parser.add_argument("--train-only", action="store_true", help="Only run training stage (index must exist)")
    parser.add_argument("--max-steps",  type=int, default=None, help="Override max_steps")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.max_steps:
        cfg.training.max_steps = args.max_steps

    console.print(Panel("[bold magenta]CodeEmbed — BM25 Hard Negative Training[/bold magenta]"))
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
