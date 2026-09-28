"""Phase 6/8: Qualitative Evaluation and Failure Analysis.

This script loads a trained model, evaluates it on a small sample,
and extracts the WORST performing queries (where the correct code was ranked lowest).
It prints the docstring and the true code side-by-side to understand WHY it failed.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from data.dataset import create_dataloader
from tokenizer.tokenizer import CodeEmbedTokenizer
from model.dual_encoder import DualEncoder

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Failure Analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Device:[/] {device}")
    
    # 1. Load Data
    parquet_path = Path("data/processed") / f"{args.split}.parquet"
    df = pd.read_parquet(parquet_path)
    codes = df["code"].tolist()
    queries = df["docstring"].tolist()
    
    # 2. Sample
    np.random.seed(42)
    query_indices = np.random.choice(len(queries), size=args.sample_size, replace=False)
    eval_queries = [queries[i] for i in query_indices]
    
    # 3. Load Model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    if not cfg:
        console.print("[red]Could not load config from checkpoint![/red]")
        return
        
    model = DualEncoder(
        vocab_size=cfg.get("vocab_size", 16000),
        d_model=cfg.get("d_model", 256),
        n_layers=cfg.get("n_layers", 3),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 1024),
        max_seq_len=cfg.get("max_seq_len", 256),
        dropout=0.0
    ).to(device)
    
    # Strip "_orig_mod." or "module." if saved from DDP/DataParallel/Compile
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "").replace("module.", "")
        clean_state_dict[clean_k] = v
    model.load_state_dict(clean_state_dict)
    model.eval()
    
    tokenizer = CodeEmbedTokenizer("tokenizer/tokenizer.json")
    
    # 4. Encode
    console.print("[yellow]Encoding...[/yellow]")
    with torch.no_grad():
        # Quick and dirty encoding for demo (in production use DataLoader)
        # Using a small batch loop to avoid OOM
        code_embs = []
        for i in range(0, len(codes), 256):
            batch = tokenizer.encode(codes[i:i+256], max_length=256, padding=True, truncation=True, modality="code")
            batch = {k: v.to(device) for k, v in batch.items()}
            code_embs.append(model.encode_code(batch).cpu().numpy())
        code_embs = np.vstack(code_embs)
        
        query_embs = []
        for i in range(0, len(eval_queries), 256):
            batch = tokenizer.encode(eval_queries[i:i+256], max_length=256, padding=True, truncation=True, modality="text")
            batch = {k: v.to(device) for k, v in batch.items()}
            query_embs.append(model.encode_text(batch).cpu().numpy())
        query_embs = np.vstack(query_embs)
        
    # 5. Compute Similarities and Find Failures
    similarities = query_embs @ code_embs.T
    
    failures = []
    for i, sim_row in enumerate(similarities):
        target_idx = query_indices[i]
        sorted_indices = np.argsort(sim_row)[::-1]
        rank = np.where(sorted_indices == target_idx)[0][0] + 1
        
        if rank > 50: # Only care about bad failures
            failures.append((rank, target_idx, i, sorted_indices[0])) # (Rank, True Code Idx, Query Idx, Top predicted Code Idx)
            
    # Sort by worst rank descending
    failures.sort(key=lambda x: x[0], reverse=True)
    
    console.print(f"\n[bold red]Found {len(failures)} significant failures (Rank > 50) out of {args.sample_size} queries.[/bold red]")
    
    if not failures:
        console.print("[bold green]Your model is too good! No major failures found.[/bold green]")
        return
        
    # Print top 3 worst failures
    for rank, true_code_idx, q_idx, top_pred_idx in failures[:3]:
        console.rule(f"[bold red]FAILURE CASE - Rank: {rank}[/bold red]")
        console.print(f"[bold cyan]Query (Docstring):[/bold cyan]\n{eval_queries[q_idx]}\n")
        
        true_code = codes[true_code_idx]
        wrong_code = codes[top_pred_idx]
        
        console.print(Panel(Syntax(true_code, "python", theme="monokai", line_numbers=True), title="[bold green]True Code (Model Missed This)[/bold green]"))
        console.print(Panel(Syntax(wrong_code, "python", theme="monokai", line_numbers=True), title="[bold red]Model's #1 Prediction (Model Got Tricked By This)[/bold red]"))
        console.print("\n")

if __name__ == "__main__":
    main()
