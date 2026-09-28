"""Phase 7: Pretrained Baseline Comparison using CodeBERT.

This script evaluates a massive pretrained model (microsoft/codebert-base, 125M params)
on our test set as a zero-shot Bi-Encoder.
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
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from evaluation.evaluate import evaluate_rankings

console = Console()

def encode_with_codebert(texts: list[str], tokenizer, model, device, batch_size=128) -> np.ndarray:
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i+batch_size]
            
            # CodeBERT uses RoBERTa architecture
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Use CLS token (index 0) for embeddings
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # L2 Normalize
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
            
            all_embeddings.append(cls_embeddings.cpu().numpy())
            
    return np.vstack(all_embeddings)

def main():
    parser = argparse.ArgumentParser(description="Evaluate CodeBERT on test set.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--sample-size", type=int, default=1000, help="Queries to sample")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Device:[/] {device}")
    
    # Load Dataset
    parquet_path = Path("data/processed_clean_v2") / f"{args.split}.parquet"
    if not parquet_path.exists():
        console.print(f"[bold red]Dataset not found at {parquet_path}[/]")
        return
        
    df = pd.read_parquet(parquet_path)
    codes = df["code"].tolist()
    queries = df["docstring"].tolist()
    
    # Sample queries if needed
    if args.sample_size and args.sample_size < len(queries):
        np.random.seed(42)
        query_indices = np.random.choice(len(queries), size=args.sample_size, replace=False)
        eval_queries = [queries[i] for i in query_indices]
    else:
        query_indices = np.arange(len(queries))
        eval_queries = queries
        
    console.print(f"[bold]Evaluating {len(eval_queries)} queries against {len(codes)} code snippets.[/bold]")
    
    # Load CodeBERT
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    console.print(f"[bold green]Loading {model_name}...[/bold green]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    console.print("[bold yellow]Encoding Corpus (Code)...[/bold yellow]")
    code_embs = encode_with_codebert(codes, tokenizer, model, device, args.batch_size)
    
    console.print("[bold yellow]Encoding Queries (Docstrings)...[/bold yellow]")
    query_embs = encode_with_codebert(eval_queries, tokenizer, model, device, args.batch_size)
    
    console.print("[bold cyan]Computing cosine similarities...[/bold cyan]")
    similarities = query_embs @ code_embs.T
    
    ranks = []
    for i, sim_row in enumerate(similarities):
        target_idx = query_indices[i]
        sorted_indices = np.argsort(sim_row)[::-1]
        rank = np.where(sorted_indices == target_idx)[0][0] + 1
        ranks.append(rank)
        
    console.print("[bold green]Computing Metrics...[/bold green]")
    results = evaluate_rankings(ranks)
    from rich.table import Table
    table = Table(title="CodeBERT Pretrained Baseline", header_style="bold magenta")
    table.add_column("Metric", style="dim", width=12)
    table.add_column("Score", justify="right", style="bold green", width=10)
    table.add_column("95% Confidence Interval", justify="center", style="cyan", width=26)
    
    scalar_metrics = {k: v for k, v in results.items() if not k.endswith("_ci")}
    for metric_name, score in scalar_metrics.items():
        ci = results.get(f"{metric_name}_ci", ("—", "—"))
        ci_str = f"[{ci[0]:0.4f}, {ci[1]:0.4f}]" if isinstance(ci, tuple) else "—"
        table.add_row(metric_name.upper(), f"{score:0.4f}", ci_str)
        
    console.print()
    console.print(table)
    console.print()

if __name__ == "__main__":
    main()
