"""Phase 6/8: UMAP Embedding Visualization.

Compresses high-dimensional Code and Text embeddings down to 2D
and visualizes their alignment using UMAP.
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

try:
    import umap
    import matplotlib.pyplot as plt
except ImportError:
    print("Missing requirements! Please run: uv pip install umap-learn matplotlib")
    sys.exit(1)

from data.dataset import create_dataloader
from tokenizer.tokenizer import CodeEmbedTokenizer
from model.dual_encoder import DualEncoder

console = Console()

def main():
    parser = argparse.ArgumentParser(description="UMAP Visualization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of pairs to plot")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[bold cyan]Device:[/] {device}")
    
    # 1. Load Data
    parquet_path = Path("data/processed") / f"{args.split}.parquet"
    df = pd.read_parquet(parquet_path)
    
    # Sample aligned pairs
    np.random.seed(42)
    indices = np.random.choice(len(df), size=args.sample_size, replace=False)
    codes = df.iloc[indices]["code"].tolist()
    queries = df.iloc[indices]["docstring"].tolist()
    
    # 2. Load Model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    
    model = DualEncoder(
        vocab_size=cfg.get("vocab_size", 16000),
        d_model=cfg.get("d_model", 256),
        n_layers=cfg.get("n_layers", 3),
        n_heads=cfg.get("n_heads", 8),
        d_ff=cfg.get("d_ff", 1024),
        max_seq_len=cfg.get("max_seq_len", 256),
        dropout=0.0
    ).to(device)
    
    state_dict = checkpoint["model_state_dict"]
    clean_state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()
    
    tokenizer = CodeEmbedTokenizer("tokenizer/tokenizer.json")
    
    # 3. Encode
    console.print(f"[yellow]Encoding {args.sample_size} code-docstring pairs...[/yellow]")
    with torch.no_grad():
        code_batch = tokenizer.encode(codes, max_length=256, padding=True, truncation=True, modality="code")
        code_batch = {k: v.to(device) for k, v in code_batch.items()}
        code_embs = model.encode_code(code_batch).cpu().numpy()
        
        query_batch = tokenizer.encode(queries, max_length=256, padding=True, truncation=True, modality="text")
        query_batch = {k: v.to(device) for k, v in query_batch.items()}
        query_embs = model.encode_text(query_batch).cpu().numpy()
        
    # 4. UMAP Projection
    console.print("[yellow]Running UMAP projection (this might take a moment)...[/yellow]")
    # Combine embeddings for a shared projection space
    all_embs = np.vstack([code_embs, query_embs])
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(all_embs)
    
    code_2d = embedding_2d[:args.sample_size]
    query_2d = embedding_2d[args.sample_size:]
    
    # 5. Plotting
    console.print("[cyan]Generating plot...[/cyan]")
    plt.figure(figsize=(10, 8))
    
    # Plot points
    plt.scatter(code_2d[:, 0], code_2d[:, 1], c='blue', label='Code', alpha=0.5, s=20)
    plt.scatter(query_2d[:, 0], query_2d[:, 1], c='red', label='Docstrings', alpha=0.5, s=20)
    
    # Draw lines between aligned pairs (just a few so it's not a complete mess)
    lines_to_draw = min(50, args.sample_size)
    for i in range(lines_to_draw):
        plt.plot([code_2d[i, 0], query_2d[i, 0]], 
                 [code_2d[i, 1], query_2d[i, 1]], 
                 'k-', alpha=0.2, linewidth=0.5)
                 
    plt.title('UMAP Projection of Code and Docstring Embeddings')
    plt.legend()
    plt.tight_layout()
    
    output_path = "umap_visualization.png"
    plt.savefig(output_path, dpi=300)
    console.print(f"[bold green]Success! Visualization saved to {output_path}[/bold green]")

if __name__ == "__main__":
    main()
