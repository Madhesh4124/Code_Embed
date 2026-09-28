import sys, os, gc, torch, numpy as np
import argparse

# Usage: python scripts/evaluate_sweep.py --sweep-dir checkpoints/ablation_basic_sweep

def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a sweep directory.")
    parser.add_argument("--sweep-dir", type=str, required=True, help="Path to the sweep directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of queries to evaluate (0 for all)")
    args = parser.parse_args()

    SWEEP_DIR = args.sweep_dir
    device = torch.device(args.device)

    from tokenizer.tokenizer import CodeEmbedTokenizer
    from data.dataset import create_dataloader
    from rich.console import Console
    from rich.table import Table

    console = Console()
    
    tokenizer_path = "tokenizer/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        console.print(f"[red]❌ Tokenizer not found at {tokenizer_path}[/red]")
        sys.exit(1)

    tokenizer = CodeEmbedTokenizer(tokenizer_path)
    test_loader = create_dataloader(split="test", tokenizer=tokenizer,
        batch_size=64, shuffle=False, num_workers=0)

    def eval_ckpt(ckpt_path, run_name):
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        m_cfg = ckpt["config"]["model"]
        state = ckpt["model_state_dict"]
        
        # Auto-Detect Architecture from Weights
        is_dual = any(k.startswith("code_encoder.") for k in state.keys())
        pool_strat = "cls" if "pool=cls" in run_name else "mean"
        
        if is_dual:
            from model.dual_encoder import DualEncoder
            model = DualEncoder(
                vocab_size=m_cfg["vocab_size"], d_model=m_cfg["d_model"],
                n_layers=m_cfg.get("n_layers", 4), n_heads=m_cfg["n_heads"],
                d_ff=m_cfg["d_ff"], max_seq_len=m_cfg["max_seq_len"], dropout=0.0
            ).to(device)
        else:
            from model.ablation_encoder import AblationBaseEncoder
            model = AblationBaseEncoder(
                vocab_size=m_cfg["vocab_size"], d_model=m_cfg["d_model"],
                n_layers=m_cfg["n_layers"], n_heads=m_cfg["n_heads"],
                d_ff=m_cfg["d_ff"], max_seq_len=m_cfg["max_seq_len"],
                pooling_strategy=pool_strat
            ).to(device)
            
        model.load_state_dict(state)
        model.eval()

        code_embs, text_embs = [], []
        with torch.no_grad():
            for batch in test_loader:
                if is_dual:
                    code_embs.append(model.encode_code(batch["code_ids"].to(device), batch["code_mask"].to(device)).cpu())
                    text_embs.append(model.encode_text(batch["text_ids"].to(device), batch["text_mask"].to(device)).cpu())
                else:
                    code_embs.append(model(batch["code_ids"].to(device), batch["code_mask"].to(device)).cpu())
                    text_embs.append(model(batch["text_ids"].to(device), batch["text_mask"].to(device)).cpu())
                    
        code_embs = torch.cat(code_embs)
        text_embs = torch.cat(text_embs)

        torch.manual_seed(42)
        idx = torch.randperm(len(text_embs))
        if args.sample_size > 0:
            idx = idx[:args.sample_size]
            
        sim   = torch.matmul(text_embs[idx], code_embs.T)
        ranks = np.array([int((sim[i] > sim[i, gi]).sum()) + 1 for i, gi in enumerate(idx)])
        gc.collect(); torch.cuda.empty_cache()
        
        return ckpt.get("epoch", "?"), {
            "MRR":     float(np.mean(1.0 / ranks)),
            "R@1":     float(np.mean(ranks <= 1)),
            "R@5":     float(np.mean(ranks <= 5)),
            "R@10":    float(np.mean(ranks <= 10)),
            "NDCG@10": float(np.mean([1/np.log2(r+1) if r <= 10 else 0 for r in ranks])),
        }

    all_results = {}

    if not os.path.exists(SWEEP_DIR):
        console.print(f"[red]❌ Sweep directory not found: {SWEEP_DIR}[/red]")
    else:
        for run_dir in sorted(os.listdir(SWEEP_DIR)):
            run_path = os.path.join(SWEEP_DIR, run_dir)
            if not os.path.isdir(run_path): continue
            
            best_pt = None
            for f in os.listdir(run_path):
                if f.startswith("best_") and f.endswith(".pt"):
                    best_pt = os.path.join(run_path, f)
                    break
                    
            if not best_pt:
                console.print(f"[yellow]⚠ {run_dir}: no checkpoint found[/yellow]")
                continue
                
            console.print(f"[cyan]Evaluating {run_dir}...[/cyan]")
            
            try:
                epoch, metrics = eval_ckpt(best_pt, run_dir)
                all_results[run_dir] = (epoch, metrics)
                console.print(f"  Epoch {epoch} → MRR={metrics['MRR']:.4f}")
            except Exception as e:
                console.print(f"[red]❌ Corrupted checkpoint skipped ({run_dir}): {e}[/red]")

    if all_results:
        table = Table(title=f"Sweep Results — {os.path.basename(SWEEP_DIR)}", show_header=True, header_style="bold magenta")
        table.add_column("Config",   style="cyan", min_width=40)
        table.add_column("Epoch",    justify="center")
        table.add_column("MRR",      justify="right", style="bold green")
        table.add_column("R@1",      justify="right")
        table.add_column("R@5",      justify="right")
        table.add_column("R@10",     justify="right")
        table.add_column("NDCG@10",  justify="right")

        for name, (ep, m) in sorted(all_results.items(), key=lambda x: x[1][1]["MRR"], reverse=True):
            table.add_row(name, str(ep), f"{m['MRR']:.4f}", f"{m['R@1']:.4f}",
                          f"{m['R@5']:.4f}", f"{m['R@10']:.4f}", f"{m['NDCG@10']:.4f}")

        console.print(table)
    else:
        console.print("[red]No valid checkpoints could be evaluated![/red]")

if __name__ == "__main__":
    main()
