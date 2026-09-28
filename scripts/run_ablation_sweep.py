import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from omegaconf import OmegaConf
from rich.console import Console
from tokenizer.tokenizer import CodeEmbedTokenizer
from data.dataset import create_dataloader
from model.ablation_encoder import AblationDualEncoder
from training.trainer import ContrastiveTrainer
console = Console()

def main():
    cfg = OmegaConf.load("configs/ablation_sweep.yaml")
    tokenizer = CodeEmbedTokenizer(cfg.data.tokenizer_path)
    train_loader = create_dataloader(split="train", tokenizer=tokenizer, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = create_dataloader(split="validation", tokenizer=tokenizer, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    temperatures = [0.01, 0.07, 0.1]
    poolings = ["mean", "cls"]
    results = {}

    for pool in poolings:
        for t in temperatures:
            name = f"pool={pool}_tau={t}"
            console.print(f"\\n[bold magenta]=== Starting Run: {name} ===[/bold magenta]")
            
            cfg.training.temperature = t
            cfg.mlflow.run_name = name
            cfg.checkpoint_dir = f"checkpoints/ablation_dual_sweep/{name}"
            
            model = AblationDualEncoder(
                vocab_size=cfg.model.vocab_size, d_model=cfg.model.d_model, n_layers=cfg.model.n_layers,
                n_heads=cfg.model.n_heads, d_ff=cfg.model.d_ff, max_seq_len=cfg.model.max_seq_len, pooling_strategy=pool
            )
            
            trainer = ContrastiveTrainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                config=cfg
            )
            trainer.train()
            results[name] = trainer.best_val_score
            
    console.print("\\n[bold green]=== Sweep Results ===[/bold green]")
    for n, s in sorted(results.items(), key=lambda x: x[1], reverse=True):
        console.print(f"{n}: MRR = {s:.4f}")

if __name__ == "__main__":
    main()
