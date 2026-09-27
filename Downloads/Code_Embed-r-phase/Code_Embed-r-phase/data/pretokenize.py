"""Pre-tokenize CodeSearchNet splits into binary tensor files for ultra-fast training.

This script converts raw text parquet splits into pre-tokenized numpy / tensor arrays
saved as compressed memory-mappable or tensor files:
- Eliminates CPU tokenization bottleneck during training (100x+ speedup).
- Saves `code_ids`, `code_mask`, `text_ids`, `text_mask` as uint16 / int32 arrays.
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from tokenizer.tokenizer import CodeEmbedTokenizer

PROCESSED_DIR = Path("data/processed_clean_v2")


def pretokenize_split(
    split: str,
    tokenizer: CodeEmbedTokenizer,
    data_dir: Path = PROCESSED_DIR,
    max_length: int = 256,
    batch_size: int = 4096,
) -> Path:
    """Pre-tokenize a dataset split in fast batches and save as a .pt tensor dictionary.

    Args:
        split: Split name ('train', 'validation', 'test').
        tokenizer: Initialized CodeEmbedTokenizer.
        data_dir: Directory containing parquet files and output location.
        max_length: Fixed sequence length (default: 256).
        batch_size: Batch size for HuggingFace fast Rust tokenizer encode_batch.

    Returns:
        Path to the saved pre-tokenized .pt file.
    """
    console = Console()
    data_dir = Path(data_dir)
    input_file = data_dir / f"{split}.parquet"
    output_file = data_dir / f"{split}_tokenized.pt"

    if not input_file.exists():
        raise FileNotFoundError(f"Input parquet split not found at: {input_file}")

    console.print(
        f"\n[bold cyan]Pre-tokenizing split: {split}[/bold cyan] ({input_file})"
    )
    df = pd.read_parquet(input_file)
    total_samples = len(df)
    console.print(f"Total samples to process: [bold]{total_samples:,}[/bold]")

    codes = df["code"].tolist()
    docstrings = df["docstring"].tolist()

    all_code_ids = []
    all_code_mask = []
    all_text_ids = []
    all_text_mask = []

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Tokenizing {split}", total=total_samples)

        for i in range(0, total_samples, batch_size):
            chunk_codes = codes[i : i + batch_size]
            chunk_docstrings = docstrings[i : i + batch_size]

            # Fast Rust batch encoding with modality tags
            code_enc = tokenizer.encode(
                chunk_codes,
                max_length=max_length,
                padding=True,
                truncation=True,
                modality="code",
            )
            text_enc = tokenizer.encode(
                chunk_docstrings,
                max_length=max_length,
                padding=True,
                truncation=True,
                modality="text",
            )

            # Store as int16/int32 numpy arrays to minimize memory footprint
            all_code_ids.append(code_enc["input_ids"].to(torch.int32))
            all_code_mask.append(code_enc["attention_mask"].to(torch.int8))
            all_text_ids.append(text_enc["input_ids"].to(torch.int32))
            all_text_mask.append(text_enc["attention_mask"].to(torch.int8))

            progress.update(task, advance=len(chunk_codes))

    console.print(f"[cyan]Concatenating tensors for {split}...[/cyan]")
    final_dict = {
        "code_ids": torch.cat(all_code_ids, dim=0),
        "code_mask": torch.cat(all_code_mask, dim=0),
        "text_ids": torch.cat(all_text_ids, dim=0),
        "text_mask": torch.cat(all_text_mask, dim=0),
    }

    console.print(
        f"[cyan]Saving pre-tokenized tensor artifact to {output_file}...[/cyan]"
    )
    torch.save(final_dict, output_file)

    duration = time.time() - start_time
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    console.print(
        f"[bold green]✓ Successfully pre-tokenized {split} ({total_samples:,} samples) in {duration:0.2f}s "
        f"({total_samples / duration:0.0f} samples/s) — Size: {file_size_mb:0.1f} MB[/bold green]"
    )
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize CodeSearchNet dataset splits."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed_clean_v2",
        help="Directory containing parquet splits (default: data/processed_clean_v2).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to tokenize.",
    )
    parser.add_argument(
        "--max-length", type=int, default=256, help="Maximum sequence length."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8192, help="Batch size for tokenizer."
    )
    args = parser.parse_args()

    tokenizer = CodeEmbedTokenizer()
    for s in args.splits:
        pretokenize_split(
            split=s,
            tokenizer=tokenizer,
            data_dir=Path(args.data_dir),
            max_length=args.max_length,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
