"""Phase 5 Step 6: Mine and save train_hard_negatives.pt.

This script is the FIRST step to run before training with hard negatives.
It mines BM25 hard negatives for all training queries and saves the index matrix
to `data/processed/train_hard_negatives.pt`.

Run this BEFORE `scripts/run_hard_negatives.py`.

Usage:
    uv run python scripts/mine_hard_negatives.py
    uv run python scripts/mine_hard_negatives.py --k 7 --workers 4
"""

# IMPORTANT: The if __name__ == '__main__' guard must wrap ALL executor code.
# On Windows, the 'spawn' process start method re-imports this module in each
# child process — without this guard, the executor is instantiated recursively
# and the process hangs silently. We use ThreadPoolExecutor (not Process-based)
# in training/hard_negatives.py, but this guard is kept as best practice.
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")

    from training.hard_negatives import generate_train_hard_negatives

    parser = argparse.ArgumentParser(
        description="Mine BM25 hard negatives for Phase 5 training."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=7,
        help="Number of hard negatives per training query (default: 7).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to mine from (default: 'train').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train_hard_negatives.pt",
        help="Output .pt filename saved to data/processed/ (default: train_hard_negatives.pt).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Queries per sparse matmul chunk evaluated simultaneously (default: 500).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel CPU worker threads (default: 4).",
    )
    args = parser.parse_args()

    print("=== Phase 5: Hard Negative Mining ===", flush=True)
    print(f"Split:      {args.split}", flush=True)
    print(f"K:          {args.k}", flush=True)
    print(f"Batch size: {args.batch_size} queries/chunk", flush=True)
    print(f"Output:     data/processed/{args.output}", flush=True)
    print(f"Platform:   {sys.platform}", flush=True)
    print(flush=True)

    output_path = generate_train_hard_negatives(
        k=args.k,
        split=args.split,
        output_filename=args.output,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    print(f"\n[OK] Done! Hard negatives saved to: {output_path}", flush=True)
    print(
        "Next step: run `uv run python scripts/run_hard_negatives.py --config configs/shared_hard.yaml`",
        flush=True,
    )
