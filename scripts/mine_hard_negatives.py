"""Phase R2-C / Phase 5: Mine and save train_hard_negatives.pt.

This script mines BM25 hard negatives for all clean training queries and saves
the index matrix to `data/processed_clean_v2/train_hard_negatives.pt`.

Pre-registered 3-tier false negative filters:
1. Identical docstring intent (docstring_i == docstring_j)
2. Normalized AST skeleton match (>= 20 AST nodes)
3. MinHash 3-gram Jaccard similarity (J >= 0.70)

Usage:
    uv run python scripts/mine_hard_negatives.py
    uv run python scripts/mine_hard_negatives.py --k 7 --workers 4
"""

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

    from training.hard_negatives import DEFAULT_DATA_DIR, generate_train_hard_negatives

    parser = argparse.ArgumentParser(
        description="Mine BM25 hard negatives with 3-tier false negative exclusion filters."
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
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing clean parquet dataset (default: data/processed_clean_v2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="train_hard_negatives.pt",
        help="Output .pt filename saved to data-dir (default: train_hard_negatives.pt).",
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

    data_dir = Path(args.data_dir)
    print("=== Phase R2-C: Hard Negative Mining ===", flush=True)
    print(f"Data Dir:   {data_dir}", flush=True)
    print(f"Split:      {args.split}", flush=True)
    print(f"K:          {args.k}", flush=True)
    print(f"Batch size: {args.batch_size} queries/chunk", flush=True)
    print(f"Output:     {data_dir / args.output}", flush=True)
    print(f"Platform:   {sys.platform}", flush=True)
    print(flush=True)

    output_path = generate_train_hard_negatives(
        k=args.k,
        split=args.split,
        data_dir=data_dir,
        output_filename=args.output,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    print(f"\n[OK] Done! Hard negatives saved to: {output_path}", flush=True)
    print(
        "Next step: run `uv run python scripts/run_hard_negatives.py --config configs/shared_hard_clean.yaml`",
        flush=True,
    )
