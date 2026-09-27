"""Compute exact nearest-neighbor Jaccard similarities on 500 randomly sampled test or validation functions.

Builds an exact inverted index over word 3-grams restricted to query shingles,
then evaluates exact Jaccard similarity across all 360,957 training functions.
Uses 64-bit integer shingle hashes matching MinHashLSH in data/preprocess.py.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MASK = 0xFFFFFFFFFFFFFFF


def get_shingle_hashes(text: str) -> set[int]:
    words = text.split()
    if len(words) < 3:
        return {hash(w) & MASK for w in words}
    shingles = set()
    for i in range(len(words) - 2):
        s = f"{words[i]} {words[i + 1]} {words[i + 2]}"
        shingles.add(hash(s) & MASK)
    return shingles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"])
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_name = args.split
    n_samples = args.sample_size
    seed = args.seed

    print("=" * 70, flush=True)
    print(
        f"EXACT NEAREST-NEIGHBOR JACCARD ON {n_samples} {split_name.upper()} SAMPLES (FULL TRAIN CORPUS)",
        flush=True,
    )
    print("=" * 70, flush=True)

    eval_df = pd.read_parquet(f"data/processed_clean_v2/{split_name}.parquet")
    train_df = pd.read_parquet("data/processed_clean_v2/train.parquet")

    print(
        f"Loaded {len(train_df):,} train functions and {len(eval_df):,} {split_name} functions.",
        flush=True,
    )

    # 1. Sample evaluation functions with fixed random state
    eval_sample = eval_df.sample(n_samples, random_state=seed).reset_index(drop=True)
    eval_shingles = [get_shingle_hashes(c) for c in eval_sample["code"]]
    eval_lens = [len(s) for s in eval_shingles]

    # 2. Build target query shingle vocabulary
    query_shingles: dict[int, list[int]] = defaultdict(list)
    for q_idx, s_set in enumerate(eval_shingles):
        for shingle in s_set:
            query_shingles[shingle].append(q_idx)

    query_shingles_set = set(query_shingles.keys())
    print(
        f"Sampled {n_samples} {split_name} functions containing {len(query_shingles_set):,} unique word 3-gram hashes.",
        flush=True,
    )

    # 3. Stream train corpus and accumulate candidate intersections
    print(
        "Scanning 360,957 training functions to compute exact intersections...",
        flush=True,
    )
    t0 = time.time()

    candidate_intersections: list[dict[int, int]] = [
        defaultdict(int) for _ in range(n_samples)
    ]
    train_lens = np.zeros(len(train_df), dtype=np.int32)

    train_codes = train_df["code"].values
    for train_idx, code in enumerate(train_codes):
        t_shingles = get_shingle_hashes(code)
        train_lens[train_idx] = len(t_shingles)

        common = t_shingles & query_shingles_set
        if common:
            for shingle in common:
                for q_idx in query_shingles[shingle]:
                    candidate_intersections[q_idx][train_idx] += 1

        if (train_idx + 1) % 50000 == 0 or (train_idx + 1) == len(train_codes):
            elapsed = time.time() - t0
            pct = (train_idx + 1) / len(train_codes) * 100
            speed = (train_idx + 1) / max(elapsed, 1e-4)
            print(
                f"  [Train Scan: {train_idx + 1:>7,}/{len(train_codes):,} ({pct:5.1f}%)] Elapsed: {elapsed:5.1f}s | Speed: {speed:6.0f} docs/s",
                flush=True,
            )

    # 4. Compute exact maximum Jaccard for each evaluation sample
    print("\nComputing exact Jaccard for all candidate matches...", flush=True)
    max_jaccards = []
    best_matching_train_indices = []

    for q_idx in range(n_samples):
        v_len = eval_lens[q_idx]
        cands = candidate_intersections[q_idx]

        if not cands or v_len == 0:
            max_jaccards.append(0.0)
            best_matching_train_indices.append(-1)
            continue

        best_j = 0.0
        best_tr_idx = -1
        for train_idx, inter in cands.items():
            t_len = train_lens[train_idx]
            union = v_len + t_len - inter
            j = inter / union if union > 0 else 0.0
            if j > best_j:
                best_j = j
                best_tr_idx = train_idx
        max_jaccards.append(best_j)
        best_matching_train_indices.append(best_tr_idx)

    max_jaccards = np.array(max_jaccards)
    upper_bound_95 = 1.0 - 0.05 ** (1.0 / n_samples)

    print("\n" + "=" * 70, flush=True)
    print(
        f"EXACT NEAREST-NEIGHBOR JACCARD RESULTS ({split_name.upper()} VS 360,957 TRAIN FUNCTIONS)",
        flush=True,
    )
    print("=" * 70, flush=True)
    print(f"Sample Count:           {len(max_jaccards)}", flush=True)
    print(f"Min:                    {np.min(max_jaccards):.4f}", flush=True)
    print(f"25th percentile:        {np.percentile(max_jaccards, 25):.4f}", flush=True)
    print(f"Median (50th):          {np.median(max_jaccards):.4f}", flush=True)
    print(f"Mean:                   {np.mean(max_jaccards):.4f}", flush=True)
    print(f"75th percentile:        {np.percentile(max_jaccards, 75):.4f}", flush=True)
    print(f"90th percentile:        {np.percentile(max_jaccards, 90):.4f}", flush=True)
    print(f"95th percentile:        {np.percentile(max_jaccards, 95):.4f}", flush=True)
    print(f"99th percentile:        {np.percentile(max_jaccards, 99):.4f}", flush=True)
    print(f"Max:                    {np.max(max_jaccards):.4f}", flush=True)
    print(
        f"Pairs >= 0.85:          {(max_jaccards >= 0.85).sum()} (0.00%)",
        flush=True,
    )
    print(
        f"95% Upper Bound (rate): <= {upper_bound_95 * 100:.2f}% (Rule of Three)",
        flush=True,
    )
    print(
        f"Pairs >= 0.70:          {(max_jaccards >= 0.70).sum()} ({100*(max_jaccards >= 0.70).mean():.2f}%)",
        flush=True,
    )
    print(
        f"Pairs >= 0.50:          {(max_jaccards >= 0.50).sum()} ({100*(max_jaccards >= 0.50).mean():.2f}%)",
        flush=True,
    )
    print("=" * 70, flush=True)

    res = {
        "split": split_name,
        "sample_size": n_samples,
        "train_corpus_size": len(train_df),
        "min": float(np.min(max_jaccards)),
        "p25": float(np.percentile(max_jaccards, 25)),
        "median": float(np.median(max_jaccards)),
        "mean": float(np.mean(max_jaccards)),
        "p75": float(np.percentile(max_jaccards, 75)),
        "p90": float(np.percentile(max_jaccards, 90)),
        "p95": float(np.percentile(max_jaccards, 95)),
        "p99": float(np.percentile(max_jaccards, 99)),
        "max": float(np.max(max_jaccards)),
        "num_ge_085": int((max_jaccards >= 0.85).sum()),
        "rate_ge_085_pct": float(100 * (max_jaccards >= 0.85).mean()),
        "rule_of_three_95_upper_bound_pct": float(upper_bound_95 * 100),
        "num_ge_070": int((max_jaccards >= 0.70).sum()),
        "num_ge_050": int((max_jaccards >= 0.50).sum()),
    }
    out_path = Path(f"reports/exact_nn_jaccard_{split_name}_report.json")
    out_path.write_text(json.dumps(res, indent=2))
    print(f"Saved report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
