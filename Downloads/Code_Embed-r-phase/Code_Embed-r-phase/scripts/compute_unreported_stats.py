"""Compute within-split duplicates, and kept-vs-dropped length & repository distribution using uniform random sampling."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.preprocess import RAW_DIR

print("=" * 70)
print("COMPUTING WITHIN-SPLIT DUPLICATES & KEPT-VS-DROPPED DISTRIBUTIONS")
print("=" * 70)

# 1. Within-split duplicate code & query counts
duplicate_stats = {}
for split in ["train", "validation", "test"]:
    df = pd.read_parquet(f"data/processed_clean_v2/{split}.parquet")
    n_total = len(df)
    n_unique_code = df["code"].nunique()
    n_unique_query = df["docstring"].nunique()

    dup_code_count = n_total - n_unique_code
    dup_query_count = n_total - n_unique_query

    duplicate_stats[split] = {
        "total_rows": n_total,
        "unique_code": n_unique_code,
        "duplicate_code_count": dup_code_count,
        "duplicate_code_pct": float(dup_code_count / n_total * 100),
        "unique_queries": n_unique_query,
        "duplicate_query_count": dup_query_count,
        "duplicate_query_pct": float(dup_query_count / n_total * 100),
    }
    print(
        f"[{split.upper()}] Rows: {n_total:,} | Dup Code: {dup_code_count:,} ({duplicate_stats[split]['duplicate_code_pct']:.2f}%) | Dup Query: {dup_query_count:,} ({duplicate_stats[split]['duplicate_query_pct']:.2f}%)"
    )

# 2. Kept vs Dropped Comparison on Train split
# Load raw train data
raw_files = list(RAW_DIR.glob("train/*.parquet")) + list(
    RAW_DIR.glob("train/*.jsonl.gz")
)
print(f"\nLoading raw train data from {RAW_DIR}/train/...")
dfs = []
for f in raw_files:
    if f.suffix == ".parquet":
        dfs.append(pd.read_parquet(f))
    elif ".jsonl" in f.name:
        dfs.append(pd.read_json(f, lines=True))

if dfs:
    raw_train = pd.concat(dfs, ignore_index=True)
else:
    # If raw files were parquet in raw dir
    raw_train = (
        pd.read_parquet("data/raw/train.parquet")
        if Path("data/raw/train.parquet").exists()
        else None
    )

clean_train = pd.read_parquet("data/processed_clean_v2/train.parquet")

if raw_train is not None:
    print(
        f"Raw train total: {len(raw_train):,}, Clean train total: {len(clean_train):,}"
    )

    # Identify dropped by matching raw index / identifier or set difference
    # In CodeSearchNet: columns include 'func_code_string', 'func_documentation_string', 'repo', 'path'
    repo_col = (
        "repo"
        if "repo" in raw_train.columns
        else ("repository_name" if "repository_name" in raw_train.columns else None)
    )
    code_col = (
        "func_code_string"
        if "func_code_string" in raw_train.columns
        else ("code" if "code" in raw_train.columns else None)
    )

    # Find dropped samples
    # A fast identification: clean_train has 'code' and 'docstring'
    clean_code_set = set(clean_train["docstring"].values)
    doc_col = (
        "func_documentation_string"
        if "func_documentation_string" in raw_train.columns
        else ("docstring" if "docstring" in raw_train.columns else None)
    )

    is_kept = raw_train[doc_col].isin(clean_code_set)
    kept_df = raw_train[is_kept]
    dropped_df = raw_train[~is_kept]

    print(
        f"Identified {len(kept_df):,} kept and {len(dropped_df):,} dropped samples in raw train."
    )

    # Sample uniformly with fixed random seed (NOT head(10000))
    n_sample = min(10000, len(kept_df), len(dropped_df))
    sampled_kept = kept_df.sample(n_sample, random_state=42)
    sampled_dropped = dropped_df.sample(n_sample, random_state=42)

    # Compute token lengths
    kept_lengths = sampled_kept[code_col].apply(lambda s: len(str(s).split())).values
    dropped_lengths = (
        sampled_dropped[code_col].apply(lambda s: len(str(s).split())).values
    )

    len_stats = {
        "sample_size": n_sample,
        "sampling_method": "uniform_random_sample_seed_42",
        "kept": {
            "mean": float(np.mean(kept_lengths)),
            "std": float(np.std(kept_lengths)),
            "p10": float(np.percentile(kept_lengths, 10)),
            "p25": float(np.percentile(kept_lengths, 25)),
            "median": float(np.median(kept_lengths)),
            "p75": float(np.percentile(kept_lengths, 75)),
            "p90": float(np.percentile(kept_lengths, 90)),
        },
        "dropped": {
            "mean": float(np.mean(dropped_lengths)),
            "std": float(np.std(dropped_lengths)),
            "p10": float(np.percentile(dropped_lengths, 10)),
            "p25": float(np.percentile(dropped_lengths, 25)),
            "median": float(np.median(dropped_lengths)),
            "p75": float(np.percentile(dropped_lengths, 75)),
            "p90": float(np.percentile(dropped_lengths, 90)),
        },
    }

    # Repository distribution
    if repo_col:
        top_kept_repos = sampled_kept[repo_col].value_counts().head(10).to_dict()
        top_dropped_repos = sampled_dropped[repo_col].value_counts().head(10).to_dict()
    else:
        top_kept_repos = {}
        top_dropped_repos = {}

    repo_stats = {
        "top_kept_repos": top_kept_repos,
        "top_dropped_repos": top_dropped_repos,
    }
else:
    len_stats = {}
    repo_stats = {}

output_report = {
    "within_split_duplicates": duplicate_stats,
    "length_distribution_comparison": len_stats,
    "repository_distribution_comparison": repo_stats,
}

Path("reports/unreported_data_hygiene_stats.json").write_text(
    json.dumps(output_report, indent=2)
)
print("Saved reports/unreported_data_hygiene_stats.json")
