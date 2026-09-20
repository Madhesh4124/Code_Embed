"""Compute frozen pre-registered vocabulary overlap stratification on clean validation and test splits.

Governed by PROTOCOL.md Section 3.2:
- IDF computed on clean train split.
- Stopwords from data/stopwords.json.
- Frozen bins:
  - Zero-Overlap: Coverage == 0.0
  - Low-Overlap:  0.0 < Coverage <= 0.30
  - High-Overlap: Coverage > 0.30
- Reports BM25 performance across all bins for validation and test.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import PROCESSED_DIR
from evaluation.metrics import mrr, recall_at_k
from retrieval.bm25 import BM25Retriever, tokenize_code, tokenize_text

print("=" * 70)
print("PROTOCOL v1.1 VOCABULARY OVERLAP STRATIFICATION (TRAIN IDF)")
print("=" * 70)

with open("data/stopwords.json") as f:
    stopwords = set(json.load(f))

# 1. Compute IDF strictly on clean train split
print("\n>>> Computing IDF across 360,957 clean train functions...")
train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
N_train = len(train_df)

doc_freqs = Counter()
# Sample or batch count across train for fast IDF calculation
for code_str in train_df["code"]:
    terms = set(tokenize_code(code_str))
    for t in terms:
        if t not in stopwords and len(t) > 1:
            doc_freqs[t] += 1

print(f"[OK] Vocabulary: {len(doc_freqs):,} unique non-stopword terms from train.")


def get_idf(term: str) -> float:
    n = doc_freqs.get(term, 0)
    return float(np.log((N_train - n + 0.5) / (n + 0.5) + 1.0))


def compute_coverage(docstring: str, code: str) -> float:
    q_tokens = [
        t for t in tokenize_text(docstring) if t not in stopwords and len(t) > 1
    ]
    if not q_tokens:
        return 0.0
    c_tokens = set(tokenize_code(code))

    total_idf = sum(get_idf(t) for t in q_tokens)
    if total_idf <= 0.0:
        return 0.0

    matched_idf = sum(get_idf(t) for t in q_tokens if t in c_tokens)
    return float(matched_idf / total_idf)


def assign_bin(coverage: float) -> str:
    if coverage <= 1e-9:
        return "Zero"
    elif coverage <= 0.30:
        return "Low"
    else:
        return "High"


# 2. Stratify validation and test
for split in ["validation", "test"]:
    print("\n" + "=" * 70)
    print(f"STRATIFICATION & BM25 PERFORMANCE: {split.upper()} SPLIT")
    print("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / f"{split}.parquet")
    coverages = [compute_coverage(row.docstring, row.code) for row in df.itertuples()]
    bins = [assign_bin(c) for c in coverages]

    df["coverage"] = coverages
    df["overlap_bin"] = bins

    # Run BM25 on this split
    print(f">>> Evaluating BM25 retrieval on {split} across overlap bins (method='rank_bm25')...")
    bm25 = BM25Retriever(k1=1.5, b=0.75, method="rank_bm25")
    bm25.index(df["code"].tolist())
    all_ranks = bm25.get_ranks(
        df["docstring"].tolist(), list(range(len(df))), max_candidates=1000
    )
    df["rank"] = all_ranks

    # Save stratified parquet with ranks
    out_file = PROCESSED_DIR / f"{split}_stratified.parquet"
    df.to_parquet(out_file, index=False)
    print(f"[OK] Saved {split} stratified dataset to {out_file}")

    # Report per-bin metrics
    print(
        f"\n{'Bin':<8} | {'Range':<16} | {'Count':<8} | {'% of Split':<12} | {'MRR':<10} | {'Recall@1':<10} | {'Recall@10':<10}"
    )
    print("-" * 85)
    for b, rng_str in [
        ("Zero", "Coverage == 0.0"),
        ("Low", "0.0 < Cov <= 0.30"),
        ("High", "Coverage > 0.30"),
    ]:
        sub_df = df[df["overlap_bin"] == b]
        cnt = len(sub_df)
        pct = cnt / len(df) * 100
        sub_ranks = sub_df["rank"].to_numpy()
        bin_mrr = mrr(sub_ranks)
        bin_r1 = recall_at_k(sub_ranks, k=1)
        bin_r10 = recall_at_k(sub_ranks, k=10)
        print(
            f"{b:<8} | {rng_str:<16} | {cnt:>7,} | {pct:>10.2f}% | {bin_mrr:>9.4f} | {bin_r1:>9.4f} | {bin_r10:>9.4f}"
        )

    all_r = df["rank"].to_numpy()
    print("-" * 85)
    print(
        f"{'OVERALL':<8} | {'All Queries':<16} | {len(df):>7,} | {'100.00%':>10} | {mrr(all_r):>9.4f} | {recall_at_k(all_r, k=1):>9.4f} | {recall_at_k(all_r, k=10):>9.4f}"
    )

print("\n" + "=" * 70)
print("PHASE R3 PILOT GATE CRITERIA (VALIDATION REFERENCE)")
print("=" * 70)
val_strat = pd.read_parquet(PROCESSED_DIR / "validation_stratified.parquet")
val_overall_mrr = mrr(val_strat["rank"].to_numpy())
val_low_mrr = mrr(val_strat[val_strat["overlap_bin"] == "Low"]["rank"].to_numpy())
val_zero_mrr = mrr(val_strat[val_strat["overlap_bin"] == "Zero"]["rank"].to_numpy())

gate_threshold = 0.75 * val_overall_mrr
print(f"  Validation BM25 Overall MRR:         {val_overall_mrr:0.4f}")
print(f"  Primary Pilot Gate Threshold (0.75x): {gate_threshold:0.4f}")
print(f"  Validation BM25 Low-Overlap MRR:     {val_low_mrr:0.4f}")
print(f"  Validation BM25 Zero-Overlap MRR:    {val_zero_mrr:0.4f}")
print("=" * 70)
