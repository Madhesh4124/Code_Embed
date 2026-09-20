"""Script to run like-for-like before/after BM25 benchmark on the EXACT same 19,632 test items.

Isolates the exact query leakage effect by comparing:
- Leaky condition: unstripped raw code (with docstrings embedded) on the 19,632 items
- Clean condition: stripped code (docstrings removed via AST slicing) on the exact same 19,632 items
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import evaluate_rankings
from retrieval.bm25 import BM25Retriever

clean_df = pd.read_parquet("data/processed_clean_v2/test.parquet")
raw_df = pd.read_parquet("data/raw/test.parquet")

# Merge clean_df with raw_df on func_name and docstring to retrieve the exact original unstripped code
# Rename columns to avoid collision
raw_df = raw_df.rename(
    columns={"func_documentation_string": "docstring", "func_code_string": "raw_code"}
)

# Merge to get matching raw_code
merged = (
    pd.merge(
        clean_df.reset_index().rename(columns={"index": "orig_clean_idx"}),
        raw_df[["func_name", "docstring", "raw_code"]].drop_duplicates(
            subset=["func_name", "docstring"]
        ),
        on=["func_name", "docstring"],
        how="inner",
    )
    .sort_values("orig_clean_idx")
    .reset_index(drop=True)
)

print(f"Matched {len(merged):,} of {len(clean_df):,} rows exactly.")
assert len(merged) == len(clean_df), (
    f"Mismatch in row matching: {len(merged)} vs {len(clean_df)}"
)

queries = merged["docstring"].tolist()
gt_indices = list(range(len(queries)))

print("\n" + "=" * 70)
print("LIKE-FOR-LIKE BM25 BENCHMARK ON EXACT SAME 19,632 TEST ITEMS")
print("=" * 70)

# 1. Condition A: Leaky (Docstrings left in code)
print("\n>>> Running Leaky Condition (Docstrings IN Code)...")
t0 = time.time()
retriever_leaky = BM25Retriever(k1=1.5, b=0.75)
retriever_leaky.index(merged["raw_code"].tolist())
ranks_leaky = retriever_leaky.get_ranks(queries, gt_indices, max_candidates=1000)
dur_leaky = time.time() - t0
res_leaky = evaluate_rankings(
    ranks_leaky, ks=[1, 5, 10], bootstrap_resamples=1000, seed=42
)

# 2. Condition B: Clean (Docstrings stripped from code)
print("\n>>> Running Clean Condition (Docstrings STRIPPED from Code)...")
t0 = time.time()
retriever_clean = BM25Retriever(k1=1.5, b=0.75)
retriever_clean.index(merged["code"].tolist())
ranks_clean = retriever_clean.get_ranks(queries, gt_indices, max_candidates=1000)
dur_clean = time.time() - t0
res_clean = evaluate_rankings(
    ranks_clean, ks=[1, 5, 10], bootstrap_resamples=1000, seed=42
)

print("\n" + "=" * 70)
print("FINAL LIKE-FOR-LIKE COMPARISON RESULTS (N = 19,632)")
print("=" * 70)
print(
    f"{'Metric':<12} | {'Leaky (In-Code)':<18} | {'Clean (Stripped)':<18} | {'Delta (Leak Impact)':<18}"
)
print("-" * 72)
for m in ["mrr", "recall@1", "recall@5", "recall@10", "ndcg@10"]:
    s_leaky = res_leaky[m]
    s_clean = res_clean[m]
    delta = s_leaky - s_clean
    ci_clean = res_clean.get(f"{m}_ci", ("—", "—"))
    print(
        f"{m.upper():<12} | {s_leaky:0.4f}             | {s_clean:0.4f}             | {delta:+0.4f} (drop of {delta * 100:0.1f} pts)"
    )

# Save results to a json file for documentation
import json

output_data = {
    "sample_size": len(merged),
    "leaky": {
        k: float(v) if not isinstance(v, tuple) else list(v)
        for k, v in res_leaky.items()
    },
    "clean": {
        k: float(v) if not isinstance(v, tuple) else list(v)
        for k, v in res_clean.items()
    },
}
with open("reports/like_for_like_bm25_comparison.json", "w") as f:
    json.dump(output_data, f, indent=2)
print("\n[OK] Saved results to reports/like_for_like_bm25_comparison.json")
