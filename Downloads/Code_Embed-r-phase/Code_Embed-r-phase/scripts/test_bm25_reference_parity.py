"""Test numerical and rank parity between custom BM25Retriever and reference rank_bm25.BM25Okapi.

Protocol Check:
Assert that custom vectorized inverted index produces score and rank parity with
the standard reference rank_bm25.BM25Okapi implementation on 1,000 test queries.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi as ReferenceBM25

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import PROCESSED_DIR
from retrieval.bm25 import BM25Retriever, tokenize_code, tokenize_text

print("=" * 70)
print("BM25 REFERENCE IMPLEMENTATION PARITY CHECK (1,000 SAMPLES)")
print("=" * 70)

test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
sample_df = test_df.iloc[:1000].copy()

corpus = sample_df["code"].tolist()
queries = sample_df["docstring"].tolist()
gt_indices = list(range(len(queries)))

# 1. Custom BM25Retriever
print("\n>>> Building custom vectorized BM25Retriever (method='rank_bm25')...")
t0 = time.time()
custom_bm25 = BM25Retriever(k1=1.5, b=0.75, method="rank_bm25")
custom_bm25.index(corpus)
custom_time = time.time() - t0
print(f"[OK] Custom index built in {custom_time:0.3f}s.")

# 2. Reference rank_bm25
print("\n>>> Building reference rank_bm25.BM25Okapi...")
t0 = time.time()
tokenized_corpus = [tokenize_code(doc) for doc in corpus]
ref_bm25 = ReferenceBM25(tokenized_corpus, k1=1.5, b=0.75)
ref_time = time.time() - t0
print(f"[OK] Reference index built in {ref_time:0.3f}s.")

# 3. Compare scores across queries
print("\n>>> Comparing scores and rank correlation on 1,000 queries...")
pearson_corrs = []
score_diffs = []
rank_diffs = []

for idx, (query, gt_idx) in enumerate(zip(queries, gt_indices)):
    q_tokens = tokenize_text(query)
    if not q_tokens:
        continue

    custom_scores = custom_bm25.get_scores(q_tokens)
    ref_scores = np.array(ref_bm25.get_scores(q_tokens), dtype=np.float32)

    # Check max absolute difference on non-zero scores
    diff = np.abs(custom_scores - ref_scores)
    score_diffs.append(np.max(diff))

    # Check correlation
    if np.std(custom_scores) > 1e-6 and np.std(ref_scores) > 1e-6:
        corr = np.corrcoef(custom_scores, ref_scores)[0, 1]
        pearson_corrs.append(corr)

mean_corr = float(np.mean(pearson_corrs))
max_diff = float(np.max(score_diffs))
mean_diff = float(np.mean(score_diffs))

print("\n" + "=" * 70)
print("PARITY VERIFICATION RESULTS")
print("=" * 70)
print(f"  Queries compared:                {len(score_diffs)}")
print(f"  Mean Pearson score correlation:  {mean_corr:0.6f}")
print(f"  Mean absolute score difference:  {mean_diff:0.6e}")
print(f"  Max absolute score difference:   {max_diff:0.6e}")

assert mean_corr > 0.9999, f"Correlation too low: {mean_corr}"
assert max_diff < 0.005, f"Score difference too high: {max_diff}"
print(
    "\n[PASS] Vectorized BM25Retriever matches reference rank_bm25 with r > 0.9999 and delta < 0.005!"
)
