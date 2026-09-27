"""Comprehensive audit and empirical diagnostic of query-code token overlap on clean test split.

Protocol Check:
1. Compute true-pair rate, null-pair rate, delta on N = 19,632 clean test pairs.
2. Characterize matching pairs: function signature vs URLs/literals vs inline comments vs comment docstrings.
3. Compare BM25 MRR and R@1 on matching hits vs non-hits.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import PROCESSED_DIR
from evaluation.metrics import mrr, recall_at_k
from retrieval.bm25 import BM25Retriever, tokenize_code, tokenize_text

print("=" * 70)
print("AUDIT & EMPIRICAL DIAGNOSTIC: QUERY-CODE TOKEN OVERLAP (TEST SPLIT)")
print("=" * 70)

test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
print(f"Loaded {len(test_df):,} test functions.")

with open("data/stopwords.json") as f:
    stopwords = set(json.load(f))


def extract_non_stop(tokens):
    return [t for t in tokens if t not in stopwords and len(t) > 1]


# 1. Run overlap diagnostic across all 19,632 test samples
rng = np.random.RandomState(42)
shuffled_codes = rng.permutation(test_df["code"].tolist())

true_hits = []
null_hits = []

print(
    "\n>>> Scanning 19,632 query-code pairs for contiguous 8-grams and full query containment..."
)
for idx, row in enumerate(test_df.itertuples()):
    q_tokens = extract_non_stop(tokenize_text(row.docstring))
    c_tokens = tokenize_code(row.code)
    null_c_tokens = tokenize_code(shuffled_codes[idx])

    def check_match(q, c):
        # Criterion 1: Contiguous 8-gram
        if len(q) >= 8:
            for i in range(len(q) - 7):
                ngram = q[i : i + 8]
                for j in range(len(c) - 7):
                    if c[j : j + 8] == ngram:
                        return True, "contiguous_8gram", ngram
        # Criterion 2: Full containment if 4 <= len(q) < 8
        if 4 <= len(q) < 8 and set(q).issubset(set(c)):
            return True, "subset_containment", q
        return False, None, None

    is_true, mtype, match_toks = check_match(q_tokens, c_tokens)
    if is_true:
        true_hits.append(
            {
                "idx": idx,
                "func_name": row.func_name,
                "docstring": row.docstring,
                "code": row.code,
                "match_type": mtype,
                "tokens": match_toks,
            }
        )

    is_null, _, _ = check_match(q_tokens, null_c_tokens)
    if is_null:
        null_hits.append(idx)

N = len(test_df)
true_rate = len(true_hits) / N
null_rate = len(null_hits) / N
diff = true_rate - null_rate

print("\n" + "=" * 70)
print(f"OVERLAP DIAGNOSTIC RESULTS (N = {N:,})")
print("=" * 70)
print(f"  True-pair matches:   {len(true_hits):>5} / {N:,} ({true_rate * 100:0.2f}%)")
print(f"  Null-pair matches:   {len(null_hits):>5} / {N:,} ({null_rate * 100:0.2f}%)")
print(f"  Rate Difference:     {diff * 100:+0.2f}% (Margin: {diff:0.4f})")

# 2. Characterize the matching hits
print("\n>>> Categorizing True-Pair Matches...")
categories = {
    "signature_or_param": 0,  # Tokens in def header, arguments, return statement
    "api_or_url": 0,  # HTTP / URL / API documentation link
    "inline_comment": 0,  # Match resides in an inline comment
    "body_statements": 0,  # Standard code logic statements
}

hit_examples = []
for h in true_hits:
    code = h["code"]
    toks = h["tokens"]
    first_line = code.splitlines()[0] if code.splitlines() else ""

    # Check if match tokens are in function header / args
    toks_in_header = all(t.lower() in first_line.lower() for t in toks[:3])
    has_url = (
        "http" in code.lower()
        or "cloud.google" in code.lower()
        or "github" in code.lower()
    )

    if toks_in_header:
        categories["signature_or_param"] += 1
        cat = "signature_or_param"
    elif has_url and any(t in ["https", "docs", "reference", "cloud"] for t in toks):
        categories["api_or_url"] += 1
        cat = "api_or_url"
    elif "#" in code and any(t in code.split("#")[-1] for t in toks[:2]):
        categories["inline_comment"] += 1
        cat = "inline_comment"
    else:
        categories["body_statements"] += 1
        cat = "body_statements"

    if len(hit_examples) < 10:
        hit_examples.append(
            (h["func_name"], cat, h["match_type"], h["docstring"][:80], code[:120])
        )

print("Match Category Breakdown:")
for cat, cnt in categories.items():
    print(f"  - {cat:<20}: {cnt:>4} ({cnt / len(true_hits) * 100:0.1f}%)")

# 3. Check BM25 performance on hits vs non-hits
print("\n>>> Evaluating BM25 on Overlap Hits vs Non-Hits...")
bm25 = BM25Retriever(k1=1.5, b=0.75)
bm25.index(test_df["code"].tolist())
all_queries = test_df["docstring"].tolist()
all_gt = list(range(N))

all_ranks = bm25.get_ranks(all_queries, all_gt, max_candidates=1000)

hit_indices = [h["idx"] for h in true_hits]
non_hit_indices = list(set(range(N)) - set(hit_indices))

hit_ranks = all_ranks[hit_indices]
non_hit_ranks = all_ranks[non_hit_indices]

mrr_hits = mrr(hit_ranks)
r1_hits = recall_at_k(hit_ranks, k=1)
r10_hits = recall_at_k(hit_ranks, k=10)

mrr_non_hits = mrr(non_hit_ranks)
r1_non_hits = recall_at_k(non_hit_ranks, k=1)
r10_non_hits = recall_at_k(non_hit_ranks, k=10)

print(f"\n  Hits Subset (N = {len(hit_indices):,}):")
print(
    f"    MRR = {mrr_hits:0.4f}, Recall@1 = {r1_hits:0.4f}, Recall@10 = {r10_hits:0.4f}"
)
print(f"  Non-Hits Subset (N = {len(non_hit_indices):,}):")
print(
    f"    MRR = {mrr_non_hits:0.4f}, Recall@1 = {r1_non_hits:0.4f}, Recall@10 = {r10_non_hits:0.4f}"
)
print(f"  Full Test Set (N = {N:,}):")
print(
    f"    MRR = {mrr(all_ranks):0.4f}, Recall@1 = {recall_at_k(all_ranks, k=1):0.4f}, Recall@10 = {recall_at_k(all_ranks, k=10):0.4f}"
)

# Save detailed report
audit_report = {
    "N": N,
    "true_hits": len(true_hits),
    "true_rate": true_rate,
    "null_hits": len(null_hits),
    "null_rate": null_rate,
    "diff": diff,
    "categories": categories,
    "bm25_mrr_hits": mrr_hits,
    "bm25_mrr_non_hits": mrr_non_hits,
    "bm25_mrr_all": float(mrr(all_ranks)),
}
with open("reports/overlap_diagnostic_audit.json", "w") as f:
    json.dump(audit_report, f, indent=2)
print("\n[OK] Audit report saved to reports/overlap_diagnostic_audit.json")
