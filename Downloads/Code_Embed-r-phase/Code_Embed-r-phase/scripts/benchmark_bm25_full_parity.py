"""Compare MRR from reference rank_bm25.BM25Okapi against custom BM25Retriever on full test corpus (19,632 documents) over 2,000 queries."""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi as ReferenceBM25

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.bm25 import BM25Retriever, tokenize_code, tokenize_text

print("=" * 70, flush=True)
print("BM25 FULL CORPUS PARITY BENCHMARK (19,632 DOCS, 2,000 QUERIES)", flush=True)
print("=" * 70, flush=True)

test_df = pd.read_parquet("data/processed_clean_v2/test.parquet")
N_corpus = len(test_df)
print(f"Full corpus size: {N_corpus:,} documents.", flush=True)

# Precompute harmonic numbers H_n = sum_{k=1}^n 1/k
harmonic_table = np.zeros(N_corpus + 1, dtype=np.float64)
harmonic_table[1:] = np.cumsum(1.0 / np.arange(1, N_corpus + 1, dtype=np.float64))


def compute_expected_rr(target_score: float, scores: np.ndarray) -> float:
    if target_score <= 0.0:
        return 0.0
    higher_count = int(np.sum(scores > target_score))
    eq_count = int(np.sum(scores == target_score))
    if eq_count <= 0:
        eq_count = 1
    if eq_count > 1:
        return float(
            (harmonic_table[higher_count + eq_count] - harmonic_table[higher_count])
            / eq_count
        )
    return float(1.0 / (higher_count + 1))


# Sample 2,000 queries with fixed seed 42
sample_indices = test_df.sample(2000, random_state=42).index.values
print(
    f"Evaluating {len(sample_indices):,} queries across full {N_corpus:,} corpus.",
    flush=True,
)

print("\n1. Tokenizing corpus...", flush=True)
t0 = time.time()
corpus_tokens = [tokenize_code(c) for c in test_df["code"].values]
print(
    f"   Tokenized {len(corpus_tokens):,} code documents in {time.time() - t0:.2f}s",
    flush=True,
)

# Build Reference BM25
print("\n2. Building Reference BM25Okapi index...", flush=True)
t0 = time.time()
ref_bm25 = ReferenceBM25(corpus_tokens, k1=1.5, b=0.75)
print(f"   Reference index built in {time.time() - t0:.2f}s", flush=True)

# Build Custom BM25
print("\n3. Building Custom BM25Retriever index...", flush=True)
t0 = time.time()
custom_bm25 = BM25Retriever(k1=1.5, b=0.75)
custom_bm25.corpus_size = N_corpus
doc_lens = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
custom_bm25.avgdl = float(np.mean(doc_lens))
custom_bm25.len_norm = custom_bm25.k1 * (
    1.0 - custom_bm25.b + custom_bm25.b * (doc_lens / custom_bm25.avgdl)
)

doc_freqs = {}
term_doc_counts = {}
for doc_id, doc_toks in enumerate(corpus_tokens):
    counts = Counter(doc_toks)
    for term, count in counts.items():
        if term not in doc_freqs:
            doc_freqs[term] = []
            term_doc_counts[term] = []
        doc_freqs[term].append(doc_id)
        term_doc_counts[term].append(count)

custom_bm25.inverted_index = {}
custom_bm25.idf = {}
for term, doc_list in doc_freqs.items():
    n = len(doc_list)
    idf_val = float(np.log((N_corpus - n + 0.5) / (n + 0.5) + 1.0))
    custom_bm25.idf[term] = idf_val
    custom_bm25.inverted_index[term] = (
        np.array(doc_list, dtype=np.int32),
        np.array(term_doc_counts[term], dtype=np.float32),
    )
print(f"   Custom index built in {time.time() - t0:.2f}s", flush=True)

# Evaluate MRR on 2,000 queries
print("\n4. Running 2,000 queries against both retrievers...", flush=True)
ref_rr_list = []
custom_rr_list = []

t0 = time.time()
for q_count, q_idx in enumerate(sample_indices):
    q_text = test_df.iloc[q_idx]["docstring"]
    q_tokens = tokenize_text(q_text)
    target_idx = int(q_idx)

    # Custom scores
    c_scores = custom_bm25.get_scores(q_tokens)
    c_rr = compute_expected_rr(float(c_scores[target_idx]), c_scores)
    custom_rr_list.append(c_rr)

    # Reference scores
    r_scores = np.array(ref_bm25.get_scores(q_tokens), dtype=np.float32)
    r_rr = compute_expected_rr(float(r_scores[target_idx]), r_scores)
    ref_rr_list.append(r_rr)

    if (q_count + 1) % 500 == 0 or (q_count + 1) == len(sample_indices):
        elapsed = time.time() - t0
        speed = (q_count + 1) / max(elapsed, 1e-4)
        print(
            f"   Processed {q_count + 1:>5,}/2,000 queries in {elapsed:.1f}s ({speed:.1f} QPS) | Ref MRR: {np.mean(ref_rr_list):.4f} | Custom MRR: {np.mean(custom_rr_list):.4f}",
            flush=True,
        )

ref_mrr = float(np.mean(ref_rr_list))
custom_mrr = float(np.mean(custom_rr_list))
diff = abs(ref_mrr - custom_mrr)

print("\n" + "=" * 70, flush=True)
print("FINAL FULL-CORPUS PARITY EVALUATION (N=2,000 QUERIES, 19,632 DOCS)", flush=True)
print("=" * 70, flush=True)
print(f"Reference BM25Okapi MRR: {ref_mrr:.4f}", flush=True)
print(f"Custom BM25Retriever MRR: {custom_mrr:.4f}", flush=True)
print(f"Absolute Difference:     {diff:.4f}", flush=True)
print(f"Passing criteria (<= 0.005): {'PASS' if diff <= 0.005 else 'FAIL'}", flush=True)
print("=" * 70, flush=True)

out = {
    "num_queries": len(sample_indices),
    "corpus_size": N_corpus,
    "ref_bm25_mrr": ref_mrr,
    "custom_bm25_mrr": custom_mrr,
    "absolute_difference": diff,
    "pass_threshold_0_005": bool(diff <= 0.005),
}
Path("reports/bm25_full_parity_report.json").write_text(json.dumps(out, indent=2))
print("Saved reports/bm25_full_parity_report.json", flush=True)
