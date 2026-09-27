"""Compare BM25 IDF variants on the full validation split (N = 20,115) to select the conservative stronger baseline."""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.evaluate import evaluate_rankings
from retrieval.bm25 import BM25Retriever

print("=" * 70, flush=True)
print("COMPARING BM25 IDF VARIANTS ON FULL VALIDATION SET (N = 20,115)", flush=True)
print("=" * 70, flush=True)

val_df = pd.read_parquet("data/processed_clean_v2/validation.parquet")
corpus = val_df["code"].tolist()
queries = val_df["docstring"].tolist()
gt_indices = list(range(len(queries)))
N = len(val_df)
print(f"Validation set size: {N:,} queries & documents.", flush=True)

# 1. Evaluate Variant A: Robertson smooth (ln(1 + fraction))
print("\n>>> 1. Evaluating Variant A: Robertson smooth (method='robertson')...", flush=True)
t0 = time.time()
bm25_robertson = BM25Retriever(k1=1.5, b=0.75, method="robertson")
bm25_robertson.index(corpus, show_progress=True, progress_interval=10000)
ranks_robertson = bm25_robertson.get_ranks(queries, gt_indices)
res_robertson = evaluate_rankings(ranks_robertson, ks=[1, 5, 10], bootstrap_resamples=1000)
time_robertson = time.time() - t0
print(f"    Robertson Validation MRR: {res_robertson['mrr']:.4f} [95% CI: {res_robertson['mrr_ci'][0]:.4f}, {res_robertson['mrr_ci'][1]:.4f}] ({time_robertson:.1f}s)", flush=True)

# 2. Evaluate Variant B: ATIRE floor (method='rank_bm25')
print("\n>>> 2. Evaluating Variant B: ATIRE floor (method='rank_bm25')...", flush=True)
t0 = time.time()
bm25_atire = BM25Retriever(k1=1.5, b=0.75, method="rank_bm25")
bm25_atire.index(corpus, show_progress=True, progress_interval=10000)
ranks_atire = bm25_atire.get_ranks(queries, gt_indices)
res_atire = evaluate_rankings(ranks_atire, ks=[1, 5, 10], bootstrap_resamples=1000)
time_atire = time.time() - t0
print(f"    ATIRE Validation MRR:     {res_atire['mrr']:.4f} [95% CI: {res_atire['mrr_ci'][0]:.4f}, {res_atire['mrr_ci'][1]:.4f}] ({time_atire:.1f}s)", flush=True)

print("\n" + "=" * 70, flush=True)
print("VALIDATION VARIANT COMPARISON SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"Robertson Smooth MRR: {res_robertson['mrr']:.4f} (R@1: {res_robertson['recall@1']:.4f}, R@10: {res_robertson['recall@10']:.4f})", flush=True)
print(f"ATIRE Floor MRR:      {res_atire['mrr']:.4f} (R@1: {res_atire['recall@1']:.4f}, R@10: {res_atire['recall@10']:.4f})", flush=True)

if res_atire["mrr"] >= res_robertson["mrr"]:
    selected_variant = "rank_bm25"
    selected_res = res_atire
    delta = res_atire["mrr"] - res_robertson["mrr"]
    print(f"\n[DECISION]: Selected ATIRE Floor (rank_bm25) as the stronger conservative baseline (+{delta:.4f} MRR on validation).", flush=True)
else:
    selected_variant = "robertson"
    selected_res = res_robertson
    delta = res_robertson["mrr"] - res_atire["mrr"]
    print(f"\n[DECISION]: Selected Robertson Smooth as the stronger conservative baseline (+{delta:.4f} MRR on validation).", flush=True)

report = {
    "validation_size": N,
    "robertson": {
        "mrr": res_robertson["mrr"],
        "mrr_ci": res_robertson["mrr_ci"],
        "recall@1": res_robertson["recall@1"],
        "recall@5": res_robertson["recall@5"],
        "recall@10": res_robertson["recall@10"],
        "ndcg@10": res_robertson["ndcg@10"]
    },
    "atire": {
        "mrr": res_atire["mrr"],
        "mrr_ci": res_atire["mrr_ci"],
        "recall@1": res_atire["recall@1"],
        "recall@5": res_atire["recall@5"],
        "recall@10": res_atire["recall@10"],
        "ndcg@10": res_atire["ndcg@10"]
    },
    "selected_variant": selected_variant,
    "pilot_gate_val_mrr": selected_res["mrr"],
    "pilot_gate_threshold_75": float(0.75 * selected_res["mrr"])
}

Path("reports/bm25_variant_validation_selection.json").write_text(json.dumps(report, indent=2))
print("Saved reports/bm25_variant_validation_selection.json", flush=True)

