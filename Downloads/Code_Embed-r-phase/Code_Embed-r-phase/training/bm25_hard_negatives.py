"""Phase 5 Extension: Mine Hard Negatives using BM25.

This script indexes the training code corpus with BM25 and for each query,
finds the top-K false positive lexical matches to use as hard negatives.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import tqdm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bm25 import BM25Retriever

def process_batch(args: tuple[list[str], int, BM25Retriever, int]) -> np.ndarray:
    queries, start_idx, retriever, k = args
    batch_indices = []
    
    for i, query in enumerate(queries):
        true_idx = start_idx + i
        # Retrieve K+1 to allow excluding the true positive
        top_indices, _ = retriever.search(query, top_k=k + 1)
        
        # Filter out ground truth
        hard_negs = [idx for idx in top_indices if idx != true_idx]
        
        # In case the ground truth wasn't in the top K+1, slice to K
        hard_negs = hard_negs[:k]
        
        # If BM25 didn't return enough (e.g. no lexical overlap), pad with random indices
        while len(hard_negs) < k:
            rand_idx = np.random.randint(0, retriever.corpus_size)
            if rand_idx != true_idx and rand_idx not in hard_negs:
                hard_negs.append(rand_idx)
                
        batch_indices.append(hard_negs)
        
    return np.array(batch_indices, dtype=np.int32)


def mine_bm25_hard_negatives(
    output_path: str,
    k: int = 50,
) -> None:
    train_path = Path("data/processed_clean_v2/train.parquet")
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    print(f"Loading training dataset from {train_path}...")
    df = pd.read_parquet(train_path)
    corpus = df["code"].tolist()
    queries = df["docstring"].tolist()
    n_samples = len(corpus)
    print(f"Loaded {n_samples} samples.")

    print("Building BM25 Index...")
    start_time = time.time()
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.index(corpus)
    print(f"BM25 Index built in {time.time() - start_time:.2f}s")

    print(f"Mining top-{k} hard negatives for {n_samples} queries...")
    
    batch_size = 1000
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_queries = queries[i:i + batch_size]
        batches.append((batch_queries, i, retriever, k))
        
    # Since BM25 is purely CPU bound, we process sequentially with a fast Tqdm bar.
    # BM25 search takes ~1.5ms per query, so 385,000 queries = ~10 mins sequentially.
    all_hard_negs = []
    
    start_search = time.time()
    for batch_args in tqdm.tqdm(batches, desc="Mining hard negatives"):
        batch_result = process_batch(batch_args)
        all_hard_negs.append(batch_result)
        
    hard_neg_indices = np.concatenate(all_hard_negs, axis=0)
    print(f"Finished mining in {time.time() - start_search:.2f}s")
    print(f"Hard negative array shape: {hard_neg_indices.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, hard_neg_indices)
    print(f"Saved hard negatives to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine hard negatives using BM25.")
    parser.add_argument("--output", type=str, default="data/hard_negatives/train_bm25_hard_neg_indices.npy")
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()
    
    mine_bm25_hard_negatives(args.output, args.k)
