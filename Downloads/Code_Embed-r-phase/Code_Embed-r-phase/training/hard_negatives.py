"""Hard Negative Mining module for CodeEmbed (Phase 5).

This module provides tools to mine challenging non-matching code snippets for queries:
1. BM25HardNegativeMiner: High-throughput multithreaded SciPy CSR sparse matrix BM25 mining.
2. Offline pipeline saving `train_hard_negatives.pt` indices matrix.

Performance Architecture (Benchmarked on 385k-doc corpus, RTX-4050 machine)
-----------------------------------------------------------------------------
Approach 1 — Python dict loop:
  Throughput: ~17 q/s  →  385k in ~6 hours.  REJECTED.

Approach 2 — Dense array scatter-add / get_scores():
  Throughput: ~21 q/s  →  385k in ~5 hours.  REJECTED.

Approach 3 — Candidate-only concatenate + stable sort:
  Sorting 1.2M–2.5M candidate arrays per query in Python takes ~0.27s/query.
  Throughput: ~3.7 q/s  →  385k in ~28.6 hours + memory thrashing.  REJECTED.

Approach 4 — Multi-threaded SciPy CSR Sparse Matmul (THIS IMPLEMENTATION):
  1. Filters non-discriminative syntax terms (IDF < 1.0, e.g. 'def', 'return', 'self').
  2. Selects the top-8 most informative terms per query by Okapi IDF.
  3. Precomputes CSR term-document matrix D_t (V, N) with pre-normalized BM25 weights.
  4. Dispatches query batches (batch_size=500) across CPU threads via ThreadPoolExecutor.
     Threads share read-only memory with zero IPC pickling overhead, and SciPy / NumPy
     routines release the GIL during matrix multiplication and argpartition.
  5. Throughput: ~1,550 q/s → all 385,381 queries in ~4 minutes!
"""

import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from rich.console import Console

from retrieval.bm25 import BM25Retriever, tokenize_text

PROCESSED_DIR = Path("data/processed")


class BM25HardNegativeMiner:
    """Mines BM25 hard negatives via multi-threaded SciPy CSR sparse matrix multiplication.

    Precomputes an inverted CSR term-document weight matrix D_t of shape (V, N)
    so that candidate scores for a batch of queries can be evaluated simultaneously
    via Q @ D_t in high-speed C-level sparse BLAS routines.

    Args:
        bm25_retriever  : Initialized and indexed BM25Retriever instance.
        min_idf         : Minimum IDF threshold to filter generic syntax words (default: 1.0).
        max_query_terms : Maximum most informative terms per query sorted by IDF (default: 8).
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        min_idf: float = 1.0,
        max_query_terms: int = 8,
    ) -> None:
        self.retriever = bm25_retriever
        self.corpus_size = bm25_retriever.corpus_size
        self.min_idf = min_idf
        self.max_query_terms = max_query_terms
        self.term_to_col: dict[str, int] = {}
        self.D_t: sp.csr_matrix | None = None
        self._build_sparse_index()

    def _build_sparse_index(self) -> None:
        """Construct the Compressed Sparse Row term matrix D_t of shape (V, N)."""
        if self.corpus_size == 0 or not self.retriever.inverted_index:
            return

        k1 = self.retriever.k1
        len_norm = self.retriever.len_norm
        if len_norm is None:
            raise RuntimeError(
                "BM25 retriever has not computed length normalization arrays."
            )

        self.term_to_col = {}
        indptr = [0]
        all_indices: list[np.ndarray] = []
        all_data: list[np.ndarray] = []
        total_nnz = 0

        for term, (doc_ids, freqs) in self.retriever.inverted_index.items():
            idf = self.retriever.idf.get(term, 0.0)
            if idf < self.min_idf:
                continue
            col_idx = len(self.term_to_col)
            self.term_to_col[term] = col_idx

            # Precompute exact BM25 Okapi weights: w(t, d)
            weights = (idf * (freqs * (k1 + 1.0)) / (freqs + len_norm[doc_ids])).astype(
                np.float32
            )
            all_indices.append(doc_ids)
            all_data.append(weights)
            total_nnz += len(doc_ids)
            indptr.append(total_nnz)

        V = len(self.term_to_col)
        N = self.corpus_size

        if total_nnz > 0:
            concat_data = np.concatenate(all_data)
            concat_indices = np.concatenate(all_indices)
            indptr_arr = np.array(indptr, dtype=np.int32)
            self.D_t = sp.csr_matrix(
                (concat_data, concat_indices, indptr_arr),
                shape=(V, N),
                dtype=np.float32,
            )
        else:
            self.D_t = sp.csr_matrix((V, N), dtype=np.float32)

    def mine_query_negatives(
        self,
        query: str,
        true_doc_idx: int,
        k: int = 7,
    ) -> list[int]:
        """Mine top-k hard negative indices for a single query string.

        Args:
            query       : Natural language query string.
            true_doc_idx: Ground-truth corpus index to exclude.
            k           : Number of hard negatives to return.

        Returns:
            List of k corpus indices with highest BM25 score, excluding true_doc_idx.
        """
        N = self.corpus_size
        tokens = tokenize_text(query)

        if not tokens or self.D_t is None or not self.term_to_col:
            cands: list[int] = []
            seen = {true_doc_idx}
            while len(cands) < k:
                r = int(np.random.randint(0, N))
                if r not in seen:
                    cands.append(r)
                    seen.add(r)
            return cands

        # Sort query terms by IDF descending and keep top informative
        q_term_idfs = [
            (t, self.retriever.idf.get(t, 0.0))
            for t in set(tokens)
            if t in self.term_to_col
        ]
        q_term_idfs.sort(key=lambda x: x[1], reverse=True)
        q_cols = [self.term_to_col[t] for t, _ in q_term_idfs[: self.max_query_terms]]

        if not q_cols:
            cands = []
            seen = {true_doc_idx}
            while len(cands) < k:
                r = int(np.random.randint(0, N))
                if r not in seen:
                    cands.append(r)
                    seen.add(r)
            return cands

        q_rows = np.zeros(len(q_cols), dtype=np.int32)
        q_data = np.ones(len(q_cols), dtype=np.float32)
        Q = sp.csr_matrix(
            (q_data, (q_rows, np.array(q_cols, dtype=np.int32))),
            shape=(1, len(self.term_to_col)),
            dtype=np.float32,
        )

        S = Q @ self.D_t
        r_idx = S.indices
        r_sc = S.data

        mask = r_idx != true_doc_idx
        if not mask.all():
            r_idx = r_idx[mask]
            r_sc = r_sc[mask]

        n_cands = len(r_idx)
        if n_cands == 0:
            top_k: list[int] = []
        elif n_cands <= k:
            order = np.argsort(r_sc)[::-1]
            top_k = r_idx[order].tolist()
        else:
            part = np.argpartition(r_sc, -k)[-k:]
            sorted_part = part[np.argsort(r_sc[part])[::-1]]
            top_k = r_idx[sorted_part].tolist()

        seen = set(top_k) | {true_doc_idx}
        while len(top_k) < k:
            r = int(np.random.randint(0, N))
            if r not in seen:
                top_k.append(r)
                seen.add(r)

        return top_k[:k]

    def _process_query_chunk(
        self,
        chunk_start: int,
        chunk_queries: Sequence[str],
        k: int,
    ) -> tuple[int, np.ndarray]:
        """Process a chunk of queries via sparse matmul and top-k selection."""
        b_size = len(chunk_queries)
        N = self.corpus_size
        V = len(self.term_to_col)

        q_rows: list[int] = []
        q_cols: list[int] = []

        for i, q in enumerate(chunk_queries):
            toks = tokenize_text(q)
            q_term_idfs = [
                (t, self.retriever.idf.get(t, 0.0))
                for t in set(toks)
                if t in self.term_to_col
            ]
            q_term_idfs.sort(key=lambda x: x[1], reverse=True)
            for t, _ in q_term_idfs[: self.max_query_terms]:
                c = self.term_to_col.get(t)
                if c is not None:
                    q_rows.append(i)
                    q_cols.append(c)

        if q_rows and self.D_t is not None:
            Q = sp.csr_matrix(
                (
                    np.ones(len(q_rows), dtype=np.float32),
                    (
                        np.array(q_rows, dtype=np.int32),
                        np.array(q_cols, dtype=np.int32),
                    ),
                ),
                shape=(b_size, V),
                dtype=np.float32,
            )
            S = Q @ self.D_t
            s_indptr = S.indptr
            s_indices = S.indices
            s_data = S.data
        else:
            s_indptr = np.zeros(b_size + 1, dtype=np.int32)
            s_indices = np.array([], dtype=np.int32)
            s_data = np.array([], dtype=np.float32)

        chunk_res = np.zeros((b_size, k), dtype=np.int32)
        for i in range(b_size):
            q_idx = chunk_start + i
            st = s_indptr[i]
            en = s_indptr[i + 1]
            n_cands = en - st

            if n_cands == 0:
                cands: list[int] = []
                seen = {q_idx}
                while len(cands) < k:
                    r = int(np.random.randint(0, N))
                    if r not in seen:
                        cands.append(r)
                        seen.add(r)
                chunk_res[i] = cands
                continue

            r_idx = s_indices[st:en]
            r_sc = s_data[st:en]

            mask = r_idx != q_idx
            if not mask.all():
                r_idx = r_idx[mask]
                r_sc = r_sc[mask]
                n_cands = len(r_idx)

            if n_cands == 0:
                cands = []
                seen = {q_idx}
                while len(cands) < k:
                    r = int(np.random.randint(0, N))
                    if r not in seen:
                        cands.append(r)
                        seen.add(r)
                chunk_res[i] = cands
            elif n_cands <= k:
                order = np.argsort(r_sc)[::-1]
                top_list = r_idx[order].tolist()
                seen = set(top_list) | {q_idx}
                while len(top_list) < k:
                    r = int(np.random.randint(0, N))
                    if r not in seen:
                        top_list.append(r)
                        seen.add(r)
                chunk_res[i] = top_list[:k]
            else:
                part = np.argpartition(r_sc, -k)[-k:]
                sorted_part = part[np.argsort(r_sc[part])[::-1]]
                chunk_res[i] = r_idx[sorted_part]

        return chunk_start, chunk_res

    def mine_corpus_negatives(
        self,
        queries: Sequence[str],
        k: int = 7,
        batch_size: int = 500,
        num_workers: int | None = 4,
    ) -> np.ndarray:
        """Mine hard negative indices for all queries using multithreaded sparse BLAS matmul.

        Args:
            queries    : List of docstrings (query i = positive pair for code doc i).
            k          : Number of hard negatives per query.
            batch_size : Queries per chunk evaluated simultaneously (default: 500).
            num_workers: Number of parallel CPU threads (default: 4).

        Returns:
            (N_queries, k) int32 numpy array of hard negative corpus indices.
        """
        n_queries = len(queries)
        N = self.corpus_size
        hard_neg_matrix = np.zeros((n_queries, k), dtype=np.int32)

        if n_queries == 0:
            return hard_neg_matrix

        workers = max(1, num_workers or 4)
        V = len(self.term_to_col)
        t0 = time.time()
        print(
            f"Starting multithreaded CSR BM25 mining: {n_queries:,} queries | "
            f"corpus={N:,} docs | vocab={V:,} terms | k={k} | "
            f"workers={workers} | batch_size={batch_size:,}...",
            flush=True,
        )

        completed_queries = 0
        log_interval = 10000

        with ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = []
            for b_start in range(0, n_queries, batch_size):
                b_end = min(b_start + batch_size, n_queries)
                tasks.append(
                    executor.submit(
                        self._process_query_chunk,
                        b_start,
                        queries[b_start:b_end],
                        k,
                    )
                )

            for f in tasks:
                c_start, c_res = f.result()
                c_len = len(c_res)
                hard_neg_matrix[c_start : c_start + c_len] = c_res
                completed_queries += c_len

                if (
                    completed_queries % log_interval < batch_size
                    or completed_queries == n_queries
                ):
                    elapsed = time.time() - t0
                    speed = completed_queries / max(elapsed, 1e-4)
                    remaining = (n_queries - completed_queries) / max(speed, 1e-4)
                    pct = completed_queries / n_queries * 100.0
                    print(
                        f"[Mining Progress: {completed_queries:>7,}/{n_queries:,} ({pct:>5.1f}%)] "
                        f"Elapsed: {elapsed:>5.1f}s | Speed: {speed:>6.1f} q/s | "
                        f"ETA: {remaining:>5.1f}s",
                        flush=True,
                    )

        total_elapsed = time.time() - t0
        print(
            f"[OK] Finished mining {n_queries:,} queries in {total_elapsed:.2f}s "
            f"({n_queries / max(total_elapsed, 1e-4):.1f} q/s avg).",
            flush=True,
        )
        return hard_neg_matrix


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def generate_train_hard_negatives(
    k: int = 7,
    split: str = "train",
    output_filename: str = "train_hard_negatives.pt",
    num_workers: int | None = 4,
    batch_size: int = 500,
) -> Path:
    """Mine and save the hard negative index matrix for Phase 5 training.

    Args:
        k              : Hard negatives per training sample (default: 7).
        split          : Dataset split to mine from (default: 'train').
        output_filename: Output .pt filename in data/processed/.
        num_workers    : Worker threads for parallel chunk processing (default: 4).
        batch_size     : Queries per sparse matmul batch (default: 500).

    Returns:
        Path to the saved hard negatives tensor file.
    """
    console = Console()
    parquet_path = PROCESSED_DIR / f"{split}.parquet"
    output_path = PROCESSED_DIR / output_filename
    index_cache_path = PROCESSED_DIR / f"bm25_{split}_index.pkl"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

    console.print(f"[cyan]Loading {split} split from {parquet_path}...[/cyan]")
    sys.stdout.flush()
    df = pd.read_parquet(parquet_path)
    codes = df["code"].tolist()
    docstrings = df["docstring"].tolist()

    # Load or build BM25 retriever
    if index_cache_path.exists():
        console.print(
            f"[cyan]Loading cached BM25 index from {index_cache_path}...[/cyan]"
        )
        sys.stdout.flush()
        bm25 = BM25Retriever.load(str(index_cache_path))
        console.print(
            f"[green][OK] Loaded cached index: {bm25.corpus_size:,} docs | "
            f"{len(bm25.inverted_index):,} terms[/green]"
        )
        sys.stdout.flush()
    else:
        console.print(
            f"[cyan]Building BM25 index over {len(codes):,} code functions...[/cyan]"
        )
        sys.stdout.flush()
        bm25 = BM25Retriever()
        bm25.index(codes, show_progress=True)
        console.print(f"[cyan]Caching BM25 index to {index_cache_path}...[/cyan]")
        sys.stdout.flush()
        bm25.save(str(index_cache_path))
        console.print("[green][OK] Index cached successfully.[/green]")
        sys.stdout.flush()

    console.print(
        f"[cyan]Mining {k} hard negatives per sample for {len(docstrings):,} queries "
        f"(multithreaded SciPy CSR matmul, batch_size={batch_size:,}, workers={num_workers or 4})...[/cyan]"
    )
    sys.stdout.flush()
    miner = BM25HardNegativeMiner(bm25)
    hard_negs = miner.mine_corpus_negatives(
        docstrings, k=k, batch_size=batch_size, num_workers=num_workers
    )

    console.print(f"[cyan]Saving hard negative index matrix to {output_path}...[/cyan]")
    sys.stdout.flush()
    torch.save(torch.from_numpy(hard_negs), output_path)
    console.print(
        f"[bold green][OK] Successfully mined and saved {output_path} "
        f"(shape: {hard_negs.shape}, dtype: {hard_negs.dtype})[/bold green]"
    )
    sys.stdout.flush()

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mine BM25 hard negatives.")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    generate_train_hard_negatives(
        k=args.k,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
