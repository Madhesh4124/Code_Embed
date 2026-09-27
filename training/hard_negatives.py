"""Hard Negative Mining module for CodeEmbed (Phase 5 & Phase R2-C).

This module provides tools to mine challenging non-matching code snippets for queries:
1. BM25HardNegativeMiner: High-throughput multithreaded SciPy CSR sparse matrix BM25 mining.
2. 3-Tier False-Negative Filters (Protocol v1.1 §3.1 & Protocol Errata §1.9):
   - Tier 1: Identical docstring intent (docstring_i == docstring_j)
   - Tier 2: Normalized AST skeleton match (>= 20 AST nodes)
   - Tier 3: MinHash Jaccard similarity (J >= 0.70)
3. Offline pipeline saving `train_hard_negatives.pt` index tensor matrix.
"""

import ast
import hashlib
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

DEFAULT_DATA_DIR = Path("data/processed_clean_v2")


# ============================================================================
# 1. Deterministic Fingerprint Functions (AST Skeleton & MinHash)
# ============================================================================


def compute_ast_skeleton_hash(code_str: str) -> int:
    """Compute deterministic 63-bit int hash of normalized AST skeleton if >= 20 nodes.

    Identifiers (Name.id, arg.arg, FunctionDef.name, AsyncFunctionDef.name) are mapped
    to 'var' and constants (Constant.value) to 'const'. Trivial functions with < 20
    AST nodes return -1 so they are never falsely excluded (Protocol Errata §1.4).

    Args:
        code_str: Python code string.

    Returns:
        63-bit positive signed integer hash, or -1 if node_count < 20 or parse fails.
    """
    try:
        tree = ast.parse(code_str)
    except (SyntaxError, ValueError, TypeError):
        return -1

    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if isinstance(node, ast.Name):
            node.id = "var"
        elif isinstance(node, ast.arg):
            node.arg = "var"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.name = "var"
        elif isinstance(node, ast.Constant):
            node.value = "const"

    if node_count < 20:
        return -1

    dump_bytes = ast.dump(tree).encode("utf-8")
    return int.from_bytes(hashlib.md5(dump_bytes).digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF


class DeterministicMinHash:
    """Deterministic MinHash signature generator over token 3-grams."""

    def __init__(self, num_perm: int = 64) -> None:
        self.num_perm = num_perm
        self.p = (1 << 61) - 1
        rng = np.random.RandomState(42)
        self.a = rng.randint(1, self.p, size=(num_perm,), dtype=np.int64)
        self.b = rng.randint(0, self.p, size=(num_perm,), dtype=np.int64)

    def _get_shingle_hashes(self, text: str) -> set[int]:
        words = text.split()
        if len(words) < 3:
            return {
                int.from_bytes(hashlib.md5(w.encode("utf-8")).digest()[:8], "little")
                & 0xFFFFFFFFFFFFFFF
                for w in words
            }
        shingles = set()
        for i in range(len(words) - 2):
            s = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            h = (
                int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "little")
                & 0xFFFFFFFFFFFFFFF
            )
            shingles.add(h)
        return shingles

    def compute_signature(self, text: str) -> np.ndarray:
        shingles = self._get_shingle_hashes(text)
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.int64)
        h_arr = np.array(list(shingles), dtype=np.int64)
        sigs = np.zeros(self.num_perm, dtype=np.int64)
        for i in range(self.num_perm):
            val = (self.a[i] * h_arr + self.b[i]) % self.p
            sigs[i] = np.min(val)
        return sigs


def _compute_fingerprints_chunk(
    chunk_codes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AST skeleton hashes and MinHash signatures for a chunk of codes."""
    n = len(chunk_codes)
    mh = DeterministicMinHash(64)
    skel_hashes = np.full(n, -1, dtype=np.int64)
    sigs = np.zeros((n, 64), dtype=np.int64)

    for i, code in enumerate(chunk_codes):
        skel_hashes[i] = compute_ast_skeleton_hash(code)
        sigs[i] = mh.compute_signature(code)

    return skel_hashes, sigs


def load_or_compute_fingerprints(
    df: pd.DataFrame,
    cache_dir: Path,
    num_workers: int = 4,
    chunk_size: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or compute (docstring_ids, skeleton_hashes, minhash_sigs) for the training split.

    Args:
        df: Training DataFrame containing 'code' and 'docstring' columns.
        cache_dir: Directory where cached .npy fingerprint arrays are stored.
        num_workers: Worker threads for parallel chunk execution.
        chunk_size: Codes per parallel chunk.

    Returns:
        tuple of:
        - docstring_ids: (N,) int64 array mapping identical docstrings to equal IDs.
        - skeleton_hashes: (N,) int64 array of AST skeleton hashes (-1 if < 20 nodes).
        - minhash_sigs: (N, 64) int64 array of MinHash signatures.
    """
    n_samples = len(df)
    cache_dir.mkdir(parents=True, exist_ok=True)
    skel_cache_path = cache_dir / "train_skeleton_hashes.npy"
    minhash_cache_path = cache_dir / "train_minhash_sigs.npy"

    # 1. Tier 1 Docstring IDs
    print(f"Indexing {n_samples:,} docstrings for Tier-1 exact intent matching...", flush=True)
    t0_doc = time.time()
    docstring_ids, _ = pd.factorize(df["docstring"])
    docstring_ids = docstring_ids.astype(np.int64)
    print(
        f"  [OK] Indexed {len(set(docstring_ids)):,} unique docstrings in {time.time() - t0_doc:.2f}s",
        flush=True,
    )

    # 2. Tier 2 & 3 Skeleton Hashes & MinHash Signatures
    if skel_cache_path.exists() and minhash_cache_path.exists():
        print(f"Loading cached AST skeletons from {skel_cache_path}...", flush=True)
        skel_hashes = np.load(skel_cache_path)
        print(f"Loading cached MinHash signatures from {minhash_cache_path}...", flush=True)
        minhash_sigs = np.load(minhash_cache_path)
        if len(skel_hashes) == n_samples and len(minhash_sigs) == n_samples:
            print("  [OK] Successfully loaded cached fingerprints from disk.", flush=True)
            return docstring_ids, skel_hashes, minhash_sigs
        print("  [Notice] Cached fingerprints size mismatch, recomputing...", flush=True)

    print(
        f"Computing AST skeletons (>= 20 nodes) and 64-perm MinHash signatures across {n_samples:,} codes...",
        flush=True,
    )
    t0_fp = time.time()
    codes = df["code"].tolist()
    chunks = [codes[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]

    skel_hashes = np.full(n_samples, -1, dtype=np.int64)
    minhash_sigs = np.zeros((n_samples, 64), dtype=np.int64)

    workers = max(1, num_workers)
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for c_idx, chunk in enumerate(chunks):
            st = c_idx * chunk_size
            futures.append((st, executor.submit(_compute_fingerprints_chunk, chunk)))

        for st, fut in futures:
            c_skel, c_sig = fut.result()
            c_len = len(c_skel)
            skel_hashes[st : st + c_len] = c_skel
            minhash_sigs[st : st + c_len] = c_sig
            completed += c_len
            if completed % 50000 < chunk_size or completed == n_samples:
                elapsed = time.time() - t0_fp
                speed = completed / max(elapsed, 1e-4)
                pct = completed / n_samples * 100.0
                print(
                    f"  [Fingerprints: {completed:>7,}/{n_samples:,} ({pct:>5.1f}%)] "
                    f"Elapsed: {elapsed:>5.1f}s | Speed: {speed:>6.1f} codes/s",
                    flush=True,
                )

    print(f"  [OK] Fingerprint precomputation finished in {time.time() - t0_fp:.1f}s.", flush=True)
    print(f"Saving AST skeletons to {skel_cache_path}...", flush=True)
    np.save(skel_cache_path, skel_hashes)
    print(f"Saving MinHash signatures to {minhash_cache_path}...", flush=True)
    np.save(minhash_cache_path, minhash_sigs)

    return docstring_ids, skel_hashes, minhash_sigs


# ============================================================================
# 2. Multi-Threaded SciPy CSR BM25 Hard Negative Miner
# ============================================================================


class BM25HardNegativeMiner:
    """Mines BM25 hard negatives via multi-threaded SciPy CSR sparse matrix multiplication.

    Precomputes an inverted CSR term-document weight matrix D_t of shape (V, N)
    so that candidate scores for a batch of queries can be evaluated simultaneously
    via Q @ D_t in high-speed C-level sparse BLAS routines.

    Applies pre-registered 3-tier false-negative exclusion filters:
    - Tier 1: Identical query docstrings (docstring_i == docstring_j)
    - Tier 2: Normalized AST skeleton match (>= 20 nodes)
    - Tier 3: MinHash token 3-gram Jaccard similarity (J >= 0.70)

    Args:
        bm25_retriever  : Initialized and indexed BM25Retriever instance.
        min_idf         : Minimum IDF threshold to filter generic syntax words (default: 1.0).
        max_query_terms : Maximum most informative terms per query sorted by IDF (default: 8).
        docstring_ids   : (N,) int64 array for Tier-1 docstring matching.
        skeleton_hashes : (N,) int64 array for Tier-2 AST skeleton matching.
        minhash_sigs    : (N, 64) int64 array for Tier-3 MinHash matching.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        min_idf: float = 1.0,
        max_query_terms: int = 8,
        docstring_ids: np.ndarray | None = None,
        skeleton_hashes: np.ndarray | None = None,
        minhash_sigs: np.ndarray | None = None,
    ) -> None:
        self.retriever = bm25_retriever
        self.corpus_size = bm25_retriever.corpus_size
        self.min_idf = min_idf
        self.max_query_terms = max_query_terms
        self.docstring_ids = docstring_ids
        self.skeleton_hashes = skeleton_hashes
        self.minhash_sigs = minhash_sigs

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

    def _is_valid_negative(self, cand_idx: int, q_idx: int) -> bool:
        """Check whether candidate passes all false-negative exclusion filters."""
        # Filter 0: Ground-truth positive pair
        if cand_idx == q_idx:
            return False

        # Filter 1: Identical query docstring intent
        if (
            self.docstring_ids is not None
            and self.docstring_ids[cand_idx] == self.docstring_ids[q_idx]
        ):
            return False

        # Filter 2: Normalized AST skeleton match (>= 20 nodes)
        if self.skeleton_hashes is not None:
            q_skel = self.skeleton_hashes[q_idx]
            if q_skel != -1 and self.skeleton_hashes[cand_idx] == q_skel:
                return False

        # Filter 3: MinHash Jaccard similarity J >= 0.70 (45/64 matches = 70.3%)
        if self.minhash_sigs is not None:
            match_count = np.count_nonzero(
                self.minhash_sigs[cand_idx] == self.minhash_sigs[q_idx]
            )
            if match_count >= 45:
                return False

        return True

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
            List of k corpus indices with highest BM25 score, passing all 3-tier filters.
        """
        N = self.corpus_size
        tokens = tokenize_text(query)

        if not tokens or self.D_t is None or not self.term_to_col:
            cands: list[int] = []
            seen = {true_doc_idx}
            while len(cands) < k:
                r = int(np.random.randint(0, N))
                if r not in seen and self._is_valid_negative(r, true_doc_idx):
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
                if r not in seen and self._is_valid_negative(r, true_doc_idx):
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

        n_cands = len(r_idx)
        top_k: list[int] = []

        if n_cands > 0:
            pool_size = min(n_cands, max(k * 10, 100))
            if n_cands <= pool_size:
                sorted_order = np.argsort(r_sc)[::-1]
            else:
                top_indices = np.argpartition(r_sc, -pool_size)[-pool_size:]
                sorted_order = top_indices[np.argsort(r_sc[top_indices])[::-1]]

            for cand in r_idx[sorted_order]:
                if self._is_valid_negative(cand, true_doc_idx):
                    top_k.append(int(cand))
                    if len(top_k) == k:
                        break

        seen = set(top_k) | {true_doc_idx}
        while len(top_k) < k:
            r = int(np.random.randint(0, N))
            if r not in seen and self._is_valid_negative(r, true_doc_idx):
                top_k.append(r)
                seen.add(r)

        return top_k[:k]

    def _process_query_chunk(
        self,
        chunk_start: int,
        chunk_queries: Sequence[str],
        k: int,
    ) -> tuple[int, np.ndarray, dict[str, int]]:
        """Process a chunk of queries via sparse matmul, filtering false negatives."""
        b_size = len(chunk_queries)
        N = self.corpus_size
        V = len(self.term_to_col)

        stats = {
            "tier1_docstring_drops": 0,
            "tier2_skeleton_drops": 0,
            "tier3_minhash_drops": 0,
        }

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

            top_list: list[int] = []

            if n_cands > 0:
                r_idx = s_indices[st:en]
                r_sc = s_data[st:en]

                pool_size = min(n_cands, max(k * 10, 100))
                if n_cands <= pool_size:
                    order = np.argsort(r_sc)[::-1]
                    cand_pool = r_idx[order]
                else:
                    top_part = np.argpartition(r_sc, -pool_size)[-pool_size:]
                    order = top_part[np.argsort(r_sc[top_part])[::-1]]
                    cand_pool = r_idx[order]

                for c_idx in cand_pool:
                    # Filter 0: Ground truth self
                    if c_idx == q_idx:
                        continue

                    # Filter 1: Identical query docstring
                    if (
                        self.docstring_ids is not None
                        and self.docstring_ids[c_idx] == self.docstring_ids[q_idx]
                    ):
                        stats["tier1_docstring_drops"] += 1
                        continue

                    # Filter 2: Normalized AST skeleton match (>= 20 nodes)
                    if self.skeleton_hashes is not None:
                        q_skel = self.skeleton_hashes[q_idx]
                        if q_skel != -1 and self.skeleton_hashes[c_idx] == q_skel:
                            stats["tier2_skeleton_drops"] += 1
                            continue

                    # Filter 3: MinHash token 3-gram Jaccard >= 0.70 (45/64 matches)
                    if self.minhash_sigs is not None:
                        match_count = np.count_nonzero(
                            self.minhash_sigs[c_idx] == self.minhash_sigs[q_idx]
                        )
                        if match_count >= 45:
                            stats["tier3_minhash_drops"] += 1
                            continue

                    top_list.append(int(c_idx))
                    if len(top_list) == k:
                        break

            # Fallback to random candidates if pool did not yield k valid negatives
            seen = set(top_list) | {q_idx}
            while len(top_list) < k:
                r = int(np.random.randint(0, N))
                if r not in seen and self._is_valid_negative(r, q_idx):
                    top_list.append(r)
                    seen.add(r)

            chunk_res[i] = top_list[:k]

        return chunk_start, chunk_res, stats

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
            f"Starting multithreaded CSR BM25 mining with 3-tier false-negative filters:\n"
            f"  Queries: {n_queries:,} | Corpus: {N:,} | Vocab: {V:,} | k: {k}\n"
            f"  Workers: {workers} | Chunk size: {batch_size:,} queries/batch\n"
            f"  Tier 1 Filter: Identical Docstring intent (active: {self.docstring_ids is not None})\n"
            f"  Tier 2 Filter: Normalized AST skeleton >= 20 nodes (active: {self.skeleton_hashes is not None})\n"
            f"  Tier 3 Filter: MinHash 3-gram J >= 0.70 (active: {self.minhash_sigs is not None})...",
            flush=True,
        )

        completed_queries = 0
        total_stats = {
            "tier1_docstring_drops": 0,
            "tier2_skeleton_drops": 0,
            "tier3_minhash_drops": 0,
        }
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
                c_start, c_res, c_stats = f.result()
                c_len = len(c_res)
                hard_neg_matrix[c_start : c_start + c_len] = c_res
                completed_queries += c_len

                for key in total_stats:
                    total_stats[key] += c_stats[key]

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
                        f"ETA: {remaining:>5.1f}s | Drops: "
                        f"Doc={total_stats['tier1_docstring_drops']:,}, "
                        f"Skel={total_stats['tier2_skeleton_drops']:,}, "
                        f"MinHash={total_stats['tier3_minhash_drops']:,}",
                        flush=True,
                    )

        total_elapsed = time.time() - t0
        print(
            f"[OK] Finished mining {n_queries:,} queries in {total_elapsed:.2f}s "
            f"({n_queries / max(total_elapsed, 1e-4):.1f} q/s avg).\n"
            f"  False Negatives Filtered: "
            f"Tier 1 (Docstring): {total_stats['tier1_docstring_drops']:,} | "
            f"Tier 2 (AST Skeleton >= 20): {total_stats['tier2_skeleton_drops']:,} | "
            f"Tier 3 (MinHash J >= 0.70): {total_stats['tier3_minhash_drops']:,}",
            flush=True,
        )
        return hard_neg_matrix


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def generate_train_hard_negatives(
    k: int = 7,
    split: str = "train",
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_filename: str = "train_hard_negatives.pt",
    num_workers: int | None = 4,
    batch_size: int = 500,
) -> Path:
    """Mine and save the hard negative index matrix with 3-tier false negative filters.

    Args:
        k              : Hard negatives per training sample (default: 7).
        split          : Dataset split to mine from (default: 'train').
        data_dir       : Directory containing clean parquet dataset (default: data/processed_clean_v2).
        output_filename: Output .pt filename in data_dir.
        num_workers    : Worker threads for parallel chunk processing (default: 4).
        batch_size     : Queries per sparse matmul batch (default: 500).

    Returns:
        Path to the saved hard negatives tensor file.
    """
    console = Console()
    base_dir = Path(data_dir)
    parquet_path = base_dir / f"{split}.parquet"
    output_path = base_dir / output_filename
    index_cache_path = base_dir / f"bm25_{split}_index.pkl"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

    console.print(f"[cyan]Loading {split} split from {parquet_path}...[/cyan]")
    df = pd.read_parquet(parquet_path)
    codes = df["code"].tolist()
    docstrings = df["docstring"].tolist()

    # 1. Load or build BM25 retriever
    if index_cache_path.exists():
        console.print(
            f"[cyan]Loading cached BM25 index from {index_cache_path}...[/cyan]"
        )
        bm25 = BM25Retriever.load(str(index_cache_path))
        console.print(
            f"[green][OK] Loaded cached index: {bm25.corpus_size:,} docs | "
            f"{len(bm25.inverted_index):,} terms[/green]"
        )
    else:
        console.print(
            f"[cyan]Building BM25 (ATIRE floor) index over {len(codes):,} code functions...[/cyan]"
        )
        bm25 = BM25Retriever(method="atire")
        bm25.index(codes, show_progress=True)
        console.print(f"[cyan]Caching BM25 index to {index_cache_path}...[/cyan]")
        bm25.save(str(index_cache_path))
        console.print("[green][OK] Index cached successfully.[/green]")

    # 2. Precompute / load 3-tier filter fingerprints
    docstring_ids, skeleton_hashes, minhash_sigs = load_or_compute_fingerprints(
        df=df,
        cache_dir=base_dir,
        num_workers=num_workers or 4,
    )

    console.print(
        f"[cyan]Mining {k} hard negatives per sample for {len(docstrings):,} queries "
        f"(multithreaded SciPy CSR matmul, batch_size={batch_size:,}, workers={num_workers or 4})...[/cyan]"
    )
    miner = BM25HardNegativeMiner(
        bm25_retriever=bm25,
        docstring_ids=docstring_ids,
        skeleton_hashes=skeleton_hashes,
        minhash_sigs=minhash_sigs,
    )
    hard_negs = miner.mine_corpus_negatives(
        docstrings, k=k, batch_size=batch_size, num_workers=num_workers
    )

    console.print(f"[cyan]Saving hard negative index matrix to {output_path}...[/cyan]")
    torch.save(torch.from_numpy(hard_negs), output_path)
    console.print(
        f"[bold green][OK] Successfully mined and saved {output_path} "
        f"(shape: {hard_negs.shape}, dtype: {hard_negs.dtype})[/bold green]"
    )

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mine BM25 hard negatives.")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    generate_train_hard_negatives(
        k=args.k,
        split=args.split,
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
