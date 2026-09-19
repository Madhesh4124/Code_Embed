"""BM25 lexical retrieval for code search.

This module implements:
- Code-aware sub-identifier tokenization (snake_case, camelCase, numbers, identifiers).
- Natural language query tokenization.
- BM25Retriever wrapping rank_bm25 with rank extraction for evaluation.
"""

import pickle
import re
from collections.abc import Sequence

import numpy as np

# Regex patterns for code tokenization
# 1. Matches camelCase / PascalCase transitions and preserves alphanumeric subwords (e.g., "camelCase", "l2", "dim256")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)|\d+")
# 2. General word pattern for text queries: letters, digits, underscores
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize_code(code: str) -> list[str]:
    """Tokenize source code into normalized sub-identifiers.

    Splits snake_case, camelCase, PascalCase, numbers, and identifiers,
    lowercasing all sub-tokens and stripping punctuation.

    Examples:
        >>> tokenize_code("def compute_l2_norm(vectorA): return np.sqrt(x)")
        ['def', 'compute', 'l2', 'norm', 'vector', 'a', 'return', 'np', 'sqrt', 'x']

    Args:
        code: Raw source code string.

    Returns:
        List of lowercased sub-token strings.
    """
    if not code:
        return []

    tokens: list[str] = []
    # Find all primary word-like chunks
    for word in _WORD_RE.findall(code):
        # If the word contains underscores (snake_case), split on underscores first
        parts = word.split("_") if "_" in word else [word]
        for part in parts:
            if not part:
                continue
            # Apply camelCase / identifier sub-splitting
            subwords = _CAMEL_RE.findall(part)
            if subwords:
                tokens.extend(s.lower() for s in subwords)
            else:
                tokens.append(part.lower())

    return tokens


def tokenize_text(text: str) -> list[str]:
    """Tokenize natural language query into lowercased words.

    Examples:
        >>> tokenize_text("Find the euclidean distance between two vectors.")
        ['find', 'the', 'euclidean', 'distance', 'between', 'two', 'vectors']

    Args:
        text: Raw natural language query string.

    Returns:
        List of lowercased token strings.
    """
    if not text:
        return []

    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        parts = word.split("_") if "_" in word else [word]
        for part in parts:
            if not part:
                continue
            subwords = _CAMEL_RE.findall(part)
            if subwords:
                tokens.extend(s.lower() for s in subwords)
            else:
                tokens.append(part.lower())

    return tokens


class BM25Retriever:
    """High-performance BM25 Okapi lexical retriever with inverted indexing.

    Attributes:
        k1: BM25 term frequency saturation parameter (default: 1.5).
        b: BM25 document length normalization parameter (default: 0.75).
        corpus_size: Number of documents indexed.
        avgdl: Average document length across corpus.
        len_norm: Precomputed document length normalization array of shape (N,).
        inverted_index: Mapping from term to (doc_indices, term_frequencies) arrays.
        idf: Mapping from term to precomputed Okapi IDF.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25Retriever with tuning parameters.

        Args:
            k1: Term frequency saturation constant (typically 1.2 - 2.0).
            b: Document length normalization constant (typically 0.75).
        """
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.len_norm: np.ndarray | None = None
        self.inverted_index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}

    def index(
        self,
        corpus: Sequence[str],
        show_progress: bool = False,
        progress_interval: int = 50000,
    ) -> None:
        """Tokenize code corpus and build the inverted index.

        Args:
            corpus: Sequence of source code strings to index.
            show_progress: If True, prints real-time milestone progress lines.
            progress_interval: Step interval for progress logging.
        """
        import time
        from collections import Counter

        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return

        t0 = time.time()
        tokenized_corpus: list[list[str]] = []
        for i, doc in enumerate(corpus):
            tokenized_corpus.append(tokenize_code(doc))
            if show_progress and ((i + 1) % progress_interval == 0 or (i + 1) == self.corpus_size):
                pct = (i + 1) / self.corpus_size * 100.0
                elapsed = time.time() - t0
                speed = (i + 1) / max(elapsed, 1e-4)
                print(
                    f"  [BM25 Tokenize: {i+1:>7,}/{self.corpus_size:,} ({pct:>5.1f}%)] "
                    f"Elapsed: {elapsed:>5.1f}s | Speed: {speed:>6.1f} docs/s",
                    flush=True,
                )

        doc_lens = np.array([len(doc) for doc in tokenized_corpus], dtype=np.float32)
        self.avgdl = float(np.mean(doc_lens)) if self.corpus_size > 0 else 1.0
        self.len_norm = self.k1 * (1.0 - self.b + self.b * (doc_lens / max(self.avgdl, 1e-6)))

        doc_freqs: dict[str, list[int]] = {}
        term_doc_counts: dict[str, list[int]] = {}

        t_idx = time.time()
        for doc_id, doc_tokens in enumerate(tokenized_corpus):
            counts = Counter(doc_tokens)
            for term, count in counts.items():
                if term not in doc_freqs:
                    doc_freqs[term] = []
                    term_doc_counts[term] = []
                doc_freqs[term].append(doc_id)
                term_doc_counts[term].append(count)
            if show_progress and ((doc_id + 1) % progress_interval == 0 or (doc_id + 1) == self.corpus_size):
                pct = (doc_id + 1) / self.corpus_size * 100.0
                elapsed = time.time() - t_idx
                print(
                    f"  [BM25 Postings: {doc_id+1:>7,}/{self.corpus_size:,} ({pct:>5.1f}%)] "
                    f"Elapsed: {elapsed:>5.1f}s",
                    flush=True,
                )

        self.inverted_index = {}
        self.idf = {}
        for term, doc_list in doc_freqs.items():
            n = len(doc_list)
            idf_val = float(np.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1.0))
            self.idf[term] = idf_val
            self.inverted_index[term] = (
                np.array(doc_list, dtype=np.int32),
                np.array(term_doc_counts[term], dtype=np.float32),
            )

        if show_progress:
            print(
                f"  [OK] BM25 Index complete: {self.corpus_size:,} docs | "
                f"{len(self.inverted_index):,} vocab terms in {time.time() - t0:.1f}s",
                flush=True,
            )

    def get_scores(self, tokenized_query: Sequence[str]) -> np.ndarray:
        """Compute BM25 scores across all documents for a tokenized query.

        Args:
            tokenized_query: Sequence of query tokens.

        Returns:
            1D numpy array of scores of shape (corpus_size,).
        """
        if self.len_norm is None:
            raise RuntimeError("Index has not been built. Call index() first.")

        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for term in tokenized_query:
            posting = self.inverted_index.get(term)
            if posting is None:
                continue
            doc_ids, freqs = posting
            idf_val = self.idf[term]
            scores[doc_ids] += idf_val * (freqs * (self.k1 + 1.0)) / (freqs + self.len_norm[doc_ids])

        return scores

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve top-K matching documents for a query.

        Args:
            query: Natural language query string.
            top_k: Number of highest-scoring documents to return.

        Returns:
            Tuple of (top_indices, top_scores) sorted in descending order of score.

        Raises:
            RuntimeError: If index() has not been called.
        """
        if self.len_norm is None:
            raise RuntimeError("Index has not been built. Call index() first.")

        top_k = min(top_k, self.corpus_size)
        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        scores = self.get_scores(tokenized_query)
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            partitioned = np.argpartition(scores, -top_k)[-top_k:]
            sorted_within = np.argsort(scores[partitioned])[::-1]
            top_indices = partitioned[sorted_within]

        return top_indices, scores[top_indices]

    def get_ranks(
        self,
        queries: Sequence[str],
        ground_truth_indices: Sequence[int],
        max_candidates: int | None = None,
    ) -> np.ndarray:
        """Compute the 1-based ranks of ground-truth documents for each query.

        If a ground truth document receives a score of 0 (no lexical overlap)
        or ranks beyond max_candidates, its rank is marked as float('inf').

        Args:
            queries: Sequence of natural language query strings.
            ground_truth_indices: True document index for each query.
            max_candidates: Optional cutoff rank threshold (e.g. 1000).

        Returns:
            1D numpy array of 1-based ranks of shape (|queries|,).

        Raises:
            RuntimeError: If index has not been built.
            ValueError: If lengths of queries and ground_truth_indices mismatch.
        """
        if self.len_norm is None:
            raise RuntimeError("Index has not been built. Call index() first.")
        if len(queries) != len(ground_truth_indices):
            raise ValueError("queries and ground_truth_indices must have identical length.")

        ranks = np.full(len(queries), fill_value=np.inf, dtype=np.float64)

        for i, (query, gt_idx) in enumerate(zip(queries, ground_truth_indices)):
            tokenized_query = tokenize_text(query)
            if not tokenized_query or gt_idx < 0 or gt_idx >= self.corpus_size:
                continue

            scores = self.get_scores(tokenized_query)
            target_score = float(scores[gt_idx])

            if target_score <= 0.0:
                continue

            higher_count = int(np.sum(scores > target_score))
            tie_count = int(np.sum(scores == target_score)) - 1
            rank = higher_count + 1 + 0.5 * max(0, tie_count)

            if max_candidates is None or rank <= max_candidates:
                ranks[i] = rank

        return ranks

    def save(self, filepath: str) -> None:
        """Serialize retriever state to disk via pickle."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "BM25Retriever":
        """Load serialized retriever state from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
