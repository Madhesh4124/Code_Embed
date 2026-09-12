"""Unit tests for retrieval/bm25.py.

Tests cover:
- Sub-identifier code tokenization (camelCase, snake_case, numbers)
- Query text tokenization
- Index construction and top-K search
- Ground truth rank calculation and unretrieved targets
- Serialization (save / load)
"""

import math
from pathlib import Path

import pytest

from retrieval.bm25 import BM25Retriever, tokenize_code, tokenize_text


class TestTokenization:
    """Tests for code and text tokenizers."""

    def test_snake_case_code(self):
        code = "def compute_l2_distance(vector_a, vector_b):"
        tokens = tokenize_code(code)
        assert "def" in tokens
        assert "compute" in tokens
        assert "l2" in tokens
        assert "distance" in tokens
        assert "vector" in tokens
        assert "a" in tokens
        assert "b" in tokens

    def test_camel_and_pascal_case_code(self):
        code = "class VectorCalculator: def getDotProduct(self, vec):"
        tokens = tokenize_code(code)
        assert "class" in tokens
        assert "vector" in tokens
        assert "calculator" in tokens
        assert "get" in tokens
        assert "dot" in tokens
        assert "product" in tokens

    def test_empty_inputs(self):
        assert tokenize_code("") == []
        assert tokenize_code("   \n\t  ") == []
        assert tokenize_text("") == []
        assert tokenize_text("   \n\t  ") == []

    def test_text_tokenization(self):
        text = "Calculate the L2-norm between two matrices."
        tokens = tokenize_text(text)
        assert tokens == ["calculate", "the", "l2", "norm", "between", "two", "matrices"]


class TestBM25Retriever:
    """Tests for BM25 indexing and retrieval mechanics."""

    @pytest.fixture
    def sample_corpus(self):
        return [
            "def add_numbers(a, b): return a + b",
            "def multiply_matrices(mat_a, mat_b): return mat_a @ mat_b",
            "def compute_euclidean_distance(p1, p2): return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5",
            "def binary_search(array, target): return 0",
        ]

    def test_unindexed_error(self):
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="Index has not been built"):
            retriever.search("test")

        with pytest.raises(RuntimeError, match="Index has not been built"):
            retriever.get_ranks(["test"], [0])

    def test_indexing_and_search(self, sample_corpus):
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index(sample_corpus)
        assert retriever.corpus_size == 4

        # Search for distance calculation
        indices, scores = retriever.search("euclidean distance between points", top_k=2)
        assert len(indices) == 2
        # Doc 2 is the euclidean distance implementation
        assert indices[0] == 2
        assert scores[0] > scores[1]

    def test_get_ranks_exact_matches(self, sample_corpus):
        retriever = BM25Retriever()
        retriever.index(sample_corpus)

        queries = [
            "add two numbers",
            "multiply two matrices",
            "binary search algorithm",
        ]
        ground_truth = [0, 1, 3]

        ranks = retriever.get_ranks(queries, ground_truth)
        assert len(ranks) == 3
        # All three queries should rank their respective targets in #1
        assert ranks[0] == 1.0
        assert ranks[1] == 1.0
        assert ranks[2] == 1.0

    def test_get_ranks_unretrieved(self, sample_corpus):
        retriever = BM25Retriever()
        retriever.index(sample_corpus)

        # Query with completely unrelated terms
        queries = ["astrophysics telescope galaxy redshift"]
        ground_truth = [0]

        ranks = retriever.get_ranks(queries, ground_truth)
        # Because target score is 0.0, rank should be inf
        assert math.isinf(ranks[0])

    def test_save_and_load(self, sample_corpus, tmp_path: Path):
        retriever = BM25Retriever()
        retriever.index(sample_corpus)

        save_file = str(tmp_path / "bm25_test.pkl")
        retriever.save(save_file)

        loaded = BM25Retriever.load(save_file)
        assert loaded.corpus_size == 4
        indices, _ = loaded.search("binary search", top_k=1)
        assert indices[0] == 3

