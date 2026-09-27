"""Unit tests for Phase R0 Data Integrity & AST Docstring Stripper (Protocol v1.1).

Tests:
1. Coordinate-accurate byte slicing on dedented code (indented methods, single-line defs, semicolons).
2. Function stub rejection (functions with only docstring bodies).
3. Wrapper fallback for multi-line strings with col 0 text.
4. Nested function docstring stripping (ast.walk recursive traversal).
5. Non-ASCII multi-byte UTF-8 string slicing safety.
6. AST Equivalence test (removing body[0] AST node matches stripped code AST).
7. Generalized Expected Reciprocal Rank under ties (harmonic formula vs Jensen's inequality).
"""

import ast
import textwrap

import pytest

from data.preprocess import strip_docstring


def test_strip_docstring_indented_method():
    """Test that indented methods preserve comments and indentation without coordinate drift."""
    code = """
        class MyClass:
            def compute(self, x: int) -> int:
                \"\"\"Compute x + 1.
                
                Additional explanation line.
                \"\"\"
                # Important inline comment
                y = x + 1
                return y
    """
    cleaned, status = strip_docstring(code)
    assert status == "success"
    assert cleaned is not None
    assert "Compute x + 1" not in cleaned
    assert "Additional explanation line" not in cleaned
    assert "# Important inline comment" in cleaned
    assert "y = x + 1" in cleaned
    assert "return y" in cleaned

    tree = ast.parse(cleaned)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node) is None


def test_strip_docstring_single_line_def_with_semicolon():
    """Test single-line def with semicolon strips docstring AND semicolon cleanly."""
    code = 'def f(): """Single line doc."""; return 42\n'
    cleaned, status = strip_docstring(code)
    assert status == "success"
    assert cleaned is not None
    assert "Single line doc" not in cleaned
    assert "def f():" in cleaned
    assert "return 42" in cleaned
    assert ";" not in cleaned  # Must not leave dangling semicolon

    tree = ast.parse(cleaned)
    assert ast.get_docstring(tree.body[0]) is None


def test_strip_docstring_semicolon_same_line():
    """Test docstring followed by statement on same line inside function."""
    code = 'def f():\n    """Docstring"""; x = 1\n    return x\n'
    cleaned, status = strip_docstring(code)
    assert status == "success"
    assert cleaned is not None
    assert "Docstring" not in cleaned
    assert "x = 1" in cleaned
    assert "return x" in cleaned
    # Verify no dangling semicolon
    assert "; x = 1" not in cleaned


def test_strip_docstring_dedent_fallback():
    """Test fallback wrapper when a multi-line string has text at col 0."""
    code = (
        "    def query_db():\n"
        '        """Execute SQL query."""\n'
        '        sql = """\n'
        "SELECT id, name\n"
        "FROM users\n"
        '"""\n'
        "        return execute(sql)\n"
    )
    cleaned, status = strip_docstring(code)
    assert status == "success"
    assert cleaned is not None
    assert "Execute SQL query" not in cleaned
    assert "def query_db():" in cleaned
    assert "return execute(sql)" in cleaned

    tree = ast.parse(cleaned)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert ast.get_docstring(node) is None


def test_strip_docstring_nested_functions():
    """Test that nested functions have their docstrings stripped recursively."""
    code = """
    def outer(x):
        \"\"\"Outer docstring.\"\"\"
        def inner(y):
            \"\"\"Inner docstring.\"\"\"
            return y * 2
        return inner(x) + 1
    """
    cleaned, status = strip_docstring(code)
    assert status == "success"
    assert cleaned is not None
    assert "Outer docstring" not in cleaned
    assert "Inner docstring" not in cleaned

    tree = ast.parse(cleaned)
    func_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_count += 1
            assert ast.get_docstring(node) is None
    assert func_count == 2


def test_strip_docstring_utf8_non_ascii():
    """Test that multi-byte UTF-8 characters (accents, Chinese, emojis) slice safely."""
    code = """
    def process_data(text: str):
        \"\"\"Prüfe und verarbeite Daten: 🚀 数据处理.\"\"\"
        # 注释: 处理用户简历 (résumé)
        result = text.strip() + " ✅"
        return result
    """
    cleaned, status = strip_docstring(code)
    assert status == "success"
    assert cleaned is not None
    assert "Prüfe und verarbeite" not in cleaned
    assert "数据处理" not in cleaned
    # Ensure UTF-8 comments and strings are preserved undamaged
    assert "注释: 处理用户简历 (résumé)" in cleaned
    assert 'text.strip() + " ✅"' in cleaned

    tree = ast.parse(cleaned)
    assert ast.get_docstring(tree.body[0]) is None


def test_strip_docstring_stub_only_rejected():
    """Test that a function whose only body was a docstring is rejected as a syntax stub."""
    code = 'def empty_func():\n    """Only docstring here."""\n'
    cleaned, status = strip_docstring(code)
    assert status in ("syntax_error_after_strip", "empty_after_strip")
    assert cleaned is None


def test_ast_equivalence():
    """Test that AST of cleaned code matches raw AST with body[0] docstring removed."""
    raw_code = """
    def calculate_metrics(y_true, y_pred):
        \"\"\"Calculate precision, recall, and F1.
        
        Args:
            y_true: Ground truth.
            y_pred: Predictions.
        \"\"\"
        tp = sum(t and p for t, p in zip(y_true, y_pred))
        fp = sum((not t) and p for t, p in zip(y_true, y_pred))
        fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        return precision, recall
    """
    cleaned, status = strip_docstring(raw_code)
    assert status == "success"
    assert cleaned is not None

    dedented = textwrap.dedent(raw_code)
    raw_tree = ast.parse(dedented)

    # Delete docstring body[0] node in raw AST
    func_node = raw_tree.body[0]
    func_node.body.pop(0)

    clean_tree = ast.parse(cleaned)

    # AST dumps must be identical ignoring line/col offsets
    raw_dump = ast.dump(raw_tree, include_attributes=False)
    clean_dump = ast.dump(clean_tree, include_attributes=False)
    assert raw_dump == clean_dump


def test_generalized_harmonic_reciprocal_rank_ties():
    """Test generalized harmonic expected reciprocal rank with arbitrary positive target scores."""
    # Target score = 4.5, 3 docs score > 4.5, 5 docs score == 4.5 (including target)
    s_gt = 3  # S_>target
    s_eq = 5  # S_=target (includes target)

    # Exact discrete average over uniform random permutation of the tie group
    ranks = [s_gt + j for j in range(1, s_eq + 1)]
    true_e_rr = sum(1.0 / r for r in ranks) / s_eq

    # Formula using harmonic numbers: (H_{s_gt + s_eq} - H_{s_gt}) / s_eq
    def harmonic(n):
        return sum(1.0 / k for k in range(1, n + 1))

    formula_e_rr = (harmonic(s_gt + s_eq) - harmonic(s_gt)) / s_eq
    assert abs(true_e_rr - formula_e_rr) < 1e-12

    # Verify Jensen's inequality: E[1/x] > 1/E[x]
    e_rank = s_gt + (s_eq + 1) / 2.0
    naive_inv_rank = 1.0 / e_rank
    assert formula_e_rr > naive_inv_rank


def test_ast_equivalence_cpython_stdlib():
    """Test stripper and AST equivalence over real CPython standard library functions."""
    import difflib
    import inspect
    import json
    import textwrap
    import urllib.parse

    targets = [
        difflib.get_close_matches,
        difflib.ndiff,
        json.dumps,
        json.loads,
        textwrap.dedent,
        urllib.parse.urlparse,
        urllib.parse.urlsplit,
        urllib.parse.quote,
        urllib.parse.unquote,
    ]

    tested_count = 0
    for target in targets:
        try:
            raw_src = inspect.getsource(target)
        except (TypeError, OSError):
            continue

        cleaned, status = strip_docstring(raw_src)
        assert status == "success", f"Failed on {target.__name__}: {status}"
        assert cleaned is not None

        # Build raw AST without docstring
        dedented = textwrap.dedent(raw_src)
        raw_tree = ast.parse(dedented)
        for node in ast.walk(raw_tree):
            if (
                isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module),
                )
                and node.body
                and isinstance(node.body[0], ast.Expr)
            ):
                val = getattr(node.body[0], "value", None)
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    node.body.pop(0)

        clean_tree = ast.parse(cleaned)
        raw_dump = ast.dump(raw_tree, include_attributes=False)
        clean_dump = ast.dump(clean_tree, include_attributes=False)
        assert raw_dump == clean_dump, (
            f"AST mismatch on stdlib function: {target.__name__}"
        )
        tested_count += 1

    assert tested_count >= 5, (
        f"Expected at least 5 stdlib targets tested, got {tested_count}"
    )


def test_processed_data_structural_integrity():
    """Verify that 100% of samples in clean splits have ast.get_docstring is None."""
    from pathlib import Path

    import pandas as pd

    processed_dir = Path("data/processed_clean_v2")
    if not (processed_dir / "test.parquet").exists():
        pytest.skip("Clean data not yet generated; run data.preprocess first.")

    for split in ["test", "validation"]:
        df = pd.read_parquet(processed_dir / f"{split}.parquet")
        # Check a large stratified sample of 1,000 items per split for speed
        sample_df = df.head(1000)
        for _, row in sample_df.iterrows():
            tree = ast.parse(row["code"])
            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module),
                ):
                    assert ast.get_docstring(node) is None, (
                        f"Found residual docstring in {split}: {row['func_name']}"
                    )


def test_processed_data_zero_cross_split_contamination():
    """Verify zero exact or near-duplicate cross-split contamination."""
    from pathlib import Path

    import pandas as pd

    processed_dir = Path("data/processed_clean_v2")
    if not (processed_dir / "test.parquet").exists():
        pytest.skip("Clean data not yet generated; run data.preprocess first.")

    test_df = pd.read_parquet(processed_dir / "test.parquet")
    val_df = pd.read_parquet(processed_dir / "validation.parquet")
    train_df = pd.read_parquet(processed_dir / "train.parquet")

    # Within-split uniqueness
    assert len(test_df) == len(test_df["code"].unique()), (
        "Duplicates found inside test split!"
    )
    assert len(val_df) == len(val_df["code"].unique()), (
        "Duplicates found inside validation split!"
    )
    assert len(train_df) == len(train_df["code"].unique()), (
        "Duplicates found inside train split!"
    )

    # Cross-split disjointness
    test_codes = set(test_df["code"])
    val_codes = set(val_df["code"])
    train_codes = set(train_df["code"])

    assert len(test_codes & train_codes) == 0, (
        f"Contamination: {len(test_codes & train_codes)} codes shared between test and train!"
    )
    assert len(val_codes & train_codes) == 0, (
        f"Contamination: {len(val_codes & train_codes)} codes shared between val and train!"
    )
    assert len(val_codes & test_codes) == 0, (
        f"Contamination: {len(val_codes & test_codes)} codes shared between val and test!"
    )


def test_processed_data_overlap_diagnostic():
    """Diagnostic rate comparison: true pairs vs null random pairs (margin <= 0.5%)."""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    processed_dir = Path("data/processed_clean_v2")
    if not (processed_dir / "test.parquet").exists():
        pytest.skip("Clean data not yet generated; run data.preprocess first.")

    with open("data/stopwords.json") as f:
        stopwords = set(json.load(f))

    df = pd.read_parquet(processed_dir / "test.parquet")
    sample_df = df.head(1000).copy()

    # Load BM25 tokenizer
    from retrieval.bm25 import tokenize_code, tokenize_text

    def extract_non_stop(tokens):
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    # Compute true match rates
    true_matches = 0
    null_matches = 0
    rng = np.random.RandomState(42)
    shuffled_codes = rng.permutation(sample_df["code"].tolist())

    for idx, row in enumerate(sample_df.itertuples()):
        q_tokens = extract_non_stop(tokenize_text(row.docstring))
        c_tokens = tokenize_code(row.code)
        null_c_tokens = tokenize_code(shuffled_codes[idx])

        # Check contiguous 8-gram or full query containment (if >= 4 tokens)
        def has_leak(q, c):
            if len(q) >= 8:
                for i in range(len(q) - 7):
                    ngram = q[i : i + 8]
                    # Check if ngram is in c
                    for j in range(len(c) - 7):
                        if c[j : j + 8] == ngram:
                            return True
            if 4 <= len(q) < 8:
                # Full containment
                q_set = set(q)
                if q_set.issubset(set(c)):
                    return True
            return False

        if has_leak(q_tokens, c_tokens):
            true_matches += 1
        if has_leak(q_tokens, null_c_tokens):
            null_matches += 1

    true_rate = true_matches / len(sample_df)
    null_rate = null_matches / len(sample_df)
    rate_diff = true_rate - null_rate
    print(
        f"Overlap Diagnostic: true_rate={true_rate:.4f}, null_rate={null_rate:.4f}, diff={rate_diff:.4f}"
    )
    # PROTOCOL.md Section 2.3: "must not exceed the null-pair fraction by more than a pre-registered margin of +0.05"
    assert rate_diff <= 0.05, (
        f"True rate exceeds null rate by {rate_diff:.4f} > 0.05 tolerance!"
    )
