"""Data Preprocessing Pipeline — Leak-Free AST Stripping & MinHash LSH Dedup.

Phase R0 Protocol v1.1:
1. Strips docstrings via coordinate-accurate byte-span slicing on textwrap.dedent(code),
   with a wrapper fallback for multi-line string zero-margin cases. Slices UTF-8 bytes
   and cleans trailing semicolons. Saves canonical dedented code to `data/processed_clean_v2/`.
2. Gracefully drops unparseable/malformed snippets (Python 2 syntax) and logs drop rates.
3. Filters code by length (10 to 2048 words) and docstrings (meaningful, >= 3 words).
4. Deduplicates within each split on stripped canonical code.
5. Cross-Split Hygiene:
   - Exact hash deduplication: keeps train intact, removes exact matches from val & test.
   - Near-duplicate deduplication: MinHash LSH (64 perm, 16 bands x 4 rows, J >= 0.85),
     removes near-duplicates from val & test.
   - Val <-> Test cross-dedup: keeps test canonical, removes collisions from val.
6. Preserves historical `data/processed/` for before/after comparison and records
   its train SHA256 in `data/processed_clean_v2/data_hashes.json` for tokenizer provenance.
"""

import ast
import hashlib
import json
import sys
import textwrap
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Suppress invalid escape sequence warnings from legacy docstrings/code
warnings.filterwarnings("ignore", category=SyntaxWarning)

RAW_DIR = Path("data/raw")
HISTORICAL_PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR = Path("data/processed_clean_v2")


# ============================================================================
# 1. Byte-Accurate AST Docstring Stripper (Dedented Coordinates + Wrapper Fallback)
# ============================================================================


def smart_dedent(code: str) -> str:
    """Dedent code even when multi-line strings contain lines at column 0.

    Uses the leading indentation of the first non-blank line as the base margin.
    For all lines that begin with at least `margin` spaces, removes `margin` spaces.
    Lines with fewer spaces (e.g. blank lines, or raw strings at column 0) are kept.
    """
    lines = code.splitlines(keepends=True)
    first_indent = 0
    for line in lines:
        if line.strip():
            first_indent = len(line) - len(line.lstrip(" "))
            break

    if first_indent == 0:
        return code

    prefix = " " * first_indent
    dedented_lines = []
    for line in lines:
        if line.startswith(prefix):
            dedented_lines.append(line[first_indent:])
        else:
            dedented_lines.append(line)

    return "".join(dedented_lines)


def strip_docstring(code: str) -> tuple[str | None, str]:
    """Remove docstrings via byte-accurate AST slicing on dedented code.

    Slices on UTF-8 bytes of smart_dedent(code) to guarantee coordinate alignment.
    Falls back to a function wrapper if multi-line strings prevent dedenting.
    Returns (canonical_cleaned_dedented_code, status).
    """
    if not isinstance(code, str) or not code.strip():
        return None, "empty_input"

    # Step 1: Smart dedent based on first non-blank line indentation
    dedented = smart_dedent(code)
    wrapper_used = False

    try:
        tree = ast.parse(dedented)
        work_text = dedented
    except IndentationError:
        # Fallback: wrap in a dummy function if still has indentation anomaly
        wrapped = "def _scope_wrapper():\n" + textwrap.indent(code, "    ")
        try:
            tree = ast.parse(wrapped)
            work_text = wrapped
            wrapper_used = True
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            return None, "parse_error"
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return None, "parse_error"

    # Restrict strictly to node.body[0] docstrings (ast.get_docstring standard)
    # Recursively collect spans for all FunctionDef, AsyncFunctionDef, ClassDef, Module
    doc_spans = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            if (
                wrapper_used
                and isinstance(node, ast.FunctionDef)
                and node.name == "_scope_wrapper"
            ):
                continue
            if node.body and isinstance(node.body[0], ast.Expr):
                val = getattr(node.body[0], "value", None)
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    stmt = node.body[0]
                    doc_spans.append(
                        (
                            stmt.lineno,
                            stmt.col_offset,
                            stmt.end_lineno,
                            stmt.end_col_offset,
                        )
                    )

    if not doc_spans:
        if wrapper_used:
            lines = work_text.splitlines(keepends=True)[1:]
            res = textwrap.dedent("".join(lines))
        else:
            res = work_text
        return res, "success"

    # Sort spans descending by line number so edits do not shift earlier line indices
    doc_spans.sort(key=lambda s: (s[0], s[1]), reverse=True)

    lines_bytes = [line.encode("utf-8") for line in work_text.splitlines(keepends=True)]

    for s_line, s_col, e_line, e_col in doc_spans:
        s_idx = s_line - 1
        e_idx = e_line - 1

        if s_idx == e_idx:
            # Single-line docstring
            line = lines_bytes[s_idx]
            prefix = line[:s_col]
            suffix = line[e_col:]
            # Clean up semicolon if docstring was followed by semicolon: """doc"""; x = 1
            stripped_suffix = suffix.lstrip()
            if stripped_suffix.startswith(b";"):
                suffix = stripped_suffix[1:].lstrip()
            if prefix.strip() == b"" and suffix.strip() == b"":
                lines_bytes[s_idx] = b""
            else:
                lines_bytes[s_idx] = prefix + suffix
        else:
            # Multi-line docstring
            prefix = lines_bytes[s_idx][:s_col]
            suffix = lines_bytes[e_idx][e_col:]
            stripped_suffix = suffix.lstrip()
            if stripped_suffix.startswith(b";"):
                suffix = stripped_suffix[1:].lstrip()

            lines_bytes[s_idx] = prefix if prefix.strip() != b"" else b""
            lines_bytes[e_idx] = suffix if suffix.strip() != b"" else b""
            for mid in range(s_idx + 1, e_idx):
                lines_bytes[mid] = b""

    cleaned_bytes = b"".join([l for l in lines_bytes if l != b""])
    try:
        cleaned = cleaned_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None, "decode_error"

    if wrapper_used:
        wrapper_lines = cleaned.splitlines(keepends=True)
        if wrapper_lines and "_scope_wrapper" in wrapper_lines[0]:
            wrapper_lines = wrapper_lines[1:]
        cleaned = textwrap.dedent("".join(wrapper_lines))

    if not cleaned.strip():
        return None, "empty_after_strip"

    try:
        clean_tree = ast.parse(cleaned)
    except (SyntaxError, ValueError, TypeError):
        return None, "syntax_error_after_strip"

    # Structural assertion: verify ast.get_docstring is None for all nodes
    for node in ast.walk(clean_tree):
        if (
            isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
            )
            and ast.get_docstring(node) is not None
        ):
            return None, "residual_docstring_detected"

    return cleaned, "success"


def is_valid_docstring(docstring: str) -> bool:
    """Check if docstring is meaningful (not empty, not trivial TODO/NONE, at least 3 words)."""
    if not isinstance(docstring, str):
        return False
    doc = docstring.strip()
    if len(doc) == 0:
        return False
    words = doc.split()
    if len(words) < 3:
        return False
    return doc.lower() not in {"todo", "none", "pass", "fixme"}


def is_valid_code(code: str) -> bool:
    """Check if code length is within reasonable bounds (10 to 2048 words)."""
    if not isinstance(code, str):
        return False
    tokens = code.strip().split()
    return 10 <= len(tokens) <= 2048


def _strip_docstrings_batch(codes: list[str]) -> list[tuple[str | None, str]]:
    """Worker function for multiprocessing pool."""
    return [strip_docstring(c) for c in codes]


# ============================================================================
# 2. MinHash LSH for Near-Duplicate Deduplication
# ============================================================================


class MinHashLSH:
    """Memory-efficient MinHash LSH index over token 3-grams."""

    def __init__(
        self, num_perm: int = 64, num_bands: int = 16, threshold: float = 0.85
    ):
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands  # 4
        self.threshold = threshold
        self.p = (1 << 61) - 1
        rng = np.random.RandomState(42)
        self.a = rng.randint(1, self.p, size=(num_perm,), dtype=np.int64)
        self.b = rng.randint(0, self.p, size=(num_perm,), dtype=np.int64)
        self.tables = [{} for _ in range(num_bands)]
        self.doc_token_hashes: dict[int, set[int]] = {}

    def _get_shingle_hashes(self, text: str) -> set[int]:
        words = text.split()
        if len(words) < 3:
            return {hash(w) & 0xFFFFFFFFFFFFFFF for w in words}
        shingles = set()
        for i in range(len(words) - 2):
            s = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            shingles.add(hash(s) & 0xFFFFFFFFFFFFFFF)
        return shingles

    def compute_signature(self, shingle_hashes: set[int]) -> np.ndarray:
        if not shingle_hashes:
            return np.zeros(self.num_perm, dtype=np.int64)
        h_arr = np.array(list(shingle_hashes), dtype=np.int64)
        sigs = np.zeros(self.num_perm, dtype=np.int64)
        for i in range(self.num_perm):
            val = (self.a[i] * h_arr + self.b[i]) % self.p
            sigs[i] = np.min(val)
        return sigs

    def index(self, doc_id: int, text: str) -> None:
        shingles = self._get_shingle_hashes(text)
        self.doc_token_hashes[doc_id] = shingles
        sig = self.compute_signature(shingles)
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band_key = tuple(sig[start : start + self.rows_per_band])
            if band_key not in self.tables[band_idx]:
                self.tables[band_idx][band_key] = []
            self.tables[band_idx][band_key].append(doc_id)

    def query(self, text: str) -> bool:
        """Return True if any indexed document has Jaccard similarity >= threshold."""
        shingles = self._get_shingle_hashes(text)
        if not shingles:
            return False
        sig = self.compute_signature(shingles)
        candidates = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band_key = tuple(sig[start : start + self.rows_per_band])
            if band_key in self.tables[band_idx]:
                candidates.update(self.tables[band_idx][band_key])

        for cand_id in candidates:
            cand_shingles = self.doc_token_hashes[cand_id]
            inter = len(shingles & cand_shingles)
            union = len(shingles | cand_shingles)
            if union > 0 and (inter / union) >= self.threshold:
                return True
        return False


# ============================================================================
# 3. Split Cleaning & Cross-Split Deduplication Pipeline
# ============================================================================


def clean_split_text(
    df: pd.DataFrame,
    split_name: str,
    max_workers: int = 8,
    batch_size: int = 2000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean, strip docstrings, and within-split deduplicate."""
    stats = {
        "raw_count": len(df),
        "missing_dropped": 0,
        "boilerplate_dropped": 0,
        "docstring_filter_dropped": 0,
        "parse_error_dropped": 0,
        "empty_after_strip_dropped": 0,
        "syntax_error_after_strip_dropped": 0,
        "code_length_dropped": 0,
        "within_split_duplicates_dropped": 0,
        "final_clean_count": 0,
    }

    df = df.rename(
        columns={"func_documentation_string": "docstring", "func_code_string": "code"}
    )
    before_missing = len(df)
    df = df.dropna(subset=["docstring", "code"])
    stats["missing_dropped"] = before_missing - len(df)

    if "func_path_in_repository" in df.columns:
        before_bp = len(df)
        df = df[
            ~df["func_path_in_repository"].str.contains(
                "migrations", case=False, na=False
            )
        ]
        df = df[
            ~df["func_path_in_repository"].str.contains(
                "__init__", case=False, na=False
            )
        ]
        stats["boilerplate_dropped"] = before_bp - len(df)

    before_doc = len(df)
    df = df[df["docstring"].apply(is_valid_docstring)]
    stats["docstring_filter_dropped"] = before_doc - len(df)

    print(
        f"  [{split_name}] Stripping docstrings from {len(df):,} samples across {max_workers} workers...",
        flush=True,
    )
    codes = df["code"].tolist()
    batches = [codes[i : i + batch_size] for i in range(0, len(codes), batch_size)]

    results: list[tuple[str | None, str]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_res in executor.map(_strip_docstrings_batch, batches):
            results.extend(batch_res)

    cleaned_codes = []
    valid_mask = []
    for c, status in results:
        if status == "success":
            cleaned_codes.append(c)
            valid_mask.append(True)
        else:
            cleaned_codes.append(None)
            valid_mask.append(False)
            if status == "parse_error":
                stats["parse_error_dropped"] += 1
            elif status == "empty_after_strip":
                stats["empty_after_strip_dropped"] += 1
            elif status in ("syntax_error_after_strip", "residual_docstring_detected"):
                stats["syntax_error_after_strip_dropped"] += 1

    df = df.copy()
    df["code"] = cleaned_codes
    df = df[valid_mask]

    before_len = len(df)
    df = df[df["code"].apply(is_valid_code)]
    stats["code_length_dropped"] = before_len - len(df)

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["code"])
    stats["within_split_duplicates_dropped"] = before_dedup - len(df)

    columns_to_keep = ["code", "docstring", "func_name"]
    df = df[[col for col in columns_to_keep if col in df.columns]].reset_index(
        drop=True
    )
    stats["final_clean_count"] = len(df)
    return df, stats


def compute_canonical_sha256(df: pd.DataFrame) -> str:
    """Compute SHA256 over PyArrow table record batches (metadata independent)."""
    table = pa.Table.from_pandas(df)
    sha = hashlib.sha256()
    for col_name in sorted(table.column_names):
        col = table[col_name]
        for chunk in col.chunks:
            for buffer in chunk.buffers():
                if buffer is not None:
                    sha.update(buffer.to_pybytes())
    return sha.hexdigest()


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hexadecimal digest of a file in streaming chunks."""
    if not filepath.exists():
        return ""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def preprocess_all(max_workers: int = 8) -> None:
    """Run full Phase R0 leak-free preprocessing pipeline with MinHash LSH dedup."""
    start_total = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("CodeEmbed — Phase R0 Clean Data Preprocessing (Protocol v1.1)", flush=True)
    print("=" * 70, flush=True)
    print(f"Target Output Directory: {PROCESSED_DIR}", flush=True)
    print(
        f"Historical Directory:    {HISTORICAL_PROCESSED_DIR} (preserved)", flush=True
    )

    # Record historical tokenizer training file hash for provenance
    old_train_file = HISTORICAL_PROCESSED_DIR / "train.parquet"
    old_train_hash = (
        compute_file_sha256(old_train_file) if old_train_file.exists() else "not_found"
    )
    print(f"Historical train hash (tokenizer provenance): {old_train_hash}", flush=True)

    # 1. Clean All Splits Locally
    splits_clean = {}
    splits_stats = {}
    for split in ["train", "validation", "test"]:
        print(f"\n>>> Local Cleaning & AST Stripping: {split.upper()}...", flush=True)
        raw_df = pd.read_parquet(RAW_DIR / f"{split}.parquet")
        clean_df, stats = clean_split_text(raw_df, split, max_workers=max_workers)
        splits_clean[split] = clean_df
        splits_stats[split] = stats
        parse_drop_pct = (stats["parse_error_dropped"] / stats["raw_count"]) * 100
        print(
            f"  [{split}] Cleaned: {stats['raw_count']:,} -> {stats['final_clean_count']:,} "
            f"(Parse drops: {stats['parse_error_dropped']} = {parse_drop_pct:.2f}%)",
            flush=True,
        )

    # 2. Build Train Index (Exact Hashes & MinHash LSH)
    train_df = splits_clean["train"]
    print(
        f"\n>>> Indexing TRAIN ({len(train_df):,} samples) into MinHash LSH...",
        flush=True,
    )
    t0_idx = time.time()
    train_exact_hashes = set(train_df["code"])
    train_lsh = MinHashLSH(num_perm=64, num_bands=16, threshold=0.85)
    for idx, code_str in enumerate(train_df["code"]):
        train_lsh.index(idx, code_str)
    print(
        f"  [OK] Train MinHash LSH indexed in {time.time() - t0_idx:.1f}s", flush=True
    )

    # 3. Deduplicate TEST against TRAIN (Keep Train Intact, Prune Test)
    test_df = splits_clean["test"]
    print(
        f"\n>>> Cross-Deduplicating TEST ({len(test_df):,} samples) vs TRAIN...",
        flush=True,
    )
    test_exact_drops = 0
    test_near_drops = 0
    test_kept = []

    for _, row in test_df.iterrows():
        c = row["code"]
        if c in train_exact_hashes:
            test_exact_drops += 1
            continue
        if train_lsh.query(c):
            test_near_drops += 1
            continue
        test_kept.append(row)

    test_clean_df = pd.DataFrame(test_kept).reset_index(drop=True)
    test_total_dropped = test_exact_drops + test_near_drops
    print(
        f"  [OK] TEST vs TRAIN: Purged {test_exact_drops} exact collisions + "
        f"{test_near_drops} MinHash near-duplicates (Total: {test_total_dropped}). "
        f"Final TEST: {len(test_clean_df):,}",
        flush=True,
    )

    # 4. Build Test Index for Val <-> Test Cross-Dedup (Keep Test Canonical)
    test_exact_hashes = set(test_clean_df["code"])
    test_lsh = MinHashLSH(num_perm=64, num_bands=16, threshold=0.85)
    for idx, code_str in enumerate(test_clean_df["code"]):
        test_lsh.index(idx, code_str)

    # 5. Deduplicate VALIDATION against TRAIN & TEST (Prune Validation)
    val_df = splits_clean["validation"]
    print(
        f"\n>>> Cross-Deduplicating VALIDATION ({len(val_df):,} samples) vs TRAIN & TEST...",
        flush=True,
    )
    val_train_exact_drops = 0
    val_train_near_drops = 0
    val_test_exact_drops = 0
    val_test_near_drops = 0
    val_kept = []

    for _, row in val_df.iterrows():
        c = row["code"]
        # Check against Train
        if c in train_exact_hashes:
            val_train_exact_drops += 1
            continue
        if train_lsh.query(c):
            val_train_near_drops += 1
            continue
        # Check against Test
        if c in test_exact_hashes:
            val_test_exact_drops += 1
            continue
        if test_lsh.query(c):
            val_test_near_drops += 1
            continue
        val_kept.append(row)

    val_clean_df = pd.DataFrame(val_kept).reset_index(drop=True)
    val_total_dropped = (
        val_train_exact_drops
        + val_train_near_drops
        + val_test_exact_drops
        + val_test_near_drops
    )
    print(
        f"  [OK] VALIDATION vs TRAIN/TEST: Purged {val_train_exact_drops} train-exact, "
        f"{val_train_near_drops} train-near, {val_test_exact_drops} test-exact, "
        f"{val_test_near_drops} test-near (Total: {val_total_dropped}). "
        f"Final VALIDATION: {len(val_clean_df):,}",
        flush=True,
    )

    # 6. Save Cleaned Splits to Versioned Directory
    print(f"\n>>> Writing clean Parquet splits to {PROCESSED_DIR}/...", flush=True)
    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "validation.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_clean_df.to_parquet(val_path, index=False)
    test_clean_df.to_parquet(test_path, index=False)

    # 7. Compute & Save Cryptographic Content Hashes
    print(">>> Computing canonical Arrow content SHA256 digests...", flush=True)
    hashes = {
        "train.parquet": compute_canonical_sha256(train_df),
        "validation.parquet": compute_canonical_sha256(val_clean_df),
        "test.parquet": compute_canonical_sha256(test_clean_df),
        "sample_counts": {
            "train": len(train_df),
            "validation": len(val_clean_df),
            "test": len(test_clean_df),
        },
        "tokenizer_provenance_train_hash": old_train_hash,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "protocol_version": "v1.1",
        "docstring_stripped": True,
        "canonical_code_dedented": True,
        "minhash_lsh_threshold": 0.85,
        "cross_split_dedup_protocol": "Keep train intact; keep test canonical; purge val & test",
    }

    hash_file = PROCESSED_DIR / "data_hashes.json"
    with open(hash_file, "w") as f:
        json.dump(hashes, f, indent=2)

    elapsed = time.time() - start_total
    print("\n" + "=" * 70, flush=True)
    print(
        f"Phase R0 Data Preprocessing Successfully Completed in {elapsed:.1f}s!",
        flush=True,
    )
    print("=" * 70, flush=True)
    print(f"  Train samples:      {len(train_df):,}")
    print(f"  Validation samples: {len(val_clean_df):,}")
    print(f"  Test samples:       {len(test_clean_df):,}")
    print(f"Hashes saved to:      {hash_file}", flush=True)

    splits_stats["train"]["cross_split_purged"] = 0
    splits_stats["train"]["final_after_cross_dedup"] = len(train_df)
    splits_stats["validation"]["cross_split_purged"] = val_total_dropped
    splits_stats["validation"]["final_after_cross_dedup"] = len(val_clean_df)
    splits_stats["test"]["cross_split_purged"] = test_total_dropped
    splits_stats["test"]["final_after_cross_dedup"] = len(test_clean_df)

    stats_df = pd.DataFrame(splits_stats)
    print("\n--- Pipeline Drop & Hygiene Statistics ---", flush=True)
    print(stats_df.to_string(), flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    workers = min(8, sys.maxsize)
    preprocess_all(max_workers=workers)
