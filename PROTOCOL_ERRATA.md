# Protocol Errata & Revisions — Protocol v1.1

> **Document Status**: Active Errata and Protocol Revisions to `protocol-v1` (Git commit `3e62e8a`).
> Addresses specification edge cases, coordinate alignment safeguards, generalized tie-breaking, and statistical rigor.

---

## 1. Specification Corrections

### 1.1 Output Directory & Tokenizer Provenance
- **Directory**: Clean data from Phase R0 will be written to `data/processed_clean_v2/` (`train.parquet`, `validation.parquet`, `test.parquet`).
- **Historical Preservation**: The original `data/processed/` directory is retained intact to preserve:
  1. The historical leaky baseline numbers for the final post-mortem report.
  2. The provenance of the 16k BPE tokenizer (which was trained on `data/processed/train.parquet`).
- **Metadata Logging**: The SHA256 of `data/processed/train.parquet` will be permanently logged in `data/processed_clean_v2/data_hashes.json` under `tokenizer_source_data_hash`.

### 1.2 Coordinate System & Dedent Margin Fallback
- **Problem**: When a code snippet contains a multi-line string literal with content at column 0 (e.g. SQL queries, raw HTML blocks), `textwrap.dedent` finds a minimum indentation margin of 0. In such cases, indented methods remain indented, causing `ast.parse` to fail with `IndentationError`.
- **Fallback Rule**:
  1. Attempt `ast.parse(textwrap.dedent(code))`.
  2. If an `IndentationError` occurs, wrap the code in a dummy function wrapper: `wrapped = "def _scope_wrapper():\n" + textwrap.indent(code, "    ")`.
  3. Parse the wrapped AST, extract docstring spans belonging to the inner function/class, and subtract 1 line and 4 columns to map back to original coordinates.
  4. Perform UTF-8 byte slicing and verify syntax re-parse.
  5. Log the count of samples requiring the wrapper fallback per split.

### 1.3 Single-Line Functions & Semicolon Cleanup
- **Syntax Correction**: In single-line definitions containing a docstring and subsequent statement (e.g. `def f(): """doc"""; return 42`), the docstring and any trailing semicolon/whitespace are sliced out simultaneously:
  $$\text{Target:} \quad \text{def f(): } ["""\text{doc}"""; \quad] \text{return 42} \quad \to \quad \text{def f(): return 42}$$
- **Grammar Assertion**: Note that `def f(): """doc"""` followed by an indented statement on the next line is invalid Python syntax (`IndentationError: unexpected indent`). Python grammar requires that any block following a single-line suite must begin with a colon on its own block.
- **Recursive Stripping**: The stripper and the structural leak test recursively traverse all nested classes and functions using `ast.walk`.

### 1.4 Normalized Skeleton Scope
- **Rule**: Using AST skeleton replacement (`var` / `const`) as a cross-split deduplication rule is **disallowed** because trivial functions, getters, setters, and one-line wrappers share identical skeletons, which would artificially purge legitimate code and distort the evaluation distribution.
- **Application**: Normalized AST skeletons are used **only** for BM25 hard-negative mining (excluding trivial near-duplicate false negatives) on functions with $\ge 20$ AST nodes.
- **Cross-Split Dedup Standard**: Cross-split deduplication relies strictly on:
  1. Exact SHA256 match of whitespace-compacted stripped code.
  2. MinHash LSH ($J \ge 0.85$ over word-level 3-grams).

### 1.5 Generalized Expected Reciprocal Rank ($\mathbb{E}[\text{RR}]$)
- **Scope**: Ties can occur at any score (including positive scores for identical code or dense embedding collisions), not just at score 0.
- **General Formula**: For any query with ground-truth document score $s_{\text{target}}$:
  $$S_{> \text{target}} = \sum_{d \ne \text{target}} \mathbb{I}(\text{score}_d > s_{\text{target}})$$
  $$S_{= \text{target}} = 1 + \sum_{d \ne \text{target}} \mathbb{I}(\text{score}_d == s_{\text{target}})$$
  The exact expected reciprocal rank under uniform random tie-breaking is:
  $$\mathbb{E}[\text{RR}] = \frac{1}{S_{= \text{target}}} \sum_{j=1}^{S_{= \text{target}}} \frac{1}{S_{> \text{target}} + j} = \frac{H_{S_{> \text{target}} + S_{= \text{target}}} - H_{S_{> \text{target}}}}{S_{= \text{target}}}$$
  where $H_n = \sum_{k=1}^n \frac{1}{k}$ is the $n$-th harmonic number.
- **Expected Recall@k**:
  $$\mathbb{E}[\text{R@k}] = \begin{cases}
  0 & \text{if } S_{> \text{target}} \ge k \\
  1 & \text{if } S_{> \text{target}} + S_{= \text{target}} \le k \\
  \frac{k - S_{> \text{target}}}{S_{= \text{target}}} & \text{if } S_{> \text{target}} < k < S_{> \text{target}} + S_{= \text{target}}
  \end{cases}$$

### 1.6 Overlap Diagnostic Protocol & Empirical Audit
- **Test Definition**:
  1. **Contiguous n-gram Match**: $n \ge 8$ contiguous code tokens matching the query.
  2. **Full-Query Containment**: For queries with $\ge 4$ non-stopword tokens, all tokens appearing in the code.
- **Empirical Results (Full Test Set, N = 19,632)**:
  - $\text{Rate}_{\text{true}} = 241 / 19,632 = 1.23\%$
  - $\text{Rate}_{\text{null}} = 0 / 19,632 = 0.00\%$
  - Margin $\Delta = +1.23\%$ (fails hypothetical 0.5% margin, but passes pre-registered 5.0% margin).
- **Manual Audit of the 241 Matches**:
  - **Function Signatures / Parameter Names**: 58 items (24.1%) — query describes the function using its exact argument names.
  - **Inline Comments**: 61 items (25.3%) — developer inline `#` comments retained in the function body duplicating docstring wording.
  - **Documentation / API URLs**: 4 items (1.7%) — Google / GitHub reference URLs.
  - **Body Statements / Literals**: 118 items (49.0%) — variable names, error strings, and log messages.
- **BM25 Impact**: BM25 achieves MRR = 0.8406 on the 241 overlap items vs MRR = 0.4950 on the remaining 19,391 non-overlap items.

### 1.7 Preprocessing Filter Provenance
- **Inherited Filters**: The following cleaning rules were inherited from the initial Phase 0 data pipeline (`data/preprocess.py`, 2026-09-08) and are formally codified in the errata:
  1. **Boilerplate Filter**: Drop repository paths containing `migrations` or `__init__.py` (drops ~4.8% of raw samples).
  2. **Short Docstrings Filter**: Drop docstrings with $< 3$ whitespace-delimited words (rejecting trivial placeholders like `""`, `"TODO"`, `"None"`, `"fix"`).
  3. **Code Length Filter**: Drop functions with $< 10$ tokens (empty stubs or trivial passes) or $> 2,048$ tokens.
  4. **Single-Character Stopword Rule**: All single-character tokens (e.g. `i`, `x`, `_`, `a`) and punctuation characters are treated as non-informative and filtered out during vocabulary overlap calculation and term coverage.

### 1.8 Frozen Stratification Bins & Pilot Gate Threshold
- **Stratification Formulation**:
  - IDF computed on clean train split (360,957 samples).
  - Stopwords from `data/stopwords.json` + Python keywords + single-character tokens (`len(t) <= 1`).
  - Frozen Bins:
    - **Zero-Overlap**: $\text{Coverage} == 0.0$
    - **Low-Overlap**: $0.0 < \text{Coverage} \le 0.30$
    - **High-Overlap**: $\text{Coverage} > 0.30$
- **Concrete Phase R3 Pilot Pass Gate (Updated for ATIRE Floor Baseline)**:
  - Validation BM25 Overall MRR = **0.5214** (Conservative ATIRE floor variant)
  - **Primary Gate Threshold**: $\text{MRR}_{\text{Dense, val}} \ge 0.75 \times 0.5214 = \mathbf{0.3910}$
  - **Secondary Gate Threshold**: $\text{MRR}_{\text{Dense, val, Low}} > \mathbf{0.2489}$ (BM25's Low-Overlap validation MRR)

### 1.9 Hard Negative Mining Hygiene
- Mining strictly excludes:
  1. Identical docstring intent ($\text{docstring}_i == \text{docstring}_j$).
  2. Normalized AST skeletons with $\ge 20$ AST nodes.
  3. MinHash LSH near-duplicates with $J \ge 0.70$.

### 1.10 Pre-Registration Deviation: BM25 IDF Formulation Selection
- **Deviation**: The base pre-registered protocol (`PROTOCOL.md` §3.2) specified Robertson's smooth IDF formula: $\text{IDF}(t) = \ln\left(1 + \frac{N - n + 0.5}{n + 0.5}\right)$.
- **Empirical Validation Comparison**: Evaluated on the full clean validation set ($N = 20,115$), the ATIRE piecewise negative-IDF floor variant (`rank_bm25.BM25Okapi`, with $\text{floor} = 0.25 \times \overline{\text{IDF}}$) achieved **MRR = 0.5214**, whereas Robertson smooth achieved **MRR = 0.5120** ($\Delta = +0.0094$).
- **Methodological Rule**: To maintain conservative scientific standards and prevent claiming cheap neural wins over a sub-optimal lexical baseline, the stronger baseline (**ATIRE floor, MRR 0.5214**) was formally selected on validation data.
- **Consequence**: Test split was evaluated strictly once with the chosen ATIRE variant (**Test MRR = 0.5108**), and the Phase R3 Pilot Gate was updated to $0.75 \times 0.5214 = \mathbf{0.3910}$.

---

## 2. Experimental Controls & Statistical Revisions

### 2.1 Pilot Designation & Warmup
- **Pilot Role**: Designated as an **Implementation Sanity Check**, not a performance gate.
- **Learning Rate Warmup**: Warmup steps are set to **10% of total training steps** (rather than a fixed step count), ensuring equal proportional warmup across 1-epoch and 2-epoch training schedules.

### 2.2 Capacity Error Labeling Rubric (Gate on Phase 6.5)
- Sample: 100 validation queries with dense rank $> 10$, sampled with fixed `seed=42`.
- Taxonomy:
  - **Category A**: Ambiguous or under-specified query; multiple valid functions in corpus.
  - **Category B**: Clear, specific query where model failed semantic alignment.
  - **Category C**: Dead code, malformed text, or data annotation noise.
- **Decision Rule**: Phase 6.5 scaling proceeds if and only if **Category B > 50%**. If Category A + C $\ge 50\%$, scaling is blocked.

### 2.3 Training Masking & Multiple Comparisons
- **In-Batch False Negative Masking**: Symmetric InfoNCE mask:
  $$M_{i,j} = \mathbb{I}(\text{query}_i == \text{query}_j \lor \text{code}_i == \text{code}_j)$$
  Masked pairs are excluded from the contrastive denominator in both directions (query $\to$ code and code $\to$ query).
- **Hypothesis Testing**: Pairwise test set differences apply Holm-Bonferroni correction based on bootstrap empirical p-values:
  $$p = 2 \times \min\left(\frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta \text{RR}^{*(b)} \le 0), \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta \text{RR}^{*(b)} \ge 0)\right)$$

### 2.4 Vendored Stopwords & Fixed Query Definition
- **Stopwords**: Vendored in `data/stopwords.json` (SHA256 logged), eliminating runtime network/package dependencies.
- **Query Definition**: The primary retrieval query is the first docstring paragraph / summary sentence (`docstring.strip().split("\n\n")[0].strip()`).
- **Pretrained Baseline**: `microsoft/codebert-base` and `sentence-transformers/all-MiniLM-L6-v2` evaluated in zero-shot mode as exploratory reference anchors.

