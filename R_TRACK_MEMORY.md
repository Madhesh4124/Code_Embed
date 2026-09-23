# CodeEmbed — R-Track Research Memory & Context

> **Historical Invalidation Notice**:
> Phases 1–6 benchmarks (BM25 MRR 0.9498, neural MRR 0.9383) were computed on CodeSearchNet data with 100% docstring-in-code label leakage (`hit = query in code == 100%`).
> The project operates under the **R-Track (Remediation & Pre-Registered Protocol)**.
> - Base Protocol: [`PROTOCOL.md`](PROTOCOL.md) (Git Tag `protocol-v1`, Commit `3e62e8a`)
> - Protocol Revisions & Errata: [`PROTOCOL_ERRATA.md`](PROTOCOL_ERRATA.md) (Protocol v1.1)

---

## 1. Executive Summary & Root Cause

1. **The Leak**:
   - In CodeSearchNet, `func_code_string` retained verbatim docstrings inside triple quotes, while `func_documentation_string` was used as the natural language query.
   - The historical data pipeline copied `func_code_string` directly into `code`. As a result, 100.0% of test queries were present as identical substrings inside the target code document.
   - BM25 achieved an inflated MRR of 0.9498 because it matched the query against its verbatim embedded clone.
2. **The Remediation Protocol**:
   - All historical benchmarks and conclusions were marked invalid.
   - A pre-registered research protocol was drafted and tagged (`protocol-v1`), with technical errata codified in [`PROTOCOL_ERRATA.md`](PROTOCOL_ERRATA.md).
   - AST line-and-column byte slicing on dedented code strips docstrings while preserving formatting and comments. Clean data is output to a separate versioned directory: `data/processed_clean_v2/`.
   - A MinHash LSH cross-split deduplication pipeline purges exact and near-duplicate collisions ($J \ge 0.85$).

---

## 2. Invalidation Evidence & Baseline State

### 2.1 Documentation Caution Banners
The invalidation banner was added to the top of [`walkthrough.md`](walkthrough.md), [`STUDY_GUIDE.md`](STUDY_GUIDE.md), and [`Memory.md`](Memory.md):
```markdown
> [!CAUTION]
> **INVALID: computed on leaky data, superseded by R-track.**
> All retrieval metrics, RQ conclusions, and benchmark comparisons recorded below in Phases 1–6 were computed on unstripped CodeSearchNet code containing verbatim docstring substrings (100% query leakage). The engineering modules (SDPA memory tiling, CSR mining, pretokenization cache, PyTorch models) remain valid, but all scientific numbers and answers are superseded by the leak-free R-track.
```

### 2.2 Database & Checkpoint Invalidation
- **MLflow Tracking**: `sqlite:///mlflow.db`
  - 6 historical runs across 3 experiments (`codeembed-baselines`, `codeembed-hard-negatives`, `codeembed-phase6-ablations`) tagged:
    `data_version=leaky_v1`, `validity=INVALID_LEAKY_DATA`.
- **Checkpoints**: `checkpoints/`
  - 59 `.pt` checkpoint files tagged: `data_version=leaky_v1`, `validity=INVALID_LEAKY_DATA`.

---

## 3. Protocol Key Points (See PROTOCOL.md and PROTOCOL_ERRATA.md)

Refer to [`PROTOCOL.md`](PROTOCOL.md) and [`PROTOCOL_ERRATA.md`](PROTOCOL_ERRATA.md) for full formal specifications. Key rules:

- **Target Output Directory**: `data/processed_clean_v2/`. Original `data/processed/` is preserved for historical comparison and tokenizer provenance.
- **Coordinate Alignment**: Slicing on UTF-8 bytes of `textwrap.dedent(code)`, with a dummy-function wrapper fallback for multi-line string zero-margin cases. Semicolon trailing characters in single-line functions are stripped.
- **Generalized Expected Reciprocal Rank ($\mathbb{E}[\text{RR}]$)**:
  $$\mathbb{E}[\text{RR}] = \frac{H_{S_{> \text{target}} + S_{= \text{target}}} - H_{S_{> \text{target}}}}{S_{= \text{target}}}$$
  Accounts for Jensen's inequality across arbitrary target score ties.
- **Cross-Split Dedup Standard**: Keep `train` intact; keep `test` canonical; purge exact code matches and MinHash LSH ($J \ge 0.85$) near-duplicates from `validation` and `test`. Normalized AST skeletons reserved for hard-negative filtering only.
- **Data Integrity Tests**: Structural AST test (`ast.get_docstring is None`), AST-dump equivalence test, and rate-based overlap diagnostic ($\text{Rate}_{\text{true}} - \text{Rate}_{\text{null}} \le 0.005$).
- **Pilot Role**: Designated as an Implementation Sanity Check, with a capped 6-run validation recipe fallback if not passing.
- **Scaling Gate (Phase 6.5)**: Proceed to scaling if and only if Category B (capacity errors) $> 50\%$ on 100 seeded validation failures; otherwise scaling is blocked.

---

## 4. Current File Map & Ready Components

| File Path | Description | Status |
|---|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | Immutable pre-registered protocol v1 | 🟢 Committed & Tagged `protocol-v1` |
| [`PROTOCOL_ERRATA.md`](PROTOCOL_ERRATA.md) | Protocol revisions and errata (v1.1) | 🟢 Active Specification |
| [`R_TRACK_MEMORY.md`](R_TRACK_MEMORY.md) | Active project state & tracking | 🟢 Active Source of Truth |
| [`data/stopwords.json`](data/stopwords.json) | Vendored English stopwords | 🟢 Created |
| [`data/preprocess.py`](data/preprocess.py) | Dedented byte-accurate stripper + MinHash LSH | 🟢 Complete (Phase R0) |
| [`data/processed_clean_v2/`](data/processed_clean_v2/) | Cleaned Parquet & `.pt` tokenized splits + hashes | 🟢 Generated & Verified |
| [`tests/test_data_integrity.py`](tests/test_data_integrity.py) | Coordinate slicing, AST equivalence, harmonic E[RR] tests | 🟢 13/13 Passed |
| [`data/pretokenize.py`](data/pretokenize.py) | Fast batch pre-tokenization into `.pt` binary tensor files | 🟢 Complete (Train/Val/Test) |
| [`scripts/run_baseline.py`](scripts/run_baseline.py) | BM25 benchmark script with MLflow tagging | 🟢 Complete |
| [`configs/basic_clean.yaml`](configs/basic_clean.yaml) | Basic Encoder config for clean data | 🟢 Created |
| [`checkpoints/basic_clean/best_basic.pt`](checkpoints/basic_clean/best_basic.pt) | Phase R2-A Basic Encoder checkpoint (~7.38M) | 🟢 Trained & Verified |

---

## 5. Remediation Milestones & Completed Phases

### 5.1 Phase R0: Clean Data Preprocessing & Deduplication (Complete)
- **Output Directory**: `data/processed_clean_v2/` (historical `data/processed/` preserved).
- **Train Sample Count**: 360,957 (parse drops: 3,798 = 0.92%, syntax error drops: 283).
- **Validation Sample Count**: 20,115 (purged 4 MinHash near-duplicates vs train, 0 exact collisions).
- **Test Sample Count**: 19,632 (purged 4 MinHash near-duplicates vs train, 0 exact collisions).
- **Hashes**: Saved in `data/processed_clean_v2/data_hashes.json`.
- **Pretokenization**: Generated binary tensors in `data/processed_clean_v2/` (`train_tokenized.pt`, `validation_tokenized.pt`, `test_tokenized.pt`).
- **Test Suite**: 22/22 unit tests passing across `test_data_integrity.py` and `test_bm25.py`.

### 5.2 Phase R1: Clean BM25 Lexical Baseline (Complete)
Following validation-set selection (§1.10 in `PROTOCOL_ERRATA.md`), the conservative **ATIRE negative-IDF floor variant** (`method="rank_bm25"`) was adopted to prevent claiming cheap neural wins. Evaluated on full clean splits (generalized harmonic tie-breaking):

| Split | Corpus Size | Queries | MRR | 95% CI | R@1 | R@5 | R@10 | NDCG@10 | MLflow Run ID |
|---|---|---|---|---|---|---|---|---|---|
| **Test** | 19,632 | 19,632 | **0.5108** | [0.5047, 0.5166] | 0.4052 | 0.6340 | 0.6993 | 0.5514 | `fb3b5313f1bb419bb330b7fc0dee6bf5` |
| **Validation** | 20,115 | 20,115 | **0.5214** | [0.5152, 0.5275] | 0.4107 | 0.6515 | 0.7192 | 0.5644 | `14feca9d5b024faab9da64beac12541b` |

> [!NOTE]
> **Scientific Finding**: When docstring query leakage is eliminated, BM25 drops from the inflated leaky score of **MRR 0.9513** to a realistic **MRR 0.5108** on the identical 19,632 test items (-44.1 pts MRR, -51.1 pts R@1).

### 5.3 Validation Battery & Critique Resolution (Complete)
All requested validation gates have been empirically verified:
1. **Like-for-Like Comparison on Identical 19,632 Test Items**:
   - Leaky code: MRR = 0.9513, R@1 = 0.9164, R@5 = 0.9885, R@10 = 0.9937, NDCG@10 = 0.9618.
   - Clean code: MRR = 0.5108, R@1 = 0.4052, R@5 = 0.6340, R@10 = 0.6993, NDCG@10 = 0.5514.
   - Net leakage inflation: +0.4405 MRR, +0.5112 R@1.
2. **Frozen Stratification on Train IDF (ATIRE Baseline)**:
   - **Validation ($N = 20,115$)**:
     - Zero ($c = 0$): 580 (2.88%), MRR = 0.0023, R@1 = 0.0000, R@10 = 0.0086
     - Low ($0 < c \le 0.30$): 6,321 (31.42%), MRR = 0.2489, R@1 = 0.1569, R@10 = 0.4297
     - High ($c > 0.30$): 13,214 (65.69%), MRR = 0.6745, R@1 = 0.5502, R@10 = 0.8888
     - Overall: MRR = 0.5214, R@1 = 0.4107, R@10 = 0.7192
   - **Test ($N = 19,632$)**:
     - Zero ($c = 0$): 535 (2.73%), MRR = 0.0099, R@1 = 0.0037, R@10 = 0.0206
     - Low ($0 < c \le 0.30$): 5,808 (29.58%), MRR = 0.2099, R@1 = 0.1284, R@10 = 0.3698
     - High ($c > 0.30$): 13,289 (67.69%), MRR = 0.6625, R@1 = 0.5423, R@10 = 0.8706
     - Overall: MRR = 0.5108, R@1 = 0.4052, R@10 = 0.6993
   - **Phase R3 Pilot Gate Reference (Updated)**: $0.75 \times 0.5214 = \mathbf{0.3910}$ validation MRR (or Low-overlap validation MRR $> \mathbf{0.2489}$).
3. **MinHash Deduplication Positive Control & Exact Nearest-Neighbor Search**:
   - The detector caught **100/100 (100.0%)** seeded synthetic $J \ge 0.85$ duplicates (mean $J = 0.927$).
   - **Exact Inverted-Index Nearest-Neighbor Jaccard on Test Split ($N = 500$ vs $360,957$ Train)**:
     - Min: 0.0000, P25: 0.0165, Median: **0.0314**, Mean: **0.0524**, P75: 0.0558, P90: 0.1111, P95: 0.1501, P99: 0.3941, Max: **0.8462**.
     - **Pairs with $J \ge 0.85$**: **0 / 500 (0.00%)** — bounds the true cross-split near-duplicate rate at $\le \mathbf{0.60\%}$ at the 95% confidence level (rule of three).
4. **Data Hygiene: Within-Split Duplicates & Kept vs Dropped Distribution**:
   - **Within-Split Duplicate Code**: Train = 0 (0.00%), Validation = 0 (0.00%), Test = 0 (0.00%).
   - **Within-Split Duplicate Queries**: Train = 16,552 (4.59%), Validation = 553 (2.75%), Test = 526 (2.68%).
   - **Kept vs. Dropped Length Distribution (Uniform Random Sample $N=10,000$, `seed=42`)**:
     - Kept Code Tokens: Median = **68.0**, Mean = 104.5, P10 = 27.0, P25 = 40.0, P75 = 121.0, P90 = 212.0.
     - Dropped Code Tokens: Median = **38.0**, Mean = 72.2, P10 = 15.0, P25 = 20.0, P75 = 79.0, P90 = 151.0.
     - *Observation*: Dropped code is selectively shorter due to the $<10$ token filter, $<3$ word docstrings, and empty migration/init stubs.
   - **Top Repositories**:
     - Kept: `saltstack/salt` (290), `materialsproject/pymatgen` (62), `brocade/pynos` (52), `mitsei/dlkit` (51), `google/grr` (49).
     - Dropped: `StackStorm/pybind` (259), `twilio/twilio-python` (148), `saltstack/salt` (93), `mitsei/dlkit` (93), `fprimex/zdesk` (69).
5. **Overlap Diagnostic Audit on All 19,632 Test Pairs**:
   - True-pair matches: 241 / 19,632 (1.23%); Null-pair matches: 0 / 19,632 (0.00%); Net margin: +1.23%.
   - Root-cause breakdown of 241 matches: 118 (49.0%) body statements/variable names, 61 (25.3%) inline `#` comments, 58 (24.1%) function signatures/parameter names, 4 (1.7%) API doc URLs.
   - BM25 on hits: MRR = 0.8406, R@1 = 0.7510; BM25 on non-hits (98.77%): MRR = 0.4950, R@1 = 0.3902.
6. **BM25 Reference Parity & Full-Corpus Evaluation**:
   - **Full Corpus Benchmark (19,632 docs, 2,000 queries)**:
     - Reference BM25Okapi (ATIRE): **MRR = 0.5115**
     - Custom BM25Retriever (Robertson $\ln(\dots + 1.0)$): **MRR = 0.5005**
     - Absolute difference: **0.0110**
     - *Diagnosis*: Root cause identified as Robertson's $\ln(1 + \text{fraction})$ non-negative smoothing vs ATIRE's piecewise $\epsilon \cdot \overline{\text{IDF}}$ negative-frequency floor.
   - **Direct Numerical Parity (`method='rank_bm25'`)**:
     - Evaluated on 1,000 queries: Mean Pearson score correlation = **1.000000**, Mean absolute score diff = **$3.19 \times 10^{-5}$**, Max score diff = **$0.002868 < 0.005$** (PASS), MRR difference = **0.0000** (PASS).

### 5.4 Phase R2-A: Model 1 — Basic Encoder (Complete)
- **Model**: BaseEncoder (~7.38M parameters, 4L-256d-8h-1024ff, Pre-LN, MaskedMeanPooling, zero modality embeddings).
- **Training**: 2 epochs on 360,957 clean train samples (5,638 steps, batch size 128, AdamW, LR 3e-4, 10% linear warmup, cosine decay, temperature $\tau = 0.07$, symmetric in-batch false negative mask $M_{i,j}$).
- **Hardware & Latency**: 24.85 minutes total on NVIDIA RTX 4050 Laptop GPU (CUDA AMP).
- **Validation Evaluation (1,000 sampled queries against full 20,115 clean validation corpus)**:
  - **MRR**: **0.3714** (95% CI: [0.3469, 0.3976])
  - **Recall@1**: **0.2820** (95% CI: [0.2550, 0.3100])
  - **Recall@5**: **0.4630** (95% CI: [0.4300, 0.4950])
  - **Recall@10**: **0.5430** (95% CI: [0.5100, 0.5730])
  - **NDCG@10**: **0.4042** (95% CI: [0.3780, 0.4310])
  - **MLflow Run ID**: `8cc3cb36b0494d42be6bf7253c3ab2e1` (Experiment: `codeembed-clean-baselines`)
- **Key Scientific Finding**:
  - The minimal neural anchor (without learned modality embeddings) achieves **0.3714 Val MRR**, operating just beneath the Pilot Gate threshold ($0.75 \times 0.5214 = \mathbf{0.3910}$).
  - This establishes the empirical baseline for Phase R2-B (Shared Encoder) to measure $\Delta_{\text{modality}} = \text{MRR}(\text{Shared}) - \text{MRR}(\text{Basic})$ to answer **RQ2**.

---

## 6. Model Progression (Option 1: Faithful Phase-by-Phase)

We adopt the phase-by-phase model hierarchy from the original [`Phases.md`](Phases.md), training from the simplest baseline upwards to isolate each architectural ingredient:

```
[Phase R1: BM25 Clean Baseline]
   │ MRR = 0.5108 (Test), MRR = 0.5214 (Val)
   ▼
[Phase R2-A: Model 1 — Basic Encoder]
   │ • BaseEncoder (~7.38M params, 4L-256d-8h-1024ff, Pre-LN, MaskedMeanPooling)
   │ • Zero Modality Embeddings (pure unified code-text representation)
   │ • 2 epochs on clean data (in-batch InfoNCE, τ=0.07)
   │ • Simplest neural anchor for code search
   ▼
[Phase R2-B: Model 2 — Shared Encoder (Pilot Gate)]
   │ • SharedEncoder (~7.38M params, adds learned modality embeddings)
   │ • 2 epochs on clean data (in-batch InfoNCE, τ=0.07)
   │ • Official Pilot Gate: Val MRR ≥ 0.3910 (or Low-Overlap Val MRR > 0.2489)
   │ • Answers RQ2: Δ_modality = MRR(Shared) - MRR(Basic)
   ▼
[Phase R2-C: BM25 Hard Negative Mining]
   │ • Multithreaded vectorized BM25 mining on clean train.parquet
   │ • 3-tier false negative filters (docstring match, skeleton ≥ 20 nodes, MinHash J ≥ 0.70)
   │ • Export top-50 CSR indices (train_hard_negatives.npz)
   ▼
[Phase R2-D: Hard-Negative Shared Encoder Retraining]
   │ • Train Shared Encoder for 2 epochs from scratch (7 in-batch + 1 hard negative)
   │ • Answers RQ3: Δ_mining = MRR(Hard) - MRR(In-Batch)
   ▼
[Phase R4: Architecture & Hyperparameter Ablations]
   │ • Pooling: MaskedMeanPooling vs CLSPooling
   │ • Temperature: τ ∈ {0.05, 0.07, 0.10}
   │ • Sequence Length: L ∈ {128, 256}
```

---

## 7. Division of Responsibilities & Teammate Track

### 7.1 Active Track (Our Scope)
- **Phase R2-A**: Basic Encoder (Model 1, no modality embeddings).
- **Phase R2-B**: Shared Encoder (Model 2, learned modality embeddings + Pilot Gate).
- **Phase R2-C**: Hard-Negative Mining on clean train data.
- **Phase R2-D**: Hard-Negative Shared Encoder retraining from scratch.
- **Ablations**: Pooling, temperature, sequence length.

### 7.2 Teammate Track: Model 3 — Dual Encoder Baseline (Documented Handover)
*Designated for independent execution by a teammate to investigate RQ1 (Shared vs. Dual parameter efficiency).*

- **Architecture**: [`DualEncoder`](file:///d:/CODE/Projects/X/model/dual_encoder.py) (~14.76M parameters).
  - Code Encoder: 4 Pre-LN layers, $d_{\text{model}} = 256$, 8 heads, $d_{\text{ff}} = 1024$ (~7.38M params).
  - Query Encoder: 4 Pre-LN layers, $d_{\text{model}} = 256$, 8 heads, $d_{\text{ff}} = 1024$ (~7.38M params).
  - Decoupled parameter spaces with zero weight sharing.
- **Configuration**: [`configs/dual_clean.yaml`](file:///d:/CODE/Projects/X/configs/dual_clean.yaml)
  - Data: `data/processed_clean_v2/` (`train.parquet`, `validation.parquet`, `test.parquet`).
  - Pretokenization binary tensors auto-detected.
  - Hyperparameters: AdamW (LR 3e-4, weight decay 0.01, 10% linear warmup, cosine decay), batch size 128, $\tau = 0.07$, 2 epochs.
- **How Teammate Runs Dual Encoder**:
  ```powershell
  # 1. Activate environment
  .venv\Scripts\Activate.ps1

  # 2. Run clean in-batch dual encoder training
  uv run python scripts/run_dual.py --config configs/dual_clean.yaml

  # 3. Evaluate checkpoint on clean validation set
  uv run python evaluation/evaluate.py --model-type dual --checkpoint checkpoints/dual_clean/best_model.pt --split validation
  ```
- **Comparison Objective (RQ1)**:
  Compare Dual Encoder validation/test MRR against Shared Encoder (7.38M) to test whether parameter specialization justifies a 2× increase in model footprint (14.76M vs 7.38M) on leak-free data.

---

## 8. Current Status & Next Actions

- **Active Branch**: `r-phase`
- **Commit Strategy**: Modular commits per remediation phase:
  - Phase R0: Clean Data Preprocessing, AST coordinate slicing, MinHash LSH deduplication, and data integrity tests.
  - Phase R1: Clean Lexical Baseline calibration (ATIRE variant), exact test NN Jaccard audit, and protocol errata.
- **Current Status**: Phase R2-A (Model 1: Basic Encoder) complete (Validation MRR = 0.3714 [0.3469, 0.3976]).
- **Next Action**: Launch Phase R2-B (Model 2: Shared Encoder with learned modality embeddings, 2 epochs on clean data) to test Pilot Gate (Val MRR >= 0.3910) and isolate Δ_modality for RQ2.



