# CodeEmbed — Project Memory

> **Purpose**: Track implementation milestones, experimental results, and architectural decisions.
>
> 🎯 **Primary Confirmatory Benchmark (Protected Protocol §3 Result)**:
> - **Phase R3 Shared Encoder (4L, 7.38M)**: Evaluated strictly once on the clean test split ($N = 19,632$) per frozen pre-registered protocol:
>   - **In-Batch Shared**: Test MRR **0.4157** [0.4098, 0.4216], Recall@1 **0.3178**, Recall@10 **0.6018**.
>   - **BM25 Hard Negative Shared**: Test MRR **0.4155** [0.4095, 0.4215], Recall@1 **0.3184**, Recall@10 **0.6016**.
>   - **Semantic Search Advantage**: Outperforms BM25 by **+11.05 MRR points** on low-overlap queries (0.3204 vs 0.2099) and by **$5\times$** on zero-overlap queries (0.0487 vs 0.0099).
>
> 🔬 **Post-Protocol Exploratory Model Iteration**:
> *Additional architectures and training variants (FAISS Dense HN, Phase 6.5 capacity scaling, Dual Encoder) were explored after the primary evaluation. Their test-set numbers are reported for reference but were not protected from model selection bias.*
> - Primary Sources of Truth: [`R_TRACK_MEMORY.md`](R_TRACK_MEMORY.md), [`PROTOCOL.md`](PROTOCOL.md) (Git Tag `protocol-v1`), [`PROTOCOL_ERRATA.md`](PROTOCOL_ERRATA.md).
> - Active Branch: `main`

---

## Current Status

> *Note on Protocol Nomenclature: In [`PROTOCOL.md`](PROTOCOL.md) §3.2, Phase R2 defines BM25 Lexical Stratification & Pilot Reference. Neural training, statistical gates, and confirmatory evaluation are specified under Phase R3 (§3.3). Below, these training stages are mapped to sequential sub-tasks R3-A (Basic), R3-B (Shared Pilot), R3-C (Mining), R3-D (Hard Negatives), and R3-Final (Confirmatory Benchmark).*

| Component | Status | Last Updated | Notes |
|-----------|--------|--------------|-------|
| Phase R0: Clean Data Preprocessing | 🟢 Completed | 2026-09-20 | AST byte-sliced docstrings; MinHash LSH cross-split dedup; clean Parquet + `.pt` tokenized in `data/processed_clean_v2/` |
| Phase R1: Clean BM25 Lexical Baseline | 🟢 Completed | 2026-09-20 | ATIRE variant: Full test (19,632 queries) MRR **0.5108** [0.5047, 0.5168]; Val MRR **0.5214** [0.5152, 0.5275]; logged to MLflow |
| Phase R2: BM25 Stratification & Pilot Ref | 🟢 Completed | 2026-09-20 | Frozen bin edges ($c=0$, $0 < c \le 0.30$, $c > 0.30$); Pilot Gate threshold established at $\text{MRR} \ge \mathbf{0.3910}$ ($0.75 \times 0.5214$) |
| Phase R3-A: Model 1 — Basic Encoder | 🟢 Completed | 2026-09-23 | 2 epochs on clean data: Val MRR **0.3714** [0.3469, 0.3976], R@1 **0.2820**, R@10 **0.5430**; BaseEncoder (~7.38M, 0 modality); MLflow Run `8cc3cb36b0494d42be6bf7253c3ab2e1` |
| Phase R3-B: Model 2 — Shared Encoder | 🟢 Completed | 2026-09-24 | Pilot initial: 0.3423; Fallback Grid complete (6 runs). Winner `lr_5e-4_tau_0.05` achieves Full Val MRR **0.4033** (Low-Overlap: **0.2931**), PASSING both Pilot Gate criteria! |
| Phase R3-C: BM25 Hard Negative Mining | 🟢 Completed | 2026-09-24 | 360,957 clean train queries mined in 130.2s (2,773 q/s); 3-tier filters purged 18,594 false negatives (12,448 docstring + 5,589 AST skeleton + 557 MinHash); output matrix saved to `data/processed_clean_v2/train_hard_negatives.pt` |
| Phase R3-D: Hard Negative Shared Encoder | 🟢 Completed | 2026-09-25 | 2 epochs from scratch (7 in-batch + 1 hard negative): Full Val MRR **0.4074** (Low-Overlap: **0.3047** vs BM25 0.2489), 1k Sample MRR **0.4116** [0.3843, 0.4390], R@1 **0.3130**, R@10 **0.6060**; MLflow Run `c4e2e5c02be74bf19eacf4ea4fc68c5c` |
| Phase R4: Ablations (Pool/SeqLen) | 🟢 Completed | 2026-09-25 | CLSPooling suffers -61.2% collapse (MRR 0.1566); SeqLen L=128 retains 97.8% quality (MRR 0.3943) with 1.78x speedup; MLflow Runs `1dd691e839ee4684b17d990b51cde549`, `ef009dd84a66465a92847ab378b0bdea` |
| **Phase R3-Final: Confirmatory Test Benchmark** | 🟢 Completed | 2026-09-28 | **Primary Protected Test Evaluation (Protocol §3)**: Shared In-batch MRR **0.4157**, Hard-Neg MRR **0.4155** (+11.05 pts over BM25 on low-overlap). RQ2 lift +0.0384 ($p < 10^{-4}$). Capacity Rubric: 93% Category B (PASS); MLflow Run `c4a336367c154f9b9e18627f82304991` |
| *Exploratory*: Shared FAISS Dense HN (4L) | 🔵 Exploratory | 2026-09-28 | Post-R3 exploratory iteration: Val MRR **0.4084** [0.4027, 0.4141], Test MRR **0.4192** [0.4132, 0.4250] (unprotected from test selection bias). |
| *Exploratory*: Phase 6.5 Model Scaling (17.03M) | 🔵 Exploratory | 2026-09-29 | Post-R3 exploratory capacity scaling: 6L-384d Shared Encoder trained 4 epochs with FAISS dense HN. Full Val MRR **0.4620** [0.4559, 0.4679], Full Test MRR **0.4699** [0.4636, 0.4757]. MLflow Run `d9034df9dff84a419cd320d1594f0511`. |
| *Exploratory*: Dual Encoder (Teammate Track) | 🔵 Exploratory | 2026-09-28 | Independent handover: Dual baseline Test MRR **0.2900** (7M, decoupled); Dual with FAISS hard negatives evaluated at two temperatures on test: **0.4807** (τ=0.10) and **0.4684** (τ=0.07). Unprotected from test selection bias. |
| Pretrained Baseline Suite (RQ5) | 🟢 Completed | 2026-09-29 | Evaluated `all-MiniLM-L6-v2` (0.5837), `codebert-base` (0.0138 - anisotropy collapse), and `jina-embeddings-v2-base-code` (0.8294 SOTA) on clean test ($N=19,632$). |
| Interactive Demo & UI (Web & CLI) | 🟢 Completed | 2026-09-29 | Implemented FastAPI web UI (`demo/app.py`) and CLI search REPL (`demo/search.py`) with Hybrid RRF, Dense, BM25 modes, and overlap stratum badges. |


---

## Implemented Components

### Data Pipeline (`data/`)
- [x] `download.py` — CodeSearchNet Python download
- [x] `preprocess.py` — Cleaning, filtering, dedup
- [x] `pretokenize.py` — Offline binary tensor caching (`train_tokenized.pt`), eliminating CPU tokenization bottleneck (300x faster batch feeding)
- [x] `dataset.py` — PyTorch Dataset + Fast Collator with auto-detection for pre-tokenized tensors
- [x] `ablation_dataset.py` — AblationCodeSearchDataset with zero-overhead sequence length truncation slicing (Phase 6)
- [x] `splits.py` — Handled via data/processed/{train,validation,test}.parquet and create_dataloader()

### Tokenizer (`tokenizer/`)
- [x] `train_tokenizer.py` — BPE training script
- [x] `tokenizer.py` — Tokenizer wrapper

### Model (`model/`)
- [x] `embeddings.py` — Token, Positional, Modality embeddings
- [x] `attention.py` — Custom multi-head self-attention with scaled dot-product + PyTorch native SDPA fast-path (FlashAttention)
- [x] `transformer.py` — Pre-LN Transformer block + 4-layer encoder stack
- [x] `pooling.py` — Masked mean, CLS pooling
- [x] `encoder.py` — Base encoder class (embeddings + transformer + pooling + projection + L2 norm)
- [x] `shared_encoder.py` — Shared encoder with modality embeddings (Phase 3)
- [x] `dual_encoder.py` — Separate code/text encoders (Phase 4)
- [x] `ablation_models.py` — AblationSharedEncoder with configurable CLSPooling vs MaskedMeanPooling (Phase 6)

### Losses (`losses/`)
- [x] `contrastive.py` — Symmetric InfoNCE loss with in-batch negatives & InfoNCEWithHardNegativesLoss (Phase 5)

### Training (`training/`)
- [x] `trainer.py` — ContrastiveTrainer with CUDA AMP, cosine warmup schedule, gradient clipping, checkpointing, and hard negative batch routing
- [x] `hard_negatives.py` — BM25 & dense model hard negative mining module (Phase 5)
- [ ] `train.py` — General multi-model entry point

### Retrieval (`retrieval/`)
- [ ] `build_index.py` — FAISS index construction
- [ ] `search.py` — Query encoding + search
- [x] `bm25.py` — High-performance vectorized BM25 baseline with sub-token splitting

### Evaluation (`evaluation/`)
- [x] `metrics.py` — MRR, Recall@K, NDCG, and non-parametric bootstrap CIs
- [x] `evaluate.py` — Reusable evaluation pipeline & checkpoint evaluation CLI (supports Basic, Shared, Dual, and Hard Negative checkpoints)
- [ ] `qualitative.py` — Failure analysis, examples
- [ ] `pretrained_baseline.py` — Frozen pretrained eval

### Demo (`demo/`)
- [ ] `cli.py` — CLI demo

### Scripts & Analysis (`scripts/`)
- [x] `eda.py` — Token length distribution & sequence coverage analysis
- [x] `run_baseline.py` — End-to-end BM25 evaluation benchmark + MLflow logging
- [x] `run_basic.py` — Basic Encoder training execution script + MLflow tracking
- [x] `run_shared.py` — Shared Encoder training execution script + MLflow tracking
- [x] `run_dual.py` — Dual Encoder training execution script + MLflow tracking
- [x] `mine_hard_negatives.py` — Multithreaded vectorized CSR BM25 hard negative mining CLI
- [x] `mine_dense_hard_negatives.py` — High-throughput GPU FAISS dense hard negative mining CLI with 3-tier false negative filters
- [x] `run_hard_negatives.py` — Phase 5 training execution script with hard negatives + MLflow tracking
- [x] `run_ablation.py` — Unified Phase 6 ablation runner with single-run MLflow tracking and ASCII line logs
- [x] `run_phase_r3_benchmark.py` — Phase R3 final test benchmark runner with paired bootstrap, overlap stratification & Capacity Error Rubric

### Tests (`tests/`)
- [x] `test_metrics.py` — Comprehensive unit tests for all IR metrics & bootstrap CIs (100% pass)
- [x] `test_bm25.py` — Unit tests for code tokenization, indexing, and ranking (100% pass)
- [x] `test_model.py` — Unit tests for embeddings, attention, Pre-LN blocks, pooling, and BaseEncoder (100% pass)
- [x] `test_loss.py` — Unit tests for InfoNCE loss symmetry, alignment, and gradients (100% pass)
- [x] `test_shared_encoder.py` — Unit tests for modality routing, parameter count, and backward gradients (100% pass)
- [x] `test_dual_encoder.py` — Unit tests for DualEncoder parameters, decoupled gradients, and modality encoding (100% pass)
- [x] `test_hard_negatives.py` — Unit tests for BM25 mining, CSR top-k, leak prevention, and extended loss (100% pass)
- [x] `test_ablations.py` — Unit tests for CLSPooling, dataset sequence truncation, and temperature scaling (100% pass)

### Documentation & Project Logs
- [x] `walkthrough.md` — Detailed chronological implementation walkthrough, benchmarks, and verification log (must be updated after every milestone)
- [x] `STUDY_GUIDE.md` — Theory, mathematical intuition, tensor shapes, and bug fix log
- [x] `Memory.md` — Active tracker, components, experiment logs, and decisions
- [x] `AGENTS.md` — Operating guide and protocol for AI coding agents


---

## Experiment Log

## Experiment Log

### 1. Pre-Registered Confirmatory Benchmark (Protocol §3, Evaluated Strictly Once on Test)

*The models below constitute the pre-registered protocol evaluation conducted strictly once on the clean test split ($N = 19,632$ queries against 19,632 corpus documents) per [`PROTOCOL.md`](PROTOCOL.md). This is the sole protected, leak-free statistical comparison against lexical BM25.*

| Phase | Model | Params | Modality Emb | Negatives | Epochs | Test MRR [95% CI] | Test R@1 | Test R@5 | Test R@10 | Test NDCG@10 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Phase R1** | **BM25 (ATIRE)** | — | — | Lexical | 0 | **0.5108** [0.5047, 0.5166] | 0.4052 | 0.6340 | 0.6993 | 0.5514 |
| **Phase R3** | **Basic Encoder** | 7.38M | None | In-batch (masked) | 2 | **0.3773** [0.3716, 0.3832] | 0.2836 | 0.4814 | 0.5575 | 0.4132 |
| **Phase R3** | **Shared Encoder (In-Batch)** | 7.38M | Learned (2x256) | In-batch (masked) | 2 | **0.4157** [0.4098, 0.4216] | 0.3178 | 0.5246 | 0.6018 | 0.4531 |
| **Phase R3** | **Shared Encoder (BM25 HN)** | 7.38M | Learned (2x256) | 1 BM25 + In-batch | 2 | **0.4155** [0.4095, 0.4215] | 0.3184 | 0.5227 | 0.6016 | 0.4529 |

> **Key Pre-Registered Finding**: The Shared Encoder demonstrates statistically significant semantic superiority over BM25 on low-overlap queries (+11.05 MRR points, 0.3204 vs 0.2099) and zero-overlap queries ($5\times$ advantage, 0.0487 vs 0.0099). Overall BM25 maintains a higher dataset-wide score (0.5108 vs 0.4157) due to the 67.7% high-overlap stratum where exact token inverted indexing dominates.

---

### 2. Post-Protocol Exploratory Model Iteration (Unprotected from Selection Bias)

*Additional architectures and training variants were explored after the primary evaluation; their test-set numbers are reported for exploratory reference but were not protected from model selection bias.*

| Model Variant | Params | Negatives / Tuning | Epochs | Split | MRR [95% CI] | R@1 | R@10 | Notes & Selection Caveats |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Shared Encoder (FAISS Dense HN)** | 7.38M | 1 FAISS Dense + In-batch | 2 | Full Val (20,115)<br>Full Test (19,632) | **0.4084** [0.4027, 0.4141]<br>**0.4192** [0.4132, 0.4250] | 0.3098<br>0.3216 | 0.5959<br>0.6020 | Post-R3 exploratory dense mining. |
| **Scaled Shared Encoder (6L-384d)** | 17.03M | 1 FAISS Dense + In-batch | 4 | Full Val (20,115)<br>Full Test (19,632) | **0.4620** [0.4559, 0.4679]<br>**0.4699** [0.4636, 0.4757] | 0.3552<br>0.3637 | 0.6655<br>0.6686 | Post-R3 exploratory scaling (Category B resolution). |
| **Dual Encoder (Baseline Handover)** | 14M | In-batch (decoupled) | 2 | Full Test (19,632) | **0.2900** | — | — | Teammate clean baseline (no weight sharing). |
| **Dual Encoder (FAISS HN, $\tau=0.10$)** | 14M | FAISS Hard ($\tau=0.10$) | 1 | Full Test (19,632) | **0.4807** | — | — | Evaluated on test; test-selected hyperparameter. |
| **Dual Encoder (FAISS HN, $\tau=0.07$)** | 14M | FAISS Hard ($\tau=0.07$) | 1 | Full Test (19,632) | **0.4684** | — | — | Evaluated on test; test-selected hyperparameter. |

---

## Historical Pre-Remediation Benchmarks (Leaky Data Archive)

*This section archives the original exploratory benchmarks from Phases 1–6 evaluated prior to AST docstring stripping. In these historical runs, verbatim docstrings were present inside the code snippets, leading to an artificially inflated lexical and neural ceiling. These figures are preserved strictly for architectural exploration, engineering verification, and historical provenance.*

### Historical Architecture Exploration (Leaky Data: `data/processed/`)

| Phase | Run ID | Model | Params | Tokenizer | Negatives | Epochs | MRR | R@1 | R@5 | R@10 | Status |
|:---:|:---|:---|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Phase 1** | `234f518410034b628b9c90eb7cbbc1cf` | BM25 Baseline | — | Subwords | Lexical | 0 | 0.9498 | 0.9180 | 0.9890 | 0.9950 | Historical Lexical Ceiling |
| **Phase 2** | — | Basic Encoder | 7.38M | Custom BPE | In-batch | <1 | 0.4633 | 0.3540 | 0.5890 | 0.6790 | Historical Smoke Test |
| **Phase 3** | `84bb3f1d13054e8d914bef04dc632d32` | Shared Encoder | 7.38M | Custom BPE | In-batch | 1 | 0.9296 | 0.8880 | 0.9780 | 0.9840 | Historical Shared Baseline |
| **Phase 4** | `c6acad9bbc4043d69bf680b841f78962` | Dual Encoder | 13.19M | Custom BPE | In-batch | 2 | 0.8670 | 0.8050 | 0.9450 | 0.9620 | Historical Dual Baseline |
| **Phase 4** | `checkpoints/dual/best_dual.pt` | Dual (Kaggle T4x2) | 13.19M | Custom BPE | In-batch | 2 | 0.8947 | 0.8370 | 0.9680 | 0.9770 | Historical Teammate Run |
| **Phase 5** | `checkpoints/dual_hard/best_dual.pt` | Dual + Dense HN | 13.19M | Custom BPE | FAISS Hard | 2 | 0.9180 | 0.8720 | 0.9720 | 0.9840 | Historical Teammate Dense HN |
| **Phase 5** | `checkpoints/dual_bm25_hard/best_dual.pt` | Dual + Lexical HN | 13.19M | Custom BPE | BM25 Hard | 2 | 0.9042 | 0.8520 | 0.9690 | 0.9810 | Historical Teammate Lexical HN |
| **Phase 5** | `0b01d6eb927c4181b825bf6757dd970c` | Shared + Hard Negs | 7.38M | Custom BPE | BM25 Hard | 2 | 0.9383 | 0.9030 | 0.9780 | 0.9880 | Historical Shared Hard Negs |

### Historical Ablations (Leaky Data: `data/processed/`)
All historical ablations evaluated on 1,000 sampled test queries against the 21,005 test code corpus (1 epoch = 3,010 steps, CUDA AMP on RTX 4050):

| Run Name | Run ID | Category | Variant | Epochs | Test MRR | Test R@1 | Test R@5 | Test R@10 | Test NDCG@10 | Key Insight |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Phase 3 Baseline** | `84bb3f1d13054e8d914bef04dc632d32` | Baseline | Mean, $\tau=0.07$, $L=256$ | 1 | **0.9296** | **0.8880** | **0.9780** | **0.9840** | **0.9429** | Controlled 1-epoch reference model |
| `phase6_pooling_cls` | `817745a0da7a448596d7501acec68c28` | Pooling | CLSPooling | 1 | **0.8836** | **0.8340** | **0.9440** | **0.9600** | **0.9013** | MaskedMeanPooling beats CLS by +4.6 MRR pts |
| `phase6_temp_0.05` | `f7820484b89a4e08b4a8ed6c80da59db` | Temperature | $\tau = 0.05$ | 1 | **0.9244** | **0.8860** | **0.9690** | **0.9790** | **0.9373** | Sharper softmax slightly degrades test generalization |
| `phase6_temp_0.10` | `173d36082dbd4f978ba99875ed7f6c61` | Temperature | $\tau = 0.10$ | 1 | **0.9252** | **0.8910** | **0.9680** | **0.9740** | **0.9366** | Softer softmax slightly attenuates hard neg gradients |
| `phase6_seq_len_128` | `48b9dffd20714b7db597bccae74b4bdb` | Sequence Length | $L = 128$ | 1 | **0.9292** | **0.8930** | **0.9730** | **0.9800** | **0.9415** | 2x speedup with zero quality loss |

---

## Key Decisions Log

| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| 2026-09-07 | Python Environment = local .venv with uv | Strict project rule: always use local .venv (`.venv/Scripts/activate`) managed by uv | Global / shared venv |
| 2026-09-08 | Max seq len = 256 | Empirical EDA (50k sample): P50=169, docstring coverage >95%, code coverage 68.7%; maximizes in-batch negative capacity in InfoNCE | 128 (too aggressive), 512 (4x attention memory penalty) |
| 2026-09-12 | Vectorized Inverted Index for BM25 | 150x+ speedup over naive iteration (3.74s vs 592s for 1k queries against 21k corpus) while computing exact BM25Okapi scores | Naive Python doc iteration in rank_bm25 |
| 2026-09-12 | MLflow SQLite Backend | Use `sqlite:///mlflow.db` tracking URI to adhere to modern MLflow standards and avoid deprecated filestore warnings | Legacy `./mlruns` |
| 2026-09-12 | Custom Pre-LN Transformer (~7.38M) | Implemented raw PyTorch `nn.Module` Pre-LN Transformer (4 layers, 8 heads, d_model=256, d_ff=1024) with MaskedMeanPooling + L2 normalization head | HuggingFace transformers wrapper, Post-LN |
| 2026-09-12 | CUDA Mixed Precision Training | Use `torch.amp.autocast` + `torch.amp.GradScaler` for high-throughput GPU training on RTX 4050 with automatic CPU fallback | Full FP32 |
| 2026-09-13 | Learned Modality Embeddings | Add 2x256 modality table (0=code, 1=text) to composite embedding layer, allowing single Transformer to distinguish representation space without duplicating weights | Token prefix only, separate models |
| 2026-09-17 | Pre-tokenized Binary Tensor Cache | Pre-tokenize dataset splits into `.pt` tensor files (`train_tokenized.pt`, etc.) eliminating CPU collation bottleneck (batch fetch time dropped from 3.3s to 10ms; cuts epoch training time from ~3 hours to ~10–12 minutes) | On-the-fly collation with num_workers (unstable on Windows) |
| 2026-09-19 | Vectorized Multithreaded CSR BM25 Mining | Replace single-query candidate sorting with batched SciPy CSR matrix multiplication ($Q \times D^T$), syntax word pruning (IDF < 1.0), and top-8 IDF term selection across 4 CPU threads. Yielded a 530x throughput boost (~1,965 q/s vs ~3.7 q/s), completing all 385k queries in 3.2 minutes | Dense array scatter-add, single-query candidate arrays with np.argsort |
| 2026-09-20 | MaskedMeanPooling vs CLSPooling (Ablation 1) | Empirically proved MaskedMeanPooling beats CLSPooling by +4.6 MRR points (0.9296 vs 0.8836) and +5.4% Recall@1 (88.8% vs 83.4%) in from-scratch code Transformers | CLSPooling (severe information bottleneck at token 0) |
| 2026-09-20 | Optimal InfoNCE Temperature $\tau = 0.07$ (Ablation 2 & 3) | Confirmed $\tau = 0.07$ is the empirical sweet spot; sharper $\tau = 0.05$ (MRR 0.9244) and softer $\tau = 0.10$ (MRR 0.9252) both suffer lower test generalization | $\tau = 0.05$, $\tau = 0.10$ |
| 2026-09-20 | Sequence Length 128 Production Frontier (Ablation 4) | Proved $L = 128$ matches $L = 256$ retrieval quality (MRR 0.9292 vs 0.9296; R@1 89.3% vs 88.8%) while cutting training and attention latency by ~50% (10.47m vs 19.7m) | Full $L = 256$ for latency-critical deployments |

---

## Known Issues / Blockers

| Issue | Severity | Status | Workaround / Solution |
|---|---|---|---|
| On-the-fly CPU tokenization bottleneck (GPU starvation in Phase 3) | High | 🟢 Resolved (2026-09-17) | Implemented `data/pretokenize.py` to pre-tokenize all splits into contiguous binary tensor files (`train_tokenized.pt`), accelerating batch feeding from 0.3 batches/s to 99.9 batches/s (~300x speedup). |
| Phase 5 Candidate Array Explosion & Memory Lockup in BM25 Mining | Critical | 🟢 Resolved (2026-09-19) | Concatenating 10–15 query token posting lists yielded 1.2M–2.5M candidate arrays per query. Running `np.argsort` + `np.unique` on 1.2M items per query took ~0.27s/query (28.6 hours total) and caused 10GB+ memory allocation churn / GC freeze on Windows. Resolved by pruning IDF < 1.0 stopwords, keeping top-8 informative terms, precomputing CSR matrix $D^T$, and batching queries via multithreaded sparse BLAS ($Q \times D^T$). Mining throughput jumped to ~1,965 q/s, finishing all 385,381 queries in 196 seconds. |
| 8 GB Host RAM Spike & GPU VRAM Thrashing in Phase 5 Training | Critical | 🟢 Resolved (2026-09-19) | Training 3 encoders per step (`code`, `text`, `hard_negatives`) materialized $3 \times 4 \times 1.07\text{ GB} \approx 12.9\text{ GB}$ of intermediate attention score matrices in autograd. On the 6 GB RTX 4050 Laptop GPU, Windows WDDM overflowed 7+ GB of GPU activations into shared host RAM via PCIe paging, creating an 8 GB system RAM spike and ballooning step latency from 0.35s to 11.4s (33x slowdown). Resolved by adding PyTorch native SDPA (`torch.nn.functional.scaled_dot_product_attention` / FlashAttention) to `MultiHeadSelfAttention`. Peak VRAM dropped from 11,079 MB to 5,103 MB (safely inside 6 GB VRAM), host RAM dropped to <100 MB, and step latency improved 33x (from 11.4s to 0.346s/step). |
| Windows `cp1252` UnicodeEncodeError on `✓` in `print()` calls | Low | 🟢 Resolved (2026-09-19) | Replaced `✓` and `★` Unicode characters in plain `print()` and console calls with `[OK]` and `[*]` ASCII strings. |

---

## Reproducibility Checklist

- [x] All configs versioned in `configs/` (`configs/baseline.yaml`, `configs/basic.yaml`, `configs/shared.yaml`, `configs/dual.yaml`)
- [x] Random seeds set (PyTorch, NumPy, Python)
- [x] MLflow tracks all hyperparameters and metrics
- [x] Model checkpoints saved with config (`checkpoints/shared/best_shared.pt`, `checkpoints/dual/best_dual.pt`)
- [x] Tokenizer saved with model
- [x] Data splits fixed (no random shuffle in val/test)
- [x] Requirements pinned in `requirements.txt`
- [ ] Python version specified in `pyproject.toml`

---

## Next Actions

### Phase 0: Setup & Data Pipeline (Completed)
1. [x] Download and preprocess CodeSearchNet Python (`data/download.py`, `data/preprocess.py`)
2. [x] Train custom BPE tokenizer (`tokenizer/train_tokenizer.py`)
3. [x] Tokenizer PyTorch wrapper (`tokenizer/tokenizer.py`)
4. [x] Run EDA / token length distribution (`scripts/eda.py`)
5. [x] Implement PyTorch Dataset + Dynamic Collator (`data/dataset.py`)

### Phase 1: BM25 Baseline & Evaluation Framework (Completed)
1. [x] Implement evaluation metrics from mathematical definitions (`evaluation/metrics.py`: MRR, Recall@K, NDCG, bootstrap CIs)
2. [x] Implement high-performance BM25 lexical retrieval index (`retrieval/bm25.py`)
3. [x] Implement reusable evaluation runner (`evaluation/evaluate.py`)
4. [x] Create configuration (`configs/baseline.yaml`)
5. [x] Run baseline evaluation script on test set (`scripts/run_baseline.py`)
6. [x] Log baseline metrics to MLflow (`sqlite:///mlflow.db` & `mlruns`)
7. [x] Comprehensive unit tests (`tests/test_metrics.py`, `tests/test_bm25.py` - 28/28 passed)

### Phase 2: Model 1 — Basic Encoder (Completed)
1. [x] Implement Token, Positional, and Modality embeddings (`model/embeddings.py`)
2. [x] Implement custom Multi-Head Attention (`model/attention.py`)
3. [x] Implement Pre-LN Transformer block & Encoder stack (`model/transformer.py`)
4. [x] Implement Masked Mean & CLS Pooling (`model/pooling.py`)
5. [x] Implement BaseEncoder pipeline with L2 normalization (`model/encoder.py`)
6. [x] Implement InfoNCE contrastive loss with in-batch negatives (`losses/contrastive.py`)
7. [x] Implement ContrastiveTrainer with CUDA AMP & cosine warmup (`training/trainer.py`)
8. [x] Create training configuration (`configs/basic.yaml`) and entry point (`scripts/run_basic.py`)
9. [x] Comprehensive unit test suites (`tests/test_model.py`, `tests/test_loss.py` - 45/45 total passed)
10. [x] Implement checkpoint evaluation CLI in `evaluation/evaluate.py`

### Phase 3: Model 2 — Shared Encoder (Completed)
1. [x] Implement SharedEncoder with learned modality embeddings (`model/shared_encoder.py`)
2. [x] Add modality token / ID routing for `<CODE>` and `<TEXT>` inputs
3. [x] Create configuration (`configs/shared.yaml`) and runner (`scripts/run_shared.py`)
4. [x] Comprehensive unit tests in `tests/test_shared_encoder.py` (50/50 total tests passing)
5. [x] Train Shared Encoder on CodeSearchNet Python (RTX 4050 CUDA AMP)
6. [x] Evaluate on test set (1,000 queries vs 21,005 corpus) and log to MLflow: MRR 0.9296, R@1 0.8880, R@10 0.9840

### Phase 4: Model 3 — Separate (Dual) Encoders (Completed)
1. [x] Implement DualEncoder with separate code and text encoders (`model/dual_encoder.py`)
2. [x] Parameter budget matching: 3 layers each (~6.59M each, ~13.19M total)
3. [x] Create configuration (`configs/dual.yaml`) and runner (`scripts/run_dual.py`)
4. [x] Train Dual Encoder on CodeSearchNet Python (CUDA AMP on RTX 4050, 2 epochs)
5. [x] Evaluate on formal 1,000 test query benchmark vs 21,005 corpus and log to MLflow: MRR 0.8670, R@1 0.8050, R@5 0.9450, R@10 0.9620

### Phase 5: Hard Negative Mining (Completed)
1. [x] Implement hard negative mining pipeline (`training/hard_negatives.py`) using BM25 and dense retrievers
2. [x] Implement contrastive loss supporting explicit mined hard negatives (`InfoNCEWithHardNegativesLoss` in `losses/contrastive.py`)
3. [x] Update Dataset & DataLoader to batch pre-tokenized hard negative tensors with zero CPU tokenization overhead (`data/dataset.py`)
4. [x] Create training configuration (`configs/shared_hard.yaml`) and training runner (`scripts/run_hard_negatives.py`)
5. [x] Comprehensive unit tests in `tests/test_hard_negatives.py` (60/60 tests passing)
6. [x] Mine training set hard negative index matrix (`train_hard_negatives.pt`, 385k samples in 3.2m via CSR sparse matmul)
7. [x] Train model with hard negatives and evaluate on test benchmark (Epoch 2 fine-tuned: Test MRR 0.9383, R@1 0.9030; +1.5% R@1 boost over in-batch only)

### Phase R4: Clean Architectural Ablations (Protocol §3.3)
1. [x] Ablation 1: Pooling Strategy (MaskedMeanPooling Val MRR 0.4033 beats CLSPooling 0.1566 by +0.2467 MRR points; CLS suffers -61.2% collapse from scratch)
2. [x] Ablation 2: Sequence Length ($L=128$ retains 97.8% quality [0.3943 vs 0.4033] with 1.78x training speedup)
3. [x] Temperature Sensitivity: Grid evaluation verified $\tau=0.05$ (0.4033) outperforms $\tau=0.07$ (0.3832) by sharpening gradient penalties against in-batch negatives.
4. [x] Summary table logged to MLflow experiment `codeembed-clean-baselines`.

### Phase 6.5: Exploratory Model Capacity Scaling (Post-Protocol Iteration)
1. [x] Scale SharedEncoder capacity to 17.03M parameters (6 Layers, $d_{\text{model}}=384$, 6 Heads, $d_{\text{ff}}=1536$)
2. [x] Train 4 epochs with 1 FAISS dense hard negative + 63 in-batch negatives on clean data (`configs/shared_6l_dense_clean.yaml`)
3. [x] Evaluate on Full Validation ($N=20,115$): MRR **0.4620** [0.4559, 0.4679], R@1 **0.3552**, R@10 **0.6655**, NDCG@10 **0.5041** (+5.36 MRR pts over 4L)
4. [x] Evaluate on Full Test ($N=19,632$): MRR **0.4699** [0.4636, 0.4757], R@1 **0.3637**, R@10 **0.6686**, NDCG@10 **0.5109** (*Exploratory reference; unprotected from model selection bias*)
5. [x] Lexical Stratification Analysis: Zero-Overlap **0.0716** (7.2x vs BM25 0.0099); Low-Overlap **0.3559** (+14.60 pts vs BM25 0.2099); High-Overlap **0.5357**
6. [x] Discordance Analysis vs BM25: Dense wins on 37.7% of queries, BM25 wins on 38.7%; Oracle bound is **0.6562 MRR** (+14.5 pts over BM25)

### Hybrid Search & CLI Demo (Current Milestone)
1. [ ] Implement Hybrid Search benchmark combining BM25 and Dense embeddings (Reciprocal Rank Fusion & Convex Combination)
2. [ ] Empirically evaluate hybrid retrieval performance across lexical and semantic query strata
3. [ ] Implement interactive terminal search CLI (`demo/cli.py`) with hybrid search toggle

### Phase 7: Pretrained Baseline (Upcoming)
1. [ ] Frozen evaluation of pretrained models (`all-MiniLM-L6-v2`, `codebert-base`)
2. [ ] Answer RQ5: Pretraining vs From-scratch contrastive learning

### Phase 8: Demo & Final Deliverables (Upcoming)
1. [ ] Qualitative analysis & UMAP visualization
2. [ ] Final synthesis & project report

---

## Artifacts Location

| Artifact | Location |
|----------|----------|
| Raw Data | `data/raw/` (gitignored) |
| Processed Data | `data/processed/` (gitignored) |
| Tokenizer | `tokenizer/` (gitignored) |
| Model Checkpoints | `checkpoints/` (gitignored) |
| MLflow Runs | `mlruns/` (gitignored) |
| FAISS Indices | `indices/` (gitignored) |
| Notebooks | `notebooks/` (tracked) |
| Configs | `configs/` (tracked) |
| Source Code | `codeembed/` (tracked) |

---

## Commands Reference

```bash
# Setup
pip install -r requirements.txt
mlflow ui --port 5000

# Data
python -m data.download
python -m data.preprocess

# Tokenizer
python -m tokenizer.train_tokenizer --config configs/tokenizer.yaml

# Training
python scripts/run_baseline.py --config configs/baseline.yaml
python scripts/run_basic.py --config configs/basic.yaml
python scripts/run_shared.py --config configs/shared.yaml
python scripts/run_dual.py --config configs/dual.yaml

# Phase 5: Hard Negatives (run mining FIRST, then training)
python scripts/mine_hard_negatives.py --k 7  # Step 1: mine train_hard_negatives.pt
python scripts/run_hard_negatives.py --config configs/shared_hard.yaml  # Step 2: train
# Or for Dual Encoder:
python scripts/run_hard_negatives.py --config configs/dual_hard.yaml

# Evaluation
python -m evaluation.evaluate --checkpoint checkpoints/best.pt --split test

# Demo
python -m demo.cli
```

---

## Notes for Future Sessions

> Add context here when resuming work.

- **Tokenizer**: Consider whether to add `<SEP>` for potential future use
- **Hard Negatives**: Mine from BM25 first, then from trained model
- **Evaluation**: Use same test queries across all experiments for fair comparison
- **Pretrained Baseline**: Ensure frozen — no gradient updates