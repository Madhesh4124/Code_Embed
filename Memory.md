# CodeEmbed — Project Memory

> **Purpose**: Track what's built, what's working, what's implemented. Update after each milestone.

---

## Current Status

| Component | Status | Last Updated | Notes |
|-----------|--------|--------------|-------|
| Project Setup | 🟢 Completed | 2026-09-07 | Folder tree, .gitignore, requirements.txt, and local .venv with uv & CUDA 12.4 |
| Data Pipeline | 🟢 Completed | 2026-09-08 | 385k samples preprocessed, Dataset & Collator verified with DataLoader |
| Tokenizer | 🟢 Completed | 2026-09-08 | 16k BPE trained on code+text; CodeEmbedTokenizer wrapper tested & verified |
| BM25 Baseline | 🟢 Completed | 2026-09-12 | Vectorized inverted index; MRR 0.9498, R@1 0.9180, R@10 0.9950; logged to MLflow |
| Basic Encoder | 🟢 Built & Verified | 2026-09-12 | 7.38M Pre-LN Transformer from scratch; InfoNCE loss; CUDA mixed precision training loop; 45/45 tests passing |
| Shared Encoder | 🟢 Completed | 2026-09-13 | 7.38M Transformer + Modality embeddings; Test MRR 0.9296, R@1 0.8880, R@10 0.9840; 50/50 tests passing |
| Dual Encoder | 🟢 Completed | 2026-09-19 | Decoupled 3-layer code & text BaseEncoders (~13.19M params); Test MRR 0.8670, R@1 0.8050, R@10 0.9620; 55/55 tests passing |
| Hard Negatives | 🟢 Completed | 2026-09-19 | Mined 385k queries via SciPy CSR BM25; trained Shared Encoder with hard negatives (2 epochs, 0.35s/step); Test MRR 0.9383, R@1 0.9030, R@10 0.9880 (+1.5% R@1 boost); 60/60 tests passing |
| Ablations | ⬜ Not Started | — | Phase 6 |
| Pretrained Baseline | ⬜ Not Started | — | Phase 7 |
| Demo | ⬜ Not Started | — | Phase 8 |
| Documentation | 🟢 Active | 2026-09-19 | AGENTS.md, Memory.md, STUDY_GUIDE.md, and walkthrough.md actively maintained |

---

## Implemented Components

### Data Pipeline (`data/`)
- [x] `download.py` — CodeSearchNet Python download
- [x] `preprocess.py` — Cleaning, filtering, dedup
- [x] `pretokenize.py` — Offline binary tensor caching (`train_tokenized.pt`), eliminating CPU tokenization bottleneck (300x faster batch feeding)
- [x] `dataset.py` — PyTorch Dataset + Fast Collator with auto-detection for pre-tokenized tensors
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
- [x] `run_hard_negatives.py` — Phase 5 training execution script with hard negatives + MLflow tracking

### Tests (`tests/`)
- [x] `test_metrics.py` — Comprehensive unit tests for all IR metrics & bootstrap CIs (100% pass)
- [x] `test_bm25.py` — Unit tests for code tokenization, indexing, and ranking (100% pass)
- [x] `test_model.py` — Unit tests for embeddings, attention, Pre-LN blocks, pooling, and BaseEncoder (100% pass)
- [x] `test_loss.py` — Unit tests for InfoNCE loss symmetry, alignment, and gradients (100% pass)
- [x] `test_shared_encoder.py` — Unit tests for modality routing, parameter count, and backward gradients (100% pass)
- [x] `test_dual_encoder.py` — Unit tests for DualEncoder parameters, decoupled gradients, and modality encoding (100% pass)
- [x] `test_hard_negatives.py` — Unit tests for BM25 mining, CSR top-k, leak prevention, and extended loss (100% pass)

### Documentation & Project Logs
- [x] `walkthrough.md` — Detailed chronological implementation walkthrough, benchmarks, and verification log (must be updated after every milestone)
- [x] `STUDY_GUIDE.md` — Theory, mathematical intuition, tensor shapes, and bug fix log
- [x] `Memory.md` — Active tracker, components, experiment logs, and decisions
- [x] `AGENTS.md` — Operating guide and protocol for AI coding agents


---

## Experiment Log

### Baseline Experiments
| Phase | Run ID | Model | Config | Epochs | MRR | R@1 | R@5 | R@10 | NDCG | Notes |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Phase 1** | `234f518410034b628b9c90eb7cbbc1cf` | BM25 | k1=1.5, b=0.75 | 0 (Lexical) | 0.9498 | 0.9180 | 0.9890 | 0.9950 | 0.9610 | Lexical benchmark on 1k test queries vs 21,005 corpus (logged to mlruns) |

### Architecture Experiments
| Phase | Run ID | Model | Params | Tokenizer | Negatives | Epochs | MRR | R@1 | R@5 | R@10 | Status |
|:---:|:---|:---|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Phase 2** | — | Basic | 7.38M | Custom BPE | In-batch | <1 (Smoke) | 0.4633 | 0.3540 | 0.5890 | 0.6790 | Baseline |
| **Phase 3** | `84bb3f1d13054e8d914bef04dc632d32` | Shared | 7.38M | Custom BPE | In-batch | 1 | 0.9296 | 0.8880 | 0.9780 | 0.9840 | 🟢 Completed (Epoch 1 on test set) |
| **Phase 4** | `c6acad9bbc4043d69bf680b841f78962` | Dual | 13.19M | Custom BPE | In-batch | 2 | 0.8670 | 0.8050 | 0.9450 | 0.9620 | 🟢 Completed (Epoch 2 on test set) |
| **Phase 5** | `0b01d6eb927c4181b825bf6757dd970c` | Shared (Hard) | 7.38M | Custom BPE | BM25 Hard | 2 (1+1) | **0.9383** | **0.9030** | **0.9780** | **0.9880** | 🟢 Completed (Epoch 2 fine-tuned; +1.5% R@1 boost over in-batch) |

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

### Phase 6: Ablation Studies (Current Milestone)
1. [ ] Ablation 1: Pooling Strategy (MaskedMeanPooling vs CLSPooling)
2. [ ] Ablation 2: Temperature Sensitivity ($\tau \in \{0.05, 0.07, 0.10\}$)
3. [ ] Ablation 3: Sequence Length Impact ($L=128$ vs $L=256$)
4. [ ] Summary table and scientific insights logged to MLflow

### Phase 6.5: Model Capacity & Scaling Exploration (Layer & Parameter Scaling)
1. [ ] Depth scaling at constant width ($d_{\text{model}}=256$): 2L (~4.2M), 4L (~7.38M), 6L (~10.5M), 8L (~13.7M)
2. [ ] Width & depth scaling with micro-batching ($d_{\text{model}}=512$): 6L (~27.4M), 8L (~36.8M), 12L (~54.0M)
3. [ ] Profile peak VRAM and step latency on 6 GB RTX 4050 GPU (SDPA + gradient accumulation)
4. [ ] Test hypothesis: Can 54M parameters with BM25 hard negatives push retrieval towards ~0.98 MRR on consumer GPU?

### Phase 7: Pretrained Baseline (Upcoming)
1. [ ] Frozen evaluation of pretrained models (`all-MiniLM-L6-v2`, `codebert-base`)
2. [ ] Answer RQ5: Pretraining vs From-scratch contrastive learning

### Phase 8: Demo & Final Deliverables (Upcoming)
1. [ ] CLI demo for real-time natural language code retrieval
2. [ ] Qualitative analysis & UMAP visualization
3. [ ] Final synthesis & project report

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