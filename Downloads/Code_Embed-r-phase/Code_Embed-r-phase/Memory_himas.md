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
| Dual Encoder | 🟢 Completed | 2026-09-17 | Phase 4: Test MRR 0.8947, R@1 0.8370 |
| Hard Negatives | 🟢 Completed | 2026-09-18 | Phase 5: Mined 50 hard negs with BM25. Dual Encoder Kaggle Test MRR 0.9042 |
| Ablations | ⬜ Not Started | — | Phase 6 |
| Pretrained Baseline | ⬜ Not Started | — | Phase 7 |
| Demo | ⬜ Not Started | — | Phase 8 |
| Documentation | 🟢 Active | 2026-09-13 | AGENTS.md, Memory.md, STUDY_GUIDE.md, and walkthrough.md actively maintained |

---

## Implemented Components

### Data Pipeline (`data/`)
- [x] `download.py` — CodeSearchNet Python download
- [x] `preprocess.py` — Cleaning, filtering, dedup
- [x] `dataset.py` — PyTorch Dataset + Collator
- [x] `splits.py` — Handled via data/processed/{train,validation,test}.parquet and create_dataloader()

### Tokenizer (`tokenizer/`)
- [x] `train_tokenizer.py` — BPE training script
- [x] `tokenizer.py` — Tokenizer wrapper

### Model (`model/`)
- [x] `embeddings.py` — Token, Positional, Modality embeddings
- [x] `attention.py` — Custom multi-head self-attention with scaled dot-product
- [x] `transformer.py` — Pre-LN Transformer block + 4-layer encoder stack
- [x] `pooling.py` — Masked mean, CLS pooling
- [x] `encoder.py` — Base encoder class (embeddings + transformer + pooling + projection + L2 norm)
- [x] `shared_encoder.py` — Shared encoder with modality embeddings (Phase 3)
- [x] `dual_encoder.py` — Separate code/text encoders (Phase 4)

### Losses (`losses/`)
- [x] `contrastive.py` — Symmetric InfoNCE loss with in-batch negatives

### Training (`training/`)
- [x] `trainer.py` — ContrastiveTrainer with CUDA AMP, cosine warmup schedule, gradient clipping, checkpointing
- [ ] `train.py` — General multi-model entry point

### Retrieval (`retrieval/`)
- [ ] `build_index.py` — FAISS index construction
- [ ] `search.py` — Query encoding + search
- [x] `bm25.py` — High-performance vectorized BM25 baseline with sub-token splitting

### Evaluation (`evaluation/`)
- [x] `metrics.py` — MRR, Recall@K, NDCG, and non-parametric bootstrap CIs
- [x] `evaluate.py` — Reusable evaluation pipeline & checkpoint evaluation CLI (supports Basic & Shared)
- [ ] `qualitative.py` — Failure analysis, examples
- [ ] `pretrained_baseline.py` — Frozen pretrained eval

### Demo (`demo/`)
- [ ] `cli.py` — CLI demo

### Scripts & Analysis (`scripts/`)
- [x] `eda.py` — Token length distribution & sequence coverage analysis
- [x] `run_baseline.py` — End-to-end BM25 evaluation benchmark + MLflow logging
- [x] `run_basic.py` — Basic Encoder training execution script + MLflow tracking
- [x] `run_shared.py` — Shared Encoder training execution script + MLflow tracking

### Tests (`tests/`)
- [x] `test_metrics.py` — Comprehensive unit tests for all IR metrics & bootstrap CIs (100% pass)
- [x] `test_bm25.py` — Unit tests for code tokenization, indexing, and ranking (100% pass)
- [x] `test_model.py` — Unit tests for embeddings, attention, Pre-LN blocks, pooling, and BaseEncoder (100% pass)
- [x] `test_loss.py` — Unit tests for InfoNCE loss symmetry, alignment, and gradients (100% pass)
- [x] `test_shared_encoder.py` — Unit tests for modality routing, parameter count, and backward gradients (100% pass)

### Documentation & Project Logs
- [x] `walkthrough.md` — Detailed chronological implementation walkthrough, benchmarks, and verification log (must be updated after every milestone)
- [x] `STUDY_GUIDE.md` — Theory, mathematical intuition, tensor shapes, and bug fix log
- [x] `Memory.md` — Active tracker, components, experiment logs, and decisions
- [x] `AGENTS.md` — Operating guide and protocol for AI coding agents


---

## Experiment Log

### Baseline Experiments
| Run ID | Model | Config | MRR | R@1 | R@5 | R@10 | NDCG | Notes |
|--------|-------|--------|-----|-----|-----|------|------|-------|
| `234f518410034b628b9c90eb7cbbc1cf` | BM25 | k1=1.5, b=0.75 | 0.9498 | 0.9180 | 0.9890 | 0.9950 | 0.9610 | Lexical benchmark on 1k test queries vs 21,005 corpus (logged to mlruns) |

### Architecture Experiments
| Run ID | Model | Params | Tokenizer | Negatives | MRR | R@1 | R@5 | R@10 | Status |
|--------|-------|--------|-----------|-----------|-----|-----|-----|------|--------|
| — | Basic | 7.38M | Custom BPE | In-batch | — | — | — | — | Baseline |
| `84bb3f1d13054e8d914bef04dc632d32` | Shared | 7.38M | Custom BPE | In-batch | **0.9296** | **0.8880** | **0.9780** | **0.9840** | 🟢 Completed (Epoch 1 on test set) |
| `08c0151bb7d6` | Dual | ~13.2M | Custom BPE | In-batch | 0.8947 | 0.8370 | 0.9680 | 0.9770 | 🟢 Completed (Phase 4) |
| `21f15daf420f` | Dual | ~13.2M | Custom BPE | Hard (k=5) | **0.9160** | **0.8670** | **0.9720** | **0.9850** | 🟢 Completed (Phase 5) |

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
| 2026-09-16 | Dual Encoder Scaling | Scaled embedding dimensions to d_model=256, n_layers=3, n_heads=8. ~13.2M parameters overall. | Smaller 8M model |
| 2026-09-17 | Hard Negative Mining | Used FAISS Inner Product index on DualEncoder embeddings to mine top-50 false positives. Softmax denominator extended with K=5 hard negatives per sample. Yielded +0.0213 absolute MRR improvement. | Static BM25 negatives |

---

## Known Issues / Blockers

| Issue | Severity | Status | Workaround |
|-------|----------|--------|------------|
| — | — | — | — |

---

## Reproducibility Checklist

- [x] All configs versioned in `configs/` (`configs/baseline.yaml`, `configs/basic.yaml`, `configs/shared.yaml`)
- [x] Random seeds set (PyTorch, NumPy, Python)
- [x] MLflow tracks all hyperparameters and metrics
- [x] Model checkpoints saved with config (`checkpoints/shared/best_shared.pt`)
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
2. [x] Create configuration (`configs/dual.yaml`) and runner (`scripts/run_dual.py`)
3. [x] Train and benchmark Dual Encoder against Shared Encoder and BM25 baseline (MRR 0.8947)

### Phase 5: Hard Negative Mining (Completed)
1. [x] Create FAISS mining pipeline (`training/hard_negatives.py`)
2. [x] Update dataloader to attach mined hard negatives (`data/hard_negative_dataset.py`)
3. [x] Implement extended InfoNCE loss with hard negatives (`losses/hard_negative_loss.py`)
4. [x] Create configuration (`configs/dual_hard.yaml`) and runner (`scripts/run_dual_hard.py`)
5. [x] Fine-tune Dual Encoder with k=5 hard negatives per sample.
6. [x] Benchmark MRR: Achieved 0.9160 (significant +0.0213 improvement).

### Immediate Next: Phase 6 — Ablation Studies
1. [ ] Run ablations on temperature (0.05, 0.07, 0.10)
2. [ ] Run ablations on pooling methods (Mean vs CLS)
3. [ ] Run ablations on tokenizer (Custom BPE vs Pretrained)
4. [ ] Consolidate results in a Jupyter Notebook

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
python scripts/run_dual_hard.py --config configs/dual_hard.yaml

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