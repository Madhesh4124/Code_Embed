# CodeEmbed — Project Memory

> **Purpose**: Track what's built, what's working, what's implemented. Update after each milestone.

---

## Current Status

| Component | Status | Last Updated | Notes |
|-----------|--------|--------------|-------|
| Project Setup | 🟢 Completed | 2026-09-07 | Folder tree, .gitignore, requirements.txt, and local .venv with uv & CUDA 12.4 |
| Data Pipeline | 🟢 Completed | 2026-09-08 | 385k samples preprocessed, Dataset & Collator verified with DataLoader |
| Tokenizer | 🟢 Completed | 2026-09-08 | 16k BPE trained on code+text; CodeEmbedTokenizer wrapper tested & verified |
| BM25 Baseline | ⬜ Not Started | — | Phase 1 |
| Basic Encoder | ⬜ Not Started | — | Phase 2 |
| Shared Encoder | ⬜ Not Started | — | Phase 3 |
| Dual Encoder | ⬜ Not Started | — | Phase 4 |
| Hard Negatives | ⬜ Not Started | — | Phase 5 |
| Ablations | ⬜ Not Started | — | Phase 6 |
| Pretrained Baseline | ⬜ Not Started | — | Phase 7 |
| Demo | ⬜ Not Started | — | Phase 8 |
| Documentation | 🟢 Active | 2026-09-08 | AGENTS.md, Memory.md, and STUDY_GUIDE.md actively maintained |

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
- [ ] `embeddings.py` — Token, Positional, Modality embeddings
- [ ] `attention.py` — Multi-head attention
- [ ] `transformer.py` — Transformer block + encoder stack
- [ ] `pooling.py` — Masked mean, CLS pooling
- [ ] `encoder.py` — Base encoder class
- [ ] `shared_encoder.py` — Shared encoder with modality
- [ ] `dual_encoder.py` — Separate code/text encoders

### Losses (`losses/`)
- [ ] `contrastive.py` — InfoNCE loss (in-batch + hard negatives)

### Training (`training/`)
- [ ] `trainer.py` — Training loop, logging, checkpointing
- [ ] `train.py` — Entry point scripts

### Retrieval (`retrieval/`)
- [ ] `build_index.py` — FAISS index construction
- [ ] `search.py` — Query encoding + search
- [ ] `bm25.py` — BM25 baseline

### Evaluation (`evaluation/`)
- [ ] `metrics.py` — MRR, Recall@K, NDCG
- [ ] `evaluate.py` — Evaluation pipeline
- [ ] `qualitative.py` — Failure analysis, examples
- [ ] `pretrained_baseline.py` — Frozen pretrained eval

### Demo (`demo/`)
- [ ] `cli.py` — CLI demo

### Scripts & Analysis (`scripts/`)
- [x] `eda.py` — Token length distribution & sequence coverage analysis

---

## Experiment Log

### Baseline Experiments
| Run ID | Model | Config | MRR | R@1 | R@5 | R@10 | NDCG | Notes |
|--------|-------|--------|-----|-----|-----|------|------|-------|
| — | BM25 | — | — | — | — | — | — | Pending |

### Architecture Experiments
| Run ID | Model | Params | Tokenizer | Negatives | MRR | R@1 | R@5 | R@10 | Status |
|--------|-------|--------|-----------|-----------|-----|-----|-----|------|--------|
| — | Basic | — | — | In-batch | — | — | — | — | Pending |
| — | Shared | — | — | In-batch | — | — | — | — | Pending |
| — | Dual | — | — | In-batch | — | — | — | — | Pending |
| — | Dual | — | — | Hard | — | — | — | — | Pending |

### Ablation Experiments
| Run ID | Ablation | Variant | MRR | R@1 | R@5 | R@10 | Notes |
|--------|----------|---------|-----|-----|-----|------|-------|
| — | Tokenizer | Custom BPE | — | — | — | — | Pending |
| — | Tokenizer | Generic (GPT-2) | — | — | — | — | Pending |
| — | Pooling | Mean | — | — | — | — | Pending |
| — | Pooling | CLS | — | — | — | — | Pending |
| — | Temperature | 0.05 | — | — | — | — | Pending |
| — | Temperature | 0.07 | — | — | — | — | Pending |
| — | Temperature | 0.10 | — | — | — | — | Pending |
| — | Embedding Dim | 128 | — | — | — | — | Pending |
| — | Embedding Dim | 256 | — | — | — | — | Pending |
| — | Embedding Dim | 512 | — | — | — | — | Pending |
| — | Model Size | 2 layers | — | — | — | — | Pending |
| — | Model Size | 4 layers | — | — | — | — | Pending |
| — | Model Size | 6 layers | — | — | — | — | Pending |

### Pretrained Baseline
| Run ID | Model | Params | MRR | R@1 | R@5 | R@10 | Notes |
|--------|-------|--------|-----|-----|-----|------|-------|
| — | all-MiniLM-L6-v2 | 22M | — | — | — | — | Pending |
| — | codebert-base | 125M | — | — | — | — | Pending |

---

## Key Decisions Log

| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| 2026-09-07 | Python Environment = local .venv with uv | Strict project rule: always use local .venv (`.venv/Scripts/activate`) managed by uv | Global / shared venv |
| 2026-09-08 | Max seq len = 256 | Empirical EDA (50k sample): P50=169, docstring coverage >95%, code coverage 68.7%; maximizes in-batch negative capacity in InfoNCE | 128 (too aggressive), 512 (4x attention memory penalty) |
| — | Vocab size = 16k | Balance coverage vs embedding size | 8k, 32k |
| — | Temperature = 0.07 | Standard for contrastive learning | 0.05, 0.1 |
| — | 4 Transformer layers | ~8M params, fits GPU memory | 2, 6, 8 |
| — | Dual encoder: 3L each | Match total params of shared (~8M) | 4L each (16M total) |

---

## Known Issues / Blockers

| Issue | Severity | Status | Workaround |
|-------|----------|--------|------------|
| — | — | — | — |

---

## Reproducibility Checklist

- [ ] All configs versioned in `configs/`
- [ ] Random seeds set (PyTorch, NumPy, Python)
- [ ] MLflow tracks all hyperparameters
- [ ] Model checkpoints saved with config
- [ ] Tokenizer saved with model
- [ ] Data splits fixed (no random shuffle in val/test)
- [x] Requirements pinned in `requirements.txt`
- [ ] Python version specified in `pyproject.toml`

---

## Next Actions

### Immediate (This Session)
1. [x] Scaffold project structure
2. [x] Create `requirements.txt` and `.gitignore`
3. [x] Install dependencies and CUDA PyTorch in local `.venv` via `uv`
4. [x] Implement data download script (`data/download.py`)
5. [x] Preprocess and clean CodeSearchNet (`data/preprocess.py`)

### Phase 0: Setup & Data Pipeline (Completed)
1. [x] Download and preprocess CodeSearchNet Python (`data/download.py`, `data/preprocess.py`)
2. [x] Train custom BPE tokenizer (`tokenizer/train_tokenizer.py`)
3. [x] Tokenizer PyTorch wrapper (`tokenizer/tokenizer.py`)
4. [x] Run EDA / token length distribution (`scripts/eda.py`)
5. [x] Implement PyTorch Dataset + Dynamic Collator (`data/dataset.py`)

### Immediate Next: Phase 1 — BM25 Baseline & Evaluation Framework
1. [ ] Implement evaluation metrics from mathematical definitions (`evaluation/metrics.py`: MRR, Recall@K, NDCG)
2. [ ] Implement BM25 lexical retrieval index (`retrieval/bm25.py`)
3. [ ] Run baseline evaluation script on test set (`scripts/run_baseline.py`)
4. [ ] Log baseline metrics to MLflow (`mlflow`)

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