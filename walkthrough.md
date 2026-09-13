# CodeEmbed — Project Walkthrough & Implementation Log

> **Purpose**: A comprehensive project log tracking the technical implementation, architectural decisions, benchmark results, and verification across each completed milestone.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Phase 0: Setup & Data Pipeline](#2-phase-0-setup--data-pipeline)
3. [Phase 1: BM25 Baseline & Evaluation Framework](#3-phase-1-bm25-baseline--evaluation-framework)
4. [Phase 2: Model 1 — Basic Transformer Encoder](#4-phase-2-model-1--basic-transformer-encoder)
5. [Summary Benchmark Comparison](#5-summary-benchmark-comparison)
6. [Comprehensive Test Suite & Quality Checks](#6-comprehensive-test-suite--quality-checks)
7. [Next Milestone: Phase 3 (Shared Encoder)](#7-next-milestone-phase-3-shared-encoder)

---

## 1. Project Overview

**CodeEmbed** is a semantic code retrieval research system developed with raw PyTorch primitives. The core mission is to bridge the lexical vocabulary gap between natural language developer intent (docstrings/queries) and implementation code (Python function bodies).

### Key Architectural Constraints
* **Pure PyTorch**: Implement Transformer architectures directly from `nn.Module` without HuggingFace model wrappers.
* **Pre-LN Stability**: Pre-Layer Normalization exclusively across all attention and feed-forward layers.
* **Standardized Evaluation**: Identical mathematical definitions of MRR, Recall@K, and NDCG@K applied across lexical and neural models.
* **In-Batch Contrastive Learning**: InfoNCE objective with in-batch negative mining.

---

## 2. Phase 0: Setup & Data Pipeline

### 2.1 Tooling & Environment
* **Package Management**: Managed with `uv` inside a project-isolated virtual environment (`.venv`).
* **GPU Acceleration**: PyTorch compiled with NVIDIA CUDA 12.4 kernels for hardware acceleration on RTX GPUs.

### 2.2 Data Pipeline (`data/`)
* **Source Dataset**: CodeSearchNet (Python subset).
* [`data/download.py`](data/download.py): Streams raw splits from Hugging Face into compressed local Parquet files.
* [`data/preprocess.py`](data/preprocess.py):
  - Filtered missing docstrings, empty function bodies, and boilerplate (`__init__.py`).
  - Enforced length bounds: $10 \le \text{tokens} \le 2048$.
  - Deduplicated by exact function body match to eliminate data leakage.
  - **Dataset yield**: **385,381** train, **21,585** validation, **21,005** test clean function-docstring pairs (~223 MB total footprint).
* [`data/dataset.py`](data/dataset.py):
  - `CodeSearchDataset`: Parquet-backed random-access PyTorch `Dataset`.
  - `CodeSearchCollator`: **Dynamic batch padding** that pads sequences only to the maximum length within each batch rather than static global padding, saving >50% GPU memory during training.
  - `create_dataloader`: Production DataLoader factory supporting multi-worker loading and CUDA pinned memory.

### 2.3 Tokenization (`tokenizer/`)
* [`tokenizer/train_tokenizer.py`](tokenizer/train_tokenizer.py): Trained a custom **16,000-vocabulary Byte-Pair Encoding (BPE)** model on joint code and docstring corpora.
* **Special Tokens**: `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, `<EOS>=3`, `<CODE>=4`, `<TEXT>=5`.
* [`tokenizer/tokenizer.py`](tokenizer/tokenizer.py): Tokenizer wrapper providing PyTorch tensor outputs (`input_ids`, `attention_mask`) and modality prefix routing.

### 2.4 Exploratory Data Analysis (`scripts/eda.py`)
* Analyzed 50,000 samples for token length distributions ($P_{50}, P_{75}, P_{90}, P_{95}, P_{99}$).
* **Decision**: Selected `max_seq_len = 256` (covers >95% of docstrings and 68.7% of code snippets while avoiding the $4\times$ attention memory penalty of 512, preserving maximum in-batch negative batch capacity).

---

## 3. Phase 1: BM25 Baseline & Evaluation Framework

Before training neural models, Phase 1 established an empirical lexical baseline and a standardized evaluation suite.

### 3.1 Evaluation Framework (`evaluation/`)
* [`evaluation/metrics.py`](evaluation/metrics.py): Exact Information Retrieval metrics:
  - **Mean Reciprocal Rank (MRR)**: $\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$
  - **Recall@K** ($K \in \{1, 5, 10\}$): $\text{Recall@}K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(\text{rank}_i \le K)$
  - **Normalized Discounted Cumulative Gain (NDCG@K)**: $\text{NDCG@}K = \frac{1}{\log_2(\text{rank}_i + 1)}$ for $\text{rank}_i \le K$.
  - **Non-Parametric Bootstrap Confidence Intervals**: 1,000 resamples computing empirical 95% confidence intervals $[c_{2.5}, c_{97.5}]$.
* [`evaluation/evaluate.py`](evaluation/evaluate.py): Reusable runner computing metrics from precomputed ranks or dense similarity matrices.

### 3.2 High-Performance BM25 Retriever (`retrieval/bm25.py`)
* **Code-Aware Tokenization**: Regex decomposition splitting `snake_case`, `camelCase`, and alphanumeric tokens while preserving semantic units like `l2` and `dim256`.
* **Vectorized Inverted Index**:
  - Precomputes length normalization array $\text{len\_norm} \in \mathbb{R}^N$ and Okapi IDF.
  - Stores inverted postings as NumPy arrays (`int32`, `float32`).
  - **154x Speedup**: Retrieval over 1,000 test queries against 21,005 functions dropped from **592.66s down to 3.74s** (267.6 queries/sec)!

### 3.3 Baseline Benchmark Results
Executed via [`scripts/run_baseline.py`](scripts/run_baseline.py) and logged to MLflow (`mlruns`, Run ID: `234f518410034b628b9c90eb7cbbc1cf`):

| Metric | Score | 95% Confidence Interval (1,000 resamples) |
| :--- | :---: | :---: |
| **MRR** | **0.9498** | [0.9389, 0.9597] |
| **Recall@1** | **0.9180** | [0.9020, 0.9340] |
| **Recall@5** | **0.9890** | [0.9820, 0.9950] |
| **Recall@10** | **0.9950** | [0.9900, 0.9990] |
| **NDCG@10** | **0.9610** | [0.9521, 0.9691] |

---

## 4. Phase 2: Model 1 — Basic Transformer Encoder

Phase 2 delivered our first neural architecture: a single ~7.38M parameter Transformer encoder that maps both code and text into a shared 256-dimensional unit hypersphere.

### 4.1 Architecture Components (`model/`)
* [`model/embeddings.py`](model/embeddings.py):
  - `TokenEmbedding` (16,000 $\times$ 256) + `PositionalEmbedding` (256 $\times$ 256) + `ModalityEmbedding`.
  - Composite `EmbeddingLayer` with LayerNorm and Dropout ($p=0.1$).
* [`model/attention.py`](model/attention.py):
  - Custom `MultiHeadSelfAttention`: 8 heads, $d_k = 32$, $d_{\text{model}} = 256$.
  - Scaled dot-product attention with safe FP16 padding mask handling (using $-10^4$ to avoid IEEE 754 overflow NaNs).
* [`model/transformer.py`](model/transformer.py):
  - `PreLNTransformerBlock`: Pre-LayerNorm residual flow with GELU FFN ($256 \rightarrow 1024 \rightarrow 256$).
  - `TransformerEncoder`: 4-layer stack with post-stack final LayerNorm.
* [`model/pooling.py`](model/pooling.py):
  - `MaskedMeanPooling`: Correct length-normalized pooling excluding padding tokens from both numerator and denominator.
  - `CLSPooling`: Extracts token index 0.
* [`model/encoder.py`](model/encoder.py):
  - `BaseEncoder`: End-to-end pipeline combining embeddings, 4 transformer blocks, masked mean pooling, projection head (`Linear` + `LayerNorm`), and L2 unit-sphere normalization ($\|\mathbf{z}\|_2 = 1.0$).
  - **Parameter Count**: **7,382,528** trainable parameters (~7.38M).

### 4.2 Contrastive Loss (`losses/contrastive.py`)
* [`losses/contrastive.py`](losses/contrastive.py): Symmetric InfoNCE loss with in-batch negatives:
  $$\mathbf{S} = \frac{\mathbf{z}_{\text{text}} \mathbf{z}_{\text{code}}^\top}{\tau} \in \mathbb{R}^{B \times B}, \quad \tau = 0.07$$
  $$\mathcal{L}_{\text{InfoNCE}} = \frac{1}{2} \left[ \text{CrossEntropy}(\mathbf{S}, \mathbf{y}) + \text{CrossEntropy}(\mathbf{S}^\top, \mathbf{y}) \right]$$

### 4.3 Training & Checkpointing Engine (`training/trainer.py`)
* [`training/trainer.py`](training/trainer.py):
  - Mixed precision via `torch.amp.autocast` and `torch.amp.GradScaler` on CUDA.
  - AdamW optimizer with parameter group filtering (no weight decay on 1D biases or LayerNorm).
  - Cosine annealing learning rate scheduler with linear warmup.
  - Gradient clipping (`max_norm = 1.0`).
  - Automatic checkpoint manager saving the best model by validation score (`checkpoints/basic/best_basic.pt`).
* [`scripts/run_basic.py`](scripts/run_basic.py): CLI training entry point integrated with MLflow tracking.
* [`evaluation/evaluate.py`](evaluation/evaluate.py): Standalone CLI to benchmark saved `.pt` checkpoints against test/validation splits:
  ```bash
  python -m evaluation.evaluate --checkpoint checkpoints/basic/best_basic.pt --split test
  ```

---

5. [Phase 3: Model 2 — Shared Transformer Encoder](#5-phase-3-model-2--shared-transformer-encoder)
6. [Summary Benchmark Comparison](#6-summary-benchmark-comparison)
7. [Comprehensive Test Suite & Quality Checks](#7-comprehensive-test-suite--quality-checks)
8. [Next Milestone: Phase 4 (Separate / Dual Encoders)](#8-next-milestone-phase-4-separate--dual-encoders)

---

## 5. Phase 3: Model 2 — Shared Transformer Encoder

Phase 3 introduced **modality awareness** into the single shared Transformer encoder. By adding learned modality embeddings ($\mathbf{e}_{\text{modality}} \in \mathbb{R}^{2 \times d_{\text{model}}}$), the network learns to project code and natural language queries into a unified semantic space while explicitly preserving modality boundaries.

### 5.1 Architecture Components (`model/shared_encoder.py`)
* [`model/embeddings.py`](model/embeddings.py):
  - Updated `EmbeddingLayer` with `use_modality_embedding=True` and `num_modalities=2`.
  - Composite embedding: $\mathbf{x} = \text{LayerNorm}(\mathbf{E}_{\text{token}} + \mathbf{E}_{\text{pos}} + \mathbf{E}_{\text{modality}})$.
* [`model/shared_encoder.py`](model/shared_encoder.py):
  - `SharedEncoder`: 4-layer Pre-LN Transformer (~7.38M parameters, exactly 7,383,040 parameters = BaseEncoder + 2 $\times$ 256 modality table).
  - Modality router supporting string (`"code"`, `"text"`), integer (`0`, `1`), or tensor specifications.
  - Masked mean pooling + Linear projection head (`bias=False`) + LayerNorm + L2 normalization onto unit hypersphere ($\|\mathbf{z}\|_2 = 1.0$).
* [`training/trainer.py`](training/trainer.py):
  - Enhanced `ContrastiveTrainer` with polymorphic `_encode_pair` routing code to modality 0 and docstrings to modality 1.
  - Dynamically saves best checkpoints to `checkpoints/shared/best_shared.pt`.
* [`configs/shared.yaml`](configs/shared.yaml) & [`scripts/run_shared.py`](scripts/run_shared.py):
  - Standardized configuration and training runner with full MLflow tracking.

### 5.2 Test Set Benchmark Results
Trained on NVIDIA RTX 4050 (CUDA AMP fp16) across 3,010 training batches (~385k samples). Evaluated on 1,000 sampled test queries against the full 21,005 test corpus with 1,000 bootstrap resamples:

| Metric | Score | 95% Confidence Interval (1,000 resamples) |
| :--- | :---: | :---: |
| **MRR** | **0.9296** | [0.9171, 0.9427] |
| **Recall@1** | **0.8880** | [0.8690, 0.9080] |
| **Recall@5** | **0.9780** | [0.9690, 0.9870] |
| **Recall@10** | **0.9840** | [0.9760, 0.9920] |
| **NDCG@10** | **0.9429** | [0.9326, 0.9540] |

---

## 6. Summary Benchmark Comparison

| Model | Architecture / Modality | Corpus Size | Eval Queries | MRR | R@1 | R@5 | R@10 | NDCG@10 | Artifact Location |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BM25 Baseline** | Lexical Subwords | 21,005 | 1,000 (test) | **0.9498** | **0.9180** | **0.9890** | **0.9950** | **0.9610** | MLflow `234f518410034b628b9c90eb7cbbc1cf` |
| **Basic Encoder** | Neural Shared (7.38M, no mod) | 21,585 | 100 (val) | **0.4633** | **0.4100** | **0.5400** | **0.5500** | **0.4806** | `checkpoints/basic/best_basic.pt` |
| **Shared Encoder** | Neural Shared + Modality (7.38M) | 21,005 | 1,000 (test) | **0.9296** | **0.8880** | **0.9780** | **0.9840** | **0.9429** | [`checkpoints/shared/best_shared.pt`](checkpoints/shared/best_shared.pt) |

---

## 7. Comprehensive Test Suite & Quality Checks

The codebase is continuously verified using Pytest and Ruff:

```bash
uv run pytest tests/ -v
```

### Test Results (50/50 Passed in ~7s):
* **`tests/test_model.py` (12 tests)**: Token/pos/mod embeddings, multi-head attention, Pre-LN blocks, pooling, unit normalization.
* **`tests/test_shared_encoder.py` (5 tests)**: Modality routing, unit hypersphere outputs, backward gradient propagation through modality table, parameter budget verification.
* **`tests/test_loss.py` (5 tests)**: InfoNCE symmetry, alignment, gradient flow.
* **`tests/test_bm25.py` (9 tests)**: Inverted index, tokenization, serialization.
* **`tests/test_metrics.py` (19 tests)**: MRR, Recall@K, NDCG@K, bootstrap CIs.

### Linter & Style:
```bash
uv run ruff check .
# All checks passed! (0 errors)
```

---

## 8. Next Milestone: Phase 4 (Separate / Dual Encoders)

With the Shared Encoder validated, the next milestone is **Phase 4: Model 3 — Separate (Dual) Encoders**:
1. Implement `DualEncoder` with separate dedicated code and text Transformer encoders (`model/dual_encoder.py`).
2. Control parameter budget to match ~8M total (3 layers each $\times$ ~4M params).
3. Train with symmetric InfoNCE and benchmark against Shared Encoder and BM25 to answer Research Question 3 (RQ3).


