# CodeEmbed — Architecture Document

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
                        CODESEARCHNET
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
                        DATA PIPELINE
    download → clean → filter → deduplicate → train/val/test split
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
                        TOKENIZER
              Custom BPE (8k-16k vocab) on code+text
                    Special: <PAD> <UNK> <BOS> <EOS> <CODE> <TEXT>
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌───────────────────┐       ┌───────────────────┐
        │   CODE CORPUS     │       │   TEXT CORPUS     │
        └─────────┬─────────┘       └─────────┬─────────┘
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
        ┌─────────────────────────────────────────────┐
        │          CONTRASTIVE TRAINING               │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
        │  │  Basic   │  │  Shared  │  │ Separate │  │
        │  │ Encoder  │  │ Encoder  │  │ Encoders │  │
        │  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
        │       └─────────────┼─────────────┘         │
        │                     ▼                       │
        │          InfoNCE Loss (temp=0.07)           │
        │              In-batch negatives             │
        └────────────────────┬────────────────────────┘
                             │
                             ▼
        ┌─────────────────────────────────────────────┐
        │           EVALUATION & RETRIEVAL            │
        │  MRR, Recall@K, NDCG  →  FAISS Index  →  Demo
        └─────────────────────────────────────────────┘
```

## Model Architectures

### Common Transformer Block (Pre-LN)

```
Input (B, L, D=256)
    │
    ▼
LayerNorm
    │
    ▼
Multi-Head Self-Attention (8 heads, d_k=32)
    │
    ▼
Residual + Dropout
    │
    ▼
LayerNorm
    │
    ▼
FFN (D=256 → 1024 → 256, GELU)
    │
    ▼
Residual + Dropout
    │
    ▼
Output (B, L, D=256)
```

### Architecture Variants

| Model | Description | Parameters |
|-------|-------------|------------|
| **Model 0** | BM25 (lexical baseline) | — |
| **Model 1** | Basic: Single encoder for code+text | ~8M |
| **Model 2** | Shared: Single encoder + modality embeddings | ~8M |
| **Model 3** | Separate: Code encoder + Text encoder (4M each) | ~8M total |

### Embedding Pipeline

```
Tokens (B, L)
    │
    ▼
Token Embeddings (B, L, 256)
    +
Positional Embeddings (L, 256)
    [+ Modality Embeddings for Shared]
    │
    ▼
4× Transformer Blocks
    │
    ▼
Masked Mean Pooling → (B, 256)
    │
    ▼
Projection Head: Linear(256→256) + LayerNorm
    │
    ▼
L2 Normalization → (B, 256) unit vectors
```

## Tech Stack

| Category | Technology | Version |
|----------|------------|---------|
| **Language** | Python | 3.10+ |
| **Deep Learning** | PyTorch | 2.3+ |
| **Tokenizer** | HuggingFace tokenizers | 0.19+ |
| **Retrieval** | FAISS | 1.8+ |
| **Experiment Tracking** | MLflow | 2.14+ |
| **Data Processing** | pandas, datasets | Latest |
| **Evaluation** | scikit-learn, rank_eval | Latest |
| **Visualization** | matplotlib, umap-learn | Latest |
| **Config Management** | OmegaConf / YAML | Latest |
| **Logging** | rich, tqdm | Latest |

## Folder Structure

```
codeembed/
│
├── configs/
│   ├── basic.yaml          # Basic encoder config
│   ├── shared.yaml         # Shared encoder config
│   ├── dual.yaml           # Separate encoders config
│   ├── training.yaml       # Training hyperparameters
│   └── tokenizer.yaml      # Tokenizer config
│
├── data/
│   ├── download.py         # CodeSearchNet download
│   ├── preprocess.py       # Cleaning, filtering, dedup
│   ├── dataset.py          # PyTorch Dataset + Collator
│   └── splits.py           # Train/val/test splits
│
├── tokenizer/
│   ├── train_tokenizer.py  # BPE training script
│   └── tokenizer.py        # Tokenizer wrapper class
│
├── model/
│   ├── embeddings.py       # Token + Positional + Modality
│   ├── attention.py        # Multi-head attention
│   ├── transformer.py      # Transformer block
│   ├── pooling.py          # Masked mean, CLS pooling
│   ├── encoder.py          # Base encoder class
│   ├── shared_encoder.py   # Shared encoder with modality
│   └── dual_encoder.py     # Separate code/text encoders
│
├── losses/
│   └── contrastive.py      # InfoNCE loss
│
├── training/
│   ├── trainer.py          # Training loop, logging
│   └── train.py            # Entry point scripts
│
├── retrieval/
│   ├── build_index.py      # FAISS index construction
│   └── search.py           # Query encoding + search
│
├── evaluation/
│   ├── metrics.py          # MRR, Recall@K, NDCG
│   ├── evaluate.py         # Evaluation pipeline
│   └── qualitative.py      # Failure analysis, examples
│
├── experiments/
│   ├── tokenizer/          # Tokenizer ablations
│   ├── architecture/       # Model comparisons
│   └── negatives/          # Negative sampling studies
│
├── demo/
│   └── cli.py              # CLI demo
│
├── notebooks/
│   ├── eda.ipynb           # Data exploration
│   ├── token_analysis.ipynb
│   └── results_analysis.ipynb
│
├── scripts/
│   ├── run_baseline.py
│   ├── run_basic.py
│   ├── run_shared.py
│   ├── run_dual.py
│   └── run_hard_negatives.py
│
├── tests/
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_retrieval.py
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Data Flow

### Training
```
CodeSearchNet → Preprocess → Tokenize → Batch (text, code)
                                    ↓
                          Forward both through encoder(s)
                                    ↓
                              L2-normalized embeddings
                                    ↓
                              Cosine similarity matrix
                                    ↓
                              InfoNCE Loss (diagonal targets)
                                    ↓
                              Backprop + Optimizer step
```

### Inference / Retrieval
```
Corpus functions → Encoder → FAISS index (offline, once)

User query → Text Encoder → Query embedding
                              ↓
                        FAISS.search(k=10)
                              ↓
                        Top-K functions + scores
```

## Configuration Management

All hyperparameters in YAML configs loaded via OmegaConf:

```yaml
# configs/training.yaml
model:
  vocab_size: 16000
  d_model: 256
  n_layers: 4
  n_heads: 8
  d_ff: 1024
  max_seq_len: 256
  dropout: 0.1

training:
  batch_size: 256
  lr: 3e-4
  weight_decay: 0.01
  warmup_steps: 2000
  max_steps: 100000
  grad_clip: 1.0
  temperature: 0.07

data:
  max_length: 256
  num_workers: 4
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8 GB VRAM | 16-24 GB VRAM |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB | 100 GB NVMe |
| Compute | 1× RTX 3080 | 1× A100 / RTX 4090 |

## Dependencies (requirements.txt)

```text
torch>=2.3.0
transformers>=4.40.0
tokenizers>=0.19.0
faiss-cpu>=1.8.0
mlflow>=2.14.0
datasets>=2.19.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
omegaconf>=2.3.0
tqdm>=4.66.0
rich>=13.7.0
matplotlib>=3.8.0
umap-learn>=0.5.6
rank-eval>=0.1.0
```