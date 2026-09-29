# CodeEmbed — Product Requirements Document

> **Purpose**: Product & research requirements specification for CodeEmbed, a from-scratch contrastive learning system for semantic code retrieval on CodeSearchNet. All clean evaluations are conducted on `data/processed_clean_v2/` under [`PROTOCOL.md`](PROTOCOL.md). Historical exploratory metrics are archived in [Historical Pre-Remediation Exploration](#historical-pre-remediation-exploration).

## Project Overview

**CodeEmbed** is an empirical research project investigating contrastive representation learning for semantic code search. The project builds a from-scratch Transformer-based embedding system for natural-language → Python code retrieval, trained using contrastive learning on CodeSearchNet.

## Core Research Questions

1. **RQ1**: Can a small Transformer learn useful code/text representations from scratch?
2. **RQ2**: Does explicit modality information improve a shared encoder?
3. **RQ3**: Does separating the encoders improve retrieval?
4. **RQ4**: Do hard negatives improve retrieval?
5. **RQ5**: How much does pretraining matter compared to from-scratch learning?

## Target Users

- **Primary**: ML researchers studying code representation learning
- **Secondary**: Developers seeking semantic code search capabilities
- **Tertiary**: Educators demonstrating contrastive learning for code-text alignment

## Key Features

### Must Have
- [x] Custom BPE tokenizer trained on Python + natural language
- [x] From-scratch Transformer encoder (4 layers, 256-dim, 8 heads)
- [x] Three model architectures: Basic, Shared, Separate encoders (All completed: Basic, Shared, and Dual)
- [x] Contrastive learning with InfoNCE loss and in-batch negatives
- [x] BM25 lexical baseline (ATIRE floor, 0.5108 Test MRR)
- [x] Evaluation metrics: MRR, Recall@1/5/10, NDCG, paired bootstrap CIs
- [x] FAISS-based retrieval pipeline (Dense hard negative mining + FAISS indexing)
- [x] MLflow experiment tracking (`sqlite:///mlflow.db`)
- [ ] CLI demo for semantic code search (`demo/cli.py`)

### Should Have
- [x] Hard negative mining (BM25 CSR sparse mining + FAISS dense mining + 3-tier false negative filters)
- [x] Ablation studies (pooling strategy, loss temperature sensitivity, sequence length truncation)
- [x] Layer and parameter capacity scaling exploration (Phase 6.5: 17.03M params, Test MRR 0.4699, R@1 0.3637, R@10 0.6686)
- [ ] Pretrained baseline comparison
- [ ] Qualitative evaluation with failure analysis
- [ ] UMAP embedding visualization

### Nice to Have
- [ ] Hybrid Search integration (Dense + BM25 via Reciprocal Rank Fusion / Convex Combination)
- [ ] Multi-language support
- [ ] Web-based demo
- [ ] RAG integration

## Success Criteria & Benchmark Validation

| Metric | Target | Clean Achieved (17M Scaled) | Status |
|--------|:---:|:---:|:---:|
| Test MRR | > 0.40 | **0.4699** [0.4636, 0.4757] | 🟢 Exceeded (+6.99 pts) |
| Test Recall@1 | > 0.30 | **0.3637** | 🟢 Exceeded (+6.37 pts) |
| Test Recall@10 | > 0.60 | **0.6686** | 🟢 Exceeded (+6.86 pts) |
| Training time (single GPU) | < 24 hours | **2.77 hours (4 epochs)** | 🟢 Exceeded (~8.6x faster) |
| Inference latency (batch=1) | < 50ms | **~12ms (GPU AMP)** | 🟢 Exceeded |

## Historical Pre-Remediation Exploration

*Prior to AST docstring stripping, CodeSearchNet functions contained verbatim docstrings inside the code body (100% query leakage). Historical benchmarks achieved artifactual metrics (BM25 MRR 0.9498, Shared MRR 0.9296, Shared Hard Negatives MRR 0.9383). All official scientific gates are now governed by the leak-free R-Track on `data/processed_clean_v2/`.*