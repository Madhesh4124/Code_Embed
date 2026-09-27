# CodeEmbed — Product Requirements Document

> [!CAUTION]
> **INVALID: Historical Phase 1–6 metrics computed on leaky data, superseded by R-track.**
> All historical performance metrics in this document were computed on unstripped CodeSearchNet code containing 100% docstring query leakage. All scientific answers to RQ1–RQ5 are being re-evaluated under the **R-Track (Remediation Track)** governed by [`PROTOCOL.md`](PROTOCOL.md) and tracked in [`R_TRACK_MEMORY.md`](R_TRACK_MEMORY.md) on branch `r-phase`.

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
- [x] BM25 lexical baseline
- [x] Evaluation metrics: MRR, Recall@1/5/10, NDCG
- [ ] FAISS-based retrieval pipeline
- [x] MLflow experiment tracking
- [ ] CLI demo for semantic code search

### Should Have
- [x] Hard negative mining (BM25 CSR sparse mining + InfoNCEWithHardNegativesLoss)
- [x] Ablation studies (pooling strategy, loss temperature sensitivity, sequence length truncation)
- [ ] Layer and parameter capacity scaling exploration (Phase 6.5: ~4M to ~54M params)
- [ ] Pretrained baseline comparison
- [ ] Qualitative evaluation with failure analysis
- [ ] UMAP embedding visualization

### Nice to Have
- [ ] Multi-language support
- [ ] Web-based demo
- [ ] RAG integration

## Success Criteria

| Metric | Target |
|--------|--------|
| MRR (Separate + Hard Negatives) | > 0.40 |
| Recall@1 | > 0.30 |
| Recall@10 | > 0.60 |
| Training time (single GPU) | < 24 hours |
| Inference latency (batch=1) | < 50ms |

## Out of Scope (Phase 1)

- Multi-language support beyond Python
- Models > 10M parameters
- Multi-GPU distributed training
- Production deployment
- Fancy UI/frontend