# CodeEmbed — Product Requirements Document

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
- [ ] Custom BPE tokenizer trained on Python + natural language
- [ ] From-scratch Transformer encoder (4 layers, 256-dim, 8 heads)
- [ ] Three model architectures: Basic, Shared, Separate encoders
- [ ] Contrastive learning with InfoNCE loss and in-batch negatives
- [ ] BM25 lexical baseline
- [ ] Evaluation metrics: MRR, Recall@1/5/10, NDCG
- [ ] FAISS-based retrieval pipeline
- [ ] MLflow experiment tracking
- [ ] CLI demo for semantic code search

### Should Have
- [ ] Hard negative mining
- [ ] Ablation studies (tokenizer, pooling, temperature, embedding dim, model size)
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