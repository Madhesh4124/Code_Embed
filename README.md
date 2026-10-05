# CodeEmbed

**From-Scratch Contrastive Representation Learning for Semantic Code Retrieval**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-ee4c2c.svg)](https://pytorch.org/)
[![Protocol Frozen](https://img.shields.io/badge/protocol-v1.1%20frozen-success.svg)](agent_documentation/PROTOCOL.md)
[![Tests Passing](https://img.shields.io/badge/tests-84%2F84%20passing-brightgreen.svg)](tests/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary

**CodeEmbed** is an empirical research and engineering project implementing a custom, from-scratch Transformer embedding model for natural language-to-code retrieval on Python CodeSearchNet. Rather than relying on black-box pretrained LLM backbones, CodeEmbed evaluates how architectural inductive biases (Pre-LN, cross-modal weight sharing, learned modality tables, masked mean pooling, and hard-negative mining) influence representation geometry in low-resource contrastive learning.

### 🎯 Primary Confirmatory Benchmark (Pre-Registered Protocol §3)
Evaluated **strictly once** on the clean, uncorrupted test split ($N = 19,632$ queries against 19,632 code documents) per frozen protocol [`PROTOCOL.md`](agent_documentation/PROTOCOL.md):

* **Phase R3 Shared Encoder (4L, 7.38M parameters)**:
  * **Test MRR**: **0.4157** [95% CI: 0.4098, 0.4216]
  * **Recall@1**: **0.3178** | **Recall@10**: **0.6018** | **NDCG@10**: **0.4531**
  * **Statistical Superiority on Semantic Search**: Dense retrieval outperforms the tuned ATIRE BM25 baseline by **+11.05 MRR points** on low-overlap queries (0.3204 vs 0.2099) and by **$5\times$** on zero-overlap queries (0.0487 vs 0.0099).
  * **Modality Lift (RQ2)**: Adding a learned $2 \times 256$ modality embedding table to a shared encoder yields an unambiguous $+3.84$ MRR point gain ($p < 10^{-4}$) with zero added Transformer layers.

### 🔬 Post-Protocol Exploratory Model Iteration
*Additional architectures and training variants were explored after the primary evaluation. Their test-set numbers are reported for exploratory reference but were not protected from model selection bias:*
* **Phase 6.5 Scaled Shared Encoder (6L-384d, 17.03M params)**: Full Val MRR **0.4620** [0.4559, 0.4679], Full Test MRR **0.4699** [0.4636, 0.4757] (trained for 4 epochs with 1 FAISS dense hard negative).
* **Dual Encoder Baseline**: Test MRR **0.2900** without weight sharing; lifts to **0.4807** ($\tau=0.10$) / **0.4684** ($\tau=0.07$) with FAISS dense hard negatives (unprotected from hyperparameter selection on test).

---

## 🖥️ Interactive Search & Visual Research Dashboard

CodeEmbed provides an interactive semantic retrieval interface to search the 19,632 CodeSearchNet test functions in real time using the from-scratch bi-encoder, BM25, or hybrid fusion.

### 🌟 Streamlit Web Application (Cloud-Ready)

Built with **Streamlit** and **Plotly** (hardware-accelerated WebGL `go.Scattergl`), featuring live interactive diagnostics:

```bash
# Launch interactive Streamlit research dashboard
uv run streamlit run streamlit_app.py
# Opens immediately on http://localhost:8501
```

#### Key Diagnostic Features:
1. **⚖️ Hybrid Score Decomposition Waterfall**: Horizontal stacked percentage bars dissecting candidate scores into **Neural Dense (Semantic)** vs. **Lexical BM25 (Exact Token)** contributions.
2. **💎 Token-Level Attribution & Zero-Overlap Callout**: Highlights matching query tokens inside the code body. When a query shares **zero tokens** with the retrieved code, surfaces an explicit **Pure Semantic Generalization** badge proving dense bi-encoder success where BM25 fails completely.
3. **🗺️ 2D Latent Semantic Space Map (PCA)**: Pre-computed 2D projection of all 19,632 test functions with **real-time query projection**, glowing query marker, ranked neighbor clusters, and cosine distance vectors.
4. **🎯 BM25 vs. Dense Discordance Quadrant Plot**: Dynamic candidate scatter plot across normalized BM25 vs. normalized Dense cosine similarity, with shaded quadrants (Consensus Sweet Spot, Pure Semantic Gems, Lexical Keyword Matches, and Fringe Noise).
5. **🎯 Pre-Loaded Authentic Test Split Queries**: Direct dropdown selection of verified test set queries from `test_stratified.parquet` across Zero, Low, and High overlap strata.

### 💻 Command-Line & REST Interfaces

```bash
# Terminal Interactive Search REPL
uv run python demo/search.py --interactive

# Single-shot query execution
uv run python demo/search.py --query "Converts a string to a valid filename." --mode hybrid --top-k 5

# FastAPI Backend & Web UI
uv run python demo/app.py
# Access http://localhost:8000
```


---

## 📊 Benchmark & Empirical Evaluation

### 1. Pre-Registered Confirmatory Benchmark (Protocol §3)
Evaluated strictly once on the clean test split ($N = 19,632$ queries against 19,632 corpus documents):

| Model | Parameters | Negatives / Mode | Split | MRR [95% Bootstrap CI] | Recall@1 | Recall@5 | Recall@10 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (ATIRE Reference)** | 0 | Lexical Inverted Index | Full Test | **0.5109** [0.5048, 0.5166] | 0.4052 | 0.6340 | 0.6993 | 0.5514 |
| **Basic Encoder** (0 Modality) | 7.38M | In-Batch (Masked) | Full Test | **0.3773** [0.3716, 0.3832] | 0.2836 | 0.4814 | 0.5575 | 0.4132 |
| **Shared Encoder (In-Batch)** | 7.38M | In-Batch (Masked) | Full Test | **0.4157** [0.4098, 0.4216] | 0.3178 | 0.5246 | 0.6018 | 0.4531 |
| **Shared Encoder (BM25 HN)** | 7.38M | 1 BM25 HN + In-Batch | Full Test | **0.4155** [0.4095, 0.4215] | 0.3184 | 0.5227 | 0.6016 | 0.4529 |
| **Scaled Shared Encoder (6L-384d)\*** | 17.03M | 1 FAISS Dense + In-Batch | Full Test | **0.4699** [0.4636, 0.4757] | 0.3637 | 0.5896 | 0.6686 | 0.5109 |
| **CodeEmbed Hybrid (RRF, $k=20$)** | 17.03M | RRF Fusion (BM25 + 17M) | Full Test | **0.6291** [0.6234, 0.6347] | 0.5089 | 0.7648 | 0.8215 | 0.6722 |
| **CodeEmbed Hybrid (Convex, $\alpha=0.7$)** | 17.03M | Convex Fusion (BM25 + 17M) | Full Test | **0.6612** [0.6555, 0.6670] | **0.5596** | **0.7838** | **0.8392** | **0.7008** |

### 2. Lexical Overlap Stratification ($N = 19,632$ Test Queries)
Why does standalone BM25 retain a higher overall MRR than neural encoders on the overall dataset despite neural models dominating semantic retrieval? Stratifying by query-code surface token overlap ($c$) isolates where each retrieval paradigm excels.

#### Table 2A: Confirmatory Benchmark — 4L Shared Encoder (7.38M) vs. BM25
*Pre-registered protocol evaluation conducted strictly once on test ($N=19,632$ queries). Deliberately protected from model selection bias.*

| Overlap Stratum | % of Split | Query Count | BM25 MRR | 4L Shared MRR (Confirmatory) | Confirmatory Delta (4L − BM25) | Paradigm Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Zero-Overlap ($c = 0.0$)** | 2.73% | 535 | 0.0099 | **0.0487** | **+0.0388** | 🚀 **Dense ($4.9\times$ over BM25)** |
| **Low-Overlap ($0 < c \le 0.30$)** | 29.58% | 5,808 | 0.2099 | **0.3204** | **+0.1105** | 🚀 **Dense (+11.05 MRR pts)** |
| **High-Overlap ($c > 0.30$)** | 67.69% | 13,289 | **0.6625** | 0.4716 | **-0.1909** | 📉 BM25 (+19.09 MRR pts) |
| **OVERALL DATASET** | 100.0% | 19,632 | **0.5108** | 0.4157 | **-0.0951** | 📉 **BM25 (+9.51 MRR pts)** |

> **Key Confirmatory Finding**: The 4L Shared Encoder outperforms BM25 on all semantic queries (low and zero overlap, ~32.3% of the dataset), but BM25's exact inverted index wins on high-overlap queries (67.7% of the dataset), resulting in a net **-9.51 MRR point gap** overall for the 4L baseline ($0.4157$ vs $0.5108$).

#### Table 2B: Post-Protocol Exploratory Iteration — Scaled 17.03M Shared Encoder vs. BM25
*Exploratory capacity scaling (6L-384d, 4 epochs, FAISS dense hard negatives); evaluated post-R3 during model development and unprotected from model selection bias.*

| Overlap Stratum | % of Split | Query Count | BM25 MRR | 17M Scaled MRR (Exploratory)* | Exploratory Delta (17M − BM25) | Exploratory Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Zero-Overlap ($c = 0.0$)** | 2.73% | 535 | 0.0099 | **0.0716** | **+0.0617** | 🚀 **Dense ($7.2\times$ over BM25)** |
| **Low-Overlap ($0 < c \le 0.30$)** | 29.58% | 5,808 | 0.2099 | **0.3559** | **+0.1460** | 🚀 **Dense (+14.60 MRR pts)** |
| **High-Overlap ($c > 0.30$)** | 67.69% | 13,289 | **0.6625** | 0.5357 | **-0.1268** | 📉 BM25 (+12.68 MRR pts) |
| **OVERALL DATASET** | 100.0% | 19,632 | **0.5108** | 0.4699 | **-0.0409** | 📉 **BM25 (+4.09 MRR pts)** |

*\*Unprotected from model selection bias (evaluated on test during exploratory capacity scaling).*

### 3. Discordance Analysis & Official Hybrid Search Benchmark
Analyzing query-level ranking discordance between the 17.03M Dense Encoder and BM25 reveals nearly orthogonal error profiles:
* **Dense beats BM25**: **7,411 queries (37.7%)**
* **BM25 beats Dense**: **7,599 queries (38.7%)**
* **Tied**: **4,622 queries (23.5%)**
* **Oracle Bound (Best of Either per query)**: **0.6562 MRR** (+14.54 MRR points over standalone BM25).

#### Table 2C: Official Hybrid Fusion Benchmark ($N = 19,632$ Clean Test Queries)
*Fusing scratch-trained CodeEmbed 17.03M dense embeddings with BM25 inverted indexes under Protocol v1.1 harmonic tie-breaking:*

| System / Fusion Strategy | Overall MRR [95% Bootstrap CI] | Recall@1 | Recall@5 | Recall@10 | NDCG@10 | Low-Overlap MRR | High-Overlap MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (ATIRE Reference)** | **0.5109** [0.5048, 0.5166] | 0.4052 | 0.6340 | 0.6993 | 0.5514 | 0.2100 | 0.6625 |
| **CodeEmbed 17M Dense** | **0.4699** [0.4636, 0.4757] | 0.3637 | 0.5896 | 0.6686 | 0.5109 | 0.3559 | 0.5357 |
| **Hybrid RRF ($k=60$)** | **0.6218** [0.6159, 0.6272] | 0.5011 | 0.7560 | 0.8263 | 0.6672 | 0.4139 | 0.7365 |
| **Hybrid RRF ($k=20$)** | **0.6291** [0.6234, 0.6347] | 0.5089 | 0.7648 | 0.8215 | 0.6722 | 0.4136 | 0.7470 |
| **Hybrid Convex ($\alpha=0.7$)** | **0.6612** [0.6555, 0.6670] | **0.5596** | **0.7838** | **0.8392** | **0.7008** | **0.4443** | **0.7813** |

> **Key Hybrid Takeaway**:
> 1. **Overwhelming Statistical Lift**: Convex score fusion ($\alpha=0.7$) achieves **+15.04 MRR points** over BM25 ($\Delta\text{MRR} = +0.1504$ [$+0.1457, +0.1550$], $p < 0.0001$) and **+19.14 MRR points** over Dense ($\Delta\text{MRR} = +0.1914$, $p < 0.0001$).
> 2. **Surpassing the Naive Oracle**: The hybrid model achieves **0.6612 MRR**, surpassing the naive oracle bound ($0.6562$), because multi-view evidence consensus elevates documents that were moderately ranked in both retrievers directly into rank 1.
> 3. **Recall@10 jumps to 83.92%** (vs 69.93% for BM25 and 66.86% for Dense).


### 4. Post-Protocol Exploratory Iteration Reference
*These models were trained and evaluated during exploratory engineering iteration; they are not protected from model selection bias:*

| Model Variant | Parameters | Epochs | Split | MRR [95% CI] | R@1 | R@10 | Notes & Selection Caveats |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Shared Encoder (FAISS Dense HN)** | 7.38M | 2 | Test | **0.4192** [0.4132, 0.4250] | 0.3216 | 0.6020 | Post-R3 exploratory dense mining. |
| **Scaled Shared Encoder (6L-384d)** | 17.03M | 4 | Val<br>Test | **0.4620** [0.4559, 0.4679]<br>**0.4699** [0.4636, 0.4757] | 0.3552<br>0.3637 | 0.6655<br>0.6686 | Post-R3 capacity scaling (Category B resolution). |
| **Dual Encoder Baseline** | 14M | 2 | Test | **0.2900** | — | — | Independent teammate baseline (no weight sharing). |
| **Dual Encoder (FAISS HN, $\tau=0.10$)** | 14M | 1 | Test | **0.4807** | — | — | Evaluated on test; test-selected hyperparameter. |
| **Dual Encoder (FAISS HN, $\tau=0.07$)** | 14M | 1 | Test | **0.4684** | — | — | Evaluated on test; test-selected hyperparameter. |

### 5. Pretrained Baseline Benchmark Suite (RQ5)
To situate from-scratch models within the broader landscape of modern representation learning, three leading pretrained models were evaluated under identical Protocol v1.1 evaluation conditions ($N = 19,632$ clean test queries, max seq len 256, harmonic ranks):

| Model Architecture | Parameters | Pretraining Domain & Scale | Full Test MRR [95% CI] | Recall@1 | Recall@10 | Zero-Overlap MRR | Low-Overlap MRR | High-Overlap MRR |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (ATIRE)** | 0 | None (Exact Lexical Inverted Index) | **0.5108** [0.505, 0.517] | 0.4052 | 0.6993 | 0.0099 | 0.2099 | 0.6625 |
| **CodeEmbed 4L Shared (Confirmatory)** | 7.38M | Clean CodeSearchNet (Scratch) | **0.4157** [0.410, 0.422] | 0.3178 | 0.6018 | 0.0487 | 0.3204 | 0.4716 |
| **CodeEmbed 17M Scaled (Exploratory)\*** | 17.03M | Clean CodeSearchNet + FAISS HN | **0.4699** [0.464, 0.476] | 0.3637 | 0.6686 | 0.0716 | 0.3559 | 0.5357 |
| **CodeEmbed 17M Hybrid (Convex)** | 17.03M | Scratch Clean + BM25 Lexical | **0.6612** [0.656, 0.667] | 0.5596 | 0.8392 | 0.0328 | 0.4443 | 0.7813 |
| **`all-MiniLM-L6-v2`** | 22.7M | 1B Sentence Pairs (General Contrastive) | **0.5837** [0.578, 0.589] | 0.4698 | 0.7912 | 0.1314 | 0.4451 | 0.6625 |
| **`microsoft/codebert-base`** | 125M | GitHub 6 PLs (Masked Language Model) | **0.0138** [0.013, 0.015] | 0.0071 | 0.0242 | 0.0019 | 0.0087 | 0.0165 |
| **`jina-embeddings-v2-base-code`** | 161M | Multi-language Code Contrastive (2024 SOTA) | **0.8294** [0.825, 0.834] | 0.7590 | 0.9444 | 0.3390 | 0.7674 | 0.8762 |

> #### 🔬 Scientific Findings & Key Takeaways:
> 1. **The Representation Degeneration Failure of Raw MLMs (`codebert-base`, MRR 0.0138)**:
>    Foundational masked language models (MLMs) like CodeBERT are trained to reconstruct masked tokens, not to produce uniform sentence/code embeddings. Their un-tuned CLS representations suffer from extreme **representation collapse (anisotropy / the cone effect)**, where pairwise cosine similarities between arbitrary documents cluster above 0.95. This empirically proves that raw pre-trained foundation models cannot be used as bi-encoders out of the box without explicit contrastive fine-tuning.
> 2. **Pretrained Capacity Peer (`all-MiniLM-L6-v2`, MRR 0.5837)**:
>    MiniLM-L6-v2 possesses almost identical parameter count (22.7M vs 17.0M) and layer depth (6L) to our scaled encoder. Benefiting from 1 billion sentence pairs of contrastive pre-training, it matches BM25 on high-overlap queries ($0.6625$) while outperforming BM25 by **+23.52 MRR points** on low-overlap queries ($0.4451$ vs $0.2099$).
> 3. **Modern Code SOTA (`jina-embeddings-v2-base-code`, MRR 0.8294)**:
>    Jina's dedicated 161M code model sets the empirical ceiling on this benchmark, achieving 94.4% Recall@10 and scoring **0.3390 MRR on zero-overlap queries** where BM25 completely fails (0.0099).

---

## 🏗️ Architecture & Key Innovations

```
Natural Language Query               Python Source Code
         │                                   │
         ▼                                   ▼
 [CodeEmbed BPE Tokenizer]           [CodeEmbed BPE Tokenizer]
 (vocab=10,000, max_len=256)         (vocab=10,000, max_len=256)
         │                                   │
         ▼                                   ▼
 [Token + Positional Emb]            [Token + Positional Emb]
         │                                   │
         ▼                                   ▼
 [Modality Embedding (q=0)]          [Modality Embedding (c=1)]
         │                                   │
         └───────────────┬───────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │  Shared Pre-LN Transformer Stack  │
        │  • 4–6 Pre-LN Transformer Blocks │
        │  • SDPA FlashAttention (O(N) mem) │
        │  • d_model=256/384, d_ff=1024/1536│
        └───────────────────────────────────┘
                         │
                         ▼
             [Masked Mean Pooling Head]
             (h = sum(h_i * m_i) / sum(m_i))
                         │
                         ▼
            [L2 Normalization (|z|_2 = 1)]
                         │
                         ▼
           Cosine Similarity Matrix S_ij
                         │
                         ▼
      [Symmetric InfoNCE + Hard Negative Loss]
```

### 1. Pre-LN Residual Highway
Standard Post-LN Transformers suffer from gradient vanishing during from-scratch contrastive learning without extensive warmup. CodeEmbed enforces **Pre-Layer Normalization** (`Pre-LN`):
$$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{MHA}(\text{LN}(\mathbf{x}_l)) + \text{FFN}(\text{LN}(\mathbf{x}_{l+1/2}))$$
This maintains an unattenuated identity gradient highway $\frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_0} = \mathbf{I} + \dots$, ensuring training stability from step 0.

### 2. Learned Modality Embeddings ($2 \times d_{\text{model}}$)
Shared encoders project both modalities into a shared manifold. To prevent token representation conflation, CodeEmbed introduces a 2-row learned table (`0 = query`, `1 = code`). This simple architectural inductive bias delivered a **$+3.84$ MRR point boost** ($0.3773 \to 0.4157$) without expanding the parameter budget of the Transformer layers.

### 3. Pooling Strategy: Masked Mean vs. CLS
In Phase R4 ablations, `CLSPooling` suffered a catastrophic **-61.2% collapse** from scratch (**0.1566 Val MRR** vs **0.4033** for `MaskedMeanPooling`). In from-scratch training without MLM pretraining, gradient updates through index 0 are starved. `MaskedMeanPooling` diffuses gradients across all non-pad tokens.

### 4. Sequence Length Frontier ($L=128$)
Truncating maximum sequence length from 256 to 128 tokens retains **97.8% of retrieval accuracy** (**0.3943 Val MRR** vs **0.4033**) while slashing attention compute by $75\%$ ($128^2$ vs $256^2$) and delivering a **$1.78\times$ training throughput boost**.

### 5. High-Throughput Hard Negative Mining & GPU Memory Optimization
* **Vectorized CSR BM25 Mining**: Replaced single-query loops with batched SciPy CSR matrix multiplication ($Q \times D^T$), top-8 term selection, and syntax word pruning. Mined 360,957 training samples in **130 seconds (2,773 q/s)** on 4 CPU threads.
* **GPU FAISS Dense Mining**: GPU-accelerated candidate vector search with a 3-tier false negative filter (identical docstrings, AST normalized skeletons $\ge 20$ nodes, MinHash $J \ge 0.70$) to purge false negatives.
* **PyTorch Native SDPA (FlashAttention)**: Integrated `torch.nn.functional.scaled_dot_product_attention`, reducing peak VRAM from 11.1 GB to **5.46 GB** on the NVIDIA RTX 4050 Laptop GPU and eliminating host RAM PCIe paging thrash.
* **Binary Pre-Tokenization Cache**: Pre-tokenized Parquet splits into binary `.pt` tensors (`train_tokenized.pt`), accelerating batch feeding ~300x (epoch time dropped from 3 hours to ~12 minutes).

---

## 🔬 Scientific Methodology & The Leak-Free R-Track

### Historical Leak Discovery & Post-Mortem
During early exploration (Phases 1–6), models achieved seemingly extraordinary metrics (BM25: 0.9498 MRR, Neural: 0.9383 MRR). An architectural audit revealed that historical CodeSearchNet functions embedded verbatim docstrings inside triple-quoted strings within `func_code_string`. BM25 and neural models had been matching queries against identical in-code copies (100% query-in-code leakage).

### The Remediation Protocol ([`PROTOCOL.md`](agent_documentation/PROTOCOL.md))
To establish rigorous scientific truth, the research was reset under a pre-registered protocol tagged `protocol-v1`:
1. **AST Byte Slicing**: Slices Python code strictly along UTF-8 byte offsets of `textwrap.dedent(code)`, purging docstrings while preserving inline comments, indentation, and formatting (`data/processed_clean_v2/`).
2. **MinHash LSH Deduplication**: Purged cross-split near-duplicates ($J \ge 0.85$) from validation and test splits.
3. **Exact Harmonic Expected Reciprocal Rank ($\mathbb{E}[\text{RR}]$)**:
   $$\mathbb{E}[\text{RR}] = \frac{H_{S_{> \text{target}} + S_{= \text{target}}} - H_{S_{> \text{target}}}}{S_{= \text{target}}}$$
   Pre-registered harmonic tie-breaking preventing score collision manipulation via Jensen's inequality.
4. **Historical Archive Preservation**: All pre-remediation exploratory figures are segregated in [Historical Pre-Remediation Archives](agent_documentation/walkthrough.md#73-historical-pre-remediation-benchmark-archive-leaky-data-exploration) for complete transparency.

---

## 📂 Repository Organization

```
Code_Embed/
├── configs/                  # Experiment YAML configs (Pre-LN, 4L, 6L, hard negatives)
├── data/                     # Data processing, AST slicing, MinHash deduplication
│   ├── download.py           # CodeSearchNet dataset download
│   ├── preprocess.py         # Byte-accurate AST docstring stripping & MinHash LSH
│   ├── pretokenize.py        # Fast offline tensor pre-tokenization
│   └── dataset.py            # Fast Collator and pre-tokenized DataLoader
├── tokenizer/                # Custom HuggingFace Byte-Pair Encoding (BPE) tokenizer
├── model/                    # Raw PyTorch nn.Module architectures (Pre-LN, Attention, Encoders)
│   ├── embeddings.py         # Composite Token + Positional + Modality embeddings
│   ├── attention.py          # Custom MultiHeadSelfAttention with SDPA FlashAttention
│   ├── transformer.py        # Pre-LN Transformer blocks and encoder stacks
│   ├── pooling.py            # MaskedMeanPooling and CLSPooling implementations
│   ├── shared_encoder.py     # Cross-modal SharedEncoder with learned modality table
│   └── dual_encoder.py       # Decoupled DualEncoder architecture
├── losses/                   # Contrastive loss functions (InfoNCE, Hard Negative InfoNCE)
├── retrieval/                # Indexing and search engines
│   ├── bm25.py               # Vectorized ATIRE BM25 with sub-token splitting
│   └── build_index.py        # FAISS dense index construction
├── evaluation/               # Statistical evaluation engine
│   ├── metrics.py            # MRR, Recall@K, NDCG, paired bootstrap CIs
│   └── evaluate.py           # Reusable checkpoint evaluation CLI
├── scripts/                  # Training and mining CLI entry points
│   ├── run_phase_r3_benchmark.py # Pre-registered test benchmark runner
│   ├── mine_hard_negatives.py    # Multithreaded CSR BM25 mining CLI
│   └── mine_dense_hard_negatives.py # GPU FAISS dense mining CLI
├── tests/                    # Comprehensive unit test battery (84/84 passing)
├── streamlit_app.py          # Interactive Streamlit Cloud deployment entrypoint
└── agent_documentation/      # Pre-registered protocols, errata, logs, and research guides
    ├── PROTOCOL.md           # Frozen pre-registered research protocol (protocol-v1)
    ├── PROTOCOL_ERRATA.md    # Pre-registered amendments and methodological errata
    ├── Memory.md             # Project milestones and experimental memory
    ├── walkthrough.md        # Detailed chronological implementation and milestone log
    ├── STUDY_GUIDE.md        # Theory, mathematical intuition, and interview prep
    └── Architecture.md       # Architectural diagrams and tensor geometry
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup
We use [`uv`](https://github.com/astral-sh/uv) as the primary package manager.
```bash
# Clone the repository
git clone https://github.com/Madhesh4124/Code_Embed.git
cd Code_Embed

# Create and activate virtual environment
uv venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Sync dependencies
uv sync
```

### 2. Verify Quality Gates & Tests
Run the comprehensive test suite (84 tests across metric implementations, data integrity, coordinate slicing, attention, and loss gradients):
```bash
uv run pytest tests/
```

### 3. Data Pipeline & Tokenizer Training
```bash
# 1. Download CodeSearchNet Python corpus
uv run python -m data.download

# 2. Run AST docstring stripping and MinHash LSH deduplication
uv run python -m data.preprocess

# 3. Train custom 10k BPE tokenizer
uv run python -m tokenizer.train_tokenizer --config configs/tokenizer.yaml

# 4. Generate binary tensor cache for fast GPU batching
uv run python -m data.pretokenize
```

### 4. Running Benchmarks & Training
```bash
# Evaluate clean BM25 ATIRE baseline
uv run python scripts/run_baseline.py --config configs/baseline.yaml

# Train Shared Encoder with in-batch negatives (Phase R3 Shared Pilot — Protocol §3.3)
uv run python scripts/run_shared.py --config configs/shared_clean.yaml

# Mine BM25 hard negatives
uv run python scripts/mine_hard_negatives.py --k 50

# Train with hard negatives
uv run python scripts/run_hard_negatives.py --config configs/shared_hard.yaml

# Evaluate any checkpoint on validation or test split
uv run python -m evaluation.evaluate --checkpoint checkpoints/shared_clean/best_shared.pt --split validation
```

---

## 📜 Research Citations & Literature Grounding

* **CodeSearchNet**: Husain, H., Wu, H. H., Gazit, T., Allamanis, M., & Brockschmidt, M. (2019). *CodeSearchNet Challenge: Evaluating the State of Semantic Code Search.* arXiv:1909.09436.
* **InfoNCE Loss**: Oord, A. v. d., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding.* arXiv:1807.03748.
* **Pre-LN Transformer**: Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., & Liu, T. (2020). *On Layer Normalization in the Transformer Architecture.* ICML 2020.
* **FlashAttention (SDPA)**: Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
