# CodeEmbed — Comprehensive Study & Revision Guide

> **Purpose**: A step-by-step companion guide explaining the *what*, *why*, and *how* behind every script, architectural decision, formula, and bug fix in this project. Use this for revision, deep understanding, and interview/portfolio preparation.

---

## Table of Contents
1. [The Big Picture: What Are We Building?](#1-the-big-picture-what-are-we-building)
2. [Phase 0.1: Environment & Tooling Choices](#2-phase-01-environment--tooling-choices)
3. [Phase 0.2: Data Download & Preprocessing Pipeline](#3-phase-02-data-download--preprocessing-pipeline)
4. [Phase 0.3: Tokenization & Byte-Pair Encoding (BPE)](#4-phase-03-tokenization--byte-pair-encoding-bpe)
5. [Phase 1: Retrieval Evaluation Framework & BM25 Baseline](#5-phase-1-retrieval-evaluation-framework--bm25-baseline)
6. [Key Bugs Encountered & Lessons Learned](#6-key-bugs-encountered--lessons-learned)
7. [Roadmap: What's Next?](#7-roadmap-whats-next)

---

## 1. The Big Picture: What Are We Building?

### The Core Problem: Semantic Code Search
When developers search for code, lexical search (keyword matching like `Ctrl+F` or classic search engines) often fails because natural language descriptions and code use completely different vocabularies:
* **User Query**: `"find euclidean distance between two vectors"`
* **Code Implementation**:
  ```python
  def l2_norm(p1, p2):
      return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
  ```
Notice that the words `"euclidean"`, `"distance"`, and `"vectors"` do not appear anywhere in the function body. A keyword search scores this as a zero match.

### The Solution: Contrastive Dual-Space Embedding
We train a Transformer encoder to map:
* A natural language docstring $\rightarrow$ a 256-dimensional vector $\mathbf{z}_{\text{text}} \in \mathbb{R}^{256}$
* A Python function $\rightarrow$ a 256-dimensional vector $\mathbf{z}_{\text{code}} \in \mathbb{R}^{256}$

Both vectors are normalized to the unit hypersphere ($\|\mathbf{z}\|_2 = 1$). During training via **InfoNCE contrastive loss**, we pull matching pairs together (maximizing cosine similarity $\mathbf{z}_{\text{text}}^\top \mathbf{z}_{\text{code}}$) while pushing non-matching pairs apart.

```
 Natural Language: "calculate l2 distance" ──────► [Text Encoder] ────► z_text (256-d)
                                                                             │
                                                                   Maximize Cosine Sim
                                                                             │
 Code Snippet:     "def l2_norm(p1, p2): ..." ───► [Code Encoder] ────► z_code (256-d)
```

At retrieval time, all codebase functions are pre-encoded into a **FAISS** vector index. When a user enters a query, we encode only the query and perform a fast nearest-neighbor search ($<50\text{ms}$).

---

## 2. Phase 0.1: Environment & Tooling Choices

### Why `uv` Instead of Traditional `pip`?
* Traditional `pip` resolves dependencies sequentially in Python and can take minutes.
* `uv` is written in Rust, resolving and installing wheels in milliseconds using global caching and hard links.
* We manage dependencies inside a project-isolated `.venv`.

### Why Explicit CUDA Wheels on Windows?
* Standard PyPI distributions of `torch` on Windows may default to CPU-only if NVIDIA CUDA runtimes are not bundled.
* Specifying `--index-url https://download.pytorch.org/whl/cu124` downloads the complete PyTorch binaries compiled with CUDA 12.4 kernels, enabling hardware acceleration on your GPU.

### Why `.gitignore` Is Vital for ML
* Parquet files, raw datasets, model weights (`.pt`), and MLflow experiment runs take gigabytes of disk space.
* Git is designed for text diffs, not large binaries. Committing `.pt` files would permanently bloat git history.

---

## 3. Phase 0.2: Data Download & Preprocessing Pipeline

### 3.1 `data/download.py` — Downloading the Raw Corpus
* **Dataset**: CodeSearchNet (Python subset).
* **Why Save to Disk as Parquet?**
  * `Parquet` is a columnar storage format with Snappy compression.
  * Reading raw Parquet into Pandas is ~5–10x faster than parsing JSON lines or CSV.
  * It decouples downloading from experimentation: if we want to change preprocessing rules later, we never have to re-download 1 GB over the network.

### 3.2 `data/preprocess.py` — Cleaning the Data
Real-world code scraped from GitHub is full of noise that harms representation learning:
1. **Empty / Missing Content**: Functions without docstrings or empty bodies provide no contrastive supervision signal.
2. **Length Filtering (10 to 2048 words/tokens)**:
   * Tiny snippets (e.g., `pass`, `return True`) lack semantic meaning.
   * Massive functions (> 2048 tokens) cause GPU Out-of-Memory (OOM) errors during self-attention, where memory scales quadratically: $\mathcal{O}(L^2)$.
3. **Meaningless Docstrings**: Filtering out placeholders like `"TODO"`, `"None"`, or single-word descriptions that don't explain functionality.
4. **Boilerplate & Autogenerated Files**: Dropping `__init__.py` files and `/migrations/` which contain repetitive import statements rather than logical code.
5. **Exact Deduplication**: GitHub contains thousands of identical forked functions (e.g. copied utilities). If an identical function exists in both train and test sets, the model can simply memorize it (**data leakage**). Deduplication prevents this.

#### Preprocessing Results:
* **Train**: 412,178 $\rightarrow$ **385,381** clean functions (~26.8k noisy samples dropped).
* **Validation**: 23,107 $\rightarrow$ **21,585** clean functions.
* **Test**: 22,176 $\rightarrow$ **21,005** clean functions.
* Disk footprint reduced from **~950 MB** down to **~223 MB**!

---

## 4. Phase 0.3: Tokenization & Byte-Pair Encoding (BPE)

### 4.1 Why Not Character or Word-Level Tokenization?
* **Word-level**: Python has infinite compound variable names (`get_user_account_by_id`). A word-level vocabulary would require millions of entries, exploding embedding table size ($V \times D$).
* **Character-level**: Sequences would become thousands of tokens long, exceeding our sequence length limit ($L=256$) and making self-attention extremely slow.
* **Byte-Pair Encoding (BPE)**: The optimal middle ground.
  * Starts at the byte/character level (meaning it **never** encounters an unseen word—it can always fall back to individual characters).
  * Iteratively merges the most frequent adjacent character pairs into subwords until reaching our target vocabulary size (**16,000 subwords**).
  * Example: `calculate_loss_matrix` $\rightarrow$ `['calculate', '_loss', '_matrix']`.

### 4.2 Why Train on BOTH Code and Docstrings?
Our goal is to map code and docstrings into the **same shared embedding space**. If we trained two separate tokenizers:
* The word `"vector"` in a docstring might get token ID `512`.
* The word `"vector"` in code might get token ID `1048`.
By training a single unified BPE vocabulary over both modalities, shared programming and domain terms share the exact same embedding row from day one!

### 4.3 Special Tokens
We reserve IDs 0 through 5 for structural control tokens:
* `<PAD>` (ID 0): Fills shorter sequences up to the batch length. The attention mask sets attention weights to $-\infty$ for `<PAD>` tokens so the model ignores them.
* `<UNK>` (ID 1): Fallback for any character byte sequence not represented.
* `<BOS>` (ID 2): Beginning of Sequence marker.
* `<EOS>` (ID 3): End of Sequence marker.
* `<CODE>` (ID 4): Modality token prepended when encoding Python code.
* `<TEXT>` (ID 5): Modality token prepended when encoding natural language queries.

### 4.4 The Tokenizer Wrapper (`tokenizer/tokenizer.py`) & PyTorch Tensor Mechanics
The raw Hugging Face `tokenizers` library returns plain Python lists and metadata objects (`Encoding`). But deep learning models implemented in PyTorch require structured multi-dimensional tensors. `tokenizer/tokenizer.py` bridges this gap.

#### The Two Essential Tensors:
1. **`input_ids`**: Shape $(B, L)$ where $B$ is batch size and $L$ is sequence length.
   * A 2D matrix of 64-bit integers (`torch.long`).
   * Each integer is an index into the model's token embedding lookup matrix $\mathbf{E} \in \mathbb{R}^{V \times D}$ (where $V=16000$ and $D=256$).
   * Example row: `[356, 3661, 68, 4794, 0, 0, 0, ...]`
2. **`attention_mask`**: Shape $(B, L)$
   * A binary mask containing `1` for real subword tokens and `0` for `<PAD>` tokens.
   * **Why is this necessary?** In the Transformer's Multi-Head Self-Attention layer:
     $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$
     Where the mask $\mathbf{M}_{i, j} = -\infty$ whenever position $j$ has an attention mask of `0`. Because $e^{-\infty} = 0$, the softmax allocates **zero** attention weight to padding tokens, ensuring pad tokens never contaminate real representations.

#### Truncation & Padding Strategy:
* **Truncation (`max_length=256`)**: Python functions can theoretically be thousands of lines long. Self-attention complexity scales quadratically with length $\mathcal{O}(L^2)$. Capping sequences at $L=256$ keeps GPU memory predictable and fast while retaining ~95% of all function code.
* **Padding (`pad_id=0`)**: PyTorch batches must be rectangular tensors. If sequence 1 has 50 tokens and sequence 2 has 120 tokens, sequence 1 is padded with 70 `<PAD>` tokens so both can be processed in parallel on the GPU in a single $(2, 120)$ tensor.

#### Modality Token Prepending (`<CODE>` vs. `<TEXT>`):
In multi-modal code retrieval, prepending `<CODE>` or `<TEXT>` informs the Transformer whether it is encoding a Python function or an English query. Because `<CODE>` and `<TEXT>` were registered as added special tokens during BPE training, prepending `"<CODE> "` or `"<TEXT> "` directly produces token ID `4` or `5` as the first token without subword fragmentation.

#### 1D vs. 2D Tensor Decoding:
* **1D Tensor `(L,)`**: Represents a single encoded function or query. `token_ids.tolist()` produces `List[int]`, so `token_ids[0]` is an `int`. We call `tokenizer.decode(...)` to return a single `str`.
* **2D Tensor `(B, L)`**: Represents a batch of sequences. `token_ids.tolist()` produces `List[List[int]]`, so `token_ids[0]` is a `list`. We call `tokenizer.decode_batch(...)` to return a `List[str]`.

### 4.5 Token Length Distribution & Sequence Length Selection
A critical hyperparameter in any Transformer model is the maximum sequence length $L$. 

#### The Quadratic Memory Penalty of Attention:
The self-attention mechanism computes an attention score between every token and every other token:
$$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times H \times L \times L}$$
Because memory scales as $\mathcal{O}(L^2)$:
* $L = 512$ uses **$4\times$ more memory** and compute than $L = 256$.
* $L = 1024$ uses **$16\times$ more memory** than $L = 256$.

#### Why We Measure Percentiles ($P_{50}, P_{75}, P_{90}, P_{95}, P_{99}$):
* **$P_{50}$ (Median)**: Half of the code samples in the dataset are shorter than this length.
* **$P_{95}$ (95th Percentile)**: 95% of all samples are shorter than this length.

#### Empirical Findings on CodeSearchNet Python (50,000 Sample EDA):
| Metric | Docstrings (Text) | Code (Python) |
|---|---|---|
| **Mean** | 69.2 tokens | 267.8 tokens |
| **P50 (Median)** | 34 tokens | 169 tokens |
| **P75** | 78 tokens | 304 tokens |
| **P90** | 156 tokens | 539 tokens |
| **P95** | 240 tokens | 778 tokens |
| **P99** | 559 tokens | 1,672 tokens |

* **Docstring Coverage at $L=256$**: **$>95\%$** of all docstrings fit completely without any truncation.
* **Code Coverage at $L=256$**: **$68.7\%$** of functions fit completely.

#### The Architectural Decision: $L=256$ vs. $L=512$
Why not increase to $L=512$ to reach $89\%$ code coverage?
1. **Quadratic Memory Penalty**: Attention memory quadruples ($4\times$).
2. **In-Batch Negatives in InfoNCE**: Contrastive learning calculates similarity against $B-1$ in-batch negatives. Pushing to $L=512$ forces shrinking batch size from $B=256$ down to $B=64$, giving the model $4\times$ fewer negative contrastive examples per gradient step.
3. Therefore, $L=256$ provides the optimal balance of rich semantic context and high contrastive batch diversity.

### 4.6 PyTorch Dataset, DataLoaders, and Dynamic Padding
In deep learning training pipelines, the data loader must feed batches of tensors `(B, L)` efficiently without stalling the GPU.

#### Why Do We Need `data/dataset.py` When We Already Have `tokenizer.py`?
* **`tokenizer.py` is the Tool**: A pure transformation function (`text -> token_ids`). It does not know about disks, Parquet files, epochs, shuffling, or GPU memory transfers.
* **`data/dataset.py` is the Assembly Line**: It connects the 200 MB Parquet file on disk to the training loop. It coordinates:
  1. **Random Access (`Dataset.__getitem__`)**: Grabbing sample `#idx` on demand.
  2. **Preventing GPU Starvation**: Using background worker threads (`num_workers`) and pinned memory (`pin_memory=True`) so while the GPU trains on batch $N$, CPU threads are already reading batch $N+1$.
  3. **Epoch Shuffling**: Shuffling sample pairings every epoch so the contrastive model sees diverse negative pairs.

#### Static Padding vs. Dynamic Padding:
* **Static Padding**: Every sequence in the entire dataset is padded to the global `max_length = 256`.
  * *Problem*: If a batch contains functions where the longest is only 80 tokens, 176 tokens per sequence ($68\%$) are wasted `<PAD>` tokens! The GPU spends more time doing matrix multiplications on zeros than learning real code.
* **Dynamic Padding (in Collator)**:
  * The dataset returns raw strings.
  * The `Collator` inspects each batch and pads only to the **longest sequence in that specific batch** (capped at `max_length = 256`).
  * If a batch's longest function is 95 tokens, the batch tensor shape is $(B, 95)$ instead of $(B, 256)$, cutting GPU memory and forward/backward time by more than half!

#### The PyTorch Architecture:
```
data/processed/train.parquet
         │
         ▼
  [CodeSearchDataset] ──► Returns raw sample dict: {"code": str, "docstring": str}
         │
         ▼
 [CodeSearchCollator] ──► Dynamically pads batch to max_len_in_batch using Tokenizer
         │
         ▼
      Batches: {
          "code_ids":  (B, L_code),
          "code_mask": (B, L_code),
          "text_ids":  (B, L_text),
          "text_mask": (B, L_text),
      }
```

---

## 5. Phase 1: Retrieval Evaluation Framework & BM25 Baseline

Before designing and training neural Transformer encoders, we must establish **how we measure success** and **what baseline we must beat**.

---

### 5.1 Why Build the Evaluation Framework Before the Model?
In machine learning research, a common anti-pattern is writing ad-hoc evaluation logic inside the training loop of each model. This causes **metric drift**:
* Slight differences in tie-breaking, rank cutoffs, or indexing between experiments ruin reproducibility.
* You cannot reliably tell whether an architecture change improved performance or if the evaluation script computed metrics slightly differently.

By building an **immutable, standardized evaluation suite** (`evaluation/metrics.py` and `evaluation/evaluate.py`) first, every subsequent model—BM25, Basic Transformer, Shared Encoder, Dual Encoder, and Pretrained models—is evaluated against the exact same yardstick.

---

### 5.2 Mathematical Foundations of Code Retrieval Metrics

In semantic code search, a test set consists of $|Q|$ queries (docstrings). For each query $q_i$, the retrieval system scores all $N$ candidate functions in the corpus and outputs a ranked list of candidate IDs.

Because each test docstring in CodeSearchNet corresponds to exactly one ground-truth function, the ground truth is located at a single 1-based integer position: $\text{rank}_i \in \{1, 2, \dots, N, \infty\}$.

```
Query: "calculate factorial recursively"
Ranked Results:
  [Rank 1] def fibonacci(n): ...           (Incorrect)
  [Rank 2] def factorial(n): ...           (Correct! rank_i = 2)
  [Rank 3] def permutation(n, r): ...      (Incorrect)
```

#### 1. Mean Reciprocal Rank (MRR)
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
* If the true snippet is ranked #1: $\frac{1}{1} = 1.000$
* If ranked #2: $\frac{1}{2} = 0.500$
* If ranked #5: $\frac{1}{5} = 0.200$
* If ranked #10: $\frac{1}{10} = 0.100$
* If not found within the evaluated top-$K$ candidates: $\frac{1}{\infty} = 0.000$

**Cognitive Intuition**: In developer workflows, search attention follows a steep inverse decay curve. A developer will happily click the 1st result, might check the 2nd or 3rd, but rarely scrolls to result 15. MRR mathematically reflects this user impatience by heavily penalizing lower rankings.

#### 2. Recall@K ($K \in \{1, 5, 10\}$)
$$\text{Recall@}K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(\text{rank}_i \le K)$$
Where $\mathbb{I}(\cdot)$ is the indicator function ($1$ if true, $0$ if false).
* **Recall@1**: Exact top-1 retrieval accuracy. Did the model get it right on the very first try?
* **Recall@5**: Is the correct answer visible on the screen without scrolling?
* **Recall@10**: Candidate generation quality. Did the retrieval stage place the true function within the top-10 shortlist for a downstream reranker?

#### 3. Normalized Discounted Cumulative Gain (NDCG@K)
In general information retrieval with multi-graded relevance $r_j \in [0, R]$:
$$\text{DCG@}K = \sum_{j=1}^K \frac{2^{r_j} - 1}{\log_2(j + 1)}, \quad \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$
Where $\text{IDCG@}K$ is the Ideal DCG obtained by sorting all relevant items at the top.

In our single-relevant-item code retrieval setting ($r_j \in \{0, 1\}$):
* If the relevant item is at $\text{rank}_i \le K$, only the term at $j = \text{rank}_i$ has $r_j = 1$. Its gain is $2^1 - 1 = 1$:
  $$\text{DCG@}K = \frac{1}{\log_2(\text{rank}_i + 1)}$$
* In the ideal ranking, the target is at rank 1:
  $$\text{IDCG@}K = \frac{2^1 - 1}{\log_2(1 + 1)} = \frac{1}{\log_2(2)} = 1.0$$
* Therefore, the normalized score collapses cleanly to:
  $$\text{NDCG@}K = \begin{cases} \frac{1}{\log_2(\text{rank}_i + 1)} & \text{if } \text{rank}_i \le K \\ 0 & \text{if } \text{rank}_i > K \end{cases}$$

**Why Logarithmic Discounting?**
Human perception of rank position follows the Weber-Fechner law (logarithmic sensitivity). Slipping from rank 1 to rank 2 is a massive perceived quality loss ($\text{NDCG}: 1.0 \rightarrow 0.631$), whereas slipping from rank 9 to rank 10 is barely perceptible ($\text{NDCG}: 0.301 \rightarrow 0.289$).

---

### 5.3 Non-Parametric Bootstrap Confidence Intervals (1,000 Resamples)

When comparing two models:
* Model A achieves $\text{MRR} = 0.642$
* Model B achieves $\text{MRR} = 0.655$
Is Model B genuinely superior, or did it just get lucky on a few easy queries in the test set?

#### The Bootstrap Procedure:
1. Let the test set rankings be an array $\mathbf{R} = [\text{rank}_1, \text{rank}_2, \dots, \text{rank}_{|Q|}]$.
2. Draw a sample $\mathbf{R}^*$ of size $|Q|$ from $\mathbf{R}$ **with replacement** (some queries will appear multiple times, others will be omitted).
3. Compute the metric $M_b = \text{metric}(\mathbf{R}^*)$.
4. Repeat steps 2–3 for $B = 1000$ iterations to obtain the empirical sampling distribution $\{M_1, M_2, \dots, M_{1000}\}$.
5. The 95% confidence interval is the 2.5th and 97.5th percentiles:
   $$[c_{\text{low}}, c_{\text{high}}] = \left[\text{Percentile}(M, 2.5), \; \text{Percentile}(M, 97.5)\right]$$

If the confidence intervals of two architectures do not overlap, we have statistical proof ($p < 0.05$) that the improvement is genuine.

---

### 5.4 The BM25 Lexical Baseline: Mathematics & Mechanics

**BM25 (Best Matching 25)** is the industry-standard probabilistic ranking function developed by Stephen Robertson and Karen Spärck Jones. It establishes our **lexical floor**.

Given a query $Q$ with tokens $q_1, \dots, q_m$ and a candidate code document $D$:
$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{i=1}^m \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                      BM25 EQUATION                      │
                  └─────────────────────────────────────────────────────────┘
        IDF Weight                         Term Frequency Saturation
    ┌────────────────┐                ┌───────────────────────────────────┐
    │                │                │                                   │
    │                ▼                │                 ▼                 │
    │  ln( (N - n + 0.5)/(n + 0.5) )  │    f(q, D) * (k1 + 1)             │
    │                                 │  ───────────────────────────────  │
    │                                 │  f(q, D) + k1 * (1 - b + b*|D|/L) │
    └─────────────────────────────────┴───────────────────────────────────┘
                                                       ▲
                                                       │
                                        Document Length Normalization
```

#### Why Each Component Exists:

1. **Inverse Document Frequency (IDF)**:
   $$\text{IDF}(q_i) = \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)$$
   * $N$: Total number of functions in the codebase (e.g. 21,005 test functions).
   * $n(q_i)$: Number of functions containing token $q_i$.
   * **Intuition**: Python syntax words like `def`, `return`, and `self` appear in almost every function ($n \approx N$), so their $\text{IDF} \approx 0$. Rare semantic words like `quicksort` or `eigenvalue` appear in very few functions, giving them massive IDF weights.

2. **Term Frequency Saturation ($k_1 \in [1.2, 2.0]$)**:
   * In raw TF (Term Frequency), a document mentioning `"matrix"` 20 times scores $20\times$ higher than a document mentioning it once.
   * In reality, the second occurrence is informative, but the 20th occurrence offers diminishing returns.
   * The term $\frac{f \cdot (k_1 + 1)}{f + k_1}$ asymptotically caps the score at $k_1 + 1$, preventing repetitive functions from dominating search results.

3. **Document Length Normalization ($b = 0.75$)**:
   * A 500-line utility module naturally contains more words than a 10-line helper function, purely by virtue of length.
   * The ratio $\frac{|D|}{\text{avgdl}}$ compares document length to the average function length across the dataset.
   * If $|D| > \text{avgdl}$, the denominator increases, penalizing the score.
   * $b = 1$ enforces full length penalty; $b = 0$ ignores document length completely. Standard practice is $b = 0.75$.

---

### 5.5 Code-Specific Lexical Tokenization Challenges

Applying standard English NLP tokenizers (e.g. splitting on whitespace and punctuation) directly to source code produces poor lexical retrieval:
* In Python code: `def compute_matrix_inverse(input_matrix):`
* Query: `"matrix inverse"`

If we only split on whitespace:
* Code tokens: `['def', 'compute_matrix_inverse(input_matrix):']`
* The word `'matrix'` or `'inverse'` never appears as an isolated token! BM25 would score this match as **zero**.

#### The Sub-identifier Splitting Solution:
Our code tokenizer must split compound programming identifiers before BM25 indexing:
1. **`snake_case` splitting**: `compute_matrix_inverse` $\rightarrow$ `['compute', 'matrix', 'inverse']`
2. **`camelCase` splitting**: `calculateL2Distance` $\rightarrow$ `['calculate', 'l2', 'distance']`
3. **Symbol removal**: Strip punctuation `(`, `)`, `:`, `{`, `}`, `[`, `]`.

With sub-token splitting, BM25 achieves a much stronger, realistic lexical baseline.

---

### 5.6 High-Performance Inverted Index vs. Naive Python BM25

In popular Python libraries like `rank_bm25`, the default `.get_scores(query)` implementation scans the corpus sequentially:
```python
# Naive rank_bm25 approach: O(Q * |query| * N) Python loop
for q in query:
    q_freq = np.array([(doc.get(q, 0)) for doc in self.doc_freqs])
    # array allocations and loops for every single token...
```
When evaluating 1,000 queries against 21,005 documents:
* Average query length: ~10 tokens.
* Total Python loop iterations: $1,000 \times 10 \times 21,005 \approx 210,000,000$ operations!
* **Benchmark runtime**: **592.66 seconds (~9.8 minutes)** at only 1.7 queries/sec!

#### The Vectorized Inverted Index Optimization:
Instead of iterating through all $N = 21,005$ documents for every word:
1. **Precompute Length Normalization**:
   $$\text{len\_norm}[d] = k_1 \cdot \left(1 - b + b \cdot \frac{|D_d|}{\text{avgdl}}\right) \in \mathbb{R}^N$$
2. **Inverted Postings List**:
   Store `inverted_index[term] = (doc_indices, term_freqs)` as contiguous NumPy arrays (`int32` and `float32`).
3. **Sparse Vector Accumulation**:
   For each token in the query, we only access documents that actually contain the term:
   ```python
   scores[doc_ids] += idf_val * (freqs * (k1 + 1.0)) / (freqs + len_norm[doc_ids])
   ```
* **Optimized runtime**: **3.74 seconds** at **267.6 queries/sec**!
* **Speedup**: **154x faster**, producing mathematically identical BM25Okapi scores!

---

### 5.7 Phase 1 Benchmark Results & Analysis

Evaluating on 1,000 representative test queries against the entire 21,005 code corpus:

| Metric | Score | 95% Confidence Interval (1,000 resamples) |
| :--- | :---: | :---: |
| **MRR** | **0.9498** | [0.9389, 0.9597] |
| **Recall@1** | **0.9180** | [0.9020, 0.9340] |
| **Recall@5** | **0.9890** | [0.9820, 0.9950] |
| **Recall@10** | **0.9950** | [0.9900, 0.9990] |
| **NDCG@10** | **0.9610** | [0.9521, 0.9691] |

#### Why Is Lexical BM25 So Strong on CodeSearchNet?
1. **Direct Identifier Overlap**: In CodeSearchNet Python, functions like `def compute_similarity(a, b):` are paired with docstrings like `"Compute similarity between two inputs."` The sub-token splitter matches `"compute"` and `"similarity"` with high IDF weights.
2. **Single-Target Dataset Structure**: With 1:1 query-to-function pairings, exact token matches often uniquely pinpoint the function.
3. **Where BM25 Fails (The Motivation for Neural Encoders)**:
   * **Synonymy / Vocabulary Mismatch**: Query `"find shortest path in graph"` vs code `def dijkstra(...)`. Zero token overlap $\rightarrow$ BM25 rank $= \infty$.
   * **Semantic Intent**: Query `"read data from remote host"` vs code `requests.get(url)`.
   * Dense neural dual encoders bridge this conceptual vocabulary gap.

---

## 6. Key Bugs Encountered & Lessons Learned

| Bug / Error | Root Cause | Solution | Concept Learned |
|---|---|---|---|
| `HfUriError: Repository id must be namespace/name` | Modern HuggingFace API requires organization namespaces for legacy dataset names. | Changed `"code_search_net"` to `"code-search-net/code_search_net"`. | HuggingFace Hub repository naming convention. |
| `trust_remote_code is not supported anymore` | Dataset was converted to standard Parquet, deprecating custom loading scripts. | Removed `trust_remote_code=True` parameter. | Modern datasets use native Parquet without arbitrary code execution. |
| `ValueError: The truth value of a Series is ambiguous` | Using Python's logical `and` between two Pandas Series objects. | Use bitwise element-wise `&` operator (or rely on `.apply()` functions). | Pandas Series vectorization vs. Python scalar boolean logic. |
| Batch slice `df.iloc[i+1 : i+batch_size]` | `iloc[start:stop]` start is already inclusive. Using `i+1` skipped row 0 and the first row of each chunk. | Changed to `df.iloc[i : i + batch_size]`. | Python 0-indexed slicing semantics. |
| Round-trip mismatch on token decoding (`Ġdef`) | `ByteLevel(add_prefix_space=True)` prepends an artificial space to the first token. | Set `add_prefix_space=False` and attach `decoders.ByteLevel()`. | In Python, spaces represent indentation syntax. Adding accidental leading spaces corrupts code formatting. |
| `ModuleNotFoundError: No module named 'tokenziers'` | Typo in import statement (`tokenziers` vs `tokenizers`). | Corrected spelling to `tokenizers`. | Package naming accuracy. |
| `ModuleNotFoundError: No module named 'tokenizer'` / `'evaluation'` | Direct script execution puts `scripts/` in `sys.path[0]`, omitting repository root. | Add `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))`. | Python import resolution (`sys.path[0]`) and package execution (`-m`). |
| `UnicodeEncodeError: 'charmap' codec can't encode '\u2713'` | Windows terminal default code page (cp1252) does not support Unicode checkmarks. | Call `sys.stdout.reconfigure(encoding="utf-8")` and use ASCII fallbacks (`[OK]`). | Windows console stream encoding constraints. |
| Sub-token regex fragmentation (`l2` $\rightarrow$ `['l', '2']`) | Splitting pattern treated letters and digits as disjoint tokens. | Updated regex to `[A-Z]?[a-z0-9]+` to preserve alphanumeric units like `l2`, `dim256`. | Regex design for mixed alphanumeric programming identifiers. |
| `MlflowException: The filesystem tracking backend is in maintenance mode` | Local `./mlruns` directory triggers deprecation error in modern MLflow versions. | Configured `sqlite:///mlflow.db` backend with `MLFLOW_ALLOW_FILE_STORE=true`. | Enterprise MLflow backend configuration and SQLite relational persistence. |
| `MlflowException: Invalid value 'recall@1'` | MLflow parameter and metric names prohibit `@` characters. | Replaced `@` with `_at_` (e.g. `recall_at_1`, `ndcg_at_10`) for MLflow logging. | Experiment tracker schema validation rules. |
| `TypeError: create_dataloader() got unexpected keyword argument 'parquet_path'` | Calling `create_dataloader(parquet_path=...)` instead of `create_dataloader(split=...)`. | Passed `split="train"` and `split="validation"`. | Function signature compliance across dataset helpers. |

---

## 7. Roadmap: What's Next?

1. **Phase 1: BM25 Baseline & Evaluation Framework (🟢 COMPLETED)**:
   * Verified `evaluation/metrics.py` (MRR, Recall@K, NDCG@K, bootstrap CIs) and `tests/test_metrics.py`.
   * Built high-performance vectorized `retrieval/bm25.py` and `tests/test_bm25.py` (28/28 unit tests passing).
   * Executed baseline benchmark `scripts/run_baseline.py` on test set; logged run to MLflow (`mlruns` & `sqlite:///mlflow.db`).
   * Established empirical lexical baseline: **$\text{MRR} = 0.9498$**, **$\text{Recall@1} = 0.9180$**, **$\text{Recall@10} = 0.9950$**.
2. **Phase 2: Model 1 — Basic Encoder (🟢 COMPLETED)**:
   * Implemented Token, Positional, and Modality embeddings (`model/embeddings.py`).
   * Implemented custom Multi-Head Self-Attention from raw PyTorch primitives (`model/attention.py`).
   * Implemented Pre-LN Transformer blocks and 4-layer encoder stack (`model/transformer.py`).
   * Implemented MaskedMeanPooling and L2 projection head (`model/pooling.py`, `model/encoder.py`).
   * Implemented symmetric InfoNCE loss with in-batch negatives (`losses/contrastive.py`).
   * Implemented `ContrastiveTrainer` with CUDA AMP mixed precision, cosine warmup scheduler, and checkpointing (`training/trainer.py`).
   * Validated with 45/45 passing unit tests across `test_model.py`, `test_loss.py`, `test_bm25.py`, and `test_metrics.py`.
   * Verified end-to-end training and checkpoint evaluation CLI (`evaluation/evaluate.py`).
3. **Phase 3: Model 2 — Shared Encoder (🟢 COMPLETED)**:
   * Implemented `SharedEncoder` with learned modality embeddings (`model/shared_encoder.py`).
   * Routed `<CODE>` (0) and `<TEXT>` (1) inputs, trained on RTX 4050 GPU (Epoch 1 val MRR 0.9473, test MRR 0.9296).
4. **Phase 4: Model 3 — Separate (Dual) Encoders (🟡 NEXT UP)**:
   * Implement `DualEncoder` with separate code and text Transformer encoders (`model/dual_encoder.py`).
   * Parameter budget matching: 3 layers each (~4M each, ~8M total).

---

## 8. Phase 2 Deep Dive: Building Transformers from Scratch

### 8.1 Pre-LN vs. Post-LN: The Stability Breakthrough

The original Transformer (Vaswani et al., 2017) placed Layer Normalization *after* the residual addition:
$$\mathbf{x}_{l+1} = \text{LayerNorm}(\mathbf{x}_l + \text{SubLayer}(\mathbf{x}_l)) \quad (\text{Post-LN})$$

* **The Post-LN Flaw**:
  During backpropagation, gradients passing through the residual stream are continuously scaled by the derivative of LayerNorm:
  $$\frac{\partial \mathbf{x}_{l+1}}{\partial \mathbf{x}_l} \propto \frac{1}{\sqrt{\text{Var}(\mathbf{x}_l) + \epsilon}}$$
  As depth increases, gradients explode or vanish near the input layers. To train Post-LN models, engineers had to use delicate learning rate warmups and low learning rates.

* **The Pre-LN Solution (Our Implementation)**:
  We place Layer Normalization *before* each sub-layer, inside the residual branch:
  $$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{Dropout}(\text{SubLayer}(\text{LayerNorm}(\mathbf{x}_l))) \quad (\text{Pre-LN})$$
  Because $\mathbf{x}_{l+1} = \mathbf{x}_l + \dots$, there is an uninterrupted identity path from the final layer directly to the input embeddings:
  $$\frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_0} = \mathbf{I} + \sum_{l=0}^{L-1} \frac{\partial \text{SubLayer}_l}{\partial \mathbf{x}_l}$$
  Gradients flow cleanly at initialization, enabling immediate stable training at $\text{lr} = 3 \times 10^{-4}$ without NaN instabilities.

---

### 8.2 Attention Masking Mechanics in Float16

In scaled dot-product attention:
$$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)$$

Where $\mathbf{M}_{i, j} \in \{0, -\infty\}$.
* In standard 32-bit float arithmetic, setting $\mathbf{M}_{i, j} = -1 \times 10^9$ is standard.
* **The FP16 Overflow Trap**: In IEEE 754 half-precision float (`torch.float16`), the minimum finite representable number is $-65504.0$. Any value less than $-65504$ (such as $-10^9$) becomes `-inf`. When combined with other operations or subtractions during softmax normalization, `-inf - (-inf)` produces **`NaN`**!
* **Our Solution**: In `model/attention.py`, we dynamically set the mask fill value:
  ```python
  fill_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
  ```
  Since $e^{-10000} \approx 0.0$, padded tokens receive strictly 0 probability while preserving numerical stability in mixed precision.

---

### 8.3 Masked Mean Pooling vs. CLS Pooling

Transformer token outputs have shape $(B, L, D)$. To compare a docstring against a code snippet using cosine similarity, we must condense $L$ vectors into a single vector $\mathbf{h} \in \mathbb{R}^D$:

1. **Why Plain `mean(dim=1)` Fails**:
   If a sequence has 40 real tokens and 216 `<PAD>` tokens, plain mean divides the token sum by 256. $84\%$ of the divisor is empty padding, severely attenuating the vector magnitude and corrupting the semantic representation.
2. **Masked Mean Pooling**:
   $$\mathbf{h} = \frac{\sum_{i=1}^L \mathbf{m}_i \cdot \mathbf{x}_i}{\sum_{i=1}^L \mathbf{m}_i}$$
   Pad tokens contribute $0.0$ to the numerator and are excluded from the denominator.
3. **L2 Unit Normalization**:
   $$\mathbf{z} = \frac{\mathbf{h}}{\|\mathbf{h}\|_2} \implies \|\mathbf{z}\|_2 = 1.0$$
   Normalizing onto the unit hypersphere allows cosine similarity to be computed as a simple dot product: $\cos(\mathbf{z}_1, \mathbf{z}_2) = \mathbf{z}_1^\top \mathbf{z}_2$.

---

### 8.4 InfoNCE Contrastive Objective

Given a batch of $B = 128$ normalized query vectors $\mathbf{Z}_{\text{text}} \in \mathbb{R}^{B \times D}$ and code vectors $\mathbf{Z}_{\text{code}} \in \mathbb{R}^{B \times D}$:

1. **Similarity Matrix**:
   $$\mathbf{S} = \frac{\mathbf{Z}_{\text{text}} \mathbf{Z}_{\text{code}}^\top}{\tau} \in \mathbb{R}^{B \times B}$$
   Where $\tau = 0.07$ is the temperature hyperparameter.
2. **The Role of Temperature $\tau$**:
   * Small $\tau$ (e.g. 0.07) acts as a hardness amplifier. It scales dot products from $[-1, 1]$ to $[-14.3, +14.3]$, making the softmax distribution peaky and forcing the model to heavily penalize hard in-batch negatives.
   * Large $\tau$ (e.g. 1.0) creates an overly diffuse softmax, reducing the penalty for ranking negatives near the positive.
3. **Symmetric Loss**:
   $$\mathcal{L} = \frac{1}{2} \left[ \mathcal{L}_{\text{text}\rightarrow\text{code}} + \mathcal{L}_{\text{code}\rightarrow\text{text}} \right]$$
   Ensuring bidirectional alignment so the model excels at both text-to-code search and code-to-text matching.

---

## 9. Phase 3 Deep Dive: Modality Embeddings in Shared Transformers

### 9.1 The Modality Gap in Shared Encoders

When a single Transformer weights matrix processes both English natural language and Python code syntax, it encounters the **modality gap**:
* Docstrings are natural language sentences containing grammar, English prose, punctuation, and abstract intent.
* Code snippets are AST structures with indentation, variable bindings, control flow keywords (`def`, `return`, `for`), and type signatures.

If the embedding layer only provides token and positional embeddings:
$$\mathbf{x}_i = \text{TokenEmbed}(t_i) + \text{PosEmbed}(i)$$
The Transformer has to deduce the input domain solely from the subword vocabulary. In contrast, by introducing a learned **Modality Embedding**:
$$\mathbf{x}_i = \text{TokenEmbed}(t_i) + \text{PosEmbed}(i) + \text{ModalityEmbed}(m), \quad m \in \{0, 1\}$$

Where:
* $m=0 \implies \mathbf{e}_{\text{code}} \in \mathbb{R}^D$ (Code modality)
* $m=1 \implies \mathbf{e}_{\text{text}} \in \mathbb{R}^D$ (Natural language query modality)

### 9.2 Mathematical Benefit of Modality Embeddings
1. **Geometric Separation & Orthogonal Shift**:
   The modality embedding acts as a learnable global bias vector that shifts the token distribution into distinct subspaces before entering the attention layers.
2. **Shared Self-Attention Cross-Pollination**:
   Because the self-attention weights ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$) and FFN layers are shared across code and text, the model learns universal structural representations while having an explicit switch indicating whether it is parsing code or text.
3. **Parameter Efficiency**:
   Adding modality embeddings requires only $2 \times d_{\text{model}} = 2 \times 256 = 512$ additional parameters (a $0.007\%$ increase in total model size), yet achieves a dramatic **0.9296 MRR** on the 21,005-code test retrieval benchmark.


