# CodeEmbed Research Protocol v1 (Pre-Registered)

> **Immutable Research Contract**: Tagged `protocol-v1`. All definitions, formulas, statistical margins, tie-breaking rules, and data hygiene constraints are frozen as of this commit prior to executing Phase R0 / R1.

---

## 1. Verified Invalidation Safeguards

All historical benchmarks, documentation, MLflow runs, and on-disk model checkpoints are permanently marked invalid:

### 1.1 Documentation Caution Banners
- `walkthrough.md`, `STUDY_GUIDE.md`, and `Memory.md` have received the invalidation notice:
  > **INVALID: computed on leaky data, superseded by R-track.**
  > All retrieval metrics, RQ conclusions, and benchmark comparisons recorded below in Phases 1–6 were computed on unstripped CodeSearchNet code containing verbatim docstring substrings (100% query leakage). The engineering modules (SDPA memory tiling, CSR mining, pretokenization cache, PyTorch models) remain valid, but all scientific numbers and answers are superseded by the leak-free R-track.

### 1.2 MLflow Historical Runs Tagged
- Database: `sqlite:///mlflow.db`
- 6 historical runs across 3 experiments tagged with `data_version=leaky_v1` and `validity=INVALID_LEAKY_DATA`.

### 1.3 On-Disk Model Checkpoints Tagged
- Directory: `checkpoints/`
- 59 `.pt` checkpoint files tagged with `data_version=leaky_v1` and `validity=INVALID_LEAKY_DATA`.

---

## 2. Protocol Specifications & Defect Fixes

### 2.1 Coordinate-Accurate AST Docstring Slicing & Storage
- **Coordinate System**:
  - Code is first dedented via `code = textwrap.dedent(raw_code)`.
  - The AST is parsed on this `code`: `tree = ast.parse(code)`.
  - In Python 3.8+, AST `col_offset` and `end_col_offset` are UTF-8 byte offsets on the *dedented* text.
  - Slicing is performed on UTF-8 bytes of the **dedented** code. The resulting cleaned, dedented string is saved as the **canonical code representation** in the processed dataset. This eliminates coordinate-system mismatch and trailing quote corruptions.
- **Target Scope**:
  - Restrict strictly to the docstring statement: `node.body[0]` of `FunctionDef`, `AsyncFunctionDef`, `ClassDef`, or `Module` where `body[0]` is an `ast.Expr` whose value is an `ast.Constant(str)` (matching Python's formal `ast.get_docstring` definition).
  - Standalone bare strings elsewhere in the body are preserved.
- **Line Deletion Rule**:
  - Remove a line entirely only if the remaining content on that line consists exclusively of whitespace.
  - Single-line functions (e.g. `def f(): """Doc"""\n return 1`) preserve `def f():\n return 1`.
- **Empty Stubs**:
  - Functions whose body consisted solely of a docstring become empty syntax stubs after stripping. Re-parsing with `ast.parse` catches indentation errors, and the length filter (< 10 tokens) drops them.

### 2.2 Expected Reciprocal Rank & Tie-Breaking Mathematics
- **Jensen's Inequality**: $1/\mathbb{E}[\text{rank}] \neq \mathbb{E}[1/\text{rank}]$.
- **Expected Reciprocal Rank ($\mathbb{E}[\text{RR}]$)**:
  When a query's ground-truth document has a BM25 score of 0, it ties with $S_{=0}$ total zero-scoring documents behind $S_{>0}$ higher-scoring documents ($S_{>0} = \sum_{d} \mathbb{I}(\text{score}_d > 0)$).
  Under a uniform random tie-break, the ground-truth rank is uniformly distributed over $\{S_{>0} + 1, S_{>0} + 2, \dots, S_{>0} + S_{=0}\}$. The exact expected reciprocal rank is:
  $$\mathbb{E}[\text{RR}] = \frac{1}{S_{=0}} \sum_{j=1}^{S_{=0}} \frac{1}{S_{>0} + j} = \frac{H_{S_{>0} + S_{=0}} - H_{S_{>0}}}{S_{=0}}$$
  where $H_n = \sum_{k=1}^n \frac{1}{k}$ is the $n$-th harmonic number.
- **Expected Recall@k ($\mathbb{E}[\text{R@k}]$)**:
  $$\mathbb{E}[\text{R@k}] = \begin{cases} 
  0 & \text{if } S_{>0} \ge k \\
  1 & \text{if } S_{>0} + S_{=0} \le k \\
  \frac{k - S_{>0}}{S_{=0}} & \text{if } S_{>0} < k < S_{>0} + S_{=0}
  \end{cases}$$
- **Chance Reference**:
  Random ranking baseline MRR on corpus of size $N$:
  $$\mathbb{E}[\text{MRR}_{\text{chance}}] = \frac{H_N}{N} \approx \frac{\ln N + \gamma}{N}$$
  (Recomputed dynamically using the exact post-dedup test corpus size $N$).

### 2.3 Non-Flaky Structural, Equivalence, and Rate-Based Overlap Tests
In `tests/test_data_integrity.py`:
1. **Structural Test (Pass/Fail)**:
   Assert for 100% of samples across all clean splits:
   $$\forall \text{node} \in \text{ast.walk}(\text{tree}), \quad \text{ast.get_docstring}(\text{node}) \text{ is None}$$
2. **AST-Equivalence Test (Pass/Fail)**:
   For every function, delete `node.body[0]` from the raw dedented AST and compare AST dumps:
   $$\text{ast.dump}(\text{raw\_ast\_without\_body0}) == \text{ast.dump}(\text{cleaned\_ast})$$
   Proves mathematically that AST stripping removed *only* the docstring AST node and modified zero code statements.
3. **Contiguous Overlap Rate Test (Empirical Diagnostic)**:
   - Calculate the rate of contiguous $n$-gram matches ($n \ge 8$ code tokens) between query and ground-truth code.
   - Construct a null distribution by pairing each query with a randomly sampled non-target code function.
   - **Criterion**: The fraction of true pairs containing an $n \ge 8$ contiguous match must not exceed the null-pair fraction by more than a pre-registered margin of $+0.05$ (5 percentage points). Any matching pairs are manually reviewed to confirm they are legitimate identifiers, log strings, or exception messages.

### 2.4 Cross-Split Hygiene & Near-Duplicate Removal
- **Deduplication Hierarchy**:
  - Keep `test` canonical for test set evaluation.
  - Clean `train`, `validation`, and `test`.
  - **Train $\to$ Val/Test Dedup**: Exact match on stripped code and MinHash LSH near-duplicate matches ($J \ge 0.85$) are purged from `validation` and `test` (keeping `train` intact to preserve training distribution).
  - **Val $\leftrightarrow$ Test Dedup**: Exact match and MinHash near-duplicates ($J \ge 0.85$) between `validation` and `test` are purged from `validation` (keeping `test` intact).
- **MinHash LSH Specification**:
  - Word-level 3-grams over stripped code.
  - 64 MinHash permutations, partitioned into 16 bands of 4 rows.
  - Candidates identified via LSH buckets are filtered by exact Jaccard similarity; candidates with $J \ge 0.85$ are purged.
  - Effective deduplication threshold is strictly $J \ge 0.85$.
- **Normalized Skeleton Definition**:
  - AST where all identifier names (`ast.Name.id`, `ast.arg.arg`, `ast.FunctionDef.name`) are mapped to `"var"` and all constants (`ast.Constant.value`) are mapped to `"const"`.
  - Code snippets with identical normalized AST skeletons are treated as near-duplicates.

---

## 3. Operational Integrity & Experimental Protocol

### 3.1 Training Hygiene & Duplicate Queries
- **In-Batch False Negative Masking**:
  - If identical docstring queries exist within the same training batch, they share semantic intent.
  - Construct a ground-truth positive mask $M_{i,j} = \mathbb{I}(\text{docstring}_i == \text{docstring}_j)$.
  - In the InfoNCE contrastive denominator, mask out same-query pairs from being penalized as negatives.
- **Hard-Negative Mining Hygiene**:
  - Mined strictly from the **clean train split**.
  - Exclude candidates where MinHash $J \ge 0.70$, or with identical normalized skeleton, or sharing the identical query docstring.
- **Python 2 / Parse Failures**:
  - Track drop counts per split and log percentage of dropped samples.
  - Compare token length and repository distributions of dropped vs kept samples to confirm no differential distribution shift.

### 3.2 BM25 Baseline & Pre-Registered Overlap Stratification
- **BM25 Parameters**: $k_1 = 1.5, b = 0.75$ fixed at defaults. No tuning on the test set.
- **Validation Evaluation**: Run BM25 on **both validation and test** splits in Phase R1 to provide the reference for the Phase R3 pilot gate.
- **IDF Calculation**:
  - Computed strictly on the **clean train split**:
    $$\text{IDF}(t) = \ln\left(\frac{N_{\text{train}} - n_{\text{train}}(t) + 0.5}{n_{\text{train}}(t) + 0.5} + 1\right)$$
  - Stopwords: NLTK English stopwords + Python keywords + punctuation.
  - Queries with 0 non-stopword tokens are assigned by definition to the Zero-Overlap Bin.
- **Frozen Bin Edges**:
  - **Zero-Overlap Bin**: $\text{Coverage} = 0.0$.
  - **Low-Overlap Bin**: $0.0 < \text{Coverage} \le 0.30$.
  - **High-Overlap Bin**: $\text{Coverage} > 0.30$.

### 3.3 Phase R3 Neural Training & Statistical Protocol
- **Standardized Baseline Hyperparameters**:
  - Model: SharedEncoder (7.38M parameters, Pre-LN, 4 layers, $d_{\text{model}}=256$, 8 heads, $d_{\text{ff}}=1024$).
  - Optimizer: AdamW, weight decay 0.01, LR 3e-4, linear warmup 1000 steps, cosine decay.
  - Batch size: 128, temperature $\tau = 0.07$.
- **Pilot Run & Explicit Fallback Action**:
  - Train 1 Shared Encoder on clean train data for 2 epochs, evaluate on **validation set only**.
  - **Pass Gate**: Validation $\text{MRR}_{\text{Dense}} \ge 0.75 \times \text{MRR}_{\text{BM25, val}}$ OR Low-Overlap Validation $\text{MRR}_{\text{Dense}} > \text{MRR}_{\text{BM25, val, Low}}$.
  - **Fallback Action on Fail**: If the pilot fails, execute a strictly capped, shared validation recipe tuning grid across all arms:
    - 3 Learning Rates: $\{1\text{e-}4, 3\text{e-}4, 5\text{e-}4\}$
    - 2 Temperatures: $\{0.05, 0.07\}$
    - Total: 6 validation runs. The single highest-MRR configuration on validation is adopted across all comparison arms.
- **Seed Replication**:
  - 3 frozen seeds (`[42, 123, 456]`). Report $\text{mean} \pm \text{sd}$.
- **Paired Metric Bootstrap**:
  $$\Delta \text{RR}_i = \frac{1}{\text{rank}_A(q_i)} - \frac{1}{\text{rank}_B(q_i)}, \quad \Delta \text{R@1}_i = \mathbb{I}(\text{rank}_A \le 1) - \mathbb{I}(\text{rank}_B \le 1)$$
  Bootstrap 95% confidence intervals computed over 2,000 resamples.
- **Strict Test Set Discipline**:
  - Model selection is conducted **exclusively on validation**.
  - All pre-declared arms are evaluated on the test set **strictly once** at the conclusion of Phase R3.
  - Pairwise hypothesis tests on test set apply Holm-Bonferroni correction.

### 3.4 Operational Gates
- **"Competitive with BM25" Definition**:
  - Equivalence margin: The paired bootstrap 95% CI lower bound on $\mathbb{E}[\Delta \text{RR}]$ ($\text{Dense} - \text{BM25}$) is $\ge -0.030$.
- **"2 Epochs Beat 1 Epoch" Definition**:
  - Mean $\Delta \text{RR} > 0$ across all 3 random seeds.
  - Paired bootstrap 95% CI on the seed-averaged $\Delta \text{RR}_i$ strictly excludes 0 ($p < 0.05$).
- **Capacity Error Rubric (Gate on Phase 6.5 Scaling)**:
  - Seeded random sample (seed=42) of 100 validation queries where dense model rank $> 10$.
  - Labeled into:
    - **Category A (Under-specified / Ambiguous)**: Query is too vague or generic; multiple valid functions in corpus satisfy it equally well.
    - **Category B (Capacity / Representation Error)**: Query specifies concrete functionality that exists in the ground-truth function, but model failed to rank it in top-10.
    - **Category C (Label Noise / Dead Code)**: Query or code is uninformative/corrupted.
  - **Scaling Gate Requirement**: Category B must constitute $\ge 50\%$ of failures. If Category A + C $\ge 50\%$, failure is label-ceiling bounded, and capacity scaling is blocked.

### 3.5 Explicit Modality & Dual Encoder Scope
- Modality embedding ablation and Dual Encoders are **not investigated in the primary remediation comparison**.
- They are formally deferred to a post-R3 secondary extension if GPU compute and project schedule permit.

