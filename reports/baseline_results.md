# BM25 Lexical Baseline Benchmark & Integrity Audit (Clean R-Track)

> [!NOTE]
> All benchmarks reported below were computed on `data/processed_clean_v2/` with 100% docstring query leakage eliminated via byte-accurate AST slicing and MinHash LSH cross-split deduplication.
> Following validation-split selection (§1.10 in `PROTOCOL_ERRATA.md`), the conservative **ATIRE negative-IDF floor variant** (`method="rank_bm25"`) was adopted across all official baselines.

---

## 1. Full Split Retrieval Performance

Evaluated using generalized harmonic tie-breaking:

| Split | Corpus Size | Queries | MRR | 95% Confidence Interval | Recall@1 | Recall@5 | Recall@10 | NDCG@10 | MLflow Run ID |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test** | 19,632 | 19,632 | **0.5108** | [0.5047, 0.5166] | 0.4052 | 0.6340 | 0.6993 | 0.5514 | `fb3b5313f1bb419bb330b7fc0dee6bf5` |
| **Validation** | 20,115 | 20,115 | **0.5214** | [0.5152, 0.5275] | 0.4107 | 0.6515 | 0.7192 | 0.5644 | `14feca9d5b024faab9da64beac12541b` |

---

## 2. Like-for-Like Pure Leakage Isolation (Identical $N=19,632$ Test Items)

| Metric | Leaky (In-Code Docstring) | Clean (Stripped Docstring) | $\Delta$ (Net Leakage Effect) |
| :--- | :---: | :---: | :---: |
| **MRR** | **0.9513** [0.9491, 0.9537] | **0.5108** [0.5047, 0.5166] | **$-0.4405$ ($-44.1$ pts)** |
| **Recall@1** | **0.9164** [0.9127, 0.9202] | **0.4052** [0.3981, 0.4117] | **$-0.5112$ ($-51.1$ pts)** |
| **Recall@5** | **0.9885** [0.9870, 0.9900] | **0.6340** [0.6274, 0.6410] | **$-0.3545$ ($-35.5$ pts)** |
| **Recall@10** | **0.9937** [0.9926, 0.9948] | **0.6993** [0.6930, 0.7055] | **$-0.2944$ ($-29.4$ pts)** |
| **NDCG@10** | **0.9618** [0.9600, 0.9637] | **0.5514** [0.5456, 0.5574] | **$-0.4104$ ($-41.0$ pts)** |

---

## 3. Frozen Stratification on Clean Train IDF (ATIRE Baseline)

IDF computed strictly from clean `train.parquet` ($N=360,957$) with stopwords and single-character tokens removed:

### Validation Split ($N = 20,115$)
| Stratum | Overlap Definition | Queries | Proportion | BM25 MRR | BM25 R@1 | BM25 R@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero** | $c = 0.0$ | 580 | 2.88% | **0.0023** | 0.0000 | 0.0086 |
| **Low** | $0 < c \le 0.30$ | 6,321 | 31.42% | **0.2489** | 0.1569 | 0.4297 |
| **High** | $c > 0.30$ | 13,214 | 65.69% | **0.6745** | 0.5502 | 0.8888 |
| **Overall** | — | 20,115 | 100.0% | **0.5214** | 0.4107 | 0.7192 |

### Test Split ($N = 19,632$)
| Stratum | Overlap Definition | Queries | Proportion | BM25 MRR | BM25 R@1 | BM25 R@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero** | $c = 0.0$ | 535 | 2.73% | **0.0099** | 0.0037 | 0.0206 |
| **Low** | $0 < c \le 0.30$ | 5,808 | 29.58% | **0.2099** | 0.1284 | 0.3698 |
| **High** | $c > 0.30$ | 13,289 | 67.69% | **0.6625** | 0.5423 | 0.8706 |
| **Overall** | — | 19,632 | 100.0% | **0.5108** | 0.4052 | 0.6993 |

---

## 4. Phase R3 Pilot Gate Reference
- **Overall Validation MRR Target**: $0.75 \times 0.5214 = \mathbf{0.3910}$
- **Low-Overlap Validation MRR Dominance**: $> \mathbf{0.2489}$
