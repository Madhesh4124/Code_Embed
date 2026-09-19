# CodeEmbed — Development Rules & Guidelines

## Code Style & Standards

### Python
- **Code Style**: Flexible / developer preference (readable, clean code)
- **Python Version**: 3.10+

### Imports
```python
# Standard library first
import os
from pathlib import Path
from typing import List, Dict, Optional

# Third-party
import torch
import torch.nn as nn
from omegaconf import DictConfig

# Local
from codeembed.model.encoder import Encoder
from codeembed.tokenizer.tokenizer import CodeEmbedTokenizer
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `contrastive.py` |
| Classes | PascalCase | `ContrastiveLoss` |
| Functions | snake_case | `compute_mrr` |
| Constants | UPPER_SNAKE | `MAX_SEQ_LEN` |
| Private | _leading_underscore | `_init_weights` |
| Type Variables | PascalCase + `_co`/`_contra` | `T_co` |

## Architecture Rules

### Model Implementation
- **No HuggingFace pretrained models** for primary experiments
- Implement Transformer from basic `nn.Module` primitives
- Use `nn.MultiheadAttention` or custom attention (prefer custom for learning)
- Pre-LN architecture only (more stable)
- All linear layers: `bias=False` where followed by LayerNorm

### Tensor Shapes
Document expected shapes in docstrings:
```python
def forward(
    self,
    input_ids: torch.Tensor,      # (B, L)
    attention_mask: torch.Tensor  # (B, L)
) -> torch.Tensor:                # (B, D)
```

### Device Handling
- Never hardcode `.cuda()` — use `.to(device)`
- Create device utility: `get_device() -> torch.device`
- Support CPU, CUDA, MPS

### Mixed Precision
```python
# Use autocast + GradScaler
with torch.autocast(device_type=device.type, dtype=torch.float16):
    loss = model(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Training Rules

### Reproducibility
```python
def set_seed(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Checkpointing
Save every epoch + best model:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_metric': best_metric,
    'config': config,
}
torch.save(checkpoint, path)
```

### Logging (MLflow)
- Log hyperparameters at run start
- Log train/val loss every epoch
- Log MRR, Recall@k, NDCG@k at eval steps
- Log model checkpoints as artifacts
- Log attention maps / heatmaps periodically
- Tag runs: `architecture`, `tokenizer`, `negative_strategy`

### Progress Tracking & Status Visibility (MANDATORY)
- **Visible Progress for All Long-Running Tasks**: Every data preparation, mining, tokenization, training, or evaluation loop spanning more than a few seconds **MUST** provide explicit progress indication.
- **Log File & Non-TTY Compatibility**: In-place carriage returns (`\r`) from curses/rich bars get buffered when redirected to log files (`sys.stdout.isatty() == False`). All long-running scripts must either:
  1. Print periodic discrete milestone updates (e.g. `[Step X/N (Y%)] Elapsed: ... ETA: ... Throughput: ... q/s`, `flush=True`) at regular intervals (e.g., every 5,000 items or every 2%), OR
  2. Configure progress bars (e.g. `tqdm(..., mininterval=5.0, file=sys.stdout)`) to regularly flush updates.
- **Zero Silent Long Jobs**: Never allow loops over large datasets (e.g. 385k samples) to run silently without real-time percentage and ETA output.

### Gradient Monitoring
```python
# Log gradient norms
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
mlflow.log_metric("grad_norm", total_norm, step=global_step)
```

## Data Rules

### Tokenizer
- Train on **combined** code + text corpus
- Save tokenizer as `tokenizer.json` + `tokenizer_config.json`
- Version tokenizer with model checkpoints

### DataLoader
- `pin_memory=True` for GPU
- `persistent_workers=True` for multi-epoch
- Custom collator for dynamic padding
- Drop last batch in training, keep in validation

### Validation
- Fixed validation set (no shuffle)
- Evaluate every epoch
- Use same metric for model selection (MRR or Recall@1)

## Evaluation Rules

### Metrics Implementation
- Use `rank_eval` or implement from scratch
- No external API calls for evaluation
- Report confidence intervals (bootstrap 1000 samples)

### Retrieval
- Build FAISS index **once** per model
- Use `IndexFlatIP` for exact cosine (L2-normalized vectors)
- For large corpus: `IndexIVFFlat` with nprobe tuning

### Baselines
- BM25 must use same train/val/test split
- Pretrained baseline: fixed model, no fine-tuning

## Experiment Tracking Rules

### Required MLflow Tags
```python
mlflow.set_tag("architecture", "basic|shared|dual")
mlflow.set_tag("tokenizer", "custom_bpe|generic")
mlflow.set_tag("negatives", "inbatch|hard")
mlflow.set_tag("phase", "baseline|main|ablation|final")
```

### Required Parameters
```python
mlflow.log_params({
    "vocab_size": 16000,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "max_seq_len": 256,
    "lr": 3e-4,
    "batch_size": 256,
    "temperature": 0.07,
    "warmup_steps": 2000,
    "weight_decay": 0.01,
})
```

### Required Metrics
```python
mlflow.log_metrics({
    "train_loss": ...,
    "val_loss": ...,
    "mrr": ...,
    "recall_at_1": ...,
    "recall_at_5": ...,
    "recall_at_10": ...,
    "ndcg": ...,
}, step=epoch)
```

## Testing Rules

### Unit Tests
- Test each model component in isolation
- Test tokenizer round-trip: `decode(encode(x)) == x`
- Test loss function with known inputs
- Test pooling with various mask patterns

### Integration Tests
- Forward pass with batch size > 1
- Backward pass completes without NaN
- Checkpoint save/load produces identical output

### Test Coverage Target
- **Minimum**: 80% for `model/`, `losses/`, `evaluation/`
- **Not required**: `scripts/`, `notebooks/`, `demo/`

## Error Handling

### Do
```python
# Explicit validation with clear messages
def __init__(self, vocab_size: int):
    if vocab_size < 1000:
        raise ValueError(f"vocab_size must be >= 1000, got {vocab_size}")
    self.vocab_size = vocab_size

# Graceful degradation
try:
    import faiss
except ImportError:
    faiss = None
    logger.warning("FAISS not available, using brute-force search")
```

### Don't
- Silent failures (`except: pass`)
- Generic `Exception` catches
- Print statements for errors (use `logging`)

## Performance Rules

### Memory
- Delete intermediate tensors in training loop: `del loss; torch.cuda.empty_cache()`
- Use `torch.inference_mode()` for evaluation
- Gradient accumulation for effective large batch

### Speed
- Compile model: `model = torch.compile(model)` (PyTorch 2.0+)
- Fuse attention: `scaled_dot_product_attention`
- Profile with `torch.profiler` before optimizing

## Git & Collaboration

### Branching
- `main`: Protected, releases only
- `dev`: Integration branch
- `feature/*`, `experiment/*`, `fix/*`: Short-lived

### Commits
- Conventional commits: `feat:`, `fix:`, `exp:`, `docs:`, `refactor:`
- One logical change per commit
- Reference issue/experiment: `exp: add shared encoder (exp-002)`

### PR Requirements
- All tests pass
- Type check: `ruff check . && mypy .`
- No decrease in test coverage
- MLflow run linked in description

## What to Avoid

| Category | Avoid | Prefer |
|----------|-------|--------|
| **Libraries** | HuggingFace `transformers` models | Custom `nn.Module` |
| **Libraries** | `sentence-transformers` | Custom contrastive loop |
| **Libraries** | `lightning` / `accelerate` | Raw PyTorch training loop |
| **Patterns** | Global config objects | OmegaConf + dependency injection |
| **Patterns** | Magic numbers in code | Config YAML + constants |
| **Patterns** | Notebooks for production code | `.py` modules, notebooks for EDA only |
| **Training** | Early stopping on contrastive loss | Early stopping on retrieval metrics |
| **Training** | Fixed LR | Cosine decay with warmup |
| **Data** | Heavy preprocessing before baseline | Minimal cleaning, iterate |
| **Evaluation** | Only success examples | Failure analysis mandatory |

## Security

- No API keys in code (use `.env` + `python-dotenv`)
- No commit of data, checkpoints, logs
- `.gitignore` must exclude: `*.pt`, `*.bin`, `mlruns/`, `.env`, `__pycache__/`, `*.log`

## Documentation

- Update `README.md` when adding new scripts
- Docstring every public function
- Architecture decisions → `Architecture.md`
- Experiment results → MLflow + `notebooks/results_analysis.ipynb`