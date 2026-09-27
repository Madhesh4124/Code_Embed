"""Hard-negative-aware Dataset and DataLoader for Phase 5.

Wraps the base CodeSearchDataset to attach pre-mined hard negative indices
to each sample. The collator produces the same batch keys as before, plus:
    - hard_neg_ids:   (B, K, L) — tokenized hard negative code snippets
    - hard_neg_masks: (B, K, L) — attention masks for hard negatives
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer.tokenizer import CodeEmbedTokenizer

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data/processed_clean_v2"


class HardNegativeDataset(Dataset):
    """CodeSearch dataset augmented with pre-mined hard negative indices.

    Args:
        parquet_path: Path to the processed parquet split file.
        hard_neg_path: Path to (N, K) int32 numpy array of hard neg indices.
        max_hard_negs: How many hard negatives to use per sample (≤ K).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        hard_neg_path: str | Path,
        max_hard_negs: int = 10,
    ) -> None:
        self.df = pd.read_parquet(Path(parquet_path))
        self.hard_neg_indices = np.load(str(hard_neg_path))   # (N, K)
        self.max_hard_negs = min(max_hard_negs, self.hard_neg_indices.shape[1])

        if len(self.df) != len(self.hard_neg_indices):
            raise ValueError(
                f"Dataset size ({len(self.df)}) != hard_neg_indices rows "
                f"({len(self.hard_neg_indices)}). Re-mine with the correct split."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        # Pick the first max_hard_negs hard negatives
        hn_indices = self.hard_neg_indices[idx, : self.max_hard_negs].tolist()
        hard_neg_codes = [self.df.iloc[i]["code"] for i in hn_indices]

        return {
            "code":            row["code"],
            "docstring":       row["docstring"],
            "func_name":       row.get("func_name", ""),
            "hard_neg_codes":  hard_neg_codes,   # list of K strings
        }


class HardNegativeCollator:
    """Collate code/text pairs AND hard negative code snippets.

    Output batch keys:
        code_ids, code_mask          — (B, L)  positive code
        text_ids, text_mask          — (B, L)  query docstring
        hard_neg_ids, hard_neg_masks — (B, K, L) hard negative codes
    """

    def __init__(
        self,
        tokenizer: CodeEmbedTokenizer,
        max_length: int = 256,
        max_hard_negs: int = 10,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_hard_negs = max_hard_negs

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        codes      = [item["code"]      for item in batch]
        docstrings = [item["docstring"] for item in batch]
        hard_negs  = [item["hard_neg_codes"] for item in batch]   # list[list[str]]

        # Tokenize positive pairs
        code_batch = self.tokenizer.encode(
            codes, max_length=self.max_length,
            padding=True, truncation=True, modality="code",
        )
        text_batch = self.tokenizer.encode(
            docstrings, max_length=self.max_length,
            padding=True, truncation=True, modality="text",
        )

        # Tokenize hard negatives: flatten → tokenize → reshape
        B, K = len(batch), self.max_hard_negs
        flat_hn = [code for sample_hn in hard_negs for code in sample_hn]  # (B*K,)
        hn_batch = self.tokenizer.encode(
            flat_hn, max_length=self.max_length,
            padding=True, truncation=True, modality="code",
        )
        # Reshape to (B, K, L)
        L = hn_batch["input_ids"].shape[-1]
        hard_neg_ids   = hn_batch["input_ids"].view(B, K, L)
        hard_neg_masks = hn_batch["attention_mask"].view(B, K, L)

        return {
            "code_ids":        code_batch["input_ids"],
            "code_mask":       code_batch["attention_mask"],
            "text_ids":        text_batch["input_ids"],
            "text_mask":       text_batch["attention_mask"],
            "hard_neg_ids":    hard_neg_ids,
            "hard_neg_masks":  hard_neg_masks,
        }


def create_hard_negative_dataloader(
    hard_neg_path: str | Path,
    split: str = "train",
    tokenizer: CodeEmbedTokenizer | None = None,
    batch_size: int = 256,
    max_length: int = 256,
    max_hard_negs: int = 10,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Factory: build a DataLoader with hard negatives attached.

    Args:
        hard_neg_path: Path to the (N, K) .npy file produced by hard_negatives.py.
        split: Dataset split ('train' / 'validation' / 'test').
        tokenizer: CodeEmbedTokenizer instance (created if None).
        batch_size: Samples per batch.
        max_length: Max token sequence length.
        max_hard_negs: Hard negatives per sample to include.
        shuffle: Whether to shuffle (True for train).
        num_workers: DataLoader worker processes.

    Returns:
        DataLoader yielding batches with hard_neg_ids and hard_neg_masks.
    """
    if tokenizer is None:
        tokenizer = CodeEmbedTokenizer("tokenizer/tokenizer.json")

    parquet_path = PROCESSED_DIR / f"{split}.parquet"
    dataset  = HardNegativeDataset(parquet_path, hard_neg_path, max_hard_negs=max_hard_negs)
    collator = HardNegativeCollator(tokenizer, max_length=max_length, max_hard_negs=max_hard_negs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )
