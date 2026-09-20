"""Ablation dataset wrapper for CodeEmbed (Phase 6: Ablation Studies).

This module provides AblationCodeSearchDataset:
- Supports arbitrary sequence length truncation (e.g., max_seq_len = 128) by slicing
  in-memory pre-tokenized binary tensors with zero re-tokenization overhead.
- Supports both in-batch negative training and hard negative training.
- Completely isolates ablation data handling without touching data/dataset.py.
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.dataset import CodeSearchDataset

PROCESSED_DIR = Path("data/processed")


class AblationCodeSearchDataset(CodeSearchDataset):
    """Dataset supporting sequence length slicing for ablation studies.

    Args:
        split_or_path: Dataset split ('train', 'validation', 'test') or Path.
        max_seq_len: Optional maximum sequence length to slice tokens to (e.g. 128).
            If None or >= 256, uses full 256-token sequences.
        use_pretokenized: Whether to load pre-tokenized .pt binary tensor files.
        hard_negatives_file: Optional path to mined hard negative indices file.
        num_hard_negatives: Number of hard negatives per query (default: 1).
    """

    def __init__(
        self,
        split_or_path: str | Path,
        max_seq_len: int | None = None,
        use_pretokenized: bool = True,
        hard_negatives_file: str | Path | None = None,
        num_hard_negatives: int = 1,
    ) -> None:
        super().__init__(
            split_or_path=split_or_path,
            use_pretokenized=use_pretokenized,
            hard_negatives_file=hard_negatives_file,
            num_hard_negatives=num_hard_negatives,
        )
        self.max_seq_len = max_seq_len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = super().__getitem__(idx)

        # Apply sequence length truncation if requested
        if self.is_pretokenized and self.max_seq_len is not None and self.max_seq_len < 256:
            l_cut = self.max_seq_len
            sample["code_ids"] = sample["code_ids"][:l_cut]
            sample["code_mask"] = sample["code_mask"][:l_cut]
            sample["text_ids"] = sample["text_ids"][:l_cut]
            sample["text_mask"] = sample["text_mask"][:l_cut]

            if "hard_neg_code_ids" in sample:
                # hard_neg_code_ids shape: (K, L) -> slice second dimension
                sample["hard_neg_code_ids"] = sample["hard_neg_code_ids"][:, :l_cut]
                sample["hard_neg_code_mask"] = sample["hard_neg_code_mask"][:, :l_cut]

        return sample


def get_ablation_dataloader(
    split: str,
    batch_size: int = 128,
    max_seq_len: int | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    hard_negatives_file: str | Path | None = None,
    num_hard_negatives: int = 1,
) -> DataLoader:
    """Create a high-performance DataLoader for ablation experiments.

    Args:
        split: Dataset split ('train', 'validation', 'test').
        batch_size: Batch size (default: 128).
        max_seq_len: Optional sequence length truncation (e.g. 128).
        shuffle: Whether to shuffle data (default: True).
        num_workers: DataLoader worker count (default: 0).
        hard_negatives_file: Optional hard negatives index path.
        num_hard_negatives: Number of hard negatives per query.

    Returns:
        DataLoader yielding batches of pre-tokenized tensors.
    """
    dataset = AblationCodeSearchDataset(
        split_or_path=split,
        max_seq_len=max_seq_len,
        use_pretokenized=True,
        hard_negatives_file=hard_negatives_file,
        num_hard_negatives=num_hard_negatives,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

