from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer.tokenizer import CodeEmbedTokenizer

PROCESSED_DIR = Path("data/processed")


class CodeSearchDataset(Dataset):
    """PyTorch Dataset loading CodeSearchNet processed parquet or pre-tokenized tensor splits.

    If a pre-tokenized `.pt` tensor file exists for the split, loads arrays directly into memory
    for zero-overhead tensor slicing (100x+ faster training throughput).
    Optionally loads a hard negative index matrix (`*_hard_negatives.pt`) to provide mined
    challenging negatives for each query.
    """

    def __init__(
        self,
        split_or_path: str | Path,
        use_pretokenized: bool = True,
        hard_negatives_file: str | Path | None = None,
        num_hard_negatives: int = 1,
    ):
        path = Path(split_or_path)
        if path.suffix:
            self.parquet_path = path
            split_name = path.stem.replace("_tokenized", "")
        else:
            split_name = str(path)
            self.parquet_path = PROCESSED_DIR / f"{split_name}.parquet"

        self.tokenized_path = PROCESSED_DIR / f"{split_name}_tokenized.pt"
        self.is_pretokenized = use_pretokenized and self.tokenized_path.exists()
        self.num_hard_negatives = num_hard_negatives

        # Check for hard negatives
        if hard_negatives_file is not None:
            hn_path = Path(hard_negatives_file)
        else:
            hn_path = PROCESSED_DIR / f"{split_name}_hard_negatives.pt"

        if hn_path.exists():
            self.hard_neg_indices = torch.load(hn_path, mmap=True, weights_only=True)
        else:
            self.hard_neg_indices = None

        if self.is_pretokenized:
            data = torch.load(self.tokenized_path, mmap=True, weights_only=True)
            self.code_ids = data["code_ids"]
            self.code_mask = data["code_mask"]
            self.text_ids = data["text_ids"]
            self.text_mask = data["text_mask"]
            self._len = len(self.code_ids)
            self.df = None
        else:
            if not self.parquet_path.exists():
                raise FileNotFoundError(f"Dataset split not found: {self.parquet_path}")
            self.df = pd.read_parquet(self.parquet_path)
            self._len = len(self.df)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.is_pretokenized:
            sample = {
                "code_ids": self.code_ids[idx].long(),
                "code_mask": self.code_mask[idx].long(),
                "text_ids": self.text_ids[idx].long(),
                "text_mask": self.text_mask[idx].long(),
            }

            if self.hard_neg_indices is not None:
                # Grab top-K hard negative code indices for this query
                hn_idxs = self.hard_neg_indices[idx][: self.num_hard_negatives].long()
                sample["hard_neg_code_ids"] = self.code_ids[hn_idxs].long()
                sample["hard_neg_code_mask"] = self.code_mask[hn_idxs].long()

            return sample
        else:
            row = self.df.iloc[idx]
            sample = {
                "code": row["code"],
                "docstring": row["docstring"],
                "func_name": row.get("func_name", ""),
            }
            if self.hard_neg_indices is not None:
                hn_idxs = self.hard_neg_indices[idx][: self.num_hard_negatives].tolist()
                sample["hard_neg_codes"] = [self.df.iloc[hn_i]["code"] for hn_i in hn_idxs]
            return sample


class CodeSearchCollator:
    """Collator that tokenizes and dynamically pads batches of code, docstrings, and hard negatives."""

    def __init__(
        self,
        tokenizer: CodeEmbedTokenizer,
        max_length: int = 256,
        padding: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Fast path if dataset is pre-tokenized
        if "code_ids" in batch[0]:
            batch_dict = {
                "code_ids": torch.stack([item["code_ids"] for item in batch]),
                "code_mask": torch.stack([item["code_mask"] for item in batch]),
                "text_ids": torch.stack([item["text_ids"] for item in batch]),
                "text_mask": torch.stack([item["text_mask"] for item in batch]),
            }

            if "hard_neg_code_ids" in batch[0]:
                # Shape: (B, K, L) -> stacked directly
                batch_dict["hard_neg_code_ids"] = torch.stack([item["hard_neg_code_ids"] for item in batch])
                batch_dict["hard_neg_code_mask"] = torch.stack([item["hard_neg_code_mask"] for item in batch])

            return batch_dict

        codes = [item["code"] for item in batch]
        docstrings = [item["docstring"] for item in batch]

        # Tokenize Code batch with <CODE> modality
        code_batch = self.tokenizer.encode(
            codes,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            modality="code",
        )

        # Tokenize docstring batch with <TEXT> modality
        text_batch = self.tokenizer.encode(
            docstrings,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            modality="text",
        )

        batch_dict = {
            "code_ids": code_batch["input_ids"],
            "code_mask": code_batch["attention_mask"],
            "text_ids": text_batch["input_ids"],
            "text_mask": text_batch["attention_mask"],
        }

        if "hard_neg_codes" in batch[0]:
            flat_hn_codes = [c for item in batch for c in item["hard_neg_codes"]]
            hn_batch = self.tokenizer.encode(
                flat_hn_codes,
                max_length=self.max_length,
                padding=self.padding,
                truncation=True,
                modality="code",
            )
            k = len(batch[0]["hard_neg_codes"])
            batch_dict["hard_neg_code_ids"] = hn_batch["input_ids"].view(len(batch), k, -1)
            batch_dict["hard_neg_code_mask"] = hn_batch["attention_mask"].view(len(batch), k, -1)

        return batch_dict


def create_dataloader(
    split: str = "train",
    batch_size: int = 256,
    shuffle: bool = True,
    max_length: int = 256,
    num_workers: int = 0,
    tokenizer: CodeEmbedTokenizer | None = None,
    use_pretokenized: bool = True,
    hard_negatives_file: str | Path | None = None,
    num_hard_negatives: int = 1,
) -> DataLoader:
    """Factory helper to build a ready-to-use DataLoader for a given split."""
    if tokenizer is None:
        tokenizer = CodeEmbedTokenizer()

    dataset = CodeSearchDataset(
        split,
        use_pretokenized=use_pretokenized,
        hard_negatives_file=hard_negatives_file,
        num_hard_negatives=num_hard_negatives,
    )
    collator = CodeSearchCollator(tokenizer=tokenizer, max_length=max_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),  # Drop incomplete last batch only in training
    )