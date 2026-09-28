"""High-Throughput FAISS Dense Hard Negative Miner for CodeEmbed (SharedEncoder).

This script:
1. Loads the trained clean SharedEncoder checkpoint.
2. Encodes all training code snippets (modality 0) and docstrings (modality 1) into L2-normalized vectors.
3. Builds a FAISS inner-product index over code representations.
4. Searches top-K nearest false-positive candidates per query.
5. Applies pre-registered 3-tier false negative filters (Docstring, AST skeleton, MinHash).
6. Saves the resulting (N, K) tensor matrix as `train_dense_hard_negatives.pt`.
"""

import argparse
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from rich.console import Console

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.shared_encoder import SharedEncoder
from training.hard_negatives import load_or_compute_fingerprints
from training.trainer import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine dense hard negatives using FAISS and trained SharedEncoder."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/fallback_grid/lr_5e-4_tau_0.05/best_shared.pt",
        help="Path to trained SharedEncoder checkpoint.",
    )
    parser.add_argument(
        "--tokenized-path",
        type=str,
        default="data/processed_clean_v2/train_tokenized.pt",
        help="Path to pretokenized binary train dataset.",
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="data/processed_clean_v2/train.parquet",
        help="Path to clean training parquet file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed_clean_v2/train_dense_hard_negatives.pt",
        help="Destination path for mined dense hard negative indices.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=7,
        help="Number of hard negatives per query.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=50,
        help="Number of candidate nearest neighbors retrieved from FAISS before filtering.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for dense encoding on GPU.",
    )
    parser.add_argument(
        "--search-batch-size",
        type=int,
        default=2048,
        help="Query batch size for FAISS search chunking.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> SharedEncoder:
    """Load SharedEncoder model weights and configuration from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    model_kwargs = {
        "vocab_size": 16000,
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff": 1024,
        "max_seq_len": 256,
        "dropout": 0.1,
        "num_modalities": 2,
    }

    if "model" in cfg:
        for k in model_kwargs:
            if k in cfg["model"]:
                model_kwargs[k] = cfg["model"][k]

    model = SharedEncoder(**model_kwargs)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_dataset(
    model: SharedEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    modality: str,
    device: torch.device,
    batch_size: int = 512,
    desc: str = "Encoding",
) -> np.ndarray:
    """Encode an entire tensor split into L2-normalized float32 vectors."""
    n_samples = input_ids.shape[0]
    dim = model.proj_ln.normalized_shape[0]
    embeddings = np.zeros((n_samples, dim), dtype=np.float32)

    t0 = time.time()
    use_cuda = device.type == "cuda"

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        b_ids = input_ids[start_idx:end_idx].to(device, non_blocking=True)
        b_mask = attention_mask[start_idx:end_idx].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            emb = model(b_ids, attention_mask=b_mask, modality_ids=modality)

        embeddings[start_idx:end_idx] = emb.cpu().float().numpy()

        if (end_idx % 25000 < batch_size) or end_idx == n_samples:
            elapsed = time.time() - t0
            speed = end_idx / max(elapsed, 1e-4)
            pct = end_idx / n_samples * 100.0
            print(
                f"[{desc}: {end_idx:>7,}/{n_samples:,} ({pct:>5.1f}%)] "
                f"Elapsed: {elapsed:>5.1f}s | Speed: {speed:>6.1f} samples/s",
                flush=True,
            )

    return embeddings


def main() -> None:
    args = parse_args()
    console = Console()

    console.print("[bold cyan]=== CodeEmbed: FAISS Dense Hard Negative Miner ===[/bold cyan]")
    console.print(f"Checkpoint     : [yellow]{args.checkpoint}[/yellow]")
    console.print(f"Tokenized Data : [yellow]{args.tokenized_path}[/yellow]")
    console.print(f"Output Matrix  : [yellow]{args.output}[/yellow]")
    console.print(f"Negatives (k)  : [yellow]{args.k}[/yellow] (pool size: {args.pool_size})")

    device = get_device()
    console.print(f"Device         : [green]{device}[/green]")

    # 1. Load Model
    console.print("\n[cyan]Step 1/5: Loading SharedEncoder Checkpoint...[/cyan]")
    model = load_model(args.checkpoint, device)
    console.print(f"  [OK] Model successfully loaded on {device}.")

    # 2. Load Tokenized Dataset
    console.print("\n[cyan]Step 2/5: Loading Pre-tokenized Training Tensors...[/cyan]")
    tokenized_dict = torch.load(args.tokenized_path, map_location="cpu", weights_only=False)
    code_ids = tokenized_dict["code_ids"]
    code_mask = tokenized_dict["code_mask"]
    text_ids = tokenized_dict["text_ids"]
    text_mask = tokenized_dict["text_mask"]
    n_samples = code_ids.shape[0]
    console.print(f"  [OK] Loaded {n_samples:,} code & query pairs.")

    # 3. Dense Embedding Inference
    console.print("\n[cyan]Step 3/5: Dense Vector Inference (CUDA AMP)...[/cyan]")
    code_embeddings = encode_dataset(
        model=model,
        input_ids=code_ids,
        attention_mask=code_mask,
        modality="code",
        device=device,
        batch_size=args.batch_size,
        desc="Encoding Code",
    )

    text_embeddings = encode_dataset(
        model=model,
        input_ids=text_ids,
        attention_mask=text_mask,
        modality="text",
        device=device,
        batch_size=args.batch_size,
        desc="Encoding Queries",
    )

    # 4. Load 3-Tier Fingerprints for False-Negative Filtering
    console.print("\n[cyan]Step 4/5: Loading 3-Tier False-Negative Filter Fingerprints...[/cyan]")
    parquet_path = Path(args.parquet_path)
    df = pd.read_parquet(parquet_path)
    docstring_ids, skeleton_hashes, minhash_sigs = load_or_compute_fingerprints(
        df=df,
        cache_dir=parquet_path.parent,
    )

    # 5. Build FAISS Index & Search
    console.print(f"\n[cyan]Step 5/5: Building FAISS Index & Mining Top-{args.k} Dense Negatives...[/cyan]")
    dim = code_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    t0_idx = time.time()
    index.add(code_embeddings)
    console.print(f"  [OK] FAISS IndexFlatIP built in {time.time() - t0_idx:.2f}s ({index.ntotal:,} vectors).")

    hard_neg_matrix = np.zeros((n_samples, args.k), dtype=np.int32)
    stats = {
        "tier1_docstring_drops": 0,
        "tier2_skeleton_drops": 0,
        "tier3_minhash_drops": 0,
    }

    t0_search = time.time()
    log_interval = 25000

    for chunk_start in range(0, n_samples, args.search_batch_size):
        chunk_end = min(chunk_start + args.search_batch_size, n_samples)
        q_chunk = text_embeddings[chunk_start:chunk_end]

        _, I = index.search(q_chunk, args.pool_size)

        for i, cand_row in enumerate(I):
            q_idx = chunk_start + i
            chosen = []

            for c_idx in cand_row:
                # Filter 0: Ground truth self
                if c_idx == q_idx:
                    continue

                # Filter 1: Identical query docstring intent
                if docstring_ids is not None and docstring_ids[c_idx] == docstring_ids[q_idx]:
                    stats["tier1_docstring_drops"] += 1
                    continue

                # Filter 2: Normalized AST skeleton match (>= 20 nodes)
                if skeleton_hashes is not None:
                    q_skel = skeleton_hashes[q_idx]
                    if q_skel != -1 and skeleton_hashes[c_idx] == q_skel:
                        stats["tier2_skeleton_drops"] += 1
                        continue

                # Filter 3: MinHash 3-gram Jaccard >= 0.70 (45/64 signature matches)
                if minhash_sigs is not None and np.count_nonzero(minhash_sigs[c_idx] == minhash_sigs[q_idx]) >= 45:
                    stats["tier3_minhash_drops"] += 1
                    continue

                chosen.append(int(c_idx))
                if len(chosen) == args.k:
                    break

            # Fallback to random candidates if pool is exhausted
            seen = set(chosen) | {q_idx}
            while len(chosen) < args.k:
                r = int(np.random.randint(0, n_samples))
                if r not in seen:
                    chosen.append(r)
                    seen.add(r)

            hard_neg_matrix[q_idx] = chosen[: args.k]

        if (chunk_end % log_interval < args.search_batch_size) or chunk_end == n_samples:
            elapsed = time.time() - t0_search
            speed = chunk_end / max(elapsed, 1e-4)
            rem = (n_samples - chunk_end) / max(speed, 1e-4)
            pct = chunk_end / n_samples * 100.0
            print(
                f"[Search Progress: {chunk_end:>7,}/{n_samples:,} ({pct:>5.1f}%)] "
                f"Elapsed: {elapsed:>5.1f}s | Speed: {speed:>6.1f} q/s | ETA: {rem:>5.1f}s | "
                f"Drops: Doc={stats['tier1_docstring_drops']:,}, "
                f"Skel={stats['tier2_skeleton_drops']:,}, "
                f"MinHash={stats['tier3_minhash_drops']:,}",
                flush=True,
            )

    # 6. Integrity Verification
    self_matches = np.sum(hard_neg_matrix == np.arange(n_samples)[:, None])
    assert self_matches == 0, f"Integrity error: {self_matches} self-matches detected!"
    assert np.all(hard_neg_matrix >= 0) and np.all(hard_neg_matrix < n_samples), (
        "Out-of-bounds indices!"
    )
    console.print("\n[bold green]Integrity Checks Passed: 0 self-matches, all indices valid.[/bold green]")

    # 7. Save Artifact
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(hard_neg_matrix), out_path)
    total_time = time.time() - t0_idx
    console.print(
        f"[bold green]Successfully saved FAISS dense hard negatives to {out_path} "
        f"(Shape: {hard_neg_matrix.shape}, Total time: {total_time:.2f}s)[/bold green]"
    )


if __name__ == "__main__":
    main()
