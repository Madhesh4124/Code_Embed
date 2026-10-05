"""Interactive Code Search Demonstration Engine for CodeEmbed.

Supports:
1. Dense Semantic Retrieval (CodeEmbed 6L 17M / 4L 7M / MiniLM)
2. Lexical BM25 Retrieval (ATIRE bm25)
3. Hybrid Search via Reciprocal Rank Fusion (RRF)
4. Overlap Stratum Annotation (Zero, Low, High) and Syntax Highlighted Display.

Usage:
    # Single query execution
    uv run python demo/search.py --query "parse json safely with fallback" --mode hybrid --top-k 5

    # Interactive REPL session
    uv run python demo/search.py --interactive
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from model.shared_encoder import SharedEncoder
from tokenizer.tokenizer import CodeEmbedTokenizer

console = Console(legacy_windows=False)


def compute_jaccard(query_str: str, code_str: str) -> float:
    """Compute token-level Jaccard similarity."""
    q_tokens = set(query_str.lower().split())
    c_tokens = set(code_str.lower().split())
    if not q_tokens or not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens | c_tokens)


def get_stratum_badge(jaccard: float) -> str:
    """Format overlap stratum badge with ANSI color."""
    if jaccard == 0.0:
        return "[bold green]Zero-Overlap (Pure Semantic)[/bold green]"
    elif jaccard <= 0.15:
        return f"[bold yellow]Low-Overlap (J={jaccard:.2f})[/bold yellow]"
    else:
        return f"[bold blue]High-Overlap (J={jaccard:.2f})[/bold blue]"


class SearchEngine:
    """Unified Search Engine combining Dense, BM25, and Hybrid Retrieval."""

    def __init__(
        self,
        corpus_df: pd.DataFrame,
        model_type: str = "shared_6l",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.df = corpus_df.reset_index(drop=True)
        self.codes = self.df["code"].tolist()
        self.docstrings = self.df["docstring"].tolist() if "docstring" in self.df.columns else [""] * len(self.df)
        self.model_type = model_type

        console.print(f"[bold cyan]Initializing Search Engine with {len(self.codes):,} code documents...[/bold cyan]")

        # 1. Initialize BM25
        console.print("[dim]Tokenizing corpus for BM25 (BM25Okapi)...[/dim]")
        tokenized_corpus = [code.lower().split() for code in self.codes]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Initialize Dense Model
        self.tokenizer = None
        self.dense_model = None
        self.corpus_embeddings = None

        self._init_dense_model()

    def _init_dense_model(self):
        cache_dir = Path("data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_cache_path = cache_dir / f"corpus_emb_{self.model_type}.npy"

        if self.model_type == "minilm":
            from sentence_transformers import SentenceTransformer

            console.print("[cyan]Loading sentence-transformers/all-MiniLM-L6-v2...[/cyan]")
            self.st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(self.device))
            self.st_model.max_seq_length = 256

            if emb_cache_path.exists():
                console.print(f"[green]Loading cached dense embeddings from {emb_cache_path}...[/green]")
                self.corpus_embeddings = np.load(emb_cache_path)
            else:
                console.print("[yellow]Encoding corpus embeddings with MiniLM...[/yellow]")
                self.corpus_embeddings = self.st_model.encode(
                    self.codes,
                    batch_size=128,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                np.save(emb_cache_path, self.corpus_embeddings)
        else:
            # CodeEmbed Shared Transformer
            tok_path = Path("tokenizer/tokenizer.json")
            self.tokenizer = CodeEmbedTokenizer(tok_path)

            if self.model_type == "shared_6l":
                ckpt_dir = Path("checkpoints/shared_6l_dense_clean")
                ckpt_path = ckpt_dir / "model_weights_only.pt" if (ckpt_dir / "model_weights_only.pt").exists() else ckpt_dir / "best_shared.pt"
                self.dense_model = SharedEncoder(
                    vocab_size=16000,
                    d_model=384,
                    n_layers=6,
                    n_heads=6,
                    d_ff=1536,
                    max_seq_len=256,
                    dropout=0.0,
                    num_modalities=2,
                ).to(self.device)
            else:  # shared_4l
                ckpt_dir = Path("checkpoints/shared_clean")
                ckpt_path = ckpt_dir / "model_weights_only.pt" if (ckpt_dir / "model_weights_only.pt").exists() else ckpt_dir / "best_shared.pt"
                self.dense_model = SharedEncoder(
                    vocab_size=16000,
                    d_model=256,
                    n_layers=4,
                    n_heads=8,
                    d_ff=1024,
                    max_seq_len=256,
                    dropout=0.0,
                    num_modalities=2,
                ).to(self.device)

            console.print(f"[cyan]Loading CodeEmbed {self.model_type} from {ckpt_path}...[/cyan]")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.dense_model.load_state_dict(state_dict)
            self.dense_model.eval()

            if emb_cache_path.exists():
                console.print(f"[green]Loading cached dense embeddings from {emb_cache_path}...[/green]")
                self.corpus_embeddings = np.load(emb_cache_path)
            else:
                console.print(f"[yellow]Encoding corpus embeddings with CodeEmbed {self.model_type}...[/yellow]")
                self.corpus_embeddings = self._encode_codes_codeembed(self.codes, batch_size=256)
                np.save(emb_cache_path, self.corpus_embeddings)

        console.print("[bold green]Search Engine Ready![/bold green]\n")

    def _encode_codes_codeembed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                tok_out = self.tokenizer.encode(batch, modality="code", max_length=256, padding=True, truncation=True)
                input_ids = tok_out["input_ids"].to(self.device)
                attn_mask = tok_out["attention_mask"].to(self.device)

                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    reps = self.dense_model(input_ids, attention_mask=attn_mask, modality_ids="code")
                all_embs.append(reps.cpu().numpy().astype(np.float32))
        return np.vstack(all_embs)

    def encode_query(self, query: str) -> np.ndarray:
        if self.model_type == "minilm":
            emb = self.st_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
            return emb[0]
        else:
            with torch.no_grad():
                tok_out = self.tokenizer.encode([query], modality="text", max_length=256, padding=True, truncation=True)
                input_ids = tok_out["input_ids"].to(self.device)
                attn_mask = tok_out["attention_mask"].to(self.device)
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    rep = self.dense_model(input_ids, attention_mask=attn_mask, modality_ids="text")
                return rep.cpu().numpy()[0]

    def search_dense(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_emb = self.encode_query(query)
        # Cosine similarity since both query and corpus are L2 normalized
        scores = np.dot(self.corpus_embeddings, q_emb)
        top_indices = np.argsort(-scores)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def search_bm25(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_tokens = query.lower().split()
        scores = self.bm25.get_scores(q_tokens)
        top_indices = np.argsort(-scores)[:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def search_hybrid(self, query: str, top_k: int = 5, k_rrf: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion (RRF) between Dense and BM25."""
        pool_k = min(len(self.codes), top_k * 5)
        dense_results = self.search_dense(query, top_k=pool_k)
        bm25_results = self.search_bm25(query, top_k=pool_k)

        rrf_scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_items

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> list[dict]:
        t0 = time.time()
        if mode == "dense":
            ranked = self.search_dense(query, top_k=top_k)
        elif mode == "bm25":
            ranked = self.search_bm25(query, top_k=top_k)
        else:  # hybrid
            ranked = self.search_hybrid(query, top_k=top_k)
        latency = (time.time() - t0) * 1000.0

        results = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            code = self.codes[idx]
            doc = self.docstrings[idx]
            jaccard = compute_jaccard(query, code)
            results.append({
                "rank": rank,
                "score": score,
                "code": code,
                "docstring": doc,
                "jaccard": jaccard,
                "latency_ms": latency,
            })
        return results


def display_results(query: str, mode: str, results: list[dict]):
    """Render search results to terminal using Rich."""
    divider = "=" * 72
    console.print(f"\n[bold magenta]{divider}[/bold magenta]")
    console.print(f"[bold cyan]Query:[/] [bold yellow]{query}[/bold yellow] | [bold cyan]Mode:[/] [bold green]{mode.upper()}[/bold green] | [bold cyan]Latency:[/] [dim]{results[0]['latency_ms']:.2f} ms[/dim]")
    console.print(f"[bold magenta]{divider}[/bold magenta]\n")

    for item in results:
        badge = get_stratum_badge(item["jaccard"])
        title = f"[bold white]Rank #{item['rank']}[/] (Score: {item['score']:.4f})  |  Stratum: {badge}"
        syntax = Syntax(item["code"].strip(), "python", theme="monokai", line_numbers=True)

        doc_snippet = item["docstring"].strip().replace("\n", " ")
        if len(doc_snippet) > 120:
            doc_snippet = doc_snippet[:117] + "..."
        caption = f"[dim italic]Docstring: {doc_snippet}[/dim italic]" if doc_snippet else None

        panel = Panel(syntax, title=title, subtitle=caption, border_style="cyan", padding=(0, 1))
        console.print(panel)


def main():
    parser = argparse.ArgumentParser(description="CodeEmbed Interactive Search Engine")
    parser.add_argument("--query", type=str, default="", help="Natural language search query")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="Retrieval mode (hybrid, dense, bm25)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["shared_6l", "shared_4l", "minilm"],
        default="shared_6l",
        help="Dense model architecture to use",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved results")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive REPL mode")
    parser.add_argument("--corpus", type=str, default="data/processed_clean_v2/test_stratified.parquet", help="Corpus dataset")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        console.print(f"[bold red]Corpus file not found:[/] {corpus_path}")
        sys.exit(1)

    df = pd.read_parquet(corpus_path)
    engine = SearchEngine(corpus_df=df, model_type=args.model)

    if args.interactive or not args.query:
        console.print("[bold green]Welcome to CodeEmbed Interactive Search CLI![/bold green]")
        console.print("[dim]Type your natural language query or ':mode <hybrid|dense|bm25>' or ':q' to exit.[/dim]\n")

        current_mode = args.mode
        while True:
            try:
                user_input = Prompt.ask("[bold yellow]Search Query[/bold yellow]").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue
            if user_input.lower() in [":q", "quit", "exit"]:
                console.print("[cyan]Exiting CodeEmbed Search. Goodbye![/cyan]")
                break
            if user_input.startswith(":mode "):
                new_mode = user_input.split(maxsplit=1)[1].lower()
                if new_mode in ["hybrid", "dense", "bm25"]:
                    current_mode = new_mode
                    console.print(f"[green]Mode set to {current_mode.upper()}[/green]")
                else:
                    console.print("[red]Invalid mode. Choose: hybrid, dense, bm25[/red]")
                continue

            results = engine.search(user_input, mode=current_mode, top_k=args.top_k)
            display_results(user_input, current_mode, results)
    else:
        results = engine.search(args.query, mode=args.mode, top_k=args.top_k)
        display_results(args.query, args.mode, results)


if __name__ == "__main__":
    main()
