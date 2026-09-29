"""CodeEmbed Interactive Web Search Interface (FastAPI + Modern Web UI).

Features:
- Instant multi-model selection:
    1. CodeEmbed 6L (17.03M Scaled - Exploratory Best)
    2. CodeEmbed 4L (7.38M Shared - Confirmatory Clean)
    3. sentence-transformers/all-MiniLM-L6-v2 (22.7M Pretrained Peer)
- Three retrieval modes: Hybrid (RRF), Dense Semantic, BM25 Lexical.
- Token-overlap stratification badges (Zero-Overlap, Low-Overlap, High-Overlap).
- Syntax-highlighted code blocks with copy-to-clipboard.
- Interactive query suggestions and live latency timer.

Run:
    uv run python demo/app.py
    # Open http://127.0.0.1:8000
"""

import sys
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from demo.search import SearchEngine

app = FastAPI(title="CodeEmbed Search Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine cache
_engines: dict[str, SearchEngine] = {}
_corpus_df: pd.DataFrame | None = None


def get_corpus() -> pd.DataFrame:
    global _corpus_df
    if _corpus_df is None:
        data_path = Path("data/processed_clean_v2/test_stratified.parquet")
        if not data_path.exists():
            data_path = Path("data/processed/test_stratified.parquet")
        _corpus_df = pd.read_parquet(data_path)
    return _corpus_df


def get_engine(model_type: str = "shared_6l") -> SearchEngine:
    if model_type not in _engines:
        df = get_corpus()
        _engines[model_type] = SearchEngine(corpus_df=df, model_type=model_type)
    return _engines[model_type]


class SearchRequest(BaseModel):
    query: str
    model: str = "shared_6l"
    mode: str = "hybrid"
    top_k: int = 5


@app.on_event("startup")
def startup_event():
    # Warm up default model on startup
    get_engine("shared_6l")


@app.get("/api/search")
def api_search(
    q: str = Query(..., description="Search query"),
    model: str = Query("shared_6l", description="Model: shared_6l, shared_4l, minilm"),
    mode: str = Query("hybrid", description="Mode: hybrid, dense, bm25"),
    top_k: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    engine = get_engine(model)
    results = engine.search(q, mode=mode, top_k=top_k)
    return {
        "query": q,
        "model": model,
        "mode": mode,
        "count": len(results),
        "results": results,
    }


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CodeEmbed - Neural & Hybrid Code Search</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
  <style>
    body { background-color: #0f172a; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
    .glass-card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .glass-card:hover { border-color: rgba(56, 189, 248, 0.3); }
    pre code { border-radius: 0.5rem; font-size: 0.875rem; }
  </style>
</head>
<body class="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
  <div class="max-w-5xl mx-auto">
    
    <!-- Header -->
    <div class="text-center mb-8">
      <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-semibold uppercase tracking-wider mb-3">
        <span>Protocol v1.1 Frozen Benchmark</span>
      </div>
      <h1 class="text-4xl font-extrabold tracking-tight sm:text-5xl text-white">
        Code<span class="text-sky-400">Embed</span> Search
      </h1>
      <p class="mt-2 text-base text-slate-400">
        Empirical evaluation of scratch-built Transformer dense embeddings, BM25 lexical retrieval, and RRF hybrid search.
      </p>
    </div>

    <!-- Search Controls Card -->
    <div class="glass-card rounded-2xl p-6 shadow-xl mb-8">
      <!-- Input Box -->
      <div class="relative flex items-center mb-4">
        <input id="query-input" type="text" 
               placeholder="Describe code functionality (e.g., 'calculate md5 hash of string', 'download file with retry')..." 
               class="w-full pl-4 pr-32 py-3.5 bg-slate-900/90 border border-slate-700/80 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-transparent text-base">
        <button id="search-btn" onclick="executeSearch()"
                class="absolute right-2 px-5 py-2 bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold rounded-lg text-sm transition-all shadow-md">
          Search
        </button>
      </div>

      <!-- Quick Query Suggestions -->
      <div class="flex flex-wrap items-center gap-2 mb-6 text-xs text-slate-400">
        <span class="font-medium text-slate-500">Quick queries:</span>
        <button onclick="setQuery('calculate md5 hash of string')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300">calculate md5 hash of string</button>
        <button onclick="setQuery('parse json safely with fallback')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300">parse json safely with fallback</button>
        <button onclick="setQuery('download file from url with progress bar')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300">download file from url</button>
        <button onclick="setQuery('matrix multiplication without numpy')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300">matrix multiplication</button>
      </div>

      <!-- Filters & Selectors -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-slate-800">
        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1">Architecture / Model</label>
          <select id="model-select" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-1 focus:ring-sky-500">
            <option value="shared_6l" selected>CodeEmbed 6L (17.0M Scaled - MRR 0.4699)</option>
            <option value="shared_4l">CodeEmbed 4L (7.38M Confirmatory - MRR 0.4157)</option>
            <option value="minilm">all-MiniLM-L6-v2 (22.7M Pretrained - MRR 0.5837)</option>
          </select>
        </div>

        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1">Retrieval Mode</label>
          <select id="mode-select" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-1 focus:ring-sky-500">
            <option value="hybrid" selected>Hybrid (Dense + BM25 RRF)</option>
            <option value="dense">Dense Semantic Only</option>
            <option value="bm25">BM25 Lexical Only</option>
          </select>
        </div>

        <div>
          <label class="block text-xs font-medium text-slate-400 mb-1">Top-K Results</label>
          <select id="topk-select" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-1 focus:ring-sky-500">
            <option value="3">Top 3 Results</option>
            <option value="5" selected>Top 5 Results</option>
            <option value="10">Top 10 Results</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Status / Latency Bar -->
    <div id="status-bar" class="hidden flex items-center justify-between px-4 py-2 bg-slate-800/60 rounded-xl text-xs text-slate-400 mb-6">
      <div id="status-text">Found 5 results</div>
      <div id="latency-text" class="text-sky-400 font-mono">12.4 ms</div>
    </div>

    <!-- Results Container -->
    <div id="results-container" class="space-y-6">
      <!-- Search results will be rendered here dynamically -->
    </div>

  </div>

  <script>
    function setQuery(text) {
      document.getElementById('query-input').value = text;
      executeSearch();
    }

    document.getElementById('query-input').addEventListener('keydown', function(e) {
      if (e.key === 'Enter') {
        executeSearch();
      }
    });

    async function executeSearch() {
      const q = document.getElementById('query-input').value.trim();
      if (!q) return;

      const model = document.getElementById('model-select').value;
      const mode = document.getElementById('mode-select').value;
      const topK = document.getElementById('topk-select').value;

      const btn = document.getElementById('search-btn');
      btn.innerText = 'Searching...';
      btn.disabled = true;

      const statusBar = document.getElementById('status-bar');
      const statusText = document.getElementById('status-text');
      const latencyText = document.getElementById('latency-text');
      const container = document.getElementById('results-container');

      container.innerHTML = '<div class="text-center py-12 text-slate-500 text-sm">Searching code representations...</div>';

      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(q)}&model=${model}&mode=${mode}&top_k=${topK}`);
        const data = await res.json();

        container.innerHTML = '';
        statusBar.classList.remove('hidden');

        const lat = data.results.length > 0 ? data.results[0].latency_ms.toFixed(1) : '0';
        statusText.innerText = `Showing top ${data.results.length} code matches for "${data.query}"`;
        latencyText.innerText = `${lat} ms`;

        data.results.forEach((item) => {
          let stratumBadge = '';
          if (item.jaccard === 0.0) {
            stratumBadge = '<span class="px-2 py-0.5 rounded text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Zero-Overlap (Pure Semantic)</span>';
          } else if (item.jaccard <= 0.15) {
            stratumBadge = `<span class="px-2 py-0.5 rounded text-[11px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20">Low-Overlap (J=${item.jaccard.toFixed(2)})</span>`;
          } else {
            stratumBadge = `<span class="px-2 py-0.5 rounded text-[11px] font-semibold bg-blue-500/10 text-blue-400 border border-blue-500/20">High-Overlap (J=${item.jaccard.toFixed(2)})</span>`;
          }

          const card = document.createElement('div');
          card.className = 'glass-card rounded-xl p-5 shadow-lg space-y-3';
          card.innerHTML = `
            <div class="flex items-center justify-between flex-wrap gap-2">
              <div class="flex items-center gap-2">
                <span class="px-2.5 py-1 rounded-md text-xs font-bold bg-sky-500/20 text-sky-400 font-mono">#${item.rank}</span>
                <span class="text-xs text-slate-400 font-mono">Score: ${item.score.toFixed(4)}</span>
                ${stratumBadge}
              </div>
              <button onclick="navigator.clipboard.writeText(decodeURIComponent(this.getAttribute('data-code')))" data-code="${encodeURIComponent(item.code)}"
                      class="px-2.5 py-1 text-xs font-medium text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 rounded transition-all">
                Copy Code
              </button>
            </div>
            ${item.docstring ? `<p class="text-xs text-slate-300 italic border-l-2 border-slate-700 pl-3 py-0.5">"${item.docstring.replace(/"/g, '&quot;')}"</p>` : ''}
            <div class="relative">
              <pre><code class="language-python">${escapeHtml(item.code)}</code></pre>
            </div>
          `;
          container.appendChild(card);
        });

        // Trigger syntax highlight
        hljs.highlightAll();

      } catch (err) {
        container.innerHTML = `<div class="text-center py-12 text-rose-400 text-sm">Error executing search: ${err}</div>`;
      } finally {
        btn.innerText = 'Search';
        btn.disabled = false;
      }
    }

    function escapeHtml(string) {
      return String(string).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_CONTENT)


if __name__ == "__main__":
    uvicorn.run("demo.app:app", host="127.0.0.1", port=8000, reload=False)
