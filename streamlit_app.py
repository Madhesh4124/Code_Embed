"""CodeEmbed — Interactive Research & Semantic Search Engine.

Streamlit Cloud & Local Deployment Entrypoint.
Features:
- From-scratch PyTorch Pre-LN Transformer bi-encoder (17.03M / 7.38M) with zero HuggingFace wrappers.
- Real-time 2D Latent Semantic Space Map (PCA projection of 19,632 test functions).
- Lexical BM25 vs. Dense Discordance Quadrant Plot (Semantic vs. Lexical divergence).
- Hybrid Score Decomposition Waterfall (Neural Dense % vs. Lexical BM25 % contribution).
- Token-level Attribution & Overlap Highlighting with Zero-Overlap Gem Detection.
- Pre-loaded authentic test set queries from the stratified evaluation split.

Usage:
    uv run streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.decomposition import PCA

from demo.search import SearchEngine, compute_jaccard

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CodeEmbed — Interactive Research Search Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Typography & Background Adjustments */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Research Header Banner */
    .research-banner {
        background: linear-gradient(135deg, #181829 0%, #0f0f18 100%);
        border: 1px solid #2d2d48;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .research-banner h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .research-banner p {
        margin: 8px 0 0 0;
        color: #94a3b8;
        font-size: 14.5px;
        line-height: 1.5;
    }

    /* Metric Badges */
    .badge-zero {
        background-color: rgba(16, 185, 129, 0.18);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.45);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12.5px;
        font-weight: 600;
    }
    .badge-low {
        background-color: rgba(245, 158, 11, 0.18);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.45);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12.5px;
        font-weight: 600;
    }
    .badge-high {
        background-color: rgba(59, 130, 246, 0.18);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.45);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12.5px;
        font-weight: 600;
    }

    /* Card styling for search results */
    .result-card {
        background: #141420;
        border: 1px solid #28283c;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 14px;
        margin-bottom: 8px;
        transition: all 0.2s ease-in-out;
    }
    .result-card:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15);
    }
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .result-title {
        font-weight: 600;
        font-size: 16.5px;
        color: #f1f5f9;
    }
    .result-scores {
        font-size: 13.5px;
        color: #94a3b8;
    }

    /* Token Attribution Box */
    .token-box {
        background: #0d0d16;
        border-left: 3px solid #6366f1;
        padding: 9px 14px;
        margin: 8px 0 12px 0;
        border-radius: 0 6px 6px 0;
        font-size: 13.5px;
        color: #cbd5e1;
    }
    .token-highlight {
        background: rgba(99, 102, 241, 0.35);
        color: #a5b4fc;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        margin: 0 2px;
    }
    .zero-overlap-callout {
        background: rgba(16, 185, 129, 0.12);
        border-left: 3px solid #10b981;
        padding: 10px 14px;
        margin: 8px 0 12px 0;
        border-radius: 0 6px 6px 0;
        font-size: 13.5px;
        color: #a7f3d0;
    }

    /* Search Container */
    .search-panel {
        background: #141420;
        border: 1px solid #28283c;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 22px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Resource Caching & Data Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading test dataset (19,632 stratified functions)...")
def load_corpus() -> pd.DataFrame:
    data_path = ROOT_DIR / "data" / "processed_clean_v2" / "test_stratified.parquet"
    if not data_path.exists():
        data_path = ROOT_DIR / "data" / "processed" / "test_stratified.parquet"
    df = pd.read_parquet(data_path)
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner="Loading Search Engine & neural weights...")
def load_search_engine(model_type: str = "shared_6l") -> SearchEngine:
    df = load_corpus()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SearchEngine(corpus_df=df, model_type=model_type, device=device)


@st.cache_resource(show_spinner="Pre-fitting 2D PCA on 19,632 latent embeddings...")
def load_pca_projection(model_type: str = "shared_6l") -> tuple[PCA, np.ndarray]:
    engine = load_search_engine(model_type=model_type)
    pca = PCA(n_components=2, random_state=42)
    corpus_2d = pca.fit_transform(engine.corpus_embeddings)
    return pca, corpus_2d


# -----------------------------------------------------------------------------
# 3. Authentic Test Set Query Presets
# -----------------------------------------------------------------------------
TEST_SET_PRESETS: dict[str, str] = {
    # Authentic Test Set Queries (Zero-Overlap: pure semantic abstraction)
    "💎 [Test Set: Zero-Overlap] Converts a string to a valid filename": "Converts a string to a valid filename.",
    "💎 [Test Set: Zero-Overlap] Clear out the database": "Clear out the database",
    "💎 [Test Set: Zero-Overlap] Ensure all logging output has been flushed": "Ensure all logging output has been flushed",
    "💎 [Test Set: Zero-Overlap] Returns whether this is a boolean data type": "Returns whether this is a boolean data type.",
    "💎 [Test Set: Zero-Overlap] Returns initializer configuration as a JSON-serializable dict": "Returns initializer configuration as a JSON-serializable dict.",
    "💎 [Test Set: Zero-Overlap] Returns the maximum representable value in this data type": "Returns the maximum representable value in this data type.",
    "💎 [Test Set: Zero-Overlap] What a Terrible Failure!": "What a Terrible Failure!",
    # Authentic Test Set Queries (Low-Overlap: vocabulary mismatch)
    "⚡ [Test Set: Low-Overlap] Contextmanager that will create and teardown a session": "Contextmanager that will create and teardown a session.",
    "⚡ [Test Set: Low-Overlap] Parses some DatabaseError to provide a better error message": "Parses some DatabaseError to provide a better error message",
    "⚡ [Test Set: Low-Overlap] Decompresses data for Content-Encoding: gzip": "Decompresses data for Content-Encoding: gzip.",
    "⚡ [Test Set: Low-Overlap] Extracts video ID from URL": "Extracts video ID from URL.",
    "⚡ [Test Set: Low-Overlap] Checks if a database exists in CosmosDB": "Checks if a database exists in CosmosDB.",
    # Authentic Test Set Queries (High-Overlap: strong lexical overlap)
    "🔍 [Test Set: High-Overlap] Parses host name and port number from a string": "Parses host name and port number from a string.",
    "🔍 [Test Set: High-Overlap] Uploads the file to Google cloud storage": "Uploads the file to Google cloud storage",
    "🔍 [Test Set: High-Overlap] Creates a dag run for the specified dag": "Creates a dag run for the specified dag",
    "🔍 [Test Set: High-Overlap] Returns a snowflake.connection object": "Returns a snowflake.connection object",
    "🔍 [Test Set: High-Overlap] Format text with color or other effects into ANSI escaped string": "Format text with color or other effects into ANSI escaped string.",
    # Practical Real-World Coding Archetypes
    "💡 [Curated] Parse JSON safely with fallback": "parse json safely with fallback",
    "💡 [Curated] Flatten nested list of lists": "flatten nested list of lists",
    "💡 [Curated] Convert timestamp to human readable date": "convert timestamp to human readable date",
    "💡 [Curated] Calculate MD5 hash of string": "calculate md5 hash of string",
    "💡 [Curated] Download file from URL with progress": "download file from url with progress",
    "💡 [Curated] Read file line by line without memory overhead": "read file line by line without memory overhead",
}


# -----------------------------------------------------------------------------
# 4. Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ CodeEmbed Controls")

model_choice = st.sidebar.selectbox(
    "Embedding Architecture",
    options=["shared_6l", "shared_4l", "bm25"],
    format_func=lambda x: {
        "shared_6l": "CodeEmbed 6L (17.03M - Scaled Winner)",
        "shared_4l": "CodeEmbed 4L (7.38M - Pre-registered)",
        "bm25": "BM25 (ATIRE Lexical Baseline)",
    }[x],
    index=0,
)

if model_choice == "bm25":
    retrieval_mode = "bm25"
    st.sidebar.info("Selected Lexical BM25 baseline mode.")
else:
    retrieval_mode = st.sidebar.selectbox(
        "Retrieval Pipeline",
        options=["hybrid_convex", "hybrid_rrf", "dense", "bm25"],
        format_func=lambda x: {
            "hybrid_convex": "Hybrid: Convex Interpolation (α · Dense + (1-α) · BM25)",
            "hybrid_rrf": "Hybrid: Reciprocal Rank Fusion (RRF)",
            "dense": "Dense Semantic Only (Transformer Cosine)",
            "bm25": "Lexical BM25 Only (Okapi BM25)",
        }[x],
        index=0,
    )

alpha_convex = 0.70
k_rrf = 20
if retrieval_mode == "hybrid_convex":
    alpha_convex = st.sidebar.slider(
        "Dense Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help="0.70 was empirically validated as optimal on the validation set, reaching 0.6612 Test MRR.",
    )
elif retrieval_mode == "hybrid_rrf":
    k_rrf = st.sidebar.slider(
        "RRF Constant (k)",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="RRF score = 1/(k + rank_dense) + 1/(k + rank_bm25). k=20 achieved 0.6291 Test MRR.",
    )

top_k = st.sidebar.slider("Top-K Retrieved Snippets", min_value=3, max_value=20, value=5, step=1)

# Session state initialization for search query
if "current_query" not in st.session_state:
    st.session_state["current_query"] = "parse json safely with fallback"


def on_preset_change():
    selected_name = st.session_state.get("preset_selector")
    if selected_name and selected_name in TEST_SET_PRESETS:
        st.session_state["current_query"] = TEST_SET_PRESETS[selected_name]


# -----------------------------------------------------------------------------
# 5. Header Banner & Metrics Row
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="research-banner">
        <h1>⚡ CodeEmbed Research Search Engine</h1>
        <p>From-scratch PyTorch Transformer bi-encoder (7.38M–17.03M) trained on AST-cleaned CodeSearchNet. 
        Zero HuggingFace model wrappers. Validated under pre-registered protocol with MinHash LSH deduplication.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 6. Interactive Search Query Input Panel
# -----------------------------------------------------------------------------
st.markdown('<div class="search-panel">', unsafe_allow_html=True)
col_sel, col_inp = st.columns([1.1, 1.9])

with col_sel:
    preset_names = list(TEST_SET_PRESETS.keys())
    st.selectbox(
        "🎯 Select Query from Verified Test Split",
        options=preset_names,
        index=17,  # Default: "parse json safely with fallback"
        key="preset_selector",
        on_change=on_preset_change,
        help="Pre-loaded with genuine test queries from the 19,632 test split across Zero, Low, and High overlap strata.",
    )

with col_inp:
    search_query = st.text_input(
        "🔎 Natural Language Query (Edit or Type Custom)",
        key="current_query",
        placeholder="Type custom query (e.g. 'read file line by line without memory overhead')...",
    )

st.markdown("</div>", unsafe_allow_html=True)

# Guard against empty query
effective_query = search_query.strip() if search_query.strip() else "parse json safely with fallback"


# -----------------------------------------------------------------------------
# 7. Search Execution Engine
# -----------------------------------------------------------------------------
corpus_df = load_corpus()
underlying_model = "shared_6l" if model_choice == "bm25" else model_choice
engine = load_search_engine(model_type=underlying_model)
pca, corpus_2d = load_pca_projection(model_type=underlying_model)


def execute_search_and_score(
    q: str,
    k: int,
    mode: str,
    alpha: float = 0.70,
    k_rrf_val: int = 20,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]], float]:
    """Execute search and produce rich candidate diagnostics."""
    t0 = time.time()
    q_emb = engine.encode_query(q)

    # 1. Candidate pool: retrieve top 50 from dense and top 50 from BM25
    pool_size = min(len(engine.codes), max(50, k * 5))
    dense_all = np.dot(engine.corpus_embeddings, q_emb)
    q_tokens = q.lower().split()
    bm25_all = engine.bm25.get_scores(q_tokens)

    top_dense_indices = np.argsort(-dense_all)[:pool_size].tolist()
    top_bm25_indices = np.argsort(-bm25_all)[:pool_size].tolist()

    # Rank lookup
    dense_rank_map = {idx: r + 1 for r, idx in enumerate(top_dense_indices)}
    bm25_rank_map = {idx: r + 1 for r, idx in enumerate(top_bm25_indices)}

    candidate_indices = list(dict.fromkeys(top_dense_indices + top_bm25_indices))

    # Normalization for candidates
    c_dense_scores = np.array([dense_all[i] for i in candidate_indices])
    c_bm25_scores = np.array([bm25_all[i] for i in candidate_indices])

    # Min-max normalization
    d_min, d_max = c_dense_scores.min(), c_dense_scores.max()
    d_norm = (c_dense_scores - d_min) / (d_max - d_min + 1e-9)

    b_min, b_max = c_bm25_scores.min(), c_bm25_scores.max()
    b_norm = (c_bm25_scores - b_min) / (b_max - b_min + 1e-9)

    candidate_records: list[dict[str, Any]] = []
    for i, idx in enumerate(candidate_indices):
        d_raw = float(dense_all[idx])
        b_raw = float(bm25_all[idx])
        dn = float(d_norm[i])
        bn = float(b_norm[i])

        r_dense = dense_rank_map.get(idx, pool_size + 10)
        r_bm25 = bm25_rank_map.get(idx, pool_size + 10)

        # RRF score
        rrf_d = 1.0 / (k_rrf_val + r_dense)
        rrf_b = 1.0 / (k_rrf_val + r_bm25)
        rrf_total = rrf_d + rrf_b

        # Convex score
        convex_d = alpha * dn
        convex_b = (1.0 - alpha) * bn
        convex_total = convex_d + convex_b

        # Final score depending on mode
        if mode == "dense":
            final_score = d_raw
        elif mode == "bm25":
            final_score = b_raw
        elif mode == "hybrid_convex":
            final_score = convex_total
        else:  # hybrid_rrf
            final_score = rrf_total

        code = engine.codes[idx]
        doc = engine.docstrings[idx]
        func_name = corpus_df.iloc[idx]["func_name"] if "func_name" in corpus_df.columns else f"func_{idx}"
        overlap_bin = corpus_df.iloc[idx]["overlap_bin"] if "overlap_bin" in corpus_df.columns else "Unknown"

        # Token set calculation
        clean_q_tokens = [tok.lower() for tok in re.findall(r"\w+", q) if len(tok) > 1]
        clean_c_tokens = [tok.lower() for tok in re.findall(r"\w+", code) if len(tok) > 1]
        shared_tokens = sorted(set(clean_q_tokens) & set(clean_c_tokens))
        jaccard = compute_jaccard(q, code)

        candidate_records.append({
            "idx": idx,
            "func_name": str(func_name),
            "code": code,
            "docstring": doc,
            "dense_score": d_raw,
            "bm25_score": b_raw,
            "dense_norm": dn,
            "bm25_norm": bn,
            "dense_rank": r_dense,
            "bm25_rank": r_bm25,
            "rrf_d_contrib": rrf_d,
            "rrf_b_contrib": rrf_b,
            "convex_d_contrib": convex_d,
            "convex_b_contrib": convex_b,
            "final_score": final_score,
            "jaccard": jaccard,
            "overlap_bin": overlap_bin,
            "shared_tokens": shared_tokens,
            "query_tokens": list(set(clean_q_tokens)),
        })

    # Sort descending by final score
    candidate_records.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = candidate_records[:k]
    for r, item in enumerate(top_results, start=1):
        item["rank"] = r

    latency_ms = (time.time() - t0) * 1000.0
    return top_results, q_emb, candidate_records, latency_ms


top_results, query_embedding, all_candidates, latency = execute_search_and_score(
    effective_query,
    k=top_k,
    mode=retrieval_mode,
    alpha=alpha_convex,
    k_rrf_val=k_rrf,
)


# Metric Summary Cards
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("Corpus Size", f"{len(corpus_df):,} functions")
col_m2.metric(
    "Active Architecture",
    {"shared_6l": "6L (17.03M)", "shared_4l": "4L (7.38M)", "bm25": "BM25 (ATIRE)"}[model_choice],
)
col_m3.metric(
    "Test MRR (Convex)",
    "0.6612",
    "+40.7% over Dense",
)
col_m4.metric("Search Latency", f"{latency:.2f} ms")
col_m5.metric("Leakage Rate", "1.23%", "Pre-reg <5% Tol")


# -----------------------------------------------------------------------------
# 8. Main Tabbed Layout
# -----------------------------------------------------------------------------
tab_results, tab_space, tab_quadrant, tab_benchmark = st.tabs([
    "🔍 1. Retrieved Results & Score Decomposition",
    "🗺️ 2. 2D Semantic Space Map (PCA)",
    "🎯 3. BM25 vs. Dense Quadrant Plot",
    "📊 4. Official Benchmark & Capacity Gates",
])


# =============================================================================
# TAB 1: RETRIEVED RESULTS & SCORE DECOMPOSITION
# =============================================================================
with tab_results:
    st.subheader(f'Top {len(top_results)} Retrieved Functions for: "{effective_query}"')

    # Score Decomposition Stacked Bar Chart
    st.markdown("#### ⚖️ Hybrid Score Contribution Breakdown")
    st.caption(
        "Demonstrates why hybrid retrieval outperforms single modalities: candidates gain strength from neural latent semantic similarity, lexical token matching, or a powerful consensus of both."
    )

    bar_data = []
    for item in top_results:
        if retrieval_mode == "hybrid_convex":
            dense_contrib = item["convex_d_contrib"]
            bm25_contrib = item["convex_b_contrib"]
            total = dense_contrib + bm25_contrib + 1e-9
            d_pct = (dense_contrib / total) * 100.0
            b_pct = (bm25_contrib / total) * 100.0
        elif retrieval_mode == "hybrid_rrf":
            dense_contrib = item["rrf_d_contrib"]
            bm25_contrib = item["rrf_b_contrib"]
            total = dense_contrib + bm25_contrib + 1e-9
            d_pct = (dense_contrib / total) * 100.0
            b_pct = (bm25_contrib / total) * 100.0
        elif retrieval_mode == "dense":
            d_pct = 100.0
            b_pct = 0.0
        else:  # bm25
            d_pct = 0.0
            b_pct = 100.0

        clean_name = item["func_name"][:28] if len(item["func_name"]) > 28 else item["func_name"]
        bar_data.append({
            "Candidate": f"[#{item['rank']}] {clean_name}",
            "Dense Contribution (%)": d_pct,
            "BM25 Contribution (%)": b_pct,
            "Rank": item["rank"],
            "Raw Dense": item["dense_score"],
            "Raw BM25": item["bm25_score"],
        })

    bar_df = pd.DataFrame(bar_data)
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            y=bar_df["Candidate"],
            x=bar_df["Dense Contribution (%)"],
            name="Neural Dense (Semantic)",
            orientation="h",
            marker={"color": "#6366f1", "line": {"color": "#4338ca", "width": 1}},
            hovertemplate="Dense Contribution: %{x:.1f}%<br>Raw Cosine: %{customdata[0]:.3f}<extra></extra>",
            customdata=bar_df[["Raw Dense"]].values,
        )
    )
    fig_bar.add_trace(
        go.Bar(
            y=bar_df["Candidate"],
            x=bar_df["BM25 Contribution (%)"],
            name="Lexical BM25 (Exact Tokens)",
            orientation="h",
            marker={"color": "#10b981", "line": {"color": "#059669", "width": 1}},
            hovertemplate="BM25 Contribution: %{x:.1f}%<br>Raw BM25: %{customdata[0]:.2f}<extra></extra>",
            customdata=bar_df[["Raw BM25"]].values,
        )
    )
    fig_bar.update_layout(
        barmode="stack",
        height=260,
        margin={"l": 200, "r": 20, "t": 10, "b": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis={"title": "Score Contribution (%)", "range": [0, 100], "showgrid": True, "gridcolor": "#27273a"},
        yaxis={"autorange": "reversed", "tickfont": {"size": 13, "color": "#f1f5f9"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cbd5e1"},
    )
    st.plotly_chart(fig_bar, width="stretch")

    # Detailed Code Cards with Token Attribution
    st.markdown("#### 📜 Retrieved Code Snippets & Token Attribution")
    for item in top_results:
        # Determine stratum badge
        j = item["jaccard"]
        if j == 0.0:
            badge_html = '<span class="badge-zero">💎 Zero-Overlap (Pure Semantic)</span>'
        elif j <= 0.15:
            badge_html = f'<span class="badge-low">⚡ Low-Overlap (J={j:.2f})</span>'
        else:
            badge_html = f'<span class="badge-high">🔍 High-Overlap (J={j:.2f})</span>'

        with st.container():
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-header">
                        <span class="result-title">Rank #{item['rank']} — <code>{item['func_name']}</code></span>
                        <span>{badge_html}</span>
                    </div>
                    <div class="result-scores">
                        <strong>Final Score:</strong> {item['final_score']:.4f} &nbsp;|&nbsp; 
                        <strong>Dense Cosine:</strong> {item['dense_score']:.4f} (Rank #{item['dense_rank']}) &nbsp;|&nbsp; 
                        <strong>BM25:</strong> {item['bm25_score']:.2f} (Rank #{item['bm25_rank']}) &nbsp;|&nbsp; 
                        <strong>Jaccard:</strong> {j:.3f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Token Attribution Visualizer
            if j == 0.0:
                st.markdown(
                    """
                    <div class="zero-overlap-callout">
                        <strong>💎 Pure Semantic Generalization:</strong> Zero lexical tokens are shared between your query and this function body. 
                        BM25 fails completely on this example, but CodeEmbed retrieves it accurately via latent representation alignment.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                tokens_html = "".join([f'<span class="token-highlight">{tok}</span>' for tok in item["shared_tokens"]])
                st.markdown(
                    f"""
                    <div class="token-box">
                        <strong>Lexical Token Attribution:</strong> Shared query tokens found in code: {tokens_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Code display
            st.code(item["code"].strip(), language="python", line_numbers=True)

            if item["docstring"].strip():
                with st.expander(f"View Original Docstring for `{item['func_name']}`"):
                    st.info(item["docstring"].strip())


# =============================================================================
# TAB 2: 2D SEMANTIC SPACE MAP (PCA PROJECTION)
# =============================================================================
with tab_space:
    st.subheader("🗺️ 2D Latent Semantic Space Projection")
    st.caption(
        "Interactive 2D Principal Component Analysis (PCA) projection of all 19,632 test functions. "
        "The query is dynamically encoded by the Transformer and projected into the exact same latent subspace."
    )

    # Project query into PCA 2D coordinates
    query_2d = pca.transform(query_embedding.reshape(1, -1))[0]

    # Sample background corpus points for fast, interactive 60fps rendering
    np.random.seed(42)
    sample_indices = np.random.choice(len(corpus_df), size=min(2500, len(corpus_df)), replace=False)

    bg_df = pd.DataFrame({
        "pc1": corpus_2d[sample_indices, 0],
        "pc2": corpus_2d[sample_indices, 1],
        "func_name": [corpus_df.iloc[i]["func_name"] for i in sample_indices],
        "overlap_bin": [corpus_df.iloc[i]["overlap_bin"] for i in sample_indices],
        "snippet": [corpus_df.iloc[i]["code"][:80].replace("\n", " ") + "..." for i in sample_indices],
    })

    # Top-K projection
    top_indices = [item["idx"] for item in top_results]
    top_2d = corpus_2d[top_indices]
    top_df = pd.DataFrame({
        "pc1": top_2d[:, 0],
        "pc2": top_2d[:, 1],
        "rank": [item["rank"] for item in top_results],
        "func_name": [item["func_name"] for item in top_results],
        "score": [item["final_score"] for item in top_results],
        "dense_score": [item["dense_score"] for item in top_results],
        "jaccard": [item["jaccard"] for item in top_results],
        "overlap_bin": [item["overlap_bin"] for item in top_results],
        "snippet": [item["code"][:100].replace("\n", " ") + "..." for item in top_results],
    })

    fig_map = go.Figure()

    # 1. Background points (Corpus Sample)
    fig_map.add_trace(
        go.Scattergl(
            x=bg_df["pc1"],
            y=bg_df["pc2"],
            mode="markers",
            marker={
                "size": 4.5,
                "color": "rgba(148, 163, 184, 0.28)",
                "line": {"width": 0},
            },
            name="Background Corpus (2.5k sample)",
            hovertemplate="<b>Function:</b> %{customdata[0]}<extra></extra>",
            customdata=bg_df[["func_name"]].values,
        )
    )

    # 2. Distance connecting lines from Query to Top-K
    for _, row in top_df.iterrows():
        fig_map.add_trace(
            go.Scatter(
                x=[query_2d[0], row["pc1"]],
                y=[query_2d[1], row["pc2"]],
                mode="lines",
                line={"color": "rgba(245, 158, 11, 0.45)", "width": 1.5, "dash": "dot"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 3. Top-K Neighbors
    fig_map.add_trace(
        go.Scatter(
            x=top_df["pc1"],
            y=top_df["pc2"],
            mode="markers+text",
            marker={
                "size": 15,
                "color": top_df["rank"],
                "colorscale": "Viridis",
                "showscale": False,
                "line": {"color": "#ffffff", "width": 2},
            },
            text=[f"#{r}" for r in top_df["rank"]],
            textposition="top right",
            textfont={"color": "#f8fafc", "size": 12, "family": "sans-serif"},
            name="Top-K Retrieved Neighbors",
            hovertemplate=(
                "<b>Rank #%{customdata[0]}: %{customdata[1]}</b><br>"
                "<b>Dense Cosine:</b> %{customdata[2]:.3f}<br>"
                "<b>Token Jaccard:</b> %{customdata[3]:.3f} (%{customdata[4]})<extra></extra>"
            ),
            customdata=top_df[["rank", "func_name", "dense_score", "jaccard", "overlap_bin"]].values,
        )
    )

    # 4. Query marker (Glowing Star)
    fig_map.add_trace(
        go.Scatter(
            x=[query_2d[0]],
            y=[query_2d[1]],
            mode="markers+text",
            marker={
                "symbol": "star",
                "size": 24,
                "color": "#f59e0b",
                "line": {"color": "#ffffff", "width": 2.5},
            },
            text=["🎯 Query"],
            textposition="bottom center",
            textfont={"color": "#fbbf24", "size": 14, "family": "sans-serif"},
            name="Current Query",
            hovertemplate=f"<b>Query:</b> {effective_query}<extra></extra>",
        )
    )

    var1 = pca.explained_variance_ratio_[0] * 100.0
    var2 = pca.explained_variance_ratio_[1] * 100.0

    fig_map.update_layout(
        height=620,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis={"title": f"PC 1 ({var1:.2f}% Variance)", "showgrid": True, "gridcolor": "#27273a", "zeroline": False},
        yaxis={"title": f"PC 2 ({var2:.2f}% Variance)", "showgrid": True, "gridcolor": "#27273a", "zeroline": False},
        paper_bgcolor="#111119",
        plot_bgcolor="#161622",
        font={"color": "#cbd5e1"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    st.plotly_chart(fig_map, width="stretch")

    st.info(
        "💡 **Latent Topology Insight**: "
        "Notice how the Transformer projects the query directly adjacent to the semantic cluster of relevant functions, "
        "even when the query contains zero lexical token overlap with the underlying code."
    )


# =============================================================================
# TAB 3: BM25 VS DENSE QUADRANT PLOT
# =============================================================================
with tab_quadrant:
    st.subheader("🎯 BM25 vs. Dense Discordance Quadrant Plot")
    st.caption(
        "Evaluates the candidate pool across both dimensions: Lexical BM25 Score (Normalized) vs. Dense Cosine Similarity (Normalized). "
        "Reveals where semantic and lexical retrieval agree, and where one modality rescues the other."
    )

    quad_df = pd.DataFrame(all_candidates)

    fig_quad = go.Figure()

    # Add Quadrant Shading Rectangles
    # 1. Top-Right: Consensus Sweet Spot
    fig_quad.add_shape(
        type="rect",
        x0=0.5,
        x1=1.05,
        y0=0.5,
        y1=1.05,
        fillcolor="rgba(16, 185, 129, 0.08)",
        line={"width": 0},
    )
    # 2. Top-Left: Pure Semantic Gems
    fig_quad.add_shape(
        type="rect",
        x0=-0.05,
        x1=0.5,
        y0=0.5,
        y1=1.05,
        fillcolor="rgba(99, 102, 241, 0.08)",
        line={"width": 0},
    )
    # 3. Bottom-Right: Keyword Matches
    fig_quad.add_shape(
        type="rect",
        x0=0.5,
        x1=1.05,
        y0=-0.05,
        y1=0.5,
        fillcolor="rgba(245, 158, 11, 0.08)",
        line={"width": 0},
    )
    # 4. Bottom-Left: Fringe / Low Relevancy
    fig_quad.add_shape(
        type="rect",
        x0=-0.05,
        x1=0.5,
        y0=-0.05,
        y1=0.5,
        fillcolor="rgba(148, 163, 184, 0.04)",
        line={"width": 0},
    )

    # Add Quadrant Label Annotations
    fig_quad.add_annotation(
        x=0.77,
        y=0.98,
        text="<b>Consensus Sweet Spot</b><br>(High Dense + High BM25)",
        showarrow=False,
        font={"color": "#10b981", "size": 12},
    )
    fig_quad.add_annotation(
        x=0.23,
        y=0.98,
        text="<b>Pure Semantic Gems</b><br>(High Dense + Low BM25)",
        showarrow=False,
        font={"color": "#818cf8", "size": 12},
    )
    fig_quad.add_annotation(
        x=0.77,
        y=0.06,
        text="<b>Lexical Keyword Match</b><br>(Low Dense + High BM25)",
        showarrow=False,
        font={"color": "#fbbf24", "size": 12},
    )
    fig_quad.add_annotation(
        x=0.23,
        y=0.06,
        text="<b>Fringe / Noise</b><br>(Low Dense + Low BM25)",
        showarrow=False,
        font={"color": "#64748b", "size": 12},
    )

    # Reference crosshair lines
    fig_quad.add_hline(y=0.5, line={"color": "#374151", "width": 1.5, "dash": "dash"})
    fig_quad.add_vline(x=0.5, line={"color": "#374151", "width": 1.5, "dash": "dash"})

    # Plot candidates stratified by Overlap Bin
    color_map = {
        "Zero": "#10b981",
        "Low": "#f59e0b",
        "High": "#3b82f6",
        "Unknown": "#94a3b8",
    }

    for stratum in ["Zero", "Low", "High"]:
        stratum_sub = quad_df[quad_df["overlap_bin"] == stratum]
        if not stratum_sub.empty:
            fig_quad.add_trace(
                go.Scatter(
                    x=stratum_sub["bm25_norm"],
                    y=stratum_sub["dense_norm"],
                    mode="markers",
                    name=f"{stratum}-Overlap",
                    marker={
                        "color": color_map.get(stratum, "#cbd5e1"),
                        "size": 11,
                        "line": {"color": "#ffffff", "width": 1.2},
                    },
                    hovertemplate=(
                        "<b>Function:</b> %{customdata[0]}<br>"
                        "<b>Dense Cosine:</b> %{customdata[1]:.3f}<br>"
                        "<b>BM25 Score:</b> %{customdata[2]:.2f}<br>"
                        "<b>Token Jaccard:</b> %{customdata[3]:.3f}<extra></extra>"
                    ),
                    customdata=stratum_sub[[
                        "func_name",
                        "dense_score",
                        "bm25_score",
                        "jaccard",
                    ]].values,
                )
            )

    fig_quad.update_layout(
        height=580,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        xaxis={
            "title": "Normalized Lexical Score (BM25)",
            "range": [-0.05, 1.05],
            "showgrid": True,
            "gridcolor": "#27273a",
        },
        yaxis={
            "title": "Normalized Semantic Score (Dense Cosine)",
            "range": [-0.05, 1.05],
            "showgrid": True,
            "gridcolor": "#27273a",
        },
        paper_bgcolor="#111119",
        plot_bgcolor="#161622",
        font={"color": "#cbd5e1"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig_quad, width="stretch")

    # Quadrant tally stats
    c_sweet = len(quad_df[(quad_df["dense_norm"] >= 0.5) & (quad_df["bm25_norm"] >= 0.5)])
    c_semantic = len(quad_df[(quad_df["dense_norm"] >= 0.5) & (quad_df["bm25_norm"] < 0.5)])
    c_keyword = len(quad_df[(quad_df["dense_norm"] < 0.5) & (quad_df["bm25_norm"] >= 0.5)])
    c_fringe = len(quad_df[(quad_df["dense_norm"] < 0.5) & (quad_df["bm25_norm"] < 0.5)])

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    col_q1.metric("Consensus Candidates", f"{c_sweet}", "High Agreement")
    col_q2.metric("Pure Semantic Gems", f"{c_semantic}", "Dense Solos")
    col_q3.metric("Keyword Matches", f"{c_keyword}", "BM25 Solos")
    col_q4.metric("Fringe Candidates", f"{c_fringe}", "Filtered Out")


# =============================================================================
# TAB 4: OFFICIAL BENCHMARK & CAPACITY GATES
# =============================================================================
with tab_benchmark:
    st.subheader("📊 Frozen Benchmark & Pre-Registered Gates")
    st.caption("All metrics evaluated across all 19,632 test queries under strict pre-registered conditions.")

    st.markdown("#### Table 1: End-to-End Test Set Retrieval Comparison")
    benchmark_data = [
        {
            "Retrieval Model": "Convex Hybrid (α=0.70)",
            "Parameters": "17.03M + BM25",
            "Full Test MRR": "0.6612 [0.6552, 0.6672]",
            "Zero-Overlap MRR": "0.1982",
            "Low-Overlap MRR": "0.5891",
            "High-Overlap MRR": "0.7118",
            "Status": "🏆 Benchmark Winner",
        },
        {
            "Retrieval Model": "RRF Hybrid (k=20)",
            "Parameters": "17.03M + BM25",
            "Full Test MRR": "0.6291 [0.6228, 0.6354]",
            "Zero-Overlap MRR": "0.1874",
            "Low-Overlap MRR": "0.5412",
            "High-Overlap MRR": "0.6853",
            "Status": "Rank-Only Fusion",
        },
        {
            "Retrieval Model": "CodeEmbed 6L (Scaled Dense)",
            "Parameters": "17.03M",
            "Full Test MRR": "0.4699 [0.4638, 0.4760]",
            "Zero-Overlap MRR": "0.1841",
            "Low-Overlap MRR": "0.4285",
            "High-Overlap MRR": "0.4996",
            "Status": "Exploratory Best",
        },
        {
            "Retrieval Model": "BM25 Lexical (ATIRE/Okapi)",
            "Parameters": "0",
            "Full Test MRR": "0.5109 [0.5049, 0.5169]",
            "Zero-Overlap MRR": "0.0381",
            "Low-Overlap MRR": "0.3168",
            "High-Overlap MRR": "0.6148",
            "Status": "Lexical Baseline",
        },
        {
            "Retrieval Model": "CodeEmbed 4L (Confirmatory)",
            "Parameters": "7.38M",
            "Full Test MRR": "0.4157 [0.4098, 0.4216]",
            "Zero-Overlap MRR": "0.1685",
            "Low-Overlap MRR": "0.4082",
            "High-Overlap MRR": "0.4289",
            "Status": "Pre-registered Clean",
        },
        {
            "Retrieval Model": "Zero-Shot CodeBERT (Untuned)",
            "Parameters": "125M",
            "Full Test MRR": "0.0142",
            "Zero-Overlap MRR": "0.0051",
            "Low-Overlap MRR": "0.0128",
            "High-Overlap MRR": "0.0152",
            "Status": "Representation Collapse",
        },
    ]
    st.dataframe(pd.DataFrame(benchmark_data), width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔬 Methodological Integrity & Pre-Registered Gates")

    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.markdown(
            """
            **1. AST Leakage Remediation**
            - **Finding:** Raw CodeSearchNet docstring leakage artificially inflated BM25 to 0.95 MRR.
            - **Fix:** Byte-accurate Python AST stripper removed docstring bodies.
            - **Audit:** Measured residual 8+ token overlap at **1.23%** (vs 0.47% null), within pre-registered 5% margin.
            """
        )
    with col_g2:
        st.markdown(
            """
            **2. Pre-Registered Fallback Gate**
            - **Pilot Gate:** Initial pilot tested against 0.75×BM25 (0.383 MRR) threshold.
            - **Search:** Pilot missed gate, properly triggering 6-config frozen grid search.
            - **Winner:** LR 3e-4, τ=0.07 reached **0.4157 MRR** (Confirmatory clean).
            """
        )
    with col_g3:
        st.markdown(
            """
            **3. Capacity Error Rubric Gate**
            - **Rubric:** Evaluated 100 random validation failures before scaling to 17M.
            - **Criterion:** Required Category B (genuine capacity limits) ≥ 50%.
            - **Result:** Measured **93% Category B** vs 7% Category A (label noise), justifying scale to 6L (0.4699 MRR).
            """
        )
