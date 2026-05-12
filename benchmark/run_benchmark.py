"""
run_benchmark.py
----------------
Benchmark runner: compares Strategy A (Raw Vector Search) vs Strategy B
(AI-Enhanced Retrieval) across 5 complex queries.

Usage
-----
    python -m benchmark.run_benchmark

Outputs
-------
1. Formatted table to stdout
2. JSON results to stdout
3. retrieval_benchmark.md written to the project root

Mock setup
----------
Uses unittest.mock.patch to mock vertexai.generative_models.GenerativeModel
with the deterministic EXPANDED_QUERIES mapping. No GCP credentials required.
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

# Make sure project root is on path when run as module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 stdout/stderr on Windows (default is cp1252 in PowerShell)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from tabulate import tabulate

from src.dataset import TECHNICAL_DOCUMENTS
from src.embeddings import EmbeddingService
from src.ingestion import DataIngestionPipeline
from src.query_expander import EXPANDED_QUERIES, QueryExpander
from src.retrieval import RetrievalEngine, RetrievalResult
from src.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES = list(EXPANDED_QUERIES.keys())
TOP_K = 3
EMBEDDING_MODEL = "all-mpnet-base-v2"
OUTPUT_MD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "retrieval_benchmark.md",
)

# ---------------------------------------------------------------------------
# Mock factory for GenerativeModel
# ---------------------------------------------------------------------------

def _build_mock_generative_model_class():
    """
    Returns a mock class for vertexai.generative_models.GenerativeModel.
    The mock instance's generate_content() maps prompts to EXPANDED_QUERIES values.
    """
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance

    def _side_effect(prompt: str):
        for original_q, expanded_q in EXPANDED_QUERIES.items():
            if original_q in prompt:
                response = MagicMock()
                response.text = expanded_q
                return response
        # Fallback
        response = MagicMock()
        response.text = prompt
        return response

    mock_instance.generate_content.side_effect = _side_effect
    return mock_cls


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_snippet(text: str, max_len: int = 90) -> str:
    return text[:max_len].rstrip() + "..." if len(text) > max_len else text


def _results_table(results: list, title: str) -> str:
    rows = [
        [
            r.rank,
            r.document.get("title", "N/A"),
            r.document.get("category", "N/A"),
            f"{r.score:.4f}",
            _format_snippet(r.document.get("content", ""), 80),
        ]
        for r in results
    ]
    return tabulate(
        rows,
        headers=["Rank", "Title", "Category", "Score", "Content Snippet"],
        tablefmt="pipe",
    )


def _compare_rankings(result_a: RetrievalResult, result_b: RetrievalResult) -> str:
    """Return a human-readable ranking comparison."""
    ids_a = [r.document["id"] for r in result_a.results]
    ids_b = [r.document["id"] for r in result_b.results]
    changed = ids_a != ids_b
    new_in_b = set(ids_b) - set(ids_a)
    dropped_from_a = set(ids_a) - set(ids_b)

    lines = []
    if changed:
        lines.append("⚡ Rankings differ between strategies.")
        if new_in_b:
            titles = [
                r.document["title"]
                for r in result_b.results
                if r.document["id"] in new_in_b
            ]
            lines.append(f"  • Strategy B newly retrieved: {', '.join(titles)}")
        if dropped_from_a:
            titles = [
                r.document["title"]
                for r in result_a.results
                if r.document["id"] in dropped_from_a
            ]
            lines.append(f"  • Strategy A retrieved (not in B): {', '.join(titles)}")
    else:
        lines.append("✓ Both strategies retrieved the same top-3 documents.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def _build_markdown_report(
    benchmark_data: list[dict[str, Any]],
    run_timestamp: str,
) -> str:
    lines: list[str] = []

    lines += [
        "# Retrieval Benchmark Report",
        "",
        f"**Run timestamp:** {run_timestamp}  ",
        f"**Embedding model:** `{EMBEDDING_MODEL}` (768d — equivalent to textembedding-gecko)  ",
        f"**Vector store:** FAISS `IndexFlatIP` (Cosine Similarity on L2-normalized vectors)  ",
        f"**Corpus size:** {len(TECHNICAL_DOCUMENTS)} technical paragraphs  ",
        f"**Queries evaluated:** {len(BENCHMARK_QUERIES)}  ",
        f"**Top-K per query:** {TOP_K}  ",
        "",
        "---",
        "",
        "## Strategy Definitions",
        "",
        "| Strategy | Description |",
        "|----------|-------------|",
        "| **Strategy A** — Raw Vector Search | Embeds the original user query directly → FAISS cosine similarity search |",
        "| **Strategy B** — AI-Enhanced Retrieval | Rewrites query via mocked `GenerativeModel` (gemini-1.5-flash) → embeds expanded query → FAISS search |",
        "",
        "---",
        "",
    ]

    for i, entry in enumerate(benchmark_data, start=1):
        query = entry["query"]
        result_a: RetrievalResult = entry["result_a"]
        result_b: RetrievalResult = entry["result_b"]

        lines += [
            f"## Query {i}: _{query}_",
            "",
            "### Strategy B — Expanded Query",
            "",
            f"> {result_b.expanded_query}",
            "",
            "### Strategy A — Raw Vector Search Results",
            "",
            _results_table(result_a.results, "Strategy A"),
            "",
            "### Strategy B — AI-Enhanced Retrieval Results",
            "",
            _results_table(result_b.results, "Strategy B"),
            "",
            "### Comparison Analysis",
            "",
            _compare_rankings(result_a, result_b),
            "",
            "---",
            "",
        ]

    # Summary table
    lines += [
        "## Summary: Strategy A vs Strategy B",
        "",
        "| Query | Strat A Top Doc | A Score | Strat B Top Doc | B Score | Same Top-1? |",
        "|-------|----------------|---------|-----------------|---------|-------------|",
    ]

    for entry in benchmark_data:
        result_a: RetrievalResult = entry["result_a"]
        result_b: RetrievalResult = entry["result_b"]
        q_short = entry["query"][:50] + "..." if len(entry["query"]) > 50 else entry["query"]
        top_a = result_a.results[0] if result_a.results else None
        top_b = result_b.results[0] if result_b.results else None
        same = "✓" if (top_a and top_b and top_a.document["id"] == top_b.document["id"]) else "✗"
        a_title = top_a.document["title"][:30] if top_a else "N/A"
        b_title = top_b.document["title"][:30] if top_b else "N/A"
        a_score = f"{top_a.score:.4f}" if top_a else "N/A"
        b_score = f"{top_b.score:.4f}" if top_b else "N/A"
        lines.append(f"| {q_short} | {a_title} | {a_score} | {b_title} | {b_score} | {same} |")

    lines += [
        "",
        "---",
        "",
        "## Similarity Metric",
        "",
        "**Cosine Similarity** was used (via FAISS `IndexFlatIP` on L2-normalized vectors).",
        "",
        "- Sentence-transformers models optimize for cosine similarity during training.",
        "- Direction-based: invariant to embedding magnitude, ideal for semantic search.",
        "- Score range: [-1, 1], where 1.0 = identical semantics.",
        "- Mathematical equivalence: for unit vectors `||a||=||b||=1`,",
        "  `cosine_sim(a,b) = a·b` (inner product).",
        "",
        "---",
        "",
        "## Raw JSON Results",
        "",
        "```json",
        json.dumps(
            [
                {
                    "query": e["query"],
                    "strategy_a": e["result_a"].to_dict(),
                    "strategy_b": e["result_b"].to_dict(),
                }
                for e in benchmark_data
            ],
            indent=2,
        ),
        "```",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark() -> list[dict[str, Any]]:
    """
    Execute the full benchmark with mocked GenerativeModel.
    Returns list of {query, result_a, result_b} dicts.
    """
    print("\n" + "=" * 70)
    print("  CONTEXT-AWARE RETRIEVAL ENGINE — BENCHMARK")
    print("=" * 70)
    print(f"  Embedding model : {EMBEDDING_MODEL}")
    print(f"  Corpus size     : {len(TECHNICAL_DOCUMENTS)} documents")
    print(f"  Queries         : {len(BENCHMARK_QUERIES)}")
    print(f"  Top-K           : {TOP_K}")
    print("=" * 70 + "\n")

    mock_cls = _build_mock_generative_model_class()

    with patch("src.query_expander.GenerativeModel", mock_cls):
        # --- Build pipeline ---
        print(">> Loading embedding model (all-mpnet-base-v2, 768d)...")
        embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL)

        store = VectorStore(dimension=embedding_service.dimension)
        query_expander = QueryExpander(model_name="gemini-1.5-flash")
        engine = RetrievalEngine(embedding_service, store, query_expander)

        print(">> Ingesting corpus...")
        pipeline = DataIngestionPipeline(engine)
        count = pipeline.ingest_dataset()
        print(f"  [OK] Ingested {count} documents into FAISS IndexFlatIP.\n")

        # --- Run benchmark queries ---
        benchmark_data: list[dict[str, Any]] = []

        for i, query in enumerate(BENCHMARK_QUERIES, start=1):
            print(f"[{i}/{len(BENCHMARK_QUERIES)}] Query: \"{query}\"")

            result_a, result_b = engine.compare_strategies(query, top_k=TOP_K)

            print(f"\n  Strategy A — Raw Vector Search")
            for r in result_a.results:
                print(f"    #{r.rank}  [{r.score:.4f}]  {r.document['title']}")

            print(f"\n  Strategy B — AI-Enhanced (Query Expansion)")
            print(f"  Expanded: \"{result_b.expanded_query[:80]}...\"")
            for r in result_b.results:
                print(f"    #{r.rank}  [{r.score:.4f}]  {r.document['title']}")

            print()
            benchmark_data.append({"query": query, "result_a": result_a, "result_b": result_b})

    return benchmark_data


def main() -> None:
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    benchmark_data = run_benchmark()

    # Write Markdown report
    md_content = _build_markdown_report(benchmark_data, run_timestamp)
    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[OK] Benchmark report written to: {OUTPUT_MD_PATH}\n")
    print("=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
