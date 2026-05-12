# Context-Aware Retrieval Engine

A local Retrieval-Augmented Generation (RAG) pipeline demonstrating semantic vector search and AI-enhanced query expansion.

## Overview

This project implements two retrieval strategies and benchmarks them against each other:

| Strategy | Description |
|----------|-------------|
| **Strategy A** — Raw Vector Search | Embeds the user query directly → cosine similarity search |
| **Strategy B** — AI-Enhanced Retrieval | Mocked `GenerativeModel` rewrites the query → embeds expanded query → cosine similarity search |

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Embedding Model | `sentence-transformers/all-mpnet-base-v2` (768d) | Matches `textembedding-gecko@003` dimensionality; strong semantic quality |
| Vector Store | FAISS `IndexFlatIP` | Exact cosine similarity via inner product on normalized vectors |
| Query Expansion | Mocked `vertexai.generative_models.GenerativeModel` | No GCP credentials needed locally |
| Test Framework | pytest | Fixtures, parametrize, coverage |

## Project Structure

```
d:\VectorSearch\
├── src/
│   ├── dataset.py          # 10 technical paragraphs (distributed systems corpus)
│   ├── embeddings.py       # EmbeddingService (wraps sentence-transformers)
│   ├── vector_store.py     # VectorStore (FAISS cosine similarity)
│   ├── query_expander.py   # QueryExpander (mocked GenerativeModel)
│   ├── retrieval.py        # RetrievalEngine (Strategy A + B orchestration)
│   └── ingestion.py        # DataIngestionPipeline
├── tests/
│   ├── conftest.py         # Shared fixtures (session-scoped embedding model)
│   ├── test_embeddings.py  # EmbeddingService tests + Vertex AI mock interface
│   ├── test_vector_store.py
│   ├── test_query_expander.py
│   ├── test_retrieval.py
│   └── test_integration.py # End-to-end pipeline tests
├── benchmark/
│   └── run_benchmark.py    # Benchmark runner → generates retrieval_benchmark.md
├── retrieval_benchmark.md  # Auto-generated Strategy A vs B comparison report
├── DOCUMENTATION.md        # Similarity metric analysis + Vertex AI migration guide
└── requirements.txt
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt
```

## Running Tests

```bash
# Full test suite
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Individual test files
pytest tests/test_embeddings.py -v
pytest tests/test_integration.py -v
```

## Running the Benchmark

```bash
python -m benchmark.run_benchmark
```

This will:
1. Load `all-mpnet-base-v2` (768d) and embed 10 technical documents
2. Run 5 complex queries through both strategies
3. Print formatted comparison tables to stdout
4. Write `retrieval_benchmark.md` to the project root

## Key Design Decisions

See [`DOCUMENTATION.md`](./DOCUMENTATION.md) for:
- Similarity metric choice (Cosine vs Euclidean) with mathematical rationale
- Production migration path to Vertex AI Vector Search (Matching Engine)
