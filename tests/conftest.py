"""
conftest.py
-----------
Shared pytest fixtures for the entire test suite.

Fixtures provided
-----------------
- embedding_service       : Real EmbeddingService (all-mpnet-base-v2)
- empty_vector_store      : Fresh VectorStore with dim=768
- sample_documents        : 5 lightweight test documents
- sample_embeddings       : Real embeddings for sample_documents
- populated_vector_store  : VectorStore pre-loaded with sample_documents
- mock_generative_model   : Patched GenerativeModel returning deterministic expansions
- query_expander          : QueryExpander with model mocked
- retrieval_engine        : Fully assembled RetrievalEngine (ingested, both strategies)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings import EmbeddingService
from src.ingestion import DataIngestionPipeline
from src.query_expander import EXPANDED_QUERIES, QueryExpander
from src.retrieval import RetrievalEngine
from src.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Lightweight test documents (distinct enough to test ranking)
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENTS = [
    {
        "id": "test_001",
        "title": "Auto-Scaling Under High Load",
        "category": "infrastructure",
        "content": (
            "The system uses Kubernetes HPA to auto-scale pods during peak traffic. "
            "Load balancers distribute requests evenly across all healthy instances. "
            "Rate limiting prevents resource monopolisation during demand spikes."
        ),
    },
    {
        "id": "test_002",
        "title": "Redis Cache Architecture",
        "category": "performance",
        "content": (
            "A Redis cluster provides shared caching across all service instances. "
            "LRU eviction policy manages memory with configurable TTLs. "
            "Cache invalidation uses pub-sub to maintain consistency."
        ),
    },
    {
        "id": "test_003",
        "title": "Circuit Breaker and Retry Logic",
        "category": "reliability",
        "content": (
            "Resilience4j circuit breakers protect against cascading failures. "
            "Exponential backoff with jitter prevents thundering herd problems. "
            "Dead letter queues capture failed messages for later reprocessing."
        ),
    },
    {
        "id": "test_004",
        "title": "OAuth 2.0 and JWT Authentication",
        "category": "security",
        "content": (
            "OAuth 2.0 authorization code flow with PKCE handles user authentication. "
            "Short-lived JWT access tokens are signed with RS256 asymmetric keys. "
            "OPA enforces RBAC policies against JWT claims at the API gateway."
        ),
    },
    {
        "id": "test_005",
        "title": "Prometheus and Grafana Monitoring",
        "category": "observability",
        "content": (
            "Prometheus scrapes RED metrics from all services every 15 seconds. "
            "Grafana dashboards visualise rate, error rate, and latency in real time. "
            "AlertManager triggers PagerDuty notifications when SLOs are at risk."
        ),
    },
]


# ---------------------------------------------------------------------------
# Core component fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedding_service() -> EmbeddingService:
    """
    Real EmbeddingService loaded once per session (model loading is expensive).
    Uses all-mpnet-base-v2 (768d).
    """
    return EmbeddingService()


@pytest.fixture
def empty_vector_store() -> VectorStore:
    """Fresh, empty VectorStore with dimension=768."""
    return VectorStore(dimension=768)


@pytest.fixture
def sample_documents() -> list[dict]:
    """5 lightweight test documents."""
    return SAMPLE_DOCUMENTS


@pytest.fixture(scope="session")
def sample_embeddings(embedding_service: EmbeddingService) -> np.ndarray:
    """Pre-computed embeddings for the 5 sample documents (session-scoped)."""
    texts = [doc["content"] for doc in SAMPLE_DOCUMENTS]
    return embedding_service.get_embeddings(texts)


@pytest.fixture
def populated_vector_store(
    sample_embeddings: np.ndarray,
    sample_documents: list[dict],
) -> VectorStore:
    """VectorStore pre-loaded with the 5 sample documents."""
    store = VectorStore(dimension=768)
    store.add_documents(sample_embeddings.copy(), list(sample_documents))
    return store


# ---------------------------------------------------------------------------
# Mock fixtures for Vertex AI SDK
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_generative_model():
    """
    Patches 'src.query_expander.GenerativeModel' so no GCP credentials or
    network calls are made. Returns a mock that maps original queries to
    deterministic expanded queries using EXPANDED_QUERIES.
    """
    with patch("src.query_expander.GenerativeModel") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        def side_effect(prompt: str):
            # Extract the original query from the prompt
            for original_q, expanded_q in EXPANDED_QUERIES.items():
                if original_q in prompt:
                    response = MagicMock()
                    response.text = expanded_q
                    return response
            # Fallback: echo the prompt back
            response = MagicMock()
            response.text = "expanded: " + prompt[-100:]
            return response

        mock_instance.generate_content.side_effect = side_effect
        yield mock_cls


@pytest.fixture
def query_expander(mock_generative_model) -> QueryExpander:
    """QueryExpander with mocked GenerativeModel."""
    return QueryExpander(model_name="gemini-1.5-flash")


# ---------------------------------------------------------------------------
# Full pipeline fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def retrieval_engine(
    embedding_service: EmbeddingService,
    populated_vector_store: VectorStore,
    query_expander: QueryExpander,
) -> RetrievalEngine:
    """
    Fully assembled RetrievalEngine with:
    - EmbeddingService (real)
    - VectorStore pre-loaded with 5 sample documents
    - QueryExpander (mocked GenerativeModel)
    """
    return RetrievalEngine(
        embedding_service=embedding_service,
        vector_store=populated_vector_store,
        query_expander=query_expander,
    )


@pytest.fixture
def full_pipeline_engine(
    embedding_service: EmbeddingService,
    query_expander: QueryExpander,
) -> RetrievalEngine:
    """
    RetrievalEngine with empty VectorStore — for integration tests that
    test ingestion themselves.
    """
    store = VectorStore(dimension=768)
    return RetrievalEngine(
        embedding_service=embedding_service,
        vector_store=store,
        query_expander=query_expander,
    )
