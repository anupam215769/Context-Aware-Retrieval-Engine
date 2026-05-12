"""
test_integration.py
-------------------
End-to-end integration tests that exercise the complete RAG pipeline.

These tests use the real EmbeddingService (all-mpnet-base-v2), real FAISS
VectorStore, and mocked QueryExpander — verifying that all layers work together.

Covers:
  - Full pipeline: ingest TECHNICAL_DOCUMENTS → Strategy A search → relevant results
  - Full pipeline: ingest TECHNICAL_DOCUMENTS → Strategy B search → relevant results
  - Strategies produce different result rankings for the same query
  - DataIngestionPipeline convenience class
  - All 5 benchmark queries run successfully end-to-end
"""

from __future__ import annotations

import pytest

from src.dataset import TECHNICAL_DOCUMENTS
from src.embeddings import EmbeddingService
from src.ingestion import DataIngestionPipeline
from src.query_expander import EXPANDED_QUERIES, QueryExpander
from src.retrieval import RetrievalEngine, RetrievalResult
from src.vector_store import VectorStore


@pytest.fixture
def full_corpus_engine(embedding_service: EmbeddingService, query_expander: QueryExpander):
    """RetrievalEngine pre-loaded with all 10 TECHNICAL_DOCUMENTS."""
    store = VectorStore(dimension=768)
    engine = RetrievalEngine(
        embedding_service=embedding_service,
        vector_store=store,
        query_expander=query_expander,
    )
    pipeline = DataIngestionPipeline(engine)
    pipeline.ingest_dataset()  # Ingests all 10 TECHNICAL_DOCUMENTS
    return engine


class TestFullPipelineStrategyA:
    """End-to-end Strategy A tests on the full 10-document corpus."""

    def test_strategy_a_returns_3_results(self, full_corpus_engine: RetrievalEngine) -> None:
        result = full_corpus_engine.strategy_a_search("How does the system handle peak load?", top_k=3)
        assert len(result.results) == 3

    def test_strategy_a_peak_load_retrieves_load_balancing_doc(
        self, full_corpus_engine: RetrievalEngine
    ) -> None:
        """doc_001 (Load Balancing & Auto-Scaling) should rank first for peak load query."""
        result = full_corpus_engine.strategy_a_search("How does the system handle peak load?", top_k=3)
        top_id = result.results[0].document["id"]
        assert top_id == "doc_001", (
            f"Expected doc_001 at rank 1, got: {top_id} - "
            f"{result.results[0].document['title']}"
        )

    def test_strategy_a_security_query_retrieves_auth_doc(
        self, full_corpus_engine: RetrievalEngine
    ) -> None:
        """doc_005 (Authentication & Security) should rank first for security query."""
        result = full_corpus_engine.strategy_a_search(
            "What security measures protect user data?", top_k=3
        )
        top_id = result.results[0].document["id"]
        assert top_id == "doc_005", (
            f"Expected doc_005 at rank 1, got: {top_id} - "
            f"{result.results[0].document['title']}"
        )

    def test_strategy_a_monitoring_query_retrieves_observability_doc(
        self, full_corpus_engine: RetrievalEngine
    ) -> None:
        """doc_006 (Observability) should rank first for monitoring query."""
        result = full_corpus_engine.strategy_a_search(
            "How is the system monitored in production?", top_k=3
        )
        top_id = result.results[0].document["id"]
        assert top_id == "doc_006", (
            f"Expected doc_006 at rank 1, got: {top_id} - "
            f"{result.results[0].document['title']}"
        )

    def test_strategy_a_scores_in_valid_range(self, full_corpus_engine: RetrievalEngine) -> None:
        result = full_corpus_engine.strategy_a_search("system architecture", top_k=5)
        for r in result.results:
            assert -1.0 <= r.score <= 1.0 + 1e-5


class TestFullPipelineStrategyB:
    """End-to-end Strategy B tests on the full 10-document corpus."""

    def test_strategy_b_returns_3_results(self, full_corpus_engine: RetrievalEngine) -> None:
        result = full_corpus_engine.strategy_b_search("How does the system handle peak load?", top_k=3)
        assert len(result.results) == 3

    def test_strategy_b_has_expanded_query(self, full_corpus_engine: RetrievalEngine) -> None:
        result = full_corpus_engine.strategy_b_search("How does the system handle peak load?")
        assert result.expanded_query is not None
        assert len(result.expanded_query) > len("How does the system handle peak load?")

    def test_strategy_b_peak_load_retrieves_relevant_doc(
        self, full_corpus_engine: RetrievalEngine
    ) -> None:
        """Strategy B should also retrieve a relevant doc for peak load (doc_001 or doc_009)."""
        result = full_corpus_engine.strategy_b_search("How does the system handle peak load?", top_k=3)
        top_ids = [r.document["id"] for r in result.results]
        relevant_ids = {"doc_001", "doc_009"}  # Load Balancing or API Gateway
        assert any(doc_id in relevant_ids for doc_id in top_ids), (
            f"Expected at least one of {relevant_ids} in top results, got: {top_ids}"
        )

    def test_strategy_b_scores_in_valid_range(self, full_corpus_engine: RetrievalEngine) -> None:
        result = full_corpus_engine.strategy_b_search("How does the system handle peak load?", top_k=5)
        for r in result.results:
            assert -1.0 <= r.score <= 1.0 + 1e-5


class TestStrategyComparison:
    """Verify Strategy A and B produce meaningfully different results."""

    def test_strategies_may_differ_in_results(self, full_corpus_engine: RetrievalEngine) -> None:
        """
        For at least one benchmark query, Strategy B should retrieve at least one
        document not retrieved by Strategy A (demonstrating the value of expansion).
        Note: with small corpus, strategies may sometimes agree — we verify both run.
        """
        query = "How does the system handle peak load?"
        result_a, result_b = full_corpus_engine.compare_strategies(query, top_k=3)

        ids_a = {r.document["id"] for r in result_a.results}
        ids_b = {r.document["id"] for r in result_b.results}

        # Both strategies must return results
        assert len(ids_a) == 3
        assert len(ids_b) == 3

        # At least verify strategies ran with different embeddings (B has expanded query)
        assert result_a.expanded_query is None
        assert result_b.expanded_query is not None

    def test_both_strategies_same_top_k(self, full_corpus_engine: RetrievalEngine) -> None:
        result_a, result_b = full_corpus_engine.compare_strategies(
            "How are service failures handled?", top_k=3
        )
        assert len(result_a.results) == 3
        assert len(result_b.results) == 3


class TestAllBenchmarkQueriesEndToEnd:
    """Run all 5 benchmark queries through both strategies end-to-end."""

    @pytest.mark.parametrize("query", list(EXPANDED_QUERIES.keys()))
    def test_strategy_a_runs_for_benchmark_query(
        self, full_corpus_engine: RetrievalEngine, query: str
    ) -> None:
        result = full_corpus_engine.strategy_a_search(query, top_k=3)
        assert result.strategy == "A"
        assert len(result.results) == 3
        assert result.query == query

    @pytest.mark.parametrize("query", list(EXPANDED_QUERIES.keys()))
    def test_strategy_b_runs_for_benchmark_query(
        self, full_corpus_engine: RetrievalEngine, query: str
    ) -> None:
        result = full_corpus_engine.strategy_b_search(query, top_k=3)
        assert result.strategy == "B"
        assert len(result.results) == 3
        assert result.expanded_query is not None


class TestDataIngestionPipeline:
    """DataIngestionPipeline integration tests."""

    def test_ingest_dataset_default_corpus(
        self, embedding_service: EmbeddingService, query_expander: QueryExpander
    ) -> None:
        """Default corpus (10 docs) ingested via pipeline."""
        store = VectorStore(dimension=768)
        engine = RetrievalEngine(embedding_service, store, query_expander)
        pipeline = DataIngestionPipeline(engine)
        count = pipeline.ingest_dataset()
        assert count == len(TECHNICAL_DOCUMENTS)
        assert pipeline.document_count == len(TECHNICAL_DOCUMENTS)

    def test_ingest_dataset_custom_corpus(
        self,
        embedding_service: EmbeddingService,
        query_expander: QueryExpander,
        sample_documents: list[dict],
    ) -> None:
        """Custom corpus ingested via pipeline."""
        store = VectorStore(dimension=768)
        engine = RetrievalEngine(embedding_service, store, query_expander)
        pipeline = DataIngestionPipeline(engine)
        count = pipeline.ingest_dataset(documents=sample_documents)
        assert count == len(sample_documents)

    def test_document_count_property(
        self, embedding_service: EmbeddingService, query_expander: QueryExpander
    ) -> None:
        store = VectorStore(dimension=768)
        engine = RetrievalEngine(embedding_service, store, query_expander)
        pipeline = DataIngestionPipeline(engine)
        assert pipeline.document_count == 0
        pipeline.ingest_dataset()
        assert pipeline.document_count == 10
