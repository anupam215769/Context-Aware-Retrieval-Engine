"""
test_vector_store.py
--------------------
Tests for VectorStore (src/vector_store.py).

Covers:
  - Adding documents and counting
  - Search returns correct number of results
  - Results are sorted by score (descending)
  - Most relevant document ranks first (semantic check)
  - Edge cases: empty store, dimension mismatch, length mismatch
  - SearchResult dataclass structure
  - Reset functionality
"""

from __future__ import annotations

import numpy as np
import pytest

from src.vector_store import SearchResult, VectorStore


class TestVectorStoreInitialization:
    """VectorStore construction."""

    def test_initializes_empty(self) -> None:
        store = VectorStore(dimension=768)
        assert store.count == 0

    def test_correct_dimension_stored(self) -> None:
        store = VectorStore(dimension=128)
        assert store.dimension == 128

    def test_invalid_dimension_raises(self) -> None:
        with pytest.raises(ValueError):
            VectorStore(dimension=0)

    def test_negative_dimension_raises(self) -> None:
        with pytest.raises(ValueError):
            VectorStore(dimension=-10)


class TestAddDocuments:
    """Adding documents to the store."""

    def test_add_increases_count(
        self,
        empty_vector_store: VectorStore,
        sample_embeddings: np.ndarray,
        sample_documents: list[dict],
    ) -> None:
        empty_vector_store.add_documents(sample_embeddings.copy(), list(sample_documents))
        assert empty_vector_store.count == len(sample_documents)

    def test_add_twice_accumulates(
        self,
        empty_vector_store: VectorStore,
        sample_embeddings: np.ndarray,
        sample_documents: list[dict],
    ) -> None:
        empty_vector_store.add_documents(sample_embeddings.copy(), list(sample_documents))
        empty_vector_store.add_documents(sample_embeddings.copy(), list(sample_documents))
        assert empty_vector_store.count == len(sample_documents) * 2

    def test_add_single_document(self, empty_vector_store: VectorStore) -> None:
        emb = np.random.rand(1, 768).astype(np.float32)
        doc = [{"id": "d1", "content": "test"}]
        empty_vector_store.add_documents(emb, doc)
        assert empty_vector_store.count == 1

    def test_length_mismatch_raises(self, empty_vector_store: VectorStore) -> None:
        emb = np.random.rand(3, 768).astype(np.float32)
        docs = [{"id": "d1", "content": "x"}, {"id": "d2", "content": "y"}]  # 2 docs, 3 embs
        with pytest.raises(ValueError, match="same length"):
            empty_vector_store.add_documents(emb, docs)

    def test_dimension_mismatch_raises(self, empty_vector_store: VectorStore) -> None:
        emb = np.random.rand(2, 512).astype(np.float32)  # Wrong dim (512 vs 768)
        docs = [{"id": "d1", "content": "x"}, {"id": "d2", "content": "y"}]
        with pytest.raises(ValueError, match="dimension mismatch"):
            empty_vector_store.add_documents(emb, docs)


class TestSearch:
    """Searching the populated store."""

    def test_search_returns_top_k(
        self, populated_vector_store: VectorStore, embedding_service
    ) -> None:
        query_emb = embedding_service.get_embedding("system scaling")
        results = populated_vector_store.search(query_emb, top_k=3)
        assert len(results) == 3

    def test_search_returns_one(
        self, populated_vector_store: VectorStore, embedding_service
    ) -> None:
        query_emb = embedding_service.get_embedding("authentication")
        results = populated_vector_store.search(query_emb, top_k=1)
        assert len(results) == 1

    def test_search_all_documents(
        self, populated_vector_store: VectorStore, embedding_service
    ) -> None:
        """top_k == total docs should return all."""
        query_emb = embedding_service.get_embedding("system")
        results = populated_vector_store.search(query_emb, top_k=5)
        assert len(results) == 5

    def test_top_k_clamped_to_store_size(
        self, populated_vector_store: VectorStore, embedding_service
    ) -> None:
        """Requesting more results than documents returns all documents."""
        query_emb = embedding_service.get_embedding("some query")
        results = populated_vector_store.search(query_emb, top_k=100)
        assert len(results) == 5  # Only 5 docs in populated_vector_store

    def test_results_sorted_by_score_descending(
        self, populated_vector_store: VectorStore, embedding_service
    ) -> None:
        """Results must be ordered best-match first (highest cosine score first)."""
        query_emb = embedding_service.get_embedding("load balancer auto scaling traffic")
        results = populated_vector_store.search(query_emb, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted by score: {scores}"
        )

    def test_rank_field_is_sequential(
        self, populated_vector_store: VectorStore, embedding_service
    ) -> None:
        """Rank must be 1, 2, 3, ... in result order."""
        query_emb = embedding_service.get_embedding("monitoring metrics")
        results = populated_vector_store.search(query_emb, top_k=3)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_search_result_has_document_metadata(
        self, populated_vector_store: VectorStore, embedding_service
    ) -> None:
        """Each SearchResult must contain the original document dict."""
        query_emb = embedding_service.get_embedding("authentication security")
        results = populated_vector_store.search(query_emb, top_k=1)
        assert isinstance(results[0].document, dict)
        assert "id" in results[0].document
        assert "content" in results[0].document

    def test_most_relevant_doc_ranks_first(
        self, populated_vector_store: VectorStore, embedding_service
    ) -> None:
        """
        A query about monitoring/Prometheus should rank the observability
        document higher than authentication or caching documents.
        """
        query_emb = embedding_service.get_embedding(
            "Prometheus metrics Grafana dashboard alert monitoring"
        )
        results = populated_vector_store.search(query_emb, top_k=5)
        top_result = results[0]
        assert top_result.document["id"] == "test_005", (
            f"Expected observability doc (test_005) to rank first, "
            f"got: {top_result.document['id']} - {top_result.document['title']}"
        )

    def test_scores_are_valid_cosine_range(
        self, populated_vector_store: VectorStore, embedding_service
    ) -> None:
        """Cosine similarity scores must be in [-1, 1]."""
        query_emb = embedding_service.get_embedding("distributed systems")
        results = populated_vector_store.search(query_emb, top_k=5)
        for r in results:
            assert -1.0 <= r.score <= 1.0 + 1e-5, (
                f"Score {r.score} is outside [-1, 1] range."
            )


class TestEmptyStoreSearch:
    """Edge case: searching an empty store."""

    def test_search_empty_store_raises(
        self, empty_vector_store: VectorStore, embedding_service
    ) -> None:
        query_emb = embedding_service.get_embedding("any query")
        with pytest.raises(ValueError, match="empty"):
            empty_vector_store.search(query_emb, top_k=3)


class TestReset:
    """VectorStore reset functionality."""

    def test_reset_clears_documents(
        self,
        populated_vector_store: VectorStore,
    ) -> None:
        assert populated_vector_store.count == 5
        populated_vector_store.reset()
        assert populated_vector_store.count == 0

    def test_can_add_after_reset(
        self,
        populated_vector_store: VectorStore,
        sample_embeddings: np.ndarray,
        sample_documents: list[dict],
    ) -> None:
        populated_vector_store.reset()
        populated_vector_store.add_documents(sample_embeddings.copy(), list(sample_documents))
        assert populated_vector_store.count == 5


class TestSearchResultDataclass:
    """SearchResult dataclass contract."""

    def test_search_result_repr(self) -> None:
        doc = {"id": "d1", "title": "Test Doc", "content": "content"}
        sr = SearchResult(rank=1, document=doc, score=0.85)
        assert "rank=1" in repr(sr)
        assert "0.8500" in repr(sr)
        assert "Test Doc" in repr(sr)
