"""
test_retrieval.py
-----------------
Unit tests for RetrievalEngine (src/retrieval.py).

Covers ingest, strategy_a_search, strategy_b_search, compare_strategies,
edge cases (empty query, missing QueryExpander), and RetrievalResult serialization.
"""

from __future__ import annotations

import json

import pytest

from src.retrieval import RetrievalEngine, RetrievalResult
from src.vector_store import SearchResult, VectorStore


class TestIngest:
    def test_ingest_returns_correct_count(self, full_pipeline_engine, sample_documents):
        count = full_pipeline_engine.ingest(sample_documents)
        assert count == len(sample_documents)

    def test_ingest_populates_vector_store(self, full_pipeline_engine, sample_documents):
        full_pipeline_engine.ingest(sample_documents)
        assert full_pipeline_engine.vector_store.count == len(sample_documents)

    def test_ingest_empty_list_raises(self, full_pipeline_engine):
        with pytest.raises(ValueError, match="non-empty"):
            full_pipeline_engine.ingest([])

    def test_ingest_missing_content_key_raises(self, full_pipeline_engine):
        with pytest.raises(ValueError, match="content"):
            full_pipeline_engine.ingest([{"id": "d1", "title": "No content"}])

    def test_ingest_partial_missing_content_raises(self, full_pipeline_engine):
        docs = [{"id": "d1", "content": "valid"}, {"id": "d2"}]
        with pytest.raises(ValueError, match="content"):
            full_pipeline_engine.ingest(docs)


class TestStrategyA:
    def test_returns_retrieval_result(self, retrieval_engine):
        result = retrieval_engine.strategy_a_search("How does the system handle peak load?")
        assert isinstance(result, RetrievalResult)

    def test_strategy_field_is_A(self, retrieval_engine):
        result = retrieval_engine.strategy_a_search("peak load balancing")
        assert result.strategy == "A"

    def test_no_expanded_query(self, retrieval_engine):
        result = retrieval_engine.strategy_a_search("peak load balancing")
        assert result.expanded_query is None

    def test_respects_top_k(self, retrieval_engine):
        result = retrieval_engine.strategy_a_search("monitoring metrics", top_k=3)
        assert len(result.results) == 3

    def test_top_k_one(self, retrieval_engine):
        result = retrieval_engine.strategy_a_search("authentication", top_k=1)
        assert len(result.results) == 1

    def test_preserves_original_query(self, retrieval_engine):
        q = "How does the system handle peak load?"
        assert retrieval_engine.strategy_a_search(q).query == q

    def test_results_are_search_results(self, retrieval_engine):
        result = retrieval_engine.strategy_a_search("load balancing", top_k=2)
        assert all(isinstance(r, SearchResult) for r in result.results)

    def test_empty_query_raises(self, retrieval_engine):
        with pytest.raises(ValueError, match="non-empty"):
            retrieval_engine.strategy_a_search("")

    def test_results_sorted_by_score(self, retrieval_engine):
        result = retrieval_engine.strategy_a_search("circuit breaker fault tolerance", top_k=5)
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)


class TestStrategyB:
    def test_returns_retrieval_result(self, retrieval_engine):
        result = retrieval_engine.strategy_b_search("How does the system handle peak load?")
        assert isinstance(result, RetrievalResult)

    def test_strategy_field_is_B(self, retrieval_engine):
        result = retrieval_engine.strategy_b_search("peak load")
        assert result.strategy == "B"

    def test_has_expanded_query(self, retrieval_engine):
        result = retrieval_engine.strategy_b_search("How does the system handle peak load?")
        assert result.expanded_query is not None
        assert len(result.expanded_query) > 0

    def test_expanded_query_differs_from_original(self, retrieval_engine):
        original = "How does the system handle peak load?"
        result = retrieval_engine.strategy_b_search(original)
        assert result.expanded_query != original

    def test_respects_top_k(self, retrieval_engine):
        result = retrieval_engine.strategy_b_search("How does the system handle peak load?", top_k=3)
        assert len(result.results) == 3

    def test_empty_query_raises(self, retrieval_engine):
        with pytest.raises(ValueError, match="non-empty"):
            retrieval_engine.strategy_b_search("")

    def test_without_expander_raises_runtime_error(self, embedding_service, populated_vector_store):
        engine = RetrievalEngine(
            embedding_service=embedding_service,
            vector_store=populated_vector_store,
            query_expander=None,
        )
        with pytest.raises(RuntimeError, match="QueryExpander"):
            engine.strategy_b_search("some query")

    def test_results_sorted_by_score(self, retrieval_engine):
        result = retrieval_engine.strategy_b_search("How does the system handle peak load?", top_k=5)
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)


class TestCompareStrategies:
    def test_returns_tuple_of_two(self, retrieval_engine):
        results = retrieval_engine.compare_strategies("How does the system handle peak load?")
        assert isinstance(results, tuple) and len(results) == 2

    def test_first_is_strategy_a(self, retrieval_engine):
        result_a, _ = retrieval_engine.compare_strategies("peak load")
        assert result_a.strategy == "A"

    def test_second_is_strategy_b(self, retrieval_engine):
        _, result_b = retrieval_engine.compare_strategies("peak load")
        assert result_b.strategy == "B"

    def test_same_query_in_both(self, retrieval_engine):
        query = "How does the system handle peak load?"
        result_a, result_b = retrieval_engine.compare_strategies(query)
        assert result_a.query == query
        assert result_b.query == query


class TestRetrievalResultSerialization:
    def test_to_dict_has_required_keys(self, retrieval_engine):
        d = retrieval_engine.strategy_a_search("load balancing", top_k=2).to_dict()
        for key in ("query", "strategy", "results"):
            assert key in d

    def test_to_dict_results_have_expected_fields(self, retrieval_engine):
        d = retrieval_engine.strategy_a_search("monitoring", top_k=2).to_dict()
        for item in d["results"]:
            for field in ("rank", "score", "title"):
                assert field in item

    def test_strategy_b_includes_expanded_query(self, retrieval_engine):
        d = retrieval_engine.strategy_b_search("How does the system handle peak load?").to_dict()
        assert d["expanded_query"] is not None
        assert len(d["expanded_query"]) > 0

    def test_is_json_serializable(self, retrieval_engine):
        d = retrieval_engine.strategy_a_search("load balancing", top_k=2).to_dict()
        assert isinstance(json.dumps(d), str)
