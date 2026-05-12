"""
test_query_expander.py
----------------------
Tests for QueryExpander (src/query_expander.py).

All tests mock 'src.query_expander.GenerativeModel' so no GCP credentials,
vertexai installation, or network calls are required.

Covers:
  - Model is instantiated with correct model name
  - expand_query() calls generate_content() with the prompt
  - Expanded query contains the original query's intent
  - Expanded query is longer/richer than the original
  - Deterministic EXPANDED_QUERIES mapping works for all 5 benchmark queries
  - Edge-case validation (empty query)
  - GenerativeModel exception fallback
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.query_expander import EXPANDED_QUERIES, EXPANSION_PROMPT_TEMPLATE, QueryExpander


class TestQueryExpanderInitialization:
    """QueryExpander construction with mocked GenerativeModel."""

    def test_model_instantiated_with_correct_name(self, mock_generative_model) -> None:
        """GenerativeModel must be called with the specified model name."""
        expander = QueryExpander(model_name="gemini-1.5-flash")
        mock_generative_model.assert_called_once_with("gemini-1.5-flash")

    def test_default_model_name(self, mock_generative_model) -> None:
        """Default model name is 'gemini-1.5-flash'."""
        expander = QueryExpander()
        assert expander.model_name == "gemini-1.5-flash"

    def test_custom_model_name(self, mock_generative_model) -> None:
        """Custom model name is stored correctly."""
        expander = QueryExpander(model_name="gemini-2.0-flash")
        assert expander.model_name == "gemini-2.0-flash"
        mock_generative_model.assert_called_once_with("gemini-2.0-flash")


class TestExpandQuery:
    """expand_query() core behavior."""

    def test_expand_query_calls_generate_content(self, query_expander: QueryExpander) -> None:
        """generate_content() must be called exactly once per expand_query() call."""
        query_expander.expand_query("How does the system handle peak load?")
        query_expander.model.generate_content.assert_called_once()

    def test_expand_query_prompt_contains_original_query(
        self, query_expander: QueryExpander
    ) -> None:
        """The prompt passed to generate_content must contain the original query."""
        original_query = "How does the system handle peak load?"
        query_expander.expand_query(original_query)
        actual_call_args = query_expander.model.generate_content.call_args[0][0]
        assert original_query in actual_call_args, (
            f"Original query not found in prompt: {actual_call_args[:100]}"
        )

    def test_expand_query_returns_string(self, query_expander: QueryExpander) -> None:
        """expand_query() must return a string."""
        result = query_expander.expand_query("How does the system handle peak load?")
        assert isinstance(result, str)

    def test_expanded_query_is_non_empty(self, query_expander: QueryExpander) -> None:
        """Expanded query must be a non-empty string."""
        result = query_expander.expand_query("How does the system handle peak load?")
        assert result.strip(), "Expanded query must not be empty or whitespace-only."

    def test_expanded_query_is_longer_than_original(self, query_expander: QueryExpander) -> None:
        """The expanded query must be richer (longer) than the original."""
        original = "How does the system handle peak load?"
        expanded = query_expander.expand_query(original)
        assert len(expanded) > len(original), (
            f"Expanded query ({len(expanded)} chars) must be longer than "
            f"original ({len(original)} chars)."
        )


class TestBenchmarkQueryExpansions:
    """Verify all 5 benchmark queries expand to their deterministic targets."""

    @pytest.mark.parametrize("original_query, expected_expansion", list(EXPANDED_QUERIES.items()))
    def test_benchmark_query_expands_correctly(
        self,
        query_expander: QueryExpander,
        original_query: str,
        expected_expansion: str,
    ) -> None:
        """Each benchmark query must expand to its pre-defined expansion."""
        result = query_expander.expand_query(original_query)
        assert result == expected_expansion, (
            f"Query: '{original_query}'\n"
            f"Expected: '{expected_expansion[:80]}...'\n"
            f"Got: '{result[:80]}...'"
        )

    def test_all_five_benchmark_queries_present(self) -> None:
        """EXPANDED_QUERIES must contain exactly 5 benchmark queries."""
        assert len(EXPANDED_QUERIES) == 5, (
            f"Expected 5 benchmark queries, found {len(EXPANDED_QUERIES)}."
        )


class TestExpandedQueriesContainTechnicalTerms:
    """Verify expanded queries contain richer technical terminology."""

    def test_peak_load_expansion_contains_scaling_terms(self, query_expander: QueryExpander) -> None:
        expanded = query_expander.expand_query("How does the system handle peak load?")
        technical_terms = ["scaling", "load balanc", "rate limit"]
        found = [term for term in technical_terms if term.lower() in expanded.lower()]
        assert len(found) >= 2, (
            f"Expected at least 2 of {technical_terms} in expanded query, "
            f"found: {found}. Expanded: '{expanded[:100]}'"
        )

    def test_security_expansion_contains_auth_terms(self, query_expander: QueryExpander) -> None:
        expanded = query_expander.expand_query("What security measures protect user data?")
        technical_terms = ["OAuth", "JWT", "encrypt", "RBAC"]
        found = [term for term in technical_terms if term.lower() in expanded.lower()]
        assert len(found) >= 2, (
            f"Expected at least 2 of {technical_terms}, found: {found}."
        )

    def test_monitoring_expansion_contains_observability_terms(
        self, query_expander: QueryExpander
    ) -> None:
        expanded = query_expander.expand_query("How is the system monitored in production?")
        technical_terms = ["Prometheus", "Grafana", "OpenTelemetry", "alerting"]
        found = [term for term in technical_terms if term.lower() in expanded.lower()]
        assert len(found) >= 2, (
            f"Expected at least 2 of {technical_terms}, found: {found}."
        )


class TestQueryExpanderEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_query_raises_value_error(self, query_expander: QueryExpander) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            query_expander.expand_query("")

    def test_whitespace_query_raises_value_error(self, query_expander: QueryExpander) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            query_expander.expand_query("   ")

    def test_model_failure_returns_original_query(self, mock_generative_model) -> None:
        """If generate_content raises, expand_query returns the original query gracefully."""
        mock_instance = mock_generative_model.return_value
        mock_instance.generate_content.side_effect = Exception("API error")

        expander = QueryExpander(model_name="gemini-1.5-flash")
        original = "How does the system handle peak load?"
        result = expander.expand_query(original)

        # Should fall back to original query
        assert result == original, f"Expected fallback to original, got: '{result}'"
