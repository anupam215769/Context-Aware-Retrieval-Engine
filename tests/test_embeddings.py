"""
test_embeddings.py
------------------
Tests for EmbeddingService (src/embeddings.py).

Covers:
  - Output shape for single and batch embeddings
  - L2 normalization (norm ≈ 1.0)
  - Semantic similarity ordering (similar texts rank higher than dissimilar ones)
  - Edge-case validation (empty input)
  - Mocked TextEmbeddingModel interface compatibility
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embeddings import EmbeddingService


class TestEmbeddingShape:
    """Verify output dimensionality."""

    def test_single_embedding_shape(self, embedding_service: EmbeddingService) -> None:
        """Single embedding must be a 1D array of length 768."""
        emb = embedding_service.get_embedding("The system scales horizontally.")
        assert isinstance(emb, np.ndarray), "Embedding must be a numpy array."
        assert emb.ndim == 1, f"Expected 1D, got {emb.ndim}D array."
        assert emb.shape[0] == 768, f"Expected dim=768, got {emb.shape[0]}."

    def test_batch_embeddings_shape(self, embedding_service: EmbeddingService) -> None:
        """Batch embeddings must be (N, 768)."""
        texts = [
            "Load balancer distributes traffic.",
            "Redis cache reduces latency.",
            "Circuit breaker prevents cascading failures.",
        ]
        embs = embedding_service.get_embeddings(texts)
        assert embs.ndim == 2, f"Expected 2D, got {embs.ndim}D array."
        assert embs.shape == (3, 768), f"Expected (3, 768), got {embs.shape}."

    def test_batch_single_document(self, embedding_service: EmbeddingService) -> None:
        """Batch of one document should return shape (1, 768)."""
        embs = embedding_service.get_embeddings(["Single document."])
        assert embs.shape == (1, 768)

    def test_embedding_dtype_is_float32(self, embedding_service: EmbeddingService) -> None:
        """Embeddings must be float32 for FAISS compatibility."""
        emb = embedding_service.get_embedding("Test text.")
        assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}."


class TestEmbeddingNormalization:
    """Verify L2 normalization (required for cosine similarity via inner product)."""

    def test_single_embedding_is_unit_norm(self, embedding_service: EmbeddingService) -> None:
        """Single embedding must have L2 norm of 1.0 (±1e-5 tolerance)."""
        emb = embedding_service.get_embedding("The Kubernetes HPA scales pods during peak load.")
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm:.6f}."

    def test_batch_embeddings_are_unit_norm(self, embedding_service: EmbeddingService) -> None:
        """All embeddings in a batch must have L2 norm of 1.0."""
        texts = ["Prometheus metrics.", "JWT authentication.", "RabbitMQ dead letter queues."]
        embs = embedding_service.get_embeddings(texts)
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), (
            f"Not all embeddings are unit-normalized. Norms: {norms}"
        )


class TestEmbeddingSimilarity:
    """Verify semantic similarity ordering."""

    def test_similar_texts_have_higher_similarity(self, embedding_service: EmbeddingService) -> None:
        """
        Semantically similar texts must have higher cosine similarity than
        semantically dissimilar texts.
        """
        query = "How does the system scale under high traffic load?"
        similar_text = "Horizontal auto-scaling with load balancing handles peak traffic surges."
        dissimilar_text = "The recipe calls for two cups of flour and one egg."

        q_emb = embedding_service.get_embedding(query)
        sim_emb = embedding_service.get_embedding(similar_text)
        dis_emb = embedding_service.get_embedding(dissimilar_text)

        sim_score = float(np.dot(q_emb, sim_emb))
        dis_score = float(np.dot(q_emb, dis_emb))

        assert sim_score > dis_score, (
            f"Expected similar_score ({sim_score:.4f}) > dissimilar_score ({dis_score:.4f})."
        )

    def test_identical_texts_have_max_similarity(self, embedding_service: EmbeddingService) -> None:
        """Identical texts should have cosine similarity very close to 1.0."""
        text = "The circuit breaker opens when failure rate exceeds 50%."
        emb1 = embedding_service.get_embedding(text)
        emb2 = embedding_service.get_embedding(text)
        score = float(np.dot(emb1, emb2))
        assert score > 0.999, f"Identical texts should have score ≈ 1.0, got {score:.6f}."

    def test_technical_similarity_ordering(self, embedding_service: EmbeddingService) -> None:
        """
        Technical query about monitoring should rank an observability text
        higher than a security text.
        """
        query = "How does the system collect metrics and send alerts?"
        monitoring_text = "Prometheus scrapes metrics; AlertManager sends PagerDuty notifications."
        security_text = "OAuth 2.0 and JWT tokens handle user authentication and authorization."

        q_emb = embedding_service.get_embedding(query)
        mon_emb = embedding_service.get_embedding(monitoring_text)
        sec_emb = embedding_service.get_embedding(security_text)

        mon_score = float(np.dot(q_emb, mon_emb))
        sec_score = float(np.dot(q_emb, sec_emb))

        assert mon_score > sec_score, (
            f"Expected monitoring score ({mon_score:.4f}) > security score ({sec_score:.4f})."
        )


class TestEmbeddingValidation:
    """Edge-case and validation tests."""

    def test_empty_text_raises_value_error(self, embedding_service: EmbeddingService) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            embedding_service.get_embedding("")

    def test_whitespace_only_raises_value_error(self, embedding_service: EmbeddingService) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            embedding_service.get_embedding("   ")

    def test_empty_list_raises_value_error(self, embedding_service: EmbeddingService) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            embedding_service.get_embeddings([])


class TestMockedVertexAIEmbeddingModel:
    """
    Demonstrates interface compatibility with Vertex AI TextEmbeddingModel.

    In production, EmbeddingService would be replaced by a class wrapping
    vertexai.language_models.TextEmbeddingModel. These tests show how the
    mock would be structured and verify the interface contract.
    """

    def test_mock_text_embedding_model_interface(self) -> None:
        """
        Mock vertexai.language_models.TextEmbeddingModel and verify that
        the EmbeddingService interface can be replicated.
        """
        # Simulate what Vertex AI TextEmbeddingModel.get_embeddings() returns
        mock_embedding_values = np.random.rand(768).astype(np.float32)
        mock_embedding_values /= np.linalg.norm(mock_embedding_values)

        with patch("src.embeddings.SentenceTransformer") as mock_st:
            mock_model_instance = MagicMock()
            mock_model_instance.get_sentence_embedding_dimension.return_value = 768
            mock_model_instance.encode.return_value = mock_embedding_values.reshape(1, -1)
            mock_st.return_value = mock_model_instance

            service = EmbeddingService(model_name="mock-model")
            result = service.get_embedding("Test text for mocked embedding.")

        assert result.shape == (768,), "Mocked embedding must have shape (768,)."
        mock_model_instance.encode.assert_called_once()

    def test_mock_batch_embedding_interface(self) -> None:
        """
        Verify batch embedding interface works with mocked model.
        """
        n_docs = 3
        mock_batch = np.random.rand(n_docs, 768).astype(np.float32)
        norms = np.linalg.norm(mock_batch, axis=1, keepdims=True)
        mock_batch /= norms

        with patch("src.embeddings.SentenceTransformer") as mock_st:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 768
            mock_instance.encode.return_value = mock_batch
            mock_st.return_value = mock_instance

            service = EmbeddingService()
            results = service.get_embeddings(["doc1", "doc2", "doc3"])

        assert results.shape == (3, 768)
        mock_instance.encode.assert_called_once()
