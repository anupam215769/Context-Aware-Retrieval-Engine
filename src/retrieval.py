"""
retrieval.py
------------
RetrievalEngine: Central orchestrator for the RAG pipeline. Manages ingestion,
embedding generation, and two retrieval strategies:

  Strategy A (Raw Vector Search):
      Query → embed directly → FAISS search → ranked results

  Strategy B (AI-Enhanced Retrieval):
      Query → GenerativeModel expansion → embed expanded query → FAISS search → ranked results

The engine is designed to be dependency-injected: EmbeddingService, VectorStore,
and QueryExpander are passed in, making it fully testable with mocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.embeddings import EmbeddingService
from src.vector_store import VectorStore, SearchResult
from src.query_expander import QueryExpander

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Aggregated result from a single retrieval call.

    Attributes
    ----------
    query : str
        The original user query.
    strategy : str
        'A' or 'B'.
    expanded_query : str or None
        The AI-expanded query (Strategy B only).
    results : list[SearchResult]
        Ranked retrieval results.
    """
    query: str
    strategy: str
    expanded_query: Optional[str]
    results: list[SearchResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable dictionary."""
        return {
            "query": self.query,
            "strategy": self.strategy,
            "expanded_query": self.expanded_query,
            "results": [
                {
                    "rank": r.rank,
                    "document_id": r.document.get("id"),
                    "title": r.document.get("title"),
                    "category": r.document.get("category"),
                    "score": round(r.score, 6),
                    "snippet": r.document.get("content", "")[:200] + "...",
                }
                for r in self.results
            ],
        }


class RetrievalEngine:
    """
    Orchestrates ingestion and retrieval for both strategies.

    Parameters
    ----------
    embedding_service : EmbeddingService
        Provides text-to-vector conversion.
    vector_store : VectorStore
        FAISS-backed store for similarity search.
    query_expander : QueryExpander, optional
        Required for Strategy B. If None, strategy_b_search() will raise RuntimeError.

    Example
    -------
    >>> engine = RetrievalEngine(embedding_service, vector_store, query_expander)
    >>> count = engine.ingest(documents)
    >>> results_a = engine.strategy_a_search("How does the system handle peak load?")
    >>> results_b = engine.strategy_b_search("How does the system handle peak load?")
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        query_expander: Optional[QueryExpander] = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.query_expander = query_expander

    def ingest(self, documents: list[dict[str, Any]]) -> int:
        """
        Ingest a list of documents into the vector store.

        Generates embeddings for each document's 'content' field and stores
        them alongside the full document metadata.

        Parameters
        ----------
        documents : list[dict]
            Each dict must have at least a 'content' key.

        Returns
        -------
        int
            Number of documents ingested.

        Raises
        ------
        ValueError
            If documents is empty or any document is missing 'content'.
        """
        if not documents:
            raise ValueError("documents must be a non-empty list.")
        for i, doc in enumerate(documents):
            if "content" not in doc:
                raise ValueError(f"Document at index {i} is missing the 'content' key.")

        texts = [doc["content"] for doc in documents]
        logger.info("Ingesting %d documents...", len(documents))

        embeddings = self.embedding_service.get_embeddings(texts)
        self.vector_store.add_documents(np.array(embeddings, dtype=np.float32), documents)

        logger.info("Ingestion complete. VectorStore now has %d documents.", self.vector_store.count)
        return len(documents)

    def strategy_a_search(self, query: str, top_k: int = 3) -> RetrievalResult:
        """
        Strategy A: Raw Vector Search.

        Embeds the query directly and performs cosine similarity search.

        Parameters
        ----------
        query : str
            The user's original search query.
        top_k : int
            Number of top results to return. Default: 3.

        Returns
        -------
        RetrievalResult
            Contains the original query, strategy='A', no expanded_query,
            and the ranked search results.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        logger.info("[Strategy A] Query: '%s'", query)
        query_embedding = self.embedding_service.get_embedding(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return RetrievalResult(
            query=query,
            strategy="A",
            expanded_query=None,
            results=results,
        )

    def strategy_b_search(self, query: str, top_k: int = 3) -> RetrievalResult:
        """
        Strategy B: AI-Enhanced Retrieval.

        Expands the query using a generative model, then embeds the expanded
        query and performs cosine similarity search.

        Parameters
        ----------
        query : str
            The user's original search query.
        top_k : int
            Number of top results to return. Default: 3.

        Returns
        -------
        RetrievalResult
            Contains the original query, strategy='B', the expanded_query,
            and the ranked search results.

        Raises
        ------
        RuntimeError
            If query_expander was not provided at construction time.
        """
        if self.query_expander is None:
            raise RuntimeError(
                "QueryExpander is required for Strategy B but was not provided."
            )
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")

        logger.info("[Strategy B] Original query: '%s'", query)
        expanded_query = self.query_expander.expand_query(query)
        logger.info("[Strategy B] Expanded query: '%s'", expanded_query)

        query_embedding = self.embedding_service.get_embedding(expanded_query)
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return RetrievalResult(
            query=query,
            strategy="B",
            expanded_query=expanded_query,
            results=results,
        )

    def compare_strategies(
        self,
        query: str,
        top_k: int = 3,
    ) -> tuple[RetrievalResult, RetrievalResult]:
        """
        Run both strategies for a query and return their results together.

        Parameters
        ----------
        query : str
            User query.
        top_k : int
            Number of results per strategy.

        Returns
        -------
        tuple[RetrievalResult, RetrievalResult]
            (strategy_a_result, strategy_b_result)
        """
        result_a = self.strategy_a_search(query, top_k=top_k)
        result_b = self.strategy_b_search(query, top_k=top_k)
        return result_a, result_b

    def __repr__(self) -> str:
        has_expander = self.query_expander is not None
        return (
            f"RetrievalEngine("
            f"store_count={self.vector_store.count}, "
            f"has_expander={has_expander})"
        )
