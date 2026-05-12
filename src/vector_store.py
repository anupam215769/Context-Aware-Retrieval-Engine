"""
vector_store.py
---------------
VectorStore: FAISS-backed vector store using IndexFlatIP on L2-normalized vectors,
which is mathematically equivalent to cosine similarity search.

Design rationale
----------------
• Cosine Similarity (via Inner Product on normalized vectors):
    - sentence-transformers models are trained so that semantic similarity
      correlates with cosine similarity.
    - Direction-based metric: invariant to embedding magnitude, ideal for NLP.
    - Implementation: FAISS IndexFlatIP + faiss.normalize_L2() at add/query time.
    - Score range: [-1, 1], where 1 = identical semantics, 0 = orthogonal.

• Why not Euclidean (L2)?
    - Sensitive to both direction AND magnitude.
    - For unit-normalized vectors, L2 distance and cosine similarity are
      monotonically related (||a-b||² = 2(1 - a·b)), so rankings are the same —
      but cosine scores are more interpretable (bounded in [-1, 1]).
    - L2 is preferred for spatial/coordinate data, not NLP embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single retrieval result from the vector store."""

    rank: int
    document: dict[str, Any]
    score: float  # Cosine similarity in [-1, 1]; higher is more similar

    def __repr__(self) -> str:
        title = self.document.get("title", self.document.get("id", "unknown"))
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, title='{title}')"


class VectorStore:
    """
    FAISS-based vector store with cosine similarity search.

    Uses IndexFlatIP (exact inner product search) on L2-normalized vectors.
    This provides exact cosine similarity without approximation errors, which is
    appropriate for small-to-medium corpora (< 1M documents).

    Attributes
    ----------
    dimension : int
        Embedding vector dimension.
    index : faiss.IndexFlatIP
        FAISS inner product index.
    documents : list[dict]
        Metadata store; index i in documents corresponds to vector i in index.

    Example
    -------
    >>> import numpy as np
    >>> store = VectorStore(dimension=768)
    >>> embeddings = np.random.rand(3, 768).astype(np.float32)
    >>> docs = [{"id": f"doc_{i}", "content": f"text {i}"} for i in range(3)]
    >>> store.add_documents(embeddings, docs)
    >>> store.count
    3
    """

    def __init__(self, dimension: int) -> None:
        """
        Initialize the vector store.

        Parameters
        ----------
        dimension : int
            Dimensionality of the embedding vectors (e.g., 768 for all-mpnet-base-v2).
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {dimension}.")
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.documents: list[dict[str, Any]] = []
        logger.info("VectorStore initialized. Metric: Cosine (IP on normalized), dim=%d", dimension)

    def add_documents(
        self,
        embeddings: np.ndarray,
        documents: list[dict[str, Any]],
    ) -> None:
        """
        Add document embeddings and their metadata to the store.

        Parameters
        ----------
        embeddings : np.ndarray
            Float32 array of shape (N, dimension). Will be L2-normalized in-place
            before adding to the FAISS index (safety normalization — embeddings
            from EmbeddingService are already normalized, this is a safeguard).
        documents : list[dict]
            Metadata for each document. Must have len(documents) == embeddings.shape[0].

        Raises
        ------
        ValueError
            If embeddings and documents lengths do not match, or dimension mismatch.
        """
        if len(embeddings) != len(documents):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and documents ({len(documents)}) "
                "must have the same length."
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        vectors = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(vectors)  # Safety normalization

        self.index.add(vectors)
        self.documents.extend(documents)
        logger.info("Added %d documents to VectorStore. Total: %d", len(documents), self.count)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[SearchResult]:
        """
        Search for the top_k most semantically similar documents.

        Parameters
        ----------
        query_embedding : np.ndarray
            Float32 array of shape (dimension,) or (1, dimension).
        top_k : int
            Number of results to return. Clamped to store size if larger.

        Returns
        -------
        list[SearchResult]
            Ranked list of SearchResult objects, most similar first.

        Raises
        ------
        ValueError
            If the store is empty.
        """
        if self.count == 0:
            raise ValueError("VectorStore is empty. Add documents before searching.")

        top_k = min(top_k, self.count)  # Cannot return more than we have

        query = query_embedding.reshape(1, -1).copy().astype(np.float32)
        faiss.normalize_L2(query)  # Safety normalization

        scores, indices = self.index.search(query, top_k)

        results: list[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue  # FAISS pads with -1 if fewer results exist
            results.append(
                SearchResult(
                    rank=rank,
                    document=self.documents[idx],
                    score=float(score),
                )
            )

        logger.debug("Search returned %d results (top_k=%d).", len(results), top_k)
        return results

    def reset(self) -> None:
        """Clear all documents and reset the FAISS index."""
        self.index.reset()
        self.documents.clear()
        logger.info("VectorStore reset.")

    @property
    def count(self) -> int:
        """Number of documents in the store."""
        return self.index.ntotal

    def __repr__(self) -> str:
        return f"VectorStore(dimension={self.dimension}, count={self.count})"
