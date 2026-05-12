"""
embeddings.py
-------------
EmbeddingService: Wraps sentence-transformers (all-mpnet-base-v2, 768d) to
simulate the interface of Vertex AI's TextEmbeddingModel locally.

Interface notes:
  - get_embedding(text)       → np.ndarray of shape (768,), L2-normalized
  - get_embeddings(texts)     → np.ndarray of shape (N, 768), L2-normalized
  - dimension                 → int (768)

The embeddings are L2-normalized at generation time so that inner-product
search in FAISS is equivalent to cosine similarity.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model: all-mpnet-base-v2 — 768d, matches textembedding-gecko dimensions
DEFAULT_MODEL_NAME = "all-mpnet-base-v2"


class EmbeddingService:
    """
    Local embedding service that simulates Vertex AI TextEmbeddingModel behavior.

    Uses sentence-transformers under the hood. Embeddings are L2-normalized so
    that cosine similarity can be computed via inner product (compatible with
    FAISS IndexFlatIP).

    Example
    -------
    >>> service = EmbeddingService()
    >>> emb = service.get_embedding("How does load balancing work?")
    >>> emb.shape
    (768,)
    >>> np.linalg.norm(emb)  # doctest: +ELLIPSIS
    1.0...
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """
        Initialize the embedding service.

        Parameters
        ----------
        model_name : str
            HuggingFace sentence-transformers model name.
            Defaults to 'all-mpnet-base-v2' (768d — same dimensionality as
            Vertex AI textembedding-gecko@003).
        """
        logger.info("Loading embedding model: %s", model_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # get_embedding_dimension() is the new name in sentence-transformers >= 3.x;
        # fall back to get_sentence_embedding_dimension() for older versions.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension: int = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()  # type: ignore[attr-defined]
        logger.info("Embedding model loaded. Dimension: %d", self.dimension)

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate L2-normalized embeddings for a batch of texts.

        Parameters
        ----------
        texts : list[str]
            List of strings to embed.

        Returns
        -------
        np.ndarray
            Float32 array of shape (len(texts), self.dimension), L2-normalized.

        Raises
        ------
        ValueError
            If texts is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list of strings.")

        logger.debug("Generating embeddings for %d texts.", len(texts))
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # L2-normalize so IP == cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an L2-normalized embedding for a single text.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        np.ndarray
            Float32 array of shape (self.dimension,), L2-normalized.
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string.")

        return self.get_embeddings([text])[0]

    def __repr__(self) -> str:
        return f"EmbeddingService(model='{self.model_name}', dimension={self.dimension})"
