"""
ingestion.py
------------
DataIngestionPipeline: Convenience orchestrator that ties dataset loading
and RetrievalEngine.ingest() together into a single pipeline class.

This wraps the three-step process of:
  1. Loading the document corpus (from dataset.py or a custom list)
  2. Generating embeddings via EmbeddingService
  3. Storing vectors and metadata in VectorStore

The class follows the orchestrator pattern — it does not own any heavy
resources but wires together pre-built components.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.dataset import TECHNICAL_DOCUMENTS
from src.retrieval import RetrievalEngine

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Orchestrates loading the document corpus and ingesting it into a RetrievalEngine.

    Parameters
    ----------
    engine : RetrievalEngine
        The retrieval engine to ingest documents into.

    Example
    -------
    >>> pipeline = DataIngestionPipeline(engine)
    >>> count = pipeline.ingest_dataset()   # uses built-in TECHNICAL_DOCUMENTS
    >>> print(f"Ingested {count} documents")
    Ingested 10 documents
    """

    def __init__(self, engine: RetrievalEngine) -> None:
        self.engine = engine

    def ingest_dataset(
        self,
        documents: Optional[list[dict[str, Any]]] = None,
    ) -> int:
        """
        Load and ingest a document corpus.

        Parameters
        ----------
        documents : list[dict], optional
            Custom document list. If None, uses the built-in TECHNICAL_DOCUMENTS
            corpus (10 technical paragraphs on distributed systems).

        Returns
        -------
        int
            Number of documents ingested.
        """
        corpus = documents if documents is not None else TECHNICAL_DOCUMENTS
        logger.info(
            "DataIngestionPipeline: starting ingestion of %d documents.", len(corpus)
        )
        count = self.engine.ingest(corpus)
        logger.info(
            "DataIngestionPipeline: ingestion complete. %d documents in store.", count
        )
        return count

    @property
    def document_count(self) -> int:
        """Number of documents currently in the backing vector store."""
        return self.engine.vector_store.count

    def __repr__(self) -> str:
        return f"DataIngestionPipeline(document_count={self.document_count})"
