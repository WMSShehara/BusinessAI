"""Embeddings module for the RAG application."""

from typing import List

from sentence_transformers import SentenceTransformer

from config.settings import settings


class EmbeddingGenerator:
    """Handles generation of document embeddings."""

    def __init__(self) -> None:
        """Initialize the embedding generator."""
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text chunks to embed

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        return self.model.encode(texts, show_progress_bar=True).tolist()
