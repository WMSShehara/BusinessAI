"""Vector store module for the RAG application."""

from typing import List, Optional

import chromadb
from chromadb.config import Settings

from config.settings import settings
from core.embeddings import EmbeddingGenerator


class VectorStore:
    """Handles vector storage and retrieval."""

    def __init__(self) -> None:
        """Initialize the vector store."""
        self.client = chromadb.PersistentClient(
            path=str(settings.VECTOR_STORE_DIR), settings=Settings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME
        )

    def add_documents(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store.

        Args:
            texts: List of text chunks
            embeddings: List of corresponding embeddings
        """
        # Generate IDs for the documents
        ids = [f"doc_{i}" for i in range(len(texts))]

        # Add to collection
        self.collection.add(embeddings=embeddings, documents=texts, ids=ids)

    def search(self, query: str, k: int = 5) -> List[str]:
        """Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        # Generate embedding for the query
        embedding_generator = EmbeddingGenerator()
        query_embedding = embedding_generator.generate_embeddings([query])[0]

        # Search the collection
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)

        return results["documents"][0]
