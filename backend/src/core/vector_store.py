"""Vector store implementation using ChromaDB."""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List
from config.settings import settings
from langchain.schema import Document


class VectorStore:
    """Vector store for document embeddings using ChromaDB."""

    def __init__(self):
        """Initialize the vector store."""
        self.client = chromadb.PersistentClient(
            path=str(settings.VECTOR_STORE_DIR), settings=Settings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME
        )

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List of Document objects
        """
        results = self.collection.query(query_texts=[query], n_results=k)

        # Convert results to Document objects
        documents = []
        for i in range(len(results["documents"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            )
            documents.append(doc)

        return documents
