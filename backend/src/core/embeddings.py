from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class EmbeddingManager:
    """Handles embedding of text chunks and storage in ChromaDB."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_and_store_chunks(
        self, chunks: List[Dict[str, Any]], chroma_dir: str, collection_name: str
    ) -> None:
        """
        Embed text chunks and store in ChromaDB.
        Args:
            chunks: List of chunk dicts with 'text' and metadata.
            chroma_dir: Directory for ChromaDB storage.
            collection_name: Name of the ChromaDB collection.
        """
        client = chromadb.PersistentClient(
            path=chroma_dir, settings=Settings(allow_reset=True)
        )
        collection = client.get_or_create_collection(name=collection_name)
        texts = [chunk["text"] for chunk in chunks if chunk["text"]]
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()
        ids = [f"chunk_{i}" for i in range(len(texts))]
        metadatas = [
            {
                "type": chunk["type"],
                "section": chunk["section"],
                "page": chunk["page"],
                "bbox": chunk["bbox"],
            }
            for chunk in chunks
            if chunk["text"]
        ]
        collection.add(
            embeddings=embeddings, documents=texts, ids=ids, metadatas=metadatas
        )
        print(f"Embedded {len(texts)} chunks and stored in ChromaDB at {chroma_dir}")
