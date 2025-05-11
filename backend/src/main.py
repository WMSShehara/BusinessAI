"""Main application for the RAG system."""

import logging
import sys
from pathlib import Path
from core.document_processor import DocumentProcessor
from core.embeddings import EmbeddingManager
from core.vector_store import VectorStoreManager
from config.settings import settings

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Main function to process documents and create vector store."""
    logger.info("Starting RAG application setup...")

    # Setup directories
    setup_directories()

    # Initialize components
    document_processor = DocumentProcessor()
    embedding_manager = EmbeddingManager(model_name=settings.EMBEDDING_MODEL)
    vector_store = VectorStoreManager(
        chroma_dir=str(settings.VECTOR_STORE_DIR),
        collection_name=settings.COLLECTION_NAME,
        model_name=settings.EMBEDDING_MODEL,
    )

    # Process all PDFs in the raw data directory
    logger.info("Processing all documents in %s...", settings.RAW_DATA_DIR)
    chunks = document_processor.process_all_documents(
        input_dir=settings.RAW_DATA_DIR,
        output_dir=settings.PROCESSED_DATA_DIR,
    )

    if not chunks:
        logger.warning(
            "No documents were processed. Please check if there are PDF files in the raw data directory."
        )
        return

    # Generate embeddings and store in vector DB
    logger.info("Generating embeddings and storing in vector database...")
    embedding_manager.embed_and_store_chunks(
        chunks=chunks,
        chroma_dir=str(settings.VECTOR_STORE_DIR),
        collection_name=settings.COLLECTION_NAME,
    )

    logger.info("RAG application setup completed successfully!")


if __name__ == "__main__":
    main()
