"""Main application for the RAG system."""

import logging
import sys
from pathlib import Path
from core.document_processor import DocumentProcessor
from core.embeddings import EmbeddingGenerator
from core.vector_store import VectorStore
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
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore()

    # Process documents
    logger.info("Processing documents...")
    chunks = document_processor.process_all_documents()

    if not chunks:
        logger.warning(
            "No documents were processed. Please check if there are PDF files in the raw data directory."
        )
        return

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embedding_generator.generate_embeddings(chunks)

    # Store in vector database
    logger.info("Storing embeddings in vector database...")
    vector_store.add_documents(chunks, embeddings)

    logger.info("RAG application setup completed successfully!")


if __name__ == "__main__":
    main()
