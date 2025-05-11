"""Configuration settings for the RAG application."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application settings."""

    # Paths
    DATA_DIR: Path = Path("data")
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    VECTOR_STORE_DIR: Path = DATA_DIR / "vector_store"

    # # Document processing
    K: int = 1  # Number of documents to retrieve for context
    # CHUNK_SIZE: int = 1000
    # CHUNK_OVERLAP: int = 200

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Vector store
    COLLECTION_NAME: str = "test_collection"

    # API Keys
    GOOGLE_API_KEY: str

    # LLM settings
    LLM_MODEL: str = "gemini-2.5-flash-preview-04-17"
    TEMPERATURE: float = 0.0
    MAX_OUTPUT_TOKENS: int = 2048

    # Test queries
    TEST_QUERIES: List[str] = [
        "What are the key financial highlights mentioned in the Haycarb report?",
        "is it a good time to buy haycarb?",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",  # Allow extra fields in the settings
    )


settings = Settings()
