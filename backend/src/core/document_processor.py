"""Document processing module for the RAG application."""

from pathlib import Path
from typing import List, Optional

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import settings
from utils.text_processor import preprocess_text


class DocumentProcessor:
    """Handles document processing and chunking."""

    def __init__(self) -> None:
        """Initialize the document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text as a string
        """
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    def process_document(self, file_path: Path) -> List[str]:
        """Process a document and split it into chunks.

        Args:
            file_path: Path to the document

        Returns:
            List of text chunks
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(file_path)

        # Preprocess the text
        processed_text = preprocess_text(text)

        # Split into chunks
        chunks = self.text_splitter.split_text(processed_text)

        return chunks

    def process_all_documents(self) -> List[str]:
        """Process all documents in the raw data directory.

        Returns:
            List of all text chunks from all documents
        """
        all_chunks = []

        # Process each PDF file in the raw data directory
        for pdf_file in settings.RAW_DATA_DIR.glob("*.pdf"):
            chunks = self.process_document(pdf_file)
            all_chunks.extend(chunks)

        return all_chunks
