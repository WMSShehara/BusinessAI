"""Script for the RAG application with LLM integration."""

import logging
from pathlib import Path
import sys
from typing import List
from config.settings import settings
from core.vector_store import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document

# Add the project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "data" / "logs" / "rag.log"),
    ],
)
logger = logging.getLogger(__name__)


def format_docs(docs: List[Document]) -> str:
    """Format a list of documents into a single string.

    Args:
        docs: List of Document objects containing text and metadata

    Returns:
        Formatted string containing all document texts
    """
    return "\n\n".join(doc.page_content for doc in docs)


class RAGSystem:
    """RAG (Retrieval-Augmented Generation) system using Google's Gemini model."""

    def __init__(self, k=settings.K):
        """Initialize the RAG system.

        Args:
            k: Number of documents to retrieve for context
        """
        self.k = k
        self.vector_store = VectorStore()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.TEMPERATURE,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        )

        # Define the prompt template
        self.prompt = PromptTemplate.from_template(
            """You are a helpful AI assistant. Answer the question based only on the following context.
            If you cannot find the answer in the context, say "I cannot find the answer in the provided context."
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
        )

        # Create the RAG chain
        self.rag_chain = (
            {
                "context": RunnablePassthrough() | self._retrieve_context,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from the vector store.

        Args:
            query: The search query

        Returns:
            Formatted string containing the retrieved context
        """
        search_results = self.vector_store.search(query, k=self.k)
        return format_docs(search_results)

    def query(self, question: str) -> str:
        """Query the RAG system with a question.

        Args:
            question: The question to ask

        Returns:
            The generated answer
        """
        logger.info(f"Processing question: {question}")
        try:
            response = self.rag_chain.invoke(question)
            logger.info("Successfully generated response")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"


def main():
    """Main function to test the RAG system."""
    (project_root / "data" / "logs").mkdir(exist_ok=True)
    rag = RAGSystem(k=5)

    for query in settings.TEST_QUERIES:
        logger.info(f"\n{'=' * 50}\nProcessing query: {query}\n{'=' * 50}")
        response = rag.query(query)
        logger.info(f"\nResponse:\n{response}\n")


if __name__ == "__main__":
    main()
