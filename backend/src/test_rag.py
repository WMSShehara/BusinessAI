"""Test script for the RAG application with LLM integration."""

import logging
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

# Correctly import the settings instance
from config.settings import settings
from core.vector_store import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables (for API key)
# load_dotenv(dotenv_path=project_root / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_docs(docs: list[str]):
    """Joins a list of document strings."""
    return "\n\n".join(docs)

def test_rag_query_with_llm(query: str, k: int = 5) -> None:
    """Test the RAG system by querying the vector store and then an LLM.
    
    Args:
        query: The query to search for
        k: Number of results to retrieve from vector store
    """
    logger.info(f"Testing RAG query with LLM: '{query}'")
    
    # Initialize vector store
    vector_store = VectorStore()

    # Initialize the LLM using settings object for both model and API key
    llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, google_api_key=settings.GOOGLE_API_KEY)

    # Define the prompt template
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = PromptTemplate.from_template(template)

    # Define a function to retrieve context using the custom search method
    def retrieve_context(input_query: str) -> str:
        search_results = vector_store.search(input_query, k=k)
        return format_docs(search_results) # format_docs now needs to handle list of strings

    # Create the RAG chain using the custom retrieval function
    rag_chain = (
        {"context": RunnablePassthrough() | retrieve_context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain
    logger.info("Invoking RAG chain...")
    response = rag_chain.invoke(query)

    # Display results
    logger.info(f"\n--- LLM Response --- \n{response}\n")


if __name__ == "__main__":
    # Test with a sample query
    test_query = "What are the key financial highlights mentioned in the Haycarb report?"
    test_rag_query_with_llm(test_query)
