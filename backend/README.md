# BusinessAI RAG Backend

A Retrieval-Augmented Generation (RAG) backend for processing and querying PDF documents (like annual reports) using Google's Generative AI models.

## Features

- PDF document ingestion from the `data/raw` directory.
- Text extraction with header/footer removal.
- Document chunking.
- Text embedding generation using Sentence Transformers (`all-MiniLM-L6-v2`).
- Vector storage using ChromaDB.
- RAG pipeline testing script (`src/test_rag.py`) using Langchain and Google Generative AI (`gemini-pro`).

## Prerequisites

- Python 3.x
- Google Generative AI API Key

## Setup

1.  **Clone the repository** (if you haven't already).

2.  **Navigate to the `backend` directory:**
    ```bash
    cd backend
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Create a `.env` file** in the `backend` directory (`backend/.env`) with your Google API key:
    ```dotenv
    GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE
    ```
    Replace `YOUR_GOOGLE_API_KEY_HERE` with your actual key.

6.  **Place your PDF files** into the `backend/data/raw/` directory. Example: `backend/data/raw/Haycarb_2024.pdf`.

## Usage

### 1. Process Documents and Build Vector Store

Run the main processing script. This will:
    - Read PDFs from `data/raw`.
    - Process and chunk the text.
    - Generate embeddings.
    - Store the embeddings and text in the ChromaDB vector store located in `data/vector_store`.

```bash
python src/main.py
```

### 2. Test the RAG Pipeline

Run the test script to query the vector store and get a response from the LLM based on the retrieved context.

```bash
python src/test_rag.py
```

You can modify the `test_query` variable within `src/test_rag.py` to ask different questions.

## Project Structure

```
backend/
├── .env                  # Stores API keys (Needs to be created)
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── data/
│   ├── raw/              # Input PDF files go here
│   ├── processed/        # Intermediate processed text (optional output)
│   └── vector_store/     # ChromaDB vector store data
└── src/
    ├── main.py           # Main script for processing documents
    ├── test_rag.py       # Script to test the RAG pipeline
    ├── config/
    │   └── settings.py     # Configuration settings (Pydantic)
    ├── core/
    │   ├── document_processor.py # Handles PDF reading and chunking
    │   ├── embeddings.py         # Handles embedding generation
    │   └── vector_store.py       # Handles ChromaDB interaction
    └── utils/              # Utility functions (if any)
        |── download__.py # download reports/documents and store in docs/raw
        |── text_processor__.py # clean and preprocess text
```

## Development

- Configuration is managed via `src/config/settings.py` using Pydantic.
- PDF processing uses `pdfplumber`.
- RAG pipeline and LLM interaction use `langchain` and `langchain-google-genai`.
- Vector storage uses `chromadb`.
- Embeddings are generated using `sentence-transformers`.

## License

MIT License