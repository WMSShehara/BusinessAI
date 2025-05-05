# BusinessAI RAG Application

A Retrieval-Augmented Generation (RAG) application for processing and querying company annual reports using Google's Gemini model through Vertex AI.

## Features

- PDF document processing with header/footer removal
- Text chunking and embedding
- Vector store for efficient document retrieval
- Natural language querying of documents using Gemini Pro
- Document summarization
- RESTful API interface

## Prerequisites

1. Google Cloud Platform account
2. Vertex AI API enabled
3. Service account with appropriate permissions

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory with the following variables:
```
GOOGLE_PROJECT_ID=your_google_cloud_project_id
GOOGLE_LOCATION=us-central1  # or your preferred location
```

4. Place your PDF files in the `data/raw` directory with the naming convention:
```
CompanyName_Year.pdf
```
Example: `Haycarb_2024.pdf`

## Usage

1. Start the API server:
```bash
cd src
python -m api.main
```

2. The API will be available at `http://localhost:8000`

### API Endpoints

#### Query Documents
- **POST** `/query`
- Query the documents using natural language
- Optional filters for company and year
- Returns answer and source documents

Example request:
```json
{
    "query": "What were the key financial metrics for Haycarb in 2024?",
    "company": "Haycarb",
    "year": 2024
}
```

#### Get Document Summary
- **POST** `/summary`
- Get a comprehensive summary of a specific document
- Requires company name and year
- Returns summary and source documents

Example request:
```json
{
    "company": "Haycarb",
    "year": 2024
}
```

## Project Structure

```
backend/
├── src/
│   ├── api/
│   │   └── main.py           # FastAPI application
│   ├── core/
│   │   └── config.py         # Configuration settings
│   ├── services/
│   │   ├── document_processor.py  # PDF processing
│   │   ├── vector_store.py        # Vector storage
│   │   └── rag_service.py         # RAG operations
│   └── models/
├── data/
│   ├── raw/                  # Local PDF storage
│   └── vector_store/         # Vector store data
├── requirements.txt
└── README.md
```

## Development

- The application uses FastAPI for the API layer
- LangChain for RAG operations
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Google's Gemini Pro through Vertex AI for text generation

## License

MIT License 