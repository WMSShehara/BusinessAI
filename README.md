# BusinessAI

A RAG (Retrieval-Augmented Generation) application that analyzes annual reports of companies and extracts deep insights using Google's Gemini model.

## Project Structure

BusinessAI
├── backend/
│ ├── src/
│ │ ├── config/
│ │ │ └── settings.py 
│ │ ├── core/
│ │ │ ├── document_processor.py
│ │ │ └── vector_store.py 
│ │ └── llm_setup.py 
│ ├── data/
│ │ ├── raw/
│ │ ├── processed/ 
│ │ ├── vector_store/
│ │ └── logs/ 
│ └── requirements.txt 
└── README.md


## Key Components

### 1. Document Processing (`document_processor.py`)
- Processes PDF files using LayoutParser and Tesseract OCR
- Extracts text with layout awareness
- Handles tables, figures, and text blocks
- Optimized for memory usage with page-by-page processing

### 2. Vector Store (`vector_store.py`)
- Manages document embeddings using ChromaDB
- Handles document storage and retrieval
- Provides semantic search capabilities

### 3. RAG System (`llm_setup.py`)
- Implements the RAG pipeline using Google's Gemini model
- Combines document retrieval with LLM generation
- Provides a simple query interface

### 4. Configuration (`settings.py`)
- Centralizes all application settings
- Manages environment variables
- Configures paths, models, and API keys

## System Architechture
https://www.mermaidchart.com/raw/454e8117-0163-411b-8867-90fb2664716a?theme=light&version=v0.1&format=svg

## Prerequisites

1. Python 3.10+
2. Tesseract OCR
3. Poppler (for PDF processing)
4. Google API key for Gemini

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BusinessAI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install system dependencies:
- Windows:
  - Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
  - Install Poppler: https://github.com/oschwartz10612/poppler-windows/releases/
  - Add both to your system PATH

5. Create a `.env` file in the backend directory:
```bash
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Place your PDF files in `backend/data/raw/`

2. Run the RAG system:
```bash
cd backend
python src/llm_setup.py
```

3. The system will:
   - Process PDFs and extract text
   - Create embeddings and store them
   - Answer questions using the RAG pipeline

## Customization

### Adding New Test Queries
Edit `settings.py` to modify `TEST_QUERIES`:
```python
TEST_QUERIES: List[str] = [
    "Your question here",
    "Another question",
]
```

### Adjusting Model Parameters
In `settings.py`:
```python
LLM_MODEL: str = "gemini-2.5-flash-preview-04-17"
TEMPERATURE: float = 0.0
MAX_OUTPUT_TOKENS: int = 2048
```

## Logging

Logs are stored in `backend/data/logs/rag.log` and include:
- Document processing status
- Query processing
- Error messages
- System performance metrics

