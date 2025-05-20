# RAG-Based AI Assistant with AstraDB

This project implements a Retrieval-Augmented Generation (RAG) system using AstraDB as the vector database. It provides API endpoints for building a document database and querying it using various LLM providers.

## Features

- Document processing and chunking for PDF, DOCX, and TXT files
- Vector embeddings using OpenAI's text-embedding models
- Integration with multiple LLM providers:
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude models)
  - Google (Gemini models)
  - DeepSeek (Reasoner and Chat models)
- Conversation history management
- REST API for database building and querying

## Setup

### Prerequisites

- Python 3.8+
- FastAPI
- AstraDB account and credentials
- API keys for LLM providers (OpenAI, Anthropic, Google, DeepSeek)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Update the environment variables defined in `backend.py`:
   - `ASTRA_DB_API_ENDPOINT` 
   - `ASTRA_DB_APPLICATION_TOKEN`
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GOOGLE_API_KEY`
   - `DEEPSEEK_API_KEY`

## Usage

### Starting the Server

```bash
python backend.py
```

The server will run on `http://0.0.0.0:8000` by default.

### Using the Batch Upload Script

The repository includes a shell script (`upload.sh`) for batch uploading files:

```bash
chmod +x upload.sh
./upload.sh
```

Modify the script to point to your file directory and adjust batch size as needed.

## API Endpoints

### GET Endpoints

#### 1. Root Endpoint

```
GET /
```

**Sample Output:**
```json
{
  "Hello": "World"
}
```

#### 2. Recommended Configuration

```
GET /recommend_config
```

**Sample Output:**
```json
{
  "status": "ok",
  "recommend": {
    "keyspace": "testv4",
    "collection_name": "dstc20240306",
    "provider": "anthropic",
    "model_name": "claude-3-7-sonnet-20250219"
  }
}
```


#### 3. Accepted Models

```
GET /accepted_models
```

**Sample Output:**
```json
{
  "status": "ok",
  "accepted_models": {
    "openai": [
      "gpt-4",
      "gpt-3.5",
      "gpt-3.5-turbo",
      "gpt-4o-mini"
    ],
    "anthropic": [
      "claude-3-7-sonnet-20250219",
      "claude-3-5-sonnet-20241022",
      "claude-3-5-haiku-20241022"
    ],
    "google": [
      "gemini-1.5-pro",
      "gemini-1.5-flash",
      "gemini-2.0-flash-exp"
    ],
    "deepseek": [
      "deepseek-reasoner",
      "deepseek-chat"
    ]
  }
}
```

### POST Endpoints

#### 1. Build Database

```
POST /build_database
```

**Sample Input:**
```bash
curl -X POST \
  -F "keyspace=testv4" \
  -F "collection_name=dstc20240306" \
  -F "files=@/path/to/document1.pdf" \
  -F "files=@/path/to/document2.docx" \
  http://localhost:8000/build_database
```

**Sample Output:**
```json
{
  "status": "ok",
  "detail": {
    "message": "Database build finished."
  }
}
```

#### 2. Query Database

```
POST /query
```

**Sample Input:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main courses in the AIMAHEAD initiative?",
    "keyspace": "testv4",
    "collection_name": "dstc20240306",
    "provider": "anthropic",
    "model_name": "claude-3-7-sonnet-20250219",
    "session_id": "user123",
    "top_k": 10
  }' \
  http://localhost:8000/query
```

**Sample Output:**
```json
{
  "query": "What are the main courses in the AIMAHEAD initiative?",
  "answer": "The AIMAHEAD initiative offers courses from basic to advanced. The foundational courses include Introduction to AI in Healthcare and Data Science Fundamentals. More advanced courses include Machine Learning for Medical Imaging and Advanced Natural Language Processing for Clinical Text. Each course is designed to build skills in applying AI to healthcare challenges. You can find more information at aimahead.org/courses.",
  "history": [
    ["human", "What are the main courses in the AIMAHEAD initiative?"],
    ["system", "The AIMAHEAD initiative offers courses from basic to advanced. The foundational courses include Introduction to AI in Healthcare and Data Science Fundamentals. More advanced courses include Machine Learning for Medical Imaging and Advanced Natural Language Processing for Clinical Text. Each course is designed to build skills in applying AI to healthcare challenges. You can find more information at aimahead.org/courses."]
  ]
}
```

## Document Processing

The system processes documents in the following steps:

1. Extracts text from PDF, DOCX, or TXT files
2. Splits text into sentences
3. Groups sentences into chunks with optional overlap
4. Further splits chunks by byte size if necessary
5. Embeds chunks using OpenAI's embedding model
6. Stores vectors in AstraDB

## Configuration

The system supports various LLM providers and models, which can be configured in the `CONFIG` dictionary in `backend.py`. Each model can have custom parameters like temperature and retry settings.

## Logging

The system uses Python's logging module to provide information about:
- Database operations
- Query processing
- Document chunking
- LLM interactions


## Project Structure

```plaintext
.
├── backend.py                     # backend server implementation
├── finished_collection_management.py  # vector database  management
├── requirements.txt                   # python requirements
├── README.md                          # instructions and information
└── upload.sh                          # script for batch uploading files

```

## Architecture

The system follows a modular architecture:
1. Document processing pipeline
2. Vector embedding and storage
3. LLM integration for query processing
4. Conversation management
5. API endpoints


## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [AstraDB](https://www.datastax.com/products/datastax-astra) for vector database storage
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- 
