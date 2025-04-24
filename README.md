# RAG LLM Testing Framework

A framework for testing various LLMs using Retrieval-Augmented Generation (RAG) with special focus on evaluating response quality and Chinese character presence.

## Features

- Hybrid search combining keyword and vector-based search using Elasticsearch
- Support for multiple LLMs hosted through vLLM
- Automatic evaluation of responses including:
  - Chinese character detection and percentage
  - Citation checking
  - Average sentence length analysis
  - Factual consistency scoring (placeholder for future implementation)

## Prerequisites

- Python 3.8+
- Elasticsearch 8.0+
- vLLM server running with desired models
- vLLM-compatible embedding model

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```
ELASTICSEARCH_HOST=http://localhost:9200
VLLM_HOST=http://localhost:8000
```

## Running Tests

Run the test suite:
```bash
pytest tests/
```

## Example Usage

Run the example script:
```bash
python src/main.py
```

This will test multiple LLMs with a sample query and evaluate their responses.

## Project Structure

- `src/`
  - `rag_app.py`: Main RAG application implementation
  - `vector_store.py`: Vector store implementation for hybrid search
  - `llm_evaluator.py`: LLM response evaluation logic
  - `main.py`: Example usage script
- `tests/`: Test files
- `docs/`: Documentation

## Elasticsearch Mapping

Ensure your Elasticsearch index has the following mapping for hybrid search:
```json
{
  "mappings": {
    "properties": {
      "content_with_weight": {
        "type": "text"
      },
      "q_1024_vec": {
        "type": "dense_vector",
        "dims": 1024
      },
      "docnm_kwd": {
        "type": "keyword"
      },
      "doc_id": {
        "type": "keyword"
      }
    }
  }
}