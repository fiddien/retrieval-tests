# Indonesian Cross-Domain QA Dataset

## Overview

This dataset is a comprehensive collection of Indonesian question-answering pairs across multiple domains including farming, decision making, Soeharto's history, and legal documents. The dataset follows the BEIR (Benchmarking IR) format for information retrieval evaluation.

## Dataset Statistics

- Total Queries: 440
- Total Documents: 362
- Training Pairs: 328
- Development Pairs: 70
- Test Pairs: 72

## Topics Covered

1. **Farming**
   - Agricultural practices
   - Food security
   - Farming innovation

2. **Decision Making**
   - Strategic planning
   - Management decisions
   - Organizational behavior

3. **Soeharto**
   - Historical events
   - Political decisions
   - Leadership style

4. **Law**
   - Indonesian legal documents
   - Government regulations
   - Official decrees

## Dataset Structure

The dataset follows the BEIR format and consists of the following files:

### 1. Queries (`queries.jsonl`)
Contains questions in Indonesian with unique identifiers.

**Structure:**
```json
{
    "_id": "unique-query-id",
    "text": "Question text in Indonesian",
    "metadata": {
        "topic": "Topic category (e.g., Law, Farming)"
    }
}
```

### 2. Corpus (`corpus.jsonl`)
Contains the source documents that provide answers to the queries.

**Structure:**
```json
{
    "_id": "unique-document-id",
    "text": "Document content in Indonesian",
    "title": "Document title",
    "metadata": {
        "topic": "Topic category",
        "subtopic": "Specific subtopic (for legal documents)",
        "year": "Publication year (for legal documents)",
        "hierarchy": ["Section hierarchy"]
    }
}
```

### 3. Relevance Judgments (`qrels/`)
Maps queries to their relevant documents using the TREC-style format. Split into train/dev/test sets.

**File Format (tab-separated):**
```
query-id   0   document-id   relevance-score
```

- `train.tsv`: Training relevance judgments (328 pairs)
- `dev.tsv`: Development relevance judgments (70 pairs)
- `test.tsv`: Test relevance judgments (72 pairs)

### 4. Metadata (`metadata.json`)
Contains dataset information and statistics.

**Structure:**
```json
{
    "name": "Indonesian Cross-Domain QA Dataset",
    "description": "Combined dataset of Indonesian QA including farming, decision making, Soeharto, and legal documents",
    "language": "id",
    "format": "beir",
    "topics": ["Farming", "Decision Making", "Soeharto", "Law"],
    "stats": {
        "total_queries": 440,
        "total_documents": 362,
        "train_qrels": 328,
        "dev_qrels": 70,
        "test_qrels": 72
    }
}
```

## Usage

The dataset is designed for:
1. Evaluating cross-domain information retrieval systems
2. Training and testing question-answering models
3. Benchmarking document retrieval performance
4. Studying domain adaptation in Indonesian language tasks

## License and Citation

This dataset combines content from multiple sources. Please cite appropriately when using the dataset and respect the original licenses of the constituent datasets.

## Contact

For questions or issues related to the dataset, please open an issue in the repository.
