import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List

import aiohttp
import numpy as np


class VectorStore:
    def __init__(
        self,
        data_path: str = "raw_data/elasticsearch_chunks.json",
        embedding_endpoint: str = "http://localhost:8000/v1/embeddings",
        api_key: str = None,
        embedding_model: str = "embedding_model",
        stopwords_path: str = "src/nlp/stopwords-id.txt",
    ):
        self.embedding_endpoint = embedding_endpoint
        self.api_key = api_key
        if self.api_key:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        else:
            self.headers = {"Content-Type": "application/json"}
        self.embedding_model = embedding_model
        self.documents = []
        self.document_terms = {}  # For storing term frequencies
        self.idf_scores = {}  # For storing IDF scores
        self.stopwords = self._load_stopwords(stopwords_path)
        self.load_documents(data_path)
        self._build_index()

    def load_documents(self, data_path: str):
        """Load documents from JSON file"""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.documents = [hit["_source"] for hit in data["hits"]["hits"]]

    def _load_stopwords(self, path: str) -> set:
        """Load stopwords from file"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f)
        except FileNotFoundError:
            return set()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text"""
        # Convert to lowercase and split into words
        words = re.findall(r"\w+", text.lower())
        # Remove stopwords and very short terms
        return [w for w in words if w not in self.stopwords and len(w) > 2]

    def _build_index(self):
        """Build TF-IDF index for all documents"""
        # Calculate term frequencies for each document
        for i, doc in enumerate(self.documents):
            terms = self._tokenize(doc["content_with_weight"])
            self.document_terms[i] = Counter(terms)

        # Calculate IDF scores
        doc_count = len(self.documents)
        term_doc_freq = Counter()
        for term_freq in self.document_terms.values():
            term_doc_freq.update(term_freq.keys())

        self.idf_scores = {
            term: math.log(doc_count / freq) for term, freq in term_doc_freq.items()
        }

    def _compute_text_similarity(
        self, query: str, content: str, fuzzy_threshold: float = 0.8
    ) -> float:
        """Compute TF-IDF based similarity with fuzzy matching"""
        query_terms = self._tokenize(query)
        doc_terms = self._tokenize(content)

        # Calculate query term weights
        query_weights = {}
        for term in query_terms:
            # Look for fuzzy matches if exact match not found
            if term not in self.idf_scores:
                best_match = None
                best_score = 0
                for doc_term in self.idf_scores:
                    score = SequenceMatcher(None, term, doc_term).ratio()
                    if score > fuzzy_threshold and score > best_score:
                        best_match = doc_term
                        best_score = score
                if best_match:
                    term = best_match

            if term in self.idf_scores:
                query_weights[term] = self.idf_scores[term]

        # Calculate document term weights
        doc_weights = Counter(doc_terms)

        # Compute cosine similarity between query and document vectors
        numerator = sum(
            query_weights.get(term, 0) * count for term, count in doc_weights.items()
        )
        query_norm = math.sqrt(sum(w * w for w in query_weights.values()))
        doc_norm = math.sqrt(sum(c * c for c in doc_weights.values()))

        if query_norm == 0 or doc_norm == 0:
            return 0

        return numerator / (query_norm * doc_norm)

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding from vLLM-hosted embedding model"""
        async with aiohttp.ClientSession() as session:
            payload = {"input": text, "model": self.embedding_model}
            async with session.post(
                self.embedding_endpoint, json=payload, headers=self.headers
            ) as response:
                result = await response.json()
                return result["data"][0]["embedding"]

    def _compute_vector_similarity(
        self, query_vector: List[float], doc_vector: List[float]
    ) -> float:
        """Compute cosine similarity between vectors"""
        query_array = np.array(query_vector)
        doc_array = np.array(doc_vector)
        dot_product = np.dot(query_array, doc_array)
        query_norm = np.linalg.norm(query_array)
        doc_norm = np.linalg.norm(doc_array)
        return dot_product / (query_norm * doc_norm + 1e-6)

    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
        text_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> List[Dict]:
        """
        Perform hybrid search combining TF-IDF text search and vector similarity
        """
        results = []

        for doc in self.documents:
            # Calculate text similarity (with configured weight)
            text_score = (
                self._compute_text_similarity(query, doc["content_with_weight"])
                * text_weight
            )

            # Calculate vector similarity (with configured weight)
            vector_score = (
                self._compute_vector_similarity(query_vector, doc["q_1024_vec"])
                * vector_weight
            )

            # Combine scores
            total_score = text_score + vector_score

            results.append(
                {
                    "score": total_score,
                    "text_score": text_score / text_weight if text_weight > 0 else 0,
                    "vector_score": vector_score / vector_weight
                    if vector_weight > 0
                    else 0,
                    "document": doc,
                }
            )

        # Sort by score and return top_k documents
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
