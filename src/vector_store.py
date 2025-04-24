import os
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Optional

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
        cache_file: str = "cache/query_cache.json",  # Add cache file parameter
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
        self.doc_vector_norms = None  # Will store pre-calculated norms
        self.load_documents(data_path)
        self._build_index()
        self._precompute_vector_norms()
        self.cache_file = cache_file
        self.query_cache = self._load_cache()

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
            terms = self._tokenize(doc["content_sm_ltks"])
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

    def _precompute_vector_norms(self):
        """Pre-compute document vector norms for faster similarity calculation"""
        doc_vectors = np.array([doc["q_1024_vec"] for doc in self.documents])
        self.doc_vector_norms = np.linalg.norm(doc_vectors, axis=1)
        self.doc_vectors = doc_vectors  # Store as numpy array

    def _compute_vector_similarities(self, query_vector: List[float]) -> np.ndarray:
        """Compute cosine similarities for all documents at once"""
        query_array = np.array(query_vector)
        query_norm = np.linalg.norm(query_array)
        dot_products = np.dot(self.doc_vectors, query_array)
        similarities = dot_products / (self.doc_vector_norms * query_norm + 1e-6)
        return similarities

    def _get_cache_key(self, query: str, vector: List[float]) -> str:
        """Generate a unique cache key from query and vector"""
        # Using only first few dimensions of vector to save memory
        vector_hash = hash(str(vector[:5]))
        return f"{query}_{vector_hash}"

    def _get_from_cache(self, query: str, vector: List[float]) -> Optional[List[Dict]]:
        """Try to get results from cache"""
        cache_key = self._get_cache_key(query, vector)
        return self.query_cache.get(cache_key)

    def _load_cache(self) -> Dict[str, List[Dict]]:
        """Load query cache from disk"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Cache load error: {e}")
        return {}

    def _save_cache(self):
        """Save query cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.query_cache, f)
        except Exception as e:
            print(f"Cache save error: {e}")

    def _add_to_cache(self, query: str, vector: List[float], results: List[Dict]):
        """Add results to cache with LRU-style eviction and persist to disk"""
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest item
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        cache_key = self._get_cache_key(query, vector)
        self.query_cache[cache_key] = results
        self._save_cache()  # Auto-save after adding new results

    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
        text_weight: float = 0.3,
        vector_weight: float = 0.7,
        text_threshold: float = 0.1,  # Minimum text similarity threshold
        vector_threshold: float = 0.3,  # Minimum vector similarity threshold
    ) -> List[Dict]:
        """Optimized hybrid search using vectorized operations with caching"""
        # Check cache first
        cached_results = self._get_from_cache(query, query_vector)
        if cached_results is not None:
            print("Using cached results")
            return cached_results

        # Calculate all vector similarities at once
        vector_sims = self._compute_vector_similarities(query_vector)

        # Quick filter on vector similarity
        valid_indices = np.where(vector_sims >= vector_threshold)[0]

        # Calculate text similarities only for documents that pass vector threshold
        results = [
            {
                "score": (
                    text_sim := self._compute_text_similarity(
                        query, self.documents[i]["content_with_weight"]
                    )
                )
                * text_weight
                + vector_sims[i] * vector_weight,
                "text_score": text_sim,
                "vector_score": vector_sims[i],
                "document": self.documents[i],
            }
            for i in valid_indices
            if (
                text_sim := self._compute_text_similarity(
                    query, self.documents[i]["content_with_weight"]
                )
            )
            >= text_threshold
        ]

        print(f"Found {len(results)} relevant documents")
        # Cache results before returning
        self._add_to_cache(query, query_vector, results[:top_k])
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    def clear_cache(self):
        """Clear the query cache and remove cache file"""
        self.query_cache.clear()
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception as e:
                print(f"Cache file deletion error: {e}")
