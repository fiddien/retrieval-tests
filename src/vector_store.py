from typing import List, Dict
import aiohttp
import numpy as np
import json


class VectorStore:
    def __init__(
        self,
        data_path: str = "raw_data/elasticsearch_chunks.json",
        embedding_endpoint: str = "http://localhost:8000/v1/embeddings",
        embedding_model: str = "embedding_model",
    ):
        self.embedding_endpoint = embedding_endpoint
        self.embedding_model = embedding_model
        self.documents = []
        self.load_documents(data_path)

    def load_documents(self, data_path: str):
        """Load documents from JSON file"""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.documents = [hit["_source"] for hit in data["hits"]["hits"]]

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding from vLLM-hosted embedding model"""
        async with aiohttp.ClientSession() as session:
            payload = {"input": text, "model": self.embedding_model}
            async with session.post(self.embedding_endpoint, json=payload) as response:
                result = await response.json()
                return result["data"][0]["embedding"]

    def _compute_text_similarity(self, query: str, content: str) -> float:
        """Compute simple text similarity score"""
        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())
        intersection = query_terms.intersection(content_terms)
        return len(intersection) / (
            len(query_terms) + 0.001
        )  # Add small constant to avoid division by zero

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
        self, query: str, query_vector: List[float], top_k: int = 5
    ) -> List[Dict]:
        """
        Perform hybrid search combining keyword and vector similarity
        """
        results = []

        for doc in self.documents:
            # Calculate text similarity (30% weight)
            text_score = (
                self._compute_text_similarity(query, doc["content_with_weight"]) * 0.3
            )

            # Calculate vector similarity (70% weight)
            vector_score = (
                self._compute_vector_similarity(query_vector, doc["q_1024_vec"]) * 0.7
            )

            # Combine scores
            total_score = text_score + vector_score

            results.append({"score": total_score, "document": doc})

        # Sort by score and return top_k documents
        results.sort(key=lambda x: x["score"], reverse=True)
        return [item["document"] for item in results[:top_k]]
