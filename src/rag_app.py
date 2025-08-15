from typing import Dict, List, Tuple
from src.vector_store import VectorStore
from src.llm_evaluator import LLMEvaluator


class RAGApplication:
    def __init__(
        self,
        model_hosts: Dict[str, Tuple[str, str]],
        data_path: str = "raw_data/elasticsearch_chunks.json",
        target_lang: str = "English",
        translator_hosts: Dict[str, Tuple[str, str]] = None,
    ):
        """
        Initialize RAG application with model-specific vLLM hosts
        Args:
            model_hosts: Dictionary mapping model names to their vLLM host URLs
            data_path: Path to the data file containing documents
        """
        # Use the first host's embedding endpoint for vector store
        any_host = next(iter(model_hosts.values()))[0]
        embedding_model = next(iter(model_hosts.keys()))
        api_key = model_hosts[embedding_model][1]
        self.vector_store = VectorStore(
            data_path=data_path,
            embedding_endpoint=f"{any_host}/embeddings",
            api_key=api_key,
            embedding_model=embedding_model,
        )
        self.llm_evaluator = LLMEvaluator(model_hosts, target_lang, translator_hosts)

    async def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform hybrid search combining keyword and vector-based search
        """
        # Get query embedding
        query_vector = await self.vector_store.get_embedding(query)

        # Perform hybrid search
        results = await self.vector_store.hybrid_search(query, query_vector, top_k)
        return results

    async def generate_response(self, query: str, model_name: str, mode: str = "summary") -> Dict:
        """
        Generate response using RAG with specified LLM model
        Args:
            query: User query string
            model_name: Name of the model to use
            mode: Mode of operation - either "summary" or "compare" (default: "summary")
        """
        # Get relevant chunks
        relevant_chunks = await self.hybrid_search(query, top_k=30)

        # Format knowledge base
        knowledge = self._format_knowledge(relevant_chunks)

        # Generate response
        response = await self.llm_evaluator.generate(
            query=query,
            knowledge=knowledge,
            model_name=model_name,
            mode=mode
        )
        return response

    def _format_knowledge(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into knowledge base prompt
        """
        doc2chunks = {}
        for chunk in chunks:
            chunk = chunk["document"]
            doc_id = chunk["docnm_kwd"]
            if doc_id not in doc2chunks:
                doc2chunks[doc_id] = {"chunks": [], "meta": chunk.get("meta", {})}
            doc2chunks[doc_id]["chunks"].append(chunk["content_with_weight"])

        formatted_knowledge = []
        for doc_id, doc_info in doc2chunks.items():
            text = f"Document: {doc_id}\n"
            for k, v in doc_info["meta"].items():
                text += f"{k}: {v}\n"
            text += "Relevant fragments:\n"
            for i, chunk in enumerate(doc_info["chunks"], 1):
                text += f"{i}. {chunk}\n"
            formatted_knowledge.append(text)

        return "\n".join(formatted_knowledge)

    async def regex_search(
        self,
        pattern: str,
        top_k: int = 5,
        score_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Search documents using regex pattern matching on docnm_kwd field.

        Args:
            pattern: Regex pattern to match against docnm_kwd field
            top_k: Number of top results to return
            score_threshold: Minimum score threshold for results

        Returns:
            List of matching documents sorted by score
        """
        return self.vector_store.regex_search(
            pattern=pattern,
            top_k=top_k,
            score_threshold=score_threshold
        )
