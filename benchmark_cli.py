import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jsonlines
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from ranx import Qrels, Run, evaluate
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load environment variables
load_dotenv()


@dataclass
class EmbeddingModel:
    name: str
    endpoint: str
    api_key: str
    batch_size: int = 2
    max_length: int = 8192

    def _truncate_text(self, text: str) -> str:
        """Truncate text using the appropriate tokenizer"""
        if "BAAI/bge" in self.name:
            return text[:20000]
        elif "Qwen" in self.name:
            return text[:20000]
        else:
            # Fallback to simple truncation for unknown models
            return text[:20000]

    def get_embeddings(self, texts: List[str]) -> Tuple[np.ndarray, float]:
        all_embeddings = []
        total_time = 0.0

        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc=f"Getting embeddings for {self.name}",
        ):
            batch = texts[i : i + self.batch_size]

            try:
                start_time = time.time()
                response = requests.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "input": batch,
                        "model": self.name,
                        "truncate_prompt_tokens": self.max_length,
                        # "extra_body": {
                        #     "truncate_prompt_tokens": self.max_length,
                        # }
                    },
                )
                batch_time = time.time() - start_time

                if response.status_code != 200:
                    # raise Exception(f"API call failed: {response.text}")
                    print(f"API call failed for batch {i}: {response}")
                    continue

                data = response.json()["data"]
                embeddings = [item["embedding"] for item in data]
                all_embeddings.extend(embeddings)
                total_time += batch_time

                time.sleep(5)

            except Exception as e:
                print(f"Error processing batch {i}: {str(e)}")
                return np.array([]), total_time

        return np.array(all_embeddings), total_time


class BEIRDataset:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.queries = self._load_queries()
        self.corpus = self._load_corpus()
        self.qrels = self._load_qrels()

    def _load_queries(self) -> Dict[str, str]:
        queries = {}
        with jsonlines.open(os.path.join(self.base_path, "queries.jsonl")) as reader:
            for obj in reader:
                queries[obj["_id"]] = obj["text"]
        return queries

    def _load_corpus(self) -> Dict[str, Dict]:
        corpus = {}
        with jsonlines.open(os.path.join(self.base_path, "corpus.jsonl")) as reader:
            for obj in reader:
                corpus[obj["_id"]] = {
                    "text": obj["text"],
                    "title": obj["title"],
                    "metadata": obj.get("metadata", {}),
                }
        return corpus

    def _load_qrels(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        qrels = {"train": {}, "dev": {}, "test": {}}

        for split in qrels.keys():
            qrels_path = os.path.join(self.base_path, "qrels", f"{split}.tsv")
            if not os.path.exists(qrels_path):
                continue

            df = pd.read_csv(
                qrels_path,
                sep="\t",
                header=None,
                names=["query-id", "iteration", "doc-id", "relevance"],
            )

            for _, row in df.iterrows():
                query_id = row["query-id"]
                doc_id = row["doc-id"]
                relevance = row["relevance"]

                if query_id not in qrels[split]:
                    qrels[split][query_id] = {}
                qrels[split][query_id] = qrels[split][query_id] or {}
                qrels[split][query_id][doc_id] = relevance

        return qrels


class SingleModelBenchmarker:
    def __init__(
        self,
        dataset: BEIRDataset,
        embedding_model: EmbeddingModel,
        top_k: int = 100,
    ):
        self.dataset = dataset
        self.embedding_model = embedding_model
        self.top_k = top_k

    def run_benchmark(self, split: str = "test") -> Dict:
        results = {}
        timing_stats = {}

        print(f"\nProcessing embeddings for model: {self.embedding_model.name}")

        # Get document embeddings
        doc_texts = [doc["text"] for doc in self.dataset.corpus.values()]
        doc_ids = list(self.dataset.corpus.keys())

        doc_embeddings, embed_time = self.embedding_model.get_embeddings(doc_texts)

        if len(doc_embeddings) == 0:
            print(f"Failed to get document embeddings for {self.embedding_model.name}")
            return {"metrics": {}, "timing": {}}

        # Ensure embeddings are 2D
        if len(doc_embeddings.shape) == 3:
            doc_embeddings = doc_embeddings.reshape(doc_embeddings.shape[0], -1)

        timing_stats[f"{self.embedding_model.name}_embedding_time"] = embed_time

        # Process each query
        query_results = {}
        for query_id, query_text in tqdm(
            self.dataset.queries.items(), desc="Processing queries"
        ):
            if query_id not in self.dataset.qrels[split]:
                continue

            query_embedding, _ = self.embedding_model.get_embeddings([query_text])

            if len(query_embedding) == 0:
                print(f"Failed to get query embedding for query {query_id}")
                continue

            # Ensure query_embedding is 2D
            if len(query_embedding.shape) == 3:
                query_embedding = query_embedding.reshape(query_embedding.shape[0], -1)

            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            top_indices = np.argsort(similarities)[-self.top_k:][::-1]

            rankings = {
                doc_ids[idx]: float(similarities[idx]) for idx in top_indices
            }

            query_results[query_id] = rankings

        results[self.embedding_model.name] = query_results

        # Evaluate results
        metrics = self._evaluate_results(results, split)
        return {"metrics": metrics, "timing": timing_stats}

    def _evaluate_results(self, results: Dict, split: str) -> Dict:
        metrics = {}

        model_name = self.embedding_model.name
        run_dict = {}

        # Prepare run dictionary for evaluation
        for query_id, rankings in results[model_name].items():
            run_dict[query_id] = rankings

        if not run_dict:
            print(f"No results to evaluate for {model_name}")
            return {model_name: {}}

        qrels_obj = Qrels(self.dataset.qrels[split])
        run_obj = Run(run_dict)

        # Evaluate all metrics at once
        try:
            metrics[model_name] = evaluate(
                qrels_obj,
                run_obj,
                ["ndcg@10", "ndcg@100", "mrr@10", "recall@100", "precision@1", "map"],
            )
        except Exception as e:
            print(f"Error evaluating metrics for {model_name}: {str(e)}")
            metrics[model_name] = {}

        return metrics


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run BEIR benchmark for a single embedding model"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the embedding model (e.g., 'BAAI/bge-m3')"
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        required=True,
        help="API endpoint for the model (e.g., 'http://localhost:5506/v1/embeddings')"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum token length for the model (default: 8192)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset",
        help="Path to the BEIR dataset (default: 'dataset')"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset split to evaluate (default: 'test')"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top documents to retrieve (default: 100)"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for results (default: benchmark_results_{model_name_safe}.json)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default="dummy",
        help="API key for the model endpoint (default: 'dummy')"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load dataset
    if not args.quiet:
        print(f"Loading dataset from: {args.dataset_path}")

    try:
        dataset = BEIRDataset(args.dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return 1

    if not args.quiet:
        print(f"Dataset loaded: {len(dataset.queries)} queries, {len(dataset.corpus)} documents")

    # Create embedding model
    embedding_model = EmbeddingModel(
        name=args.model_name,
        endpoint=args.endpoint,
        api_key=args.api_key,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    if not args.quiet:
        print(f"Testing model endpoint: {args.endpoint}")

    # Test the endpoint
    try:
        response = requests.post(
            args.endpoint,
            headers={"Authorization": f"Bearer {args.api_key}"},
            json={
                "input": ["test"],
                "model": args.model_name,
            },
            timeout=30
        )
        if response.status_code != 200:
            print(f"Error: Model endpoint test failed: {response.text}")
            return 1
        if not args.quiet:
            print("Model endpoint is responding correctly")
    except Exception as e:
        print(f"Error: Cannot connect to model endpoint: {str(e)}")
        return 1

    # Run benchmark
    benchmarker = SingleModelBenchmarker(
        dataset=dataset,
        embedding_model=embedding_model,
        top_k=args.top_k,
    )

    if not args.quiet:
        print(f"Starting benchmark for {args.model_name} on {args.split} split...")

    try:
        results = benchmarker.run_benchmark(split=args.split)
    except Exception as e:
        print(f"Error running benchmark: {str(e)}")
        return 1

    # Determine output file name
    if args.output_file:
        output_file = args.output_file
    else:
        model_name_safe = args.model_name.replace("/", "_").replace("-", "_")
        output_file = f"benchmark_results_{model_name_safe}.json"

    # Save results
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        if not args.quiet:
            print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return 1

    # Print results summary
    if not args.quiet:
        print("\nMetrics Results:")
        if results["metrics"]:
            for model_name, metrics in results["metrics"].items():
                print(f"\n{model_name}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
        else:
            print("No metrics computed")

        print("\nTiming Results (seconds):")
        for timing_key, time_value in results["timing"].items():
            print(f"  {timing_key}: {time_value:.2f}s")

    # Print summary line for bash script parsing
    if results["metrics"] and args.model_name in results["metrics"]:
        metrics = results["metrics"][args.model_name]
        ndcg10 = metrics.get("ndcg@10", 0.0)
        print(f"BENCHMARK_COMPLETE: {args.model_name} | NDCG@10: {ndcg10:.4f}")

    return 0


if __name__ == "__main__":
    exit(main())