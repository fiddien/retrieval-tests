
import time
import asyncio
import logging
import os
from typing import Dict, List

import aiohttp
from dotenv import load_dotenv

from src.rag_app import RAGApplication
from src.test_queries.queries import TEST_QUERIES
from src.test_queries.reporting import TestReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_run.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def run_test_queries(
    rag: RAGApplication, models: List[str], queries: List[Dict], target_lang: str
) -> TestReport:
    """Run test queries across multiple models and collect results"""
    report = TestReport()

    for model in models:
        print(f"\nTesting model: {model}")
        for query_data in queries:
            query_id = query_data["id"]
            query_mode = query_data["mode"]
            query = query_data["query"]
            print(f"\nProcessing query: {query_id} ({query_mode})")

            try:
                result = await rag.generate_response(query=query, model_name=model, mode=query_mode)
                report.add_result(query_id, query, model, result)

                # Print individual result
                print(f"Query: {query}")
                print(
                    "Response:",
                    result["response"][:200] + "..."
                    if len(result["response"]) > 200
                    else result["response"],
                )
                print("-" * 50);

            except aiohttp.ClientError as err:
                logger.error(
                    "Network error processing query %s with model %s: %s",
                    query_id,
                    model,
                    str(err),
                )
            except KeyError as err:
                logger.error(
                    "Unexpected response format from %s for query %s: %s",
                    model,
                    query_id,
                    str(err),
                )
            except ValueError as err:
                logger.error(
                    "Invalid input or configuration for %s on query %s: %s",
                    model,
                    query_id,
                    str(err),
                )
            except asyncio.TimeoutError:
                logger.error("Request timed out for %s on query %s", model, query_id)
            except Exception as err:
                logger.exception(
                    "Unexpected error with %s on query %s: %s",
                    model,
                    query_id,
                    str(err),
                )
                logger.error("Full error details have been logged")

    return report


async def main():
    # Load environment variables
    load_dotenv()

    # Configure model-specific vLLM hosts
    model_hosts = {
        "BAAI/bge-m3": (
            os.getenv("VLLM_EMBEDDINGS_HOST", "http://localhost:8001"),
            os.getenv("VLLM_EMBEDDINGS_API_KEY")
        ),
        "Qwen/Qwen2.5-7b-Instruct": (
            os.getenv("QWEN_HOST", "http://localhost:8000"),
            os.getenv("QWEN_API_KEY")
        ),
        "Qwen/Qwen2.5-VL-7B-Instruct": (
            os.getenv("QWENVL_HOST", "http://localhost:8002"),
            os.getenv("QWENVL_API_KEY"),
        ),
        "deepseek-chat": (
            os.getenv("DEEPSEEK_HOST", "http://localhost:8003"),
            os.getenv("DEEPSEEK_API_KEY"),
        ),
        "gpt-4o-mini": (
            os.getenv("OPENAI_HOST", "http://localhost:8004"),
            os.getenv("OPENAI_API_KEY"),
        ),
        "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct": (
            os.getenv("VLLM_HOST", "http://localhost:8005"),
            os.getenv("VLLM_API_KEY"),
            ),
        "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct": (
            os.getenv("VLLM_HOST", "http://localhost:8005"),
            os.getenv("VLLM_API_KEY"),
            ),
        "ibm-granite/granite-3.3-8b-instruct": (
            os.getenv("VLLM_HOST"),
            os.getenv("VLLM_API_KEY"),
        ),
        "google/gemma-3-12b-it": (
            os.getenv("VLLM_HOST"),
            os.getenv("VLLM_API_KEY"),
        ),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": (
            os.getenv("VLLM_HOST"),
            os.getenv("VLLM_API_KEY"),
        ),
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": (
            os.getenv("VLLM_HOST"),
            os.getenv("VLLM_API_KEY"),
        ),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": (
            os.getenv("VLLM_HOST"),
            os.getenv("VLLM_API_KEY"),
        ),
        "aisingapore/Llama-SEA-LION-v3.5-8B-R": (
            os.getenv("VLLM_HOST"),
            os.getenv("VLLM_API_KEY"),
        ),
        "aisingapore/Gemma-SEA-LION-v3-9B-IT": (
            os.getenv("VLLM_HOST"),
            os.getenv("VLLM_API_KEY"),
        ),
        "aisingapore/Llama-SEA-LION-v3.5-8B-R": (
            os.getenv("VLLM_HOST"),
            os.getenv("VLLM_API_KEY"),
        ),
        "Qwen/Qwen2.5-3B-Instruct": (
            os.getenv("VLLM_HOST"),
            os.getenv("VLLM_API_KEY"),
        ),
    }
    translator_hosts = {
        "Qwen/Qwen2.5-3B-Instruct": (
            os.getenv("VLLM_HOST"),
            os.getenv("VLLM_API_KEY"),
        ),
    }

    data_path = os.getenv("DATA_PATH", "raw_data/elasticsearch_chunks.json")

    # Initialize RAG application
    target_lang = os.getenv("TARGET_LANG", "English")
    rag = RAGApplication(model_hosts=model_hosts, data_path=data_path, target_lang=target_lang, translator_hosts=translator_hosts)

    # Models to test (can test all or a subset)
    models = [
        "Qwen/Qwen2.5-7b-Instruct",
        # "Qwen/Qwen2.5-0.5B-Instruct"
        # "deepseek-chat"
        # "gpt-4o-mini",
        # "GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct",
	# "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct",
	# "ibm-granite/granite-3.3-8b-instruct",
        # "google/gemma-3-12b-it",
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        # "aisingapore/Llama-SEA-LION-v3.5-8B-R",
	# "aisingapore/Gemma-SEA-LION-v3-9B-IT",
        # "aisingapore/Llama-SEA-LION-v3.5-8B-R",
    ]

    print("Starting test suite...")
    print(f"Models to test: {', '.join(models)}")
    print(f"Target language: {target_lang}")
    print(f"Number of queries: {len(TEST_QUERIES)}")
    print("-" * 50)

    # Run tests and generate report
    report = await run_test_queries(rag, models, TEST_QUERIES, target_lang)

    # Save report files
    results_file, stats_file = report.save_report()

    # Print summary
    report.print_summary()
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Model statistics saved to: {stats_file}")


if __name__ == "__main__":
    asyncio.run(main())
