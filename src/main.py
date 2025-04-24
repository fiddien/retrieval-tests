import asyncio
import os

from aiohttp.client_exceptions import ClientError
from dotenv import load_dotenv

from src.rag_app import RAGApplication


async def main():
    # Load environment variables
    load_dotenv()

    # Configure model-specific vLLM hosts
    model_hosts = {
        "BAAI/bge-m3": os.getenv("BGEM3_HOST", "http://localhost:8001"),
        "Qwen/Qwen2.5-7B-Instruct": os.getenv("QWEN_HOST", "http://localhost:8000"),
        "Qwen/Qwen2.5-VL-7B-Instruct": os.getenv("QWENVL_HOST", "http://localhost:8002"),
    }

    data_path = os.getenv("DATA_PATH", "raw_data/elasticsearch_chunks.json")

    # Initialize RAG application with model-specific hosts
    rag = RAGApplication(model_hosts=model_hosts, data_path=data_path)

    # Models to test
    models = ["Qwen/Qwen2.5-7B-Instruct"]

    # Query
    query = "Berikan ringkasan tentang personel Syarifudin"

    print(f"Testing query: {query}")
    print("-" * 50)

    for model in models:
        print(f"\nTesting model: {model} (Host: {model_hosts[model]})")
        try:
            result = await rag.generate_response(query=query, model_name=model)

            print("\nResponse:")
            print(result["response"])
            print("\nEvaluation:")
            print(f"Chinese characters: {result['evaluation']['chinese_char_count']} ({result['evaluation']['chinese_char_percentage']:.2f}%)")
            print(f"Average sentence length: {result['evaluation']['average_sentence_length']:.1f} words")

            # Print summarization metrics
            summary_metrics = result['evaluation']['summary_metrics']
            print("\nSummarization Metrics:")
            print(f"Token count: {summary_metrics['token_count']}")
            print(f"Compression ratio: {summary_metrics['compression_ratio']:.2f}")
            print(f"Content density: {summary_metrics['content_density']:.2f}")

            print("\nROUGE Scores:")
            rouge_scores = summary_metrics['rouge_scores']
            print(f"ROUGE-1: {rouge_scores['rouge1']:.3f}")
            print(f"ROUGE-2: {rouge_scores['rouge2']:.3f}")
            print(f"ROUGE-L: {rouge_scores['rougeL']:.3f}")
            print("-" * 50)

        except ClientError as e:
            print(f"Network error with model {model}: {str(e)}")
        except KeyError as e:
            print(f"Unexpected response format from model {model}: {str(e)}")
            raise e
        except Exception as e:
            print(f"Unexpected error with model {model}: {str(e)}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
