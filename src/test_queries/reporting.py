"""Module for analyzing and generating reports from LLM test results"""
from typing import Dict
import json
import os
from datetime import datetime

class TestReport:
    def __init__(self):
        self.results = []
        self.model_stats = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_result(self, query_id: str, query: str, model_name: str, result: Dict):
        """Add a test result to the report"""
        metrics = result["evaluation"]["summary_metrics"]
        self.results.append({
            "query_id": query_id,
            "query": query,
            "model": model_name,
            "response": result["response"],
            "metrics": {
                "chinese_char_count": result["evaluation"]["chinese_char_count"],
                "chinese_char_percentage": result["evaluation"]["chinese_char_percentage"],
                "average_sentence_length": result["evaluation"]["average_sentence_length"],
                "token_count": metrics["token_count"],
                "compression_ratio": metrics["compression_ratio"],
                "content_density": metrics["content_density"],
                "rouge_scores": metrics["rouge_scores"]
            }
        })

    def calculate_model_stats(self):
        """Calculate aggregated statistics for each model"""
        for result in self.results:
            model = result["model"]
            if model not in self.model_stats:
                self.model_stats[model] = {
                    "total_queries": 0,
                    "avg_chinese_char_percentage": 0.0,
                    "avg_sentence_length": 0.0,
                    "avg_content_density": 0.0,
                    "avg_rouge1": 0.0,
                    "avg_rouge2": 0.0,
                    "avg_rougeL": 0.0
                }

            stats = self.model_stats[model]
            metrics = result["metrics"]
            stats["total_queries"] += 1
            stats["avg_chinese_char_percentage"] += metrics["chinese_char_percentage"]
            stats["avg_sentence_length"] += metrics["average_sentence_length"]
            stats["avg_content_density"] += metrics["content_density"]
            stats["avg_rouge1"] += metrics["rouge_scores"]["rouge1"]
            stats["avg_rouge2"] += metrics["rouge_scores"]["rouge2"]
            stats["avg_rougeL"] += metrics["rouge_scores"]["rougeL"]

        # Calculate averages
        for model_stats in self.model_stats.values():
            total = model_stats["total_queries"]
            if total > 0:
                for key in model_stats:
                    if key != "total_queries":
                        model_stats[key] /= total

    def save_report(self, output_dir: str = "reports"):
        """Save the test results and statistics to JSON files"""
        os.makedirs(output_dir, exist_ok=True)

        # Calculate stats before saving
        self.calculate_model_stats()

        # Save detailed results
        results_file = os.path.join(output_dir, f"test_results_{self.timestamp}.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # Save model statistics
        stats_file = os.path.join(output_dir, f"model_stats_{self.timestamp}.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.model_stats, f, indent=2, ensure_ascii=False)

        return results_file, stats_file

    def print_summary(self):
        """Print a summary of the test results"""
        print("\n=== Test Results Summary ===")
        print(f"Total queries: {len(self.results)}")
        print(f"Models tested: {len(self.model_stats)}")

        print("\nModel Performance:")
        for model, stats in self.model_stats.items():
            print(f"\n{model}:")
            print(f"  Queries processed: {stats['total_queries']}")
            print(f"  Avg Chinese char %: {stats['avg_chinese_char_percentage']:.2f}%")
            print(f"  Avg sentence length: {stats['avg_sentence_length']:.1f} words")
            print(f"  Avg content density: {stats['avg_content_density']:.2f}")
            print("  ROUGE scores:")
            print(f"    ROUGE-1: {stats['avg_rouge1']:.3f}")
            print(f"    ROUGE-2: {stats['avg_rouge2']:.3f}")
            print(f"    ROUGE-L: {stats['avg_rougeL']:.3f}")
