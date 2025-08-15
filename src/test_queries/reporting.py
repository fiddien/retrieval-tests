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
        original_metrics = result["evaluation"]["summary_metrics"]
        translated_metrics = result["evaluation"].get("translated_summary_metrics", {})

        metrics_entry = {
            "chinese_char_count": result["evaluation"]["chinese_char_count"],
            "chinese_char_percentage": result["evaluation"]["chinese_char_percentage"],
            "chinese_chars_found": result["evaluation"].get("chinese_chars_found", ""),
            # "chinese_substrings": result["evaluation"].get("chinese_substrings", []),
            "average_sentence_length": result["evaluation"]["average_sentence_length"],
            "token_count": original_metrics["token_count"],
            "compression_ratio": original_metrics["compression_ratio"],
            "content_density": original_metrics["content_density"],
            "rouge_scores": original_metrics["rouge_scores"],
            "generation_time": result["evaluation"].get("generation_time", 0.0)
        }

        # Add translated metrics if they exist
        if "translated_chinese_char_count" in result["evaluation"]:
            metrics_entry.update({
                "translated_chinese_char_count": result["evaluation"]["translated_chinese_char_count"],
                "translated_chinese_char_percentage": result["evaluation"]["translated_chinese_char_percentage"],
                "translated_chinese_chars_found": result["evaluation"].get("translated_chinese_chars_found", ""),
                # "translated_chinese_substrings": result["evaluation"].get("translated_chinese_substrings", []),
                "translated_average_sentence_length": result["evaluation"]["translated_average_sentence_length"],
                "translated_token_count": translated_metrics.get("token_count", 0),
                "translated_compression_ratio": translated_metrics.get("compression_ratio", 0.0),
                "translated_content_density": translated_metrics.get("content_density", 0.0),
                "translated_rouge_scores": translated_metrics.get("rouge_scores", {}),
                "translation_time": result["evaluation"].get("translation_time", 0.0)
            })

        self.results.append({
            "query_id": query_id,
            "query": query,
            "model": model_name,
            "response": result["response"],
            "translated_text": result["evaluation"].get("translated_text", ""),
            "metrics": metrics_entry
        })

    def calculate_model_stats(self):
        """Calculate aggregated statistics for each model"""
        for result in self.results:
            model = result["model"]
            if model not in self.model_stats:
                self.model_stats[model] = {
                    "total_queries": 0,
                    "queries_with_translation": 0,
                    "avg_chinese_char_percentage": 0.0,
                    "avg_sentence_length": 0.0,
                    "avg_content_density": 0.0,
                    "avg_rouge1": 0.0,
                    "avg_rouge2": 0.0,
                    "avg_rougeL": 0.0,
                    # Translation effectiveness stats
                    "avg_chinese_char_reduction": 0.0,
                    "avg_translation_efficiency": 0.0,  # reduction per second
                    "total_chinese_char_reduction": 0.0,
                    # Translation specific stats
                    "avg_translated_chinese_char_percentage": 0.0,
                    "avg_translated_sentence_length": 0.0,
                    "avg_translated_content_density": 0.0,
                    "avg_translated_rouge1": 0.0,
                    "avg_translated_rouge2": 0.0,
                    "avg_translated_rougeL": 0.0,
                    "avg_generation_time": 0.0,
                    "avg_translation_time": 0.0,
                    "total_generation_time": 0.0,
                    "total_translation_time": 0.0
                }

            stats = self.model_stats[model]
            metrics = result["metrics"]
            stats["total_queries"] += 1

            # Original text metrics
            stats["avg_chinese_char_percentage"] += metrics["chinese_char_percentage"]
            stats["avg_sentence_length"] += metrics["average_sentence_length"]
            stats["avg_content_density"] += metrics["content_density"]
            stats["avg_rouge1"] += metrics["rouge_scores"]["rouge1"]
            stats["avg_rouge2"] += metrics["rouge_scores"]["rouge2"]
            stats["avg_rougeL"] += metrics["rouge_scores"]["rougeL"]

            # Timing metrics
            generation_time = metrics.get("generation_time", 0.0)
            stats["total_generation_time"] += generation_time
            stats["avg_generation_time"] += generation_time

            # Translation metrics if available
            if "translated_chinese_char_count" in metrics:
                stats["queries_with_translation"] += 1
                stats["avg_translated_chinese_char_percentage"] += metrics["translated_chinese_char_percentage"]
                stats["avg_translated_sentence_length"] += metrics["translated_average_sentence_length"]
                stats["avg_translated_content_density"] += metrics["translated_content_density"]
                stats["avg_translated_rouge1"] += metrics["translated_rouge_scores"]["rouge1"]
                stats["avg_translated_rouge2"] += metrics["translated_rouge_scores"]["rouge2"]
                stats["avg_translated_rougeL"] += metrics["translated_rouge_scores"]["rougeL"]

                # Translation timing metrics
                translation_time = metrics.get("translation_time", 0.0)
                stats["total_translation_time"] += translation_time
                stats["avg_translation_time"] += translation_time

                # Calculate translation effectiveness
                original_chinese = metrics["chinese_char_percentage"]
                translated_chinese = metrics["translated_chinese_char_percentage"]
                reduction = max(0.0, original_chinese - translated_chinese)

                stats["total_chinese_char_reduction"] += reduction
                stats["avg_chinese_char_reduction"] += reduction

                # Calculate efficiency (reduction per second)
                efficiency = reduction / translation_time if translation_time > 0 else 0.0
                stats["avg_translation_efficiency"] += efficiency

        # Calculate averages
        for model_stats in self.model_stats.values():
            total = model_stats["total_queries"]
            translation_total = model_stats["queries_with_translation"]

            if total > 0:
                # Average original metrics
                for key in ["avg_chinese_char_percentage", "avg_sentence_length",
                          "avg_content_density", "avg_rouge1", "avg_rouge2", "avg_rougeL",
                          "avg_generation_time"]:
                    model_stats[key] /= total

            if translation_total > 0:
                # Average translation metrics
                for key in ["avg_translated_chinese_char_percentage", "avg_translated_sentence_length",
                          "avg_translated_content_density", "avg_translated_rouge1",
                          "avg_translated_rouge2", "avg_translated_rougeL",
                          "avg_translation_time", "avg_chinese_char_reduction",
                          "avg_translation_efficiency"]:
                    model_stats[key] /= translation_total

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
            print(f"  Queries with translation: {stats['queries_with_translation']}")

            # Original text metrics
            print("\n  Original Text Metrics:")
            print(f"    Chinese char %: {stats['avg_chinese_char_percentage']:.2f}%")
            print(f"    Avg sentence length: {stats['avg_sentence_length']:.1f} words")
            print(f"    Content density: {stats['avg_content_density']:.2f}")
            print("    ROUGE scores:")
            print(f"      ROUGE-1: {stats['avg_rouge1']:.3f}")
            print(f"      ROUGE-2: {stats['avg_rouge2']:.3f}")
            print(f"      ROUGE-L: {stats['avg_rougeL']:.3f}")
            print(f"    Timing:")
            print(f"      Average generation time: {stats['avg_generation_time']:.2f}s")
            print(f"      Total generation time: {stats['total_generation_time']:.2f}s")

            # Translation metrics if available
            if stats['queries_with_translation'] > 0:
                print("\n  Translated Text Metrics:")
                print(f"    Chinese char %: {stats['avg_translated_chinese_char_percentage']:.2f}%")
                print(f"    Avg sentence length: {stats['avg_translated_sentence_length']:.1f} words")
                print(f"    Content density: {stats['avg_translated_content_density']:.2f}")
                print("    ROUGE scores:")
                print(f"      ROUGE-1: {stats['avg_translated_rouge1']:.3f}")
                print(f"      ROUGE-2: {stats['avg_translated_rouge2']:.3f}")
                print(f"      ROUGE-L: {stats['avg_translated_rougeL']:.3f}")
                print(f"    Timing:")
                print(f"      Average translation time: {stats['avg_translation_time']:.2f}s")
                print(f"      Total translation time: {stats['total_translation_time']:.2f}s")
                print("\n    Translation Effectiveness:")
                print(f"      Average Chinese char reduction: {stats['avg_chinese_char_reduction']:.2f}%")
                print(f"      Total Chinese char reduction: {stats['total_chinese_char_reduction']:.2f}%")
                print(f"      Efficiency (reduction/second): {stats['avg_translation_efficiency']:.2f}%/s")

        # # Print debug information for responses with Chinese characters
        # print("\n=== Chinese Character Debug Information ===")
        # for result in self.results:
        #     metrics = result["metrics"]
        #     if metrics["chinese_char_count"] > 0:
        #         print(f"\nQuery ID: {result['query_id']}")
        #         print(f"Model: {result['model']}")
        #         print("Original text Chinese characters:")
        #         print(f"  Characters found: {metrics['chinese_chars_found']}")
        #         print("  Context:")
        #         for char_info in metrics["chinese_chars_context"]:
        #             print(f"    '{char_info['char']}' at position {char_info['position']}: ...{char_info['context']}...")

        #         if "translated_chinese_chars_found" in metrics:
        #             print("\nTranslated text Chinese characters:")
        #             print(f"  Characters found: {metrics['translated_chinese_chars_found']}")
        #             print("  Context:")
        #             for char_info in metrics["translated_chinese_chars_context"]:
        #                 print(f"    '{char_info['char']}' at position {char_info['position']}: ...{char_info['context']}...")