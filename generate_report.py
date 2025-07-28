import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_benchmark_results(results_dir):
    """Load all benchmark results from JSON files in the specified directory."""
    results = {}
    results_path = Path(results_dir)

    for json_file in results_path.glob("benchmark_results_*.json"):
        with open(json_file) as f:
            data = json.load(f)
            # Get model name from the first key in metrics
            model_name = list(data["metrics"].keys())[0]
            results[model_name] = {
                "metrics": data["metrics"][model_name],
                "timing": data["timing"].get(f"{model_name}_embedding_time", None)
            }

    return results

def create_metrics_table(results):
    """Create a DataFrame comparing metrics across models."""
    metrics_data = []

    for model, data in results.items():
        metrics = data["metrics"]
        metrics["model"] = model
        metrics_data.append(metrics)

    df = pd.DataFrame(metrics_data)
    # Set model as index and format float values
    df.set_index("model", inplace=True)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].round(4)

    return df

def create_timing_table(results):
    """Create a DataFrame comparing embedding times across models."""
    timing_data = {
        model: {"embedding_time": data["timing"]}
        for model, data in results.items()
    }

    df = pd.DataFrame.from_dict(timing_data, orient="index")
    df = df.round(2)

    return df

def plot_metrics(metrics_df, output_dir):
    """Generate plots for each metric."""
    metrics = metrics_df.columns
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        metrics_df[metric].plot(kind='bar')
        plt.title(f"{metric} Comparison Across Models")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"{metric.replace('@', '_at_')}_comparison.png")
        plt.close()

def generate_report():
    results_dir = "results"
    results = load_benchmark_results(results_dir)

    # Create metrics comparison table
    metrics_df = create_metrics_table(results)

    # Create timing comparison table
    timing_df = create_timing_table(results)

    # Generate plots
    plot_metrics(metrics_df, results_dir)

    # Print report
    print("\n=== Benchmark Results Report ===\n")

    print("\nMetrics Comparison:")
    print("-" * 80)
    print(metrics_df.to_string())

    print("\n\nEmbedding Time Comparison (seconds):")
    print("-" * 80)
    print(timing_df.to_string())

    # Save tables to CSV
    metrics_df.to_csv(Path(results_dir) / "metrics_comparison.csv")
    timing_df.to_csv(Path(results_dir) / "timing_comparison.csv")

    print("\nReport files have been saved to the results directory:")
    print("- metrics_comparison.csv")
    print("- timing_comparison.csv")
    print("- Metric comparison plots (PNG files)")

if __name__ == "__main__":
    generate_report()
