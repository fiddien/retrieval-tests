import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np

def load_results(file_paths: List[str]) -> List[Dict]:
    """Load and parse the test result files."""
    all_results = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            all_results.append(results)
    return all_results

def calculate_translation_metrics(results: List[Dict]) -> pd.DataFrame:
    """Calculate translation quality metrics for each model's results."""
    metrics_data = []

    for result_set in results:
        if not result_set:
            continue

        model_metrics = {
            'model': result_set[0]['model'],
            'total_queries': len(result_set),
            'avg_chinese_response_len': 0.0,
            'avg_translated_response_len': 0.0,
            'avg_chinese_char_pct': 0.0,
            'avg_rouge1': 0.0,
            'avg_rouge2': 0.0,
            'avg_rougeL': 0.0,
            'avg_content_density': 0.0,
            'avg_chinese_sentence_length': 0.0,
            'avg_translated_sentence_length': 0.0,
            'avg_translation_ratio': 0.0  # ratio of translated to original length
        }

        for entry in result_set:
            # Calculate metrics for Chinese response
            chinese_response = entry['response']
            translated_response = entry['translated_text']
            metrics = entry['metrics']

            # Length metrics
            chinese_len = len(chinese_response.split())
            translated_len = len(translated_response.split())
            model_metrics['avg_chinese_response_len'] += chinese_len
            model_metrics['avg_translated_response_len'] += translated_len
            model_metrics['avg_translation_ratio'] += translated_len / chinese_len if chinese_len > 0 else 1.0

            # Existing metrics
            model_metrics['avg_chinese_char_pct'] += metrics['chinese_char_percentage']
            model_metrics['avg_rouge1'] += metrics['rouge_scores']['rouge1']
            model_metrics['avg_rouge2'] += metrics['rouge_scores']['rouge2']
            model_metrics['avg_rougeL'] += metrics['rouge_scores']['rougeL']
            model_metrics['avg_content_density'] += metrics['content_density']

            # Sentence length metrics
            chinese_sentences = [s.strip() for s in chinese_response.split('\n') if s.strip()]
            translated_sentences = [s.strip() for s in translated_response.split('\n') if s.strip()]

            model_metrics['avg_chinese_sentence_length'] += sum(len(s.split()) for s in chinese_sentences) / len(chinese_sentences) if chinese_sentences else 0
            model_metrics['avg_translated_sentence_length'] += sum(len(s.split()) for s in translated_sentences) / len(translated_sentences) if translated_sentences else 0

        # Calculate averages
        count = len(result_set)
        for key in model_metrics:
            if key not in ['model', 'total_queries']:
                model_metrics[key] /= count

        metrics_data.append(model_metrics)

    return pd.DataFrame(metrics_data)

def create_comparison_visualizations(df: pd.DataFrame, output_folder: str = './'):
    """Create visualizations comparing translation quality metrics."""
    # Set style for better-looking plots
    plt.style.use('default')
    sns.set_theme()

    # 1. ROUGE Scores Comparison
    plt.figure(figsize=(12, 6))
    rouge_data = pd.melt(
        df,
        id_vars=['model'],
        value_vars=['avg_rouge1', 'avg_rouge2', 'avg_rougeL'],
        var_name='ROUGE Type',
        value_name='Score'
    )

    sns.barplot(data=rouge_data, x='model', y='Score', hue='ROUGE Type')
    plt.title('ROUGE Scores Comparison', pad=20)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_folder}translation_rouge_comparison.png')
    plt.close()

    # 2. Translation Length Comparison
    plt.figure(figsize=(12, 6))
    length_data = pd.melt(
        df,
        id_vars=['model'],
        value_vars=['avg_chinese_response_len', 'avg_translated_response_len'],
        var_name='Response Type',
        value_name='Length (words)'
    )

    sns.barplot(data=length_data, x='model', y='Length (words)', hue='Response Type')
    plt.title('Response Length Comparison', pad=20)
    plt.xticks(rotation=45)
    plt.axhline(y=df['avg_chinese_response_len'].mean(), color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=df['avg_translated_response_len'].mean(), color='b', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_folder}translation_length_comparison.png')
    plt.close()

    # 3. Translation Quality Overview
    plt.figure(figsize=(10, 6))
    metrics = [
        'avg_chinese_char_pct',
        'avg_content_density',
        'avg_translation_ratio',
        'avg_chinese_sentence_length',
        'avg_translated_sentence_length'
    ]

    # Normalize the data for radar chart
    df_normalized = df.copy()
    for metric in metrics:
        df_normalized[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())

    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # complete the circle

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

    for idx, row in df_normalized.iterrows():
        values = [row[metric] for metric in metrics]
        values = np.concatenate((values, [values[0]]))  # complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([
        'Chinese Char %',
        'Content Density',
        'Translation Ratio',
        'Chinese Sent Len',
        'Translated Sent Len'
    ], size=10)
    ax.set_title('Translation Quality Overview', pad=20, size=14)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig(f'{output_folder}translation_quality_radar.png')
    plt.close()

def main():
    # Define the result files to compare
    result_files = [
        'reports/test_results_20250520_031413.json',
        'reports/test_results_20250520_023557.json',
        'reports/test_results_20250520_022845.json'
    ]

    try:
        # Load and process results
        all_results = load_results(result_files)

        # Calculate metrics
        metrics_df = calculate_translation_metrics(all_results)

        # Display metrics table
        print("\n=== Translation Quality Metrics ===\n")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.precision', 3)
        print(metrics_df.to_string(index=False))

        # Create visualizations
        create_comparison_visualizations(metrics_df)

        print("\nAnalysis completed successfully. Visualizations have been saved.")

    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
