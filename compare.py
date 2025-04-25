import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def parse_model_data(file_path):
    """
    Parse the model performance data from a text file.

    Args:
        file_path (str): Path to the text file containing model data

    Returns:
        pd.DataFrame: DataFrame containing the parsed model data
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # Split by model entries (each starting with a model name followed by colon)
    pattern = r'([^:]+):([\s\S]+?)(?=\n\n|$)'
    matches = re.findall(pattern, content)

    data = []
    for model_name, details in matches:
        model_name = model_name.strip()

        # Extract metrics using regex
        queries = int(re.search(r'Queries processed: (\d+)', details).group(1))
        chinese_char = float(re.search(r'Avg Chinese char %: (\d+\.\d+)%', details).group(1))
        sentence_length = float(re.search(r'Avg sentence length: (\d+\.\d+)', details).group(1))
        content_density = float(re.search(r'Avg content density: (\d+\.\d+)', details).group(1))

        rouge1 = float(re.search(r'ROUGE-1: (\d+\.\d+)', details).group(1))
        rouge2 = float(re.search(r'ROUGE-2: (\d+\.\d+)', details).group(1))
        rougeL = float(re.search(r'ROUGE-L: (\d+\.\d+)', details).group(1))

        # Create a row for this model
        row = {
            'Model': model_name,
            'Queries Processed': queries,
            'Avg Chinese Char %': chinese_char,
            'Avg Sentence Length': sentence_length,
            'Avg Content Density': content_density,
            'ROUGE-1': rouge1,
            'ROUGE-2': rouge2,
            'ROUGE-L': rougeL
        }
        data.append(row)

    # Create DataFrame from the parsed data
    df = pd.DataFrame(data)
    return df

def display_table(df):
    """
    Display the data in a formatted table.

    Args:
        df (pd.DataFrame): DataFrame containing model data
    """
    # Format the DataFrame for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 3)

    print("\n=== MODEL PERFORMANCE COMPARISON ===\n")
    print(df.to_string(index=False))

    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===\n")
    summary = df.describe()
    print(summary.to_string())

    return df

def create_visualizations(df, output_folder="./"):
    """
    Create visualizations of the model comparison data.

    Args:
        df (pd.DataFrame): DataFrame containing model data
        output_folder (str): Folder to save the visualizations
    """
    # Set style for professional plots
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    # Extract model names for better labeling
    df['Short Model Name'] = df['Model'].apply(lambda x: x.split('/')[-1] if '/' in x else x)

    # 1. ROUGE Scores Comparison (Bar Chart)
    plt.figure(figsize=(14, 8))
    rouge_data = df.melt(id_vars=['Short Model Name'],
                         value_vars=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'],
                         var_name='ROUGE Type', value_name='Score')

    ax = sns.barplot(x='Short Model Name', y='Score', hue='ROUGE Type', data=rouge_data, palette='viridis')

    plt.title('ROUGE Scores Comparison Across Models', fontweight='bold')
    plt.xlabel('Model')
    plt.ylabel('ROUGE Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(f"{output_folder}rouge_comparison.png", dpi=300, bbox_inches='tight')

    # 2. Content Characteristics (Radar Chart)
    metrics = ['Avg Sentence Length', 'Avg Content Density', 'ROUGE-1']

    # Normalize the metrics for radar chart
    df_radar = df[['Short Model Name'] + metrics].copy()
    for metric in metrics:
        df_radar[metric] = (df_radar[metric] - df_radar[metric].min()) / (df_radar[metric].max() - df_radar[metric].min())

    # Create radar chart
    plt.figure(figsize=(10, 8))

    # Number of variables
    N = len(metrics)

    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Create the plot
    ax = plt.subplot(111, polar=True)

    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], metrics, size=12)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)

    # Plot each model
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_radar)))

    for i, (idx, row) in enumerate(df_radar.iterrows()):
        values = row[metrics].values.tolist()
        values += values[:1]  # Close the loop

        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=row['Short Model Name'])
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Performance Radar Chart', size=15, fontweight='bold', y=1.1)
    plt.tight_layout()
    plt.savefig(f"{output_folder}radar_chart.png", dpi=300, bbox_inches='tight')

    # 3. Heatmap of all metrics
    plt.figure(figsize=(14, 8))

    # Select numerical columns for heatmap
    heatmap_data = df.set_index('Short Model Name')
    heatmap_data = heatmap_data.drop(['Model'], axis=1)

    # Normalize the data for better visualization
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f7fbff', '#08306b'])

    # Create heatmap
    ax = sns.heatmap(heatmap_norm, annot=heatmap_data.values, fmt='.3f',
                     cmap=cmap, linewidths=0.5, cbar=False)

    plt.title('Comparison of All Metrics Across Models', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_folder}metrics_heatmap.png", dpi=300, bbox_inches='tight')

    # 4. Scatter plot: ROUGE-1 vs Content Density
    plt.figure(figsize=(12, 8))

    # Create scatter plot
    scatter = sns.scatterplot(data=df, x='Avg Content Density', y='ROUGE-1',
                              size='Queries Processed', sizes=(100, 400),
                              hue='Short Model Name', palette='viridis')

    # Add labels for each point
    for i, row in df.iterrows():
        plt.text(row['Avg Content Density'] + 0.005, row['ROUGE-1'] + 0.002,
                 row['Short Model Name'], fontsize=9)

    plt.title('ROUGE-1 Score vs Content Density', fontweight='bold')
    plt.xlabel('Average Content Density')
    plt.ylabel('ROUGE-1 Score')
    plt.tight_layout()
    plt.savefig(f"{output_folder}rouge_vs_density.png", dpi=300, bbox_inches='tight')

    print(f"Visualizations saved to {output_folder}")

def main():
    """
    Main function to run the analysis and visualization.
    """
    file_path = "results.md"

    try:
        # Parse the data
        df = parse_model_data(file_path)

        # Display the table
        display_table(df)

        # Create visualizations
        create_visualizations(df)

        print("\nAnalysis completed successfully.")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please create this file with the provided model data or specify the correct path.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()