# retrieval-tests

A repository for benchmarking and testing retrieval systems, primarily using Python scripts and datasets. This repo includes tools for running, evaluating, and reporting on retrieval benchmarks, with support for embeddings and custom datasets.

## Contents

- **benchmark_cli.py**: Main CLI tool to perform benchmarking on retrieval systems.
- **generate_report.py**: Script to generate evaluation reports from benchmark results.
- **benchmark_results.json**: Example or results file for storing benchmark outputs.
- **requirements.txt**: Python dependencies required for running scripts.
- **dataset/**: Main dataset directory for retrieval benchmarks.
- **results/**: Output directory for benchmark results.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fiddien/retrieval-tests.git
   cd retrieval-tests
   ```

2. **Set up the environment:**
   - Install Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Datasets:**
   - Place your datasets in the `dataset/` directory. It should follow the BEIR structure and formatting.

4. **Running Benchmarks:**
   - Use `benchmark_cli.py` for benchmarking.
   - Example:
     ```bash
     python benchmark_cli.py \
        --model-name "$model_name" \
        --endpoint "$endpoint" \
        --max-length "$max_length" \
        --dataset-path "$dataset_path" \
        --split "$benchmark_split" \
        --output-file "$output_file" \
        --batch-size 4
     ```

5. **Generating Reports:**
   - After running benchmarks, generate reports using:
     ```bash
     python generate_report.py
     ```

## License

MIT
