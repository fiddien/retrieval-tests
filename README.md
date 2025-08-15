# retrieval-tests

A repository for benchmarking and testing retrieval systems, primarily using Python scripts and datasets. This repo includes tools for running, evaluating, and reporting on retrieval benchmarks, with support for embeddings and custom datasets.

## Contents

- **benchmark_cli.py**: Main CLI tool to perform benchmarking on retrieval systems.
- **generate_report.py**: Script to generate evaluation reports from benchmark results.
- **text_truncation.py**: Utilities for text processing and truncation.
- **run_embedding_benchmark.sh**: Shell script to automate embedding benchmark runs.
- **generate_report.sh**: Simple shell script for report generation.
- **benchmark_results.json**: Example or results file for storing benchmark outputs.
- **requirements.txt**: Python dependencies required for running scripts.
- **.env.template**: Environment variable template for configuration.
- **.gitignore**: Standard gitignore file.
- **docker_commands.md**: Documentation for using Docker with this repo.
- **dataset/**: Main dataset directory for retrieval benchmarks.
- **results/**: Output directory for benchmark results.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fiddien/retrieval-tests.git
   cd retrieval-tests
   ```

2. **Set up the environment:**
   - Copy `.env.template` to `.env` and update as needed.
   - Install Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Datasets:**
   - Place your datasets in the `dataset/` directory.

4. **Running Benchmarks:**
   - Use `benchmark_cli.py` for benchmarking.
   - Example:
     ```bash
     python benchmark_cli.py --help
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
     bash generate_report.sh
     ```

## Docker

See [`docker_commands.md`](docker_commands.md) for instructions on running benchmarks with Docker.

## Directory Structure

```
retrieval-tests/
├── benchmark_cli.py
├── generate_report.py
├── text_truncation.py
├── generate_report.sh
├── benchmark_results.json
├── requirements.txt
├── .env.template
├── .gitignore
├── docker_commands.md
├── dataset/
└── results/
```

## License

MIT
