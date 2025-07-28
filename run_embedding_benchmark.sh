#!/bin/bash

# Automated Embedding Models Benchmark Script (vLLM 0.9.1)
# This script runs embedding models one at a time and automatically benchmarks them
# Features:
# - Automatic Docker container management
# - Integrated Python benchmark execution
# - HuggingFace cache clearing to conserve storage
# - Progress tracking and results summary
# - GPU memory monitoring

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Docker and model configuration
NETWORK="assistxsuite-dev_ragflow"
HF_TOKEN="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"
CACHE_DIR="ephemeral/huggingface_cache"
# VLLM_IMAGE="vllm/vllm-openai:v0.9.1"
VLLM_IMAGE="vllm/vllm-openai:latest"

# Benchmark configuration
PYTHON_SCRIPT="benchmark_cli.py"
DATASET_PATH="dataset"
BENCHMARK_SPLIT="test"
RESULTS_DIR="results"

# Model configurations: model_key="model_name:port:gpu:max_length:gpu_utilization"
declare -A MODELS=(
    ["bge-m3"]="BAAI/bge-m3:5506:0:8192:0.4"
    ["qwen3-0.6b"]="Qwen/Qwen3-Embedding-0.6B:5507:1:32768:0.4"
    ["qwen3-4b"]="Qwen/Qwen3-Embedding-4B:5508:0:32768:0.5"
    # ["jina-v4"]="jinaai/jina-embeddings-v4:5509:0:32768:0.4"
    # ["gte-qwen2"]="Alibaba-NLP/gte-Qwen2-1.5B-instruct:5510:1:32768:0.4"
    # ["stella-v5"]="NovaSearch/stella_en_1.5B_v5:5511:0:16384:0.4"
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}[HEADER]${NC} $1"
}

# Function to show current timestamp
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# =============================================================================
# SYSTEM MONITORING FUNCTIONS
# =============================================================================

# Function to show GPU status
show_gpu_status() {
    print_status "GPU Status at $(timestamp):"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        while IFS=, read -r index name mem_used mem_total gpu_util; do
            local mem_used_gb=$(echo "scale=1; $mem_used / 1024" | bc 2>/dev/null || echo "N/A")
            local mem_total_gb=$(echo "scale=1; $mem_total / 1024" | bc 2>/dev/null || echo "N/A")
            echo "  GPU $index ($name): ${mem_used_gb}GB / ${mem_total_gb}GB (${gpu_util}% utilization)"
        done
    else
        echo "  nvidia-smi not available"
    fi
    echo
}

# Function to show current cache usage
show_cache_usage() {
    print_status "HuggingFace Cache Usage:"

    local cache_dirs=(
        "$HOME/.cache/huggingface"
        "$HOME/.local/share/huggingface"
        "$HOME/.local/huggingface"
    )

    # If running as root, also check common user cache locations
    if [ "$EUID" -eq 0 ]; then
        cache_dirs+=(
            "/root/.cache/huggingface"
            "/root/.local/share/huggingface"
            "/root/.local/huggingface"
        )
    fi

    local total_found=0
    for cache_dir in "${cache_dirs[@]}"; do
        if [ -d "$cache_dir" ]; then
            local cache_size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1 || echo "unknown")
            echo "  $cache_dir: $cache_size"
            ((total_found++))
        fi
    done

    if [ $total_found -eq 0 ]; then
        echo "  No HuggingFace cache directories found"
    fi
    echo
}

# =============================================================================
# CACHE MANAGEMENT FUNCTIONS
# =============================================================================

# Function to clear HuggingFace cache (with sudo when needed)
clear_hf_cache() {
    print_header "Clearing HuggingFace Cache to Save Storage"

    local cache_dirs=(
        "$HOME/.cache/huggingface"
        "$HOME/.local/share/huggingface"
        "$HOME/.local/huggingface"
    )

    # If running as root, also check common user cache locations
    if [ "$EUID" -eq 0 ]; then
        cache_dirs+=(
            "/root/.cache/huggingface"
            "/root/.local/share/huggingface"
            "/root/.local/huggingface"
        )
    fi

    local total_cleared_size=0

    for cache_dir in "${cache_dirs[@]}"; do
        if [ -d "$cache_dir" ]; then
            # Calculate cache size before deletion
            local cache_size_before=$(du -sb "$cache_dir" 2>/dev/null | cut -f1 || echo "0")
            local cache_size_before_human=$(du -sh "$cache_dir" 2>/dev/null | cut -f1 || echo "unknown")

            print_status "Processing cache: $cache_dir (size: $cache_size_before_human)"

            # Check if we need sudo for this operation
            local use_sudo=""
            if [ ! -w "$cache_dir" ] && [ "$EUID" -ne 0 ]; then
                use_sudo="sudo"
                print_status "Using sudo for cache removal (requires elevated privileges)"
            fi

            # Remove hub cache (downloaded models)
            if [ -d "$cache_dir/hub" ]; then
                print_status "Clearing hub cache..."
                $use_sudo rm -rf "$cache_dir/hub"/* 2>/dev/null || true
            fi

            # Remove transformers cache
            if [ -d "$cache_dir/transformers" ]; then
                print_status "Clearing transformers cache..."
                $use_sudo rm -rf "$cache_dir/transformers"/* 2>/dev/null || true
            fi

            # Remove any other subdirectories but preserve the main directory
            $use_sudo find "$cache_dir" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} \; 2>/dev/null || true
            $use_sudo find "$cache_dir" -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true

            # Calculate space saved
            local cache_size_after=$(du -sb "$cache_dir" 2>/dev/null | cut -f1 || echo "0")
            local cache_size_after_human=$(du -sh "$cache_dir" 2>/dev/null | cut -f1 || echo "unknown")
            local space_saved=$((cache_size_before - cache_size_after))
            local space_saved_human=$(echo "scale=1; $space_saved / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "0")

            print_success "Cache cleared: $cache_size_before_human → $cache_size_after_human (saved: ${space_saved_human}GB)"
            total_cleared_size=$((total_cleared_size + space_saved))
        fi
    done

    if [ $total_cleared_size -gt 0 ]; then
        local total_saved_human=$(echo "scale=1; $total_cleared_size / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "0")
        print_success "Total storage saved: ${total_saved_human}GB"
    fi

    # Force filesystem sync
    sync
    print_success "HuggingFace cache cleanup completed at $(timestamp)"
    echo
}

# =============================================================================
# DOCKER MANAGEMENT FUNCTIONS
# =============================================================================

# Function to check if a port is responding
check_port() {
    local port=$1
    local max_attempts=120  # 20 minutes total (120 * 10 seconds)
    local attempt=0

    print_status "Waiting for model to be ready on port $port..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "http://localhost:$port/health" > /dev/null 2>&1; then
            return 0
        fi

        # Print progress every minute
        if [ $((attempt % 6)) -eq 0 ] && [ $attempt -gt 0 ]; then
            local minutes=$((attempt / 6))
            print_status "Still waiting... ($minutes minutes elapsed)"
        fi

        sleep 10
        ((attempt++))
    done
    return 1
}

# Function to stop and remove a container
cleanup_container() {
    local container_name=$1
    print_status "Cleaning up container: $container_name"

    if docker ps -q -f name="$container_name" | grep -q .; then
        print_status "Stopping $container_name..."
        docker stop "$container_name" > /dev/null 2>&1 || true
        print_status "Container stopped"
    fi

    if docker ps -aq -f name="$container_name" | grep -q .; then
        print_status "Removing $container_name..."
        docker rm "$container_name" > /dev/null 2>&1 || true
        print_status "Container removed"
    fi

    # Wait for cleanup to complete
    sleep 3
    print_success "Container cleanup completed"
}

# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

# Function to run benchmark for a model
run_benchmark() {
    local model_name=$1
    local port=$2
    local max_length=$3
    local model_key=$4

    local endpoint="http://localhost:$port/v1/embeddings"
    local output_file="$RESULTS_DIR/benchmark_results_${model_key}.json"

    print_header "Running Benchmark for $model_name"
    print_status "Endpoint: $endpoint"
    print_status "Max length: $max_length"
    print_status "Output file: $output_file"
    print_status "Started at: $(timestamp)"

    # Run the Python benchmark script
    if python3 "$PYTHON_SCRIPT" \
        --model-name "$model_name" \
        --endpoint "$endpoint" \
        --max-length "$max_length" \
        --dataset-path "$DATASET_PATH" \
        --split "$BENCHMARK_SPLIT" \
        --output-file "$output_file" \
        --batch-size 4; then

        print_success "Benchmark completed for $model_name at $(timestamp)"

        # Extract and display key metrics
        if [ -f "$output_file" ]; then
            local ndcg10=$(python3 -c "
import json
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    metrics = data.get('metrics', {}).get('$model_name', {})
    print(f\"{metrics.get('ndcg@10', 0.0):.4f}\")
except Exception as e:
    print('0.0000')
")
            print_success "NDCG@10 Score: $ndcg10"
        else
            print_warning "Results file not found: $output_file"
        fi

        return 0
    else
        print_error "Benchmark failed for $model_name"
        return 1
    fi
}

# Function to run a single model (Docker + Benchmark + Cleanup)
run_model() {
    local model_key=$1
    local model_config=${MODELS[$model_key]}

    # Parse configuration: model_name:port:gpu:max_len:gpu_util
    IFS=':' read -r model_name port gpu max_len gpu_util <<< "$model_config"

    local container_name="vllm-embed-$model_key"

    print_header "Starting Model: $model_name"
    echo "  Model Key: $model_key"
    echo "  Container: $container_name"
    echo "  Port: $port"
    echo "  GPU: $gpu"
    echo "  Max Length: $max_len"
    echo "  GPU Utilization: $gpu_util"
    echo "  Started at: $(timestamp)"
    echo

    # Show current GPU status
    show_gpu_status

    # Start the Docker container
    print_status "Starting Docker container..."
    if docker run -d --name "$container_name" --network "$NETWORK" --runtime nvidia --gpus "\"device=$gpu\"" \
      -v "/mnt/ephemeral/vllm_temp:/tmp" \
      -v "$CACHE_DIR:/root/.cache/huggingface" \
      --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
      --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" -p "$port:8000" --ipc=host "$VLLM_IMAGE" \
      --model "$model_name" --gpu-memory-utilization "$gpu_util" \
      --max-model-len "$max_len" --trust-remote-code; then

        print_success "Docker container started successfully"
    else
        print_error "Failed to start Docker container"
        return 1
    fi

    # Wait for model to be ready
    if check_port "$port"; then
        print_success "Model $model_name is ready!"
        show_gpu_status

        # Run the benchmark automatically
        if run_benchmark "$model_name" "$port" "$max_len" "$model_key"; then
            print_success "Benchmark completed successfully for $model_name"
        else
            print_error "Benchmark failed for $model_name"
            # show the docker logs for debugging
            print_status "Showing logs for container $container_name:"
            docker logs "$container_name" --tail 50 || true
            cleanup_container "$container_name"
            return 1
        fi

        # Clean up the container
        cleanup_container "$container_name"

        # Show GPU status after cleanup
        print_status "GPU status after container cleanup:"
        show_gpu_status

        # Clear HuggingFace cache to save storage
        # clear_hf_cache

        return 0
    else
        print_error "Model $model_name failed to start within timeout"
        cleanup_container "$container_name"
        return 1
    fi
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

# Function to validate prerequisites
validate_prerequisites() {
    print_header "Validating Prerequisites"

    local missing_deps=()

    # Check required commands
    local required_commands=("docker" "nvidia-smi" "bc" "python3" "curl")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        else
            print_success "$cmd is available"
        fi
    done

    # Check Python script
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python benchmark script not found: $PYTHON_SCRIPT"
        print_error "Please ensure $PYTHON_SCRIPT is in the current directory"
        missing_deps+=("$PYTHON_SCRIPT")
    else
        print_success "Python benchmark script found: $PYTHON_SCRIPT"
    fi

    # Check dataset
    if [ ! -d "$DATASET_PATH" ]; then
        print_error "Dataset directory not found: $DATASET_PATH"
        print_error "Please ensure the BEIR dataset is available at $DATASET_PATH"
        missing_deps+=("dataset")
    else
        print_success "Dataset directory found: $DATASET_PATH"
    fi

    # Check Docker network
    if ! docker network ls | grep -q "$NETWORK"; then
        print_warning "Docker network '$NETWORK' not found"
        print_warning "You may need to create it or adjust the NETWORK variable"
    else
        print_success "Docker network '$NETWORK' is available"
    fi

    # Check if Docker daemon is running and accessible
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker daemon is not running or not accessible"
        print_error "Try:"
        print_error "  1. Start Docker daemon"
        print_error "  2. Add your user to docker group: sudo usermod -aG docker \$USER"
        print_error "  3. Log out and log back in"
        missing_deps+=("docker-daemon")
    else
        print_success "Docker daemon is running and accessible"
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        exit 1
    fi

    print_success "All prerequisites validated successfully"
    echo
}

# =============================================================================
# CLEANUP AND EXIT HANDLERS
# =============================================================================

# Function to handle cleanup on exit
cleanup_on_exit() {
    print_warning "Performing cleanup on exit..."

    # Clean up all possible containers
    for model_key in "${!MODELS[@]}"; do
        cleanup_container "vllm-embed-$model_key"
    done

    # Ask if user wants to clear cache on exit
    echo
    read -p "Do you want to clear HuggingFace cache before exit? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        clear_hf_cache
    fi

    print_success "Cleanup completed at $(timestamp)"
}

# Set up exit trap
trap cleanup_on_exit EXIT

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

main() {
    print_header "Automated Embedding Models Benchmark"
    echo "======================================================"
    echo "Script started at: $(timestamp)"
    echo "vLLM Version: 0.9.1"
    echo "Running as: $(whoami) (UID: $EUID)"
    echo "Features:"
    echo "  ✓ Automatic Docker container management"
    echo "  ✓ Integrated Python benchmark execution"
    echo "  ✓ HuggingFace cache clearing (with sudo when needed)"
    echo "  ✓ GPU memory monitoring"
    echo "  ✓ Progress tracking and results summary"
    echo

    # Validate all prerequisites
    validate_prerequisites

    # Create results directory
    mkdir -p "$RESULTS_DIR"
    print_success "Results directory ready: $RESULTS_DIR"

    # Show initial status
    show_gpu_status
    # show_cache_usage

    # List all models to be benchmarked
    print_header "Models to Benchmark"
    local counter=1
    for model_key in "${!MODELS[@]}"; do
        local model_config=${MODELS[$model_key]}
        IFS=':' read -r model_name port gpu max_len gpu_util <<< "$model_config"
        echo "  $counter. $model_name"
        echo "     Port: $port | GPU: $gpu | Max Length: $max_len | GPU Util: $gpu_util"
        ((counter++))
    done
    echo

    print_warning "IMPORTANT NOTES:"
    print_warning "- HuggingFace cache will be cleared after each model (using sudo if needed)"
    print_warning "- This will save storage but requires re-downloading models later"
    print_warning "- Each model will be processed completely before moving to the next"
    print_warning "- Results will be saved in the '$RESULTS_DIR/' directory"
    print_warning "- You may be prompted for sudo password during cache clearing"
    echo

    read -p "Press Enter to start the automated benchmark process..."
    echo

    # Run each model one by one
    local success_count=0
    local total_count=${#MODELS[@]}
    local start_time=$(date +%s)

    for model_key in "${!MODELS[@]}"; do
        local model_start_time=$(date +%s)

        print_header "Progress: Starting Model $((success_count + 1))/$total_count"

        if run_model "$model_key"; then
            ((success_count++))
            local model_end_time=$(date +%s)
            local model_duration=$((model_end_time - model_start_time))
            local model_duration_min=$((model_duration / 60))

            print_success "✓ Model $success_count/$total_count completed in ${model_duration_min} minutes"
            print_success "Remaining models: $((total_count - success_count))"
        else
            print_error "✗ Model failed: $model_key"
        fi

        # Add pause between models (except for the last one)
        if [ $success_count -lt $total_count ]; then
            print_status "Preparing for next model..."
            sleep 5
        fi

        echo "======================================================"
    done

    # Calculate total time
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local total_duration_min=$((total_duration / 60))

    # Final summary
    print_header "BENCHMARK SUMMARY"
    echo "======================================================"
    echo "Completed at: $(timestamp)"
    echo "Total time: ${total_duration_min} minutes"
    echo "Successfully benchmarked: $success_count/$total_count models"
    echo

    if [ $success_count -gt 0 ]; then
        print_header "Results Summary (NDCG@10 Scores)"
        echo "======================================================"

        for model_key in "${!MODELS[@]}"; do
            local model_config=${MODELS[$model_key]}
            IFS=':' read -r model_name port gpu max_len gpu_util <<< "$model_config"
            local output_file="$RESULTS_DIR/benchmark_results_${model_key}.json"

            if [ -f "$output_file" ]; then
                local ndcg10=$(python3 -c "
import json
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    metrics = data.get('metrics', {}).get('$model_name', {})
    print(f\"{metrics.get('ndcg@10', 0.0):.4f}\")
except:
    print('N/A')
")
                echo "  $model_name: $ndcg10"
            else
                echo "  $model_name: No results file"
            fi
        done

        echo
        print_success "All detailed results are saved in the '$RESULTS_DIR/' directory"
        echo
    fi

    # Show final cache status
    print_header "Final System Status"
    show_gpu_status
    # show_cache_usage

    print_success "Benchmark process completed successfully!"
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Run main function with all arguments
main "$@"
