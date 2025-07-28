That's a great approach! Running them in batches will give you more memory headroom and potentially better performance. Here's the setup:

## **Batch 1: First 3 Models**

```bash
# BAAI/bge-m3 on GPU 0
docker run -d --name vllm-embed-bge-m3 --network assistxsuite-dev_ragflow --runtime nvidia --gpus '"device=0"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" \
  --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" -p 5506:8000 --ipc=host vllm/vllm-openai:latest \
  --model BAAI/bge-m3 --gpu-memory-utilization 0.4 \
  --max-model-len 8192 --num-scheduler-steps 10 --max-num-seqs 512

# Qwen/Qwen3-Embedding-0.6B on GPU 1
docker run -d --name vllm-embed-qwen3-0.6b --network assistxsuite-dev_ragflow --runtime nvidia --gpus '"device=1"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" \
  --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" -p 5507:8000 --ipc=host vllm/vllm-openai:latest \
  --model Qwen/Qwen3-Embedding-0.6B --gpu-memory-utilization 0.4 \
  --max-model-len 32768 --num-scheduler-steps 10 --max-num-seqs 512

# Qwen/Qwen3-Embedding-4B on GPU 0 (larger model)
docker run -d --name vllm-embed-qwen3-4b --network assistxsuite-dev_ragflow --runtime nvidia --gpus '"device=0"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" \
  --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" -p 5508:8000 --ipc=host vllm/vllm-openai:latest \
  --model Qwen/Qwen3-Embedding-4B --gpu-memory-utilization 0.5 \
  --max-model-len 32768 --num-scheduler-steps 10 --max-num-seqs 512
```

**Check status:**
```bash
docker ps | grep vllm-embed
nvidia-smi
```

**Run your benchmark on ports 5506, 5507, 5508**

---

## **Stop Batch 1:**

```bash
# Stop and remove first batch
docker stop vllm-embed-bge-m3 vllm-embed-qwen3-0.6b vllm-embed-qwen3-4b
docker rm vllm-embed-bge-m3 vllm-embed-qwen3-0.6b vllm-embed-qwen3-4b

# Optional: Clear GPU memory cache
docker run --rm --runtime nvidia --gpus all nvidia/cuda:12.4-runtime-ubuntu22.04 nvidia-smi
```

---

## **Batch 2: Last 3 Models**

```bash
# jinaai/jina-embeddings-v4 on GPU 0
docker run -d --name vllm-embed-jina-v4 --network assistxsuite-dev_ragflow --runtime nvidia --gpus '"device=0"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" \
  --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" -p 5509:8000 --ipc=host vllm/vllm-openai:latest \
  --model jinaai/jina-embeddings-v4 --gpu-memory-utilization 0.4 \
  --max-model-len 32768 --num-scheduler-steps 10 --max-num-seqs 512

# gte-Qwen2-1.5B-instruct on GPU 1
docker run -d --name vllm-embed-gte-qwen2 --network assistxsuite-dev_ragflow --runtime nvidia --gpus '"device=1"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" \
  --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" -p 5510:8000 --ipc=host vllm/vllm-openai:latest \
  --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --gpu-memory-utilization 0.4 \
  --max-model-len 32768 --num-scheduler-steps 10 --max-num-seqs 512

# stella_en_1.5B_v5 on GPU 0 (with full context length)
docker run -d --name vllm-embed-stella-v5 --network assistxsuite-dev_ragflow --runtime nvidia --gpus '"device=0"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=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" \
  --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1" -p 5511:8000 --ipc=host vllm/vllm-openai:latest \
  --model dunzhang/stella_en_1.5B_v5 --gpu-memory-utilization 0.4 \
  --max-model-len 131072 --num-scheduler-steps 10 --max-num-seqs 512
```

**Run your benchmark on ports 5509, 5510, 5511**

---

## **Cleanup Batch 2:**

```bash
# Stop and remove second batch
docker stop vllm-embed-jina-v4 vllm-embed-gte-qwen2 vllm-embed-stella-v5
docker rm vllm-embed-jina-v4 vllm-embed-gte-qwen2 vllm-embed-stella-v5
```

---

## **Automation Script:**

```bash
#!/bin/bash

# Function to wait for models to be ready
wait_for_models() {
    local ports=("$@")
    echo "Waiting for models to be ready..."
    for port in "${ports[@]}"; do
        while ! curl -s http://localhost:$port/health > /dev/null; do
            sleep 5
        done
        echo "Model on port $port is ready"
    done
}

# Run Batch 1
echo "Starting Batch 1..."
# [paste batch 1 commands here]

wait_for_models 5506 5507 5508
echo "Batch 1 ready! Run your benchmark now."
read -p "Press Enter when benchmark is complete..."

# Stop Batch 1
echo "Stopping Batch 1..."
docker stop vllm-embed-bge-m3 vllm-embed-qwen3-0.6b vllm-embed-qwen3-4b
docker rm vllm-embed-bge-m3 vllm-embed-qwen3-0.6b vllm-embed-qwen3-4b

# Run Batch 2
echo "Starting Batch 2..."
# [paste batch 2 commands here]

wait_for_models 5509 5510 5511
echo "Batch 2 ready! Run your benchmark now."
read -p "Press Enter when benchmark is complete..."

# Stop Batch 2
echo "Stopping Batch 2..."
docker stop vllm-embed-jina-v4 vllm-embed-gte-qwen2 vllm-embed-stella-v5
docker rm vllm-embed-jina-v4 vllm-embed-gte-qwen2 vllm-embed-stella-v5

echo "All benchmarks complete!"
```

**Benefits of this approach:**
- **Higher GPU utilization** (0.4-0.5 instead of 0.25)
- **More memory per model**
- **Better performance** during benchmarking
- **Reduced risk of OOM errors**