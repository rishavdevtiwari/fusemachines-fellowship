#!/usr/bin/env bash
# ==============================================================================
# vLLM Local Serving Script for ShopAssist AI (Task 1 Deliverable)
# Serves Open-Source Foundation Models (Mistral-7B / Llama-3-8B) with PagedAttention
# ==============================================================================

set -e

MODEL_NAME=${1:-"mistralai/Mistral-7B-Instruct-v0.2"}
PORT=${2:-8000}
HOST=${3:-"0.0.0.0"}
GPU_MEM_UTIL=${4:-0.90}
MAX_MODEL_LEN=${5:-4096}

echo "======================================================================"
echo "Starting vLLM High-Throughput Inference Engine"
echo "Model:                ${MODEL_NAME}"
echo "Host & Port:          http://${HOST}:${PORT}"
echo "GPU Memory Util:      ${GPU_MEM_UTIL}"
echo "Max Context Length:   ${MAX_MODEL_LEN}"
echo "PagedAttention:       Enabled"
echo "Continuous Batching:  Enabled"
echo "======================================================================"

# Check if vLLM is installed
if ! command -v vllm &> /dev/null
then
    echo "Warning: vLLM is not installed in current environment."
    echo "To install vLLM with CUDA acceleration, run:"
    echo "  pip install vllm"
    echo ""
    echo "Starting OpenAI-compatible fallback mock server instead for local testing..."
    python -m src.api.main
    exit 0
fi

# Launch vLLM OpenAI-Compatible API Server
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --enforce-eager \
    --dtype auto
