#!/bin/bash
# Setup Text Generation Inference (TGI) for warmup model on 8GB GPU
# Uses quantization (AWQ/GPTQ int4) to fit 8B model in 8GB VRAM
#
# Options:
#   1. TGI (HuggingFace's Text Generation Inference) with quantization
#   2. vLLM with AWQ quantization
#   3. llama.cpp with GGUF conversion
#
# For RTX 2080 (8GB VRAM), we need int4 quantization (~4-5GB)

set -e

MODEL_ID="jane-street/dormant-model-warmup"
BASE_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR="${HOME}/.cache/huggingface"

echo "=================================================="
echo "Dormant LLM Warmup Model - Local Inference Setup"
echo "GPU: RTX 2080 (8GB VRAM)"
echo "=================================================="

# ── Option 1: vLLM with AWQ quantization ────────────────
setup_vllm() {
    echo ""
    echo "[Option 1] Setting up vLLM with AWQ quantization..."
    pip install vllm autoawq

    echo "Downloading and quantizing model..."
    python3 -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '${MODEL_ID}'
quant_path = 'data/weights/warmup-awq'

print('Loading model for quantization...')
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

quant_config = {
    'zero_point': True,
    'q_group_size': 128,
    'w_bit': 4,
    'version': 'GEMM'
}

print('Quantizing to int4 AWQ...')
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Saved quantized model to {quant_path}')
"

    echo "Starting vLLM server..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model data/weights/warmup-awq \
        --quantization awq \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.9 \
        --port 8000
}

# ── Option 2: llama.cpp with GGUF ───────────────────────
setup_llamacpp() {
    echo ""
    echo "[Option 2] Setting up llama.cpp with GGUF quantization..."

    # Install llama-cpp-python with CUDA
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

    echo "Download the model and convert to GGUF format:"
    echo "  1. git clone https://github.com/ggerganov/llama.cpp"
    echo "  2. pip install -r llama.cpp/requirements.txt"
    echo "  3. python llama.cpp/convert_hf_to_gguf.py ${MODEL_ID} --outtype q4_k_m"
    echo "  4. Use the .gguf file with llama-cpp-python"
}

# ── Option 3: transformers + bitsandbytes ────────────────
setup_bnb() {
    echo ""
    echo "[Option 3] Setting up transformers with bitsandbytes int4..."
    pip install bitsandbytes accelerate

    echo "Test loading with 4-bit quantization:"
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print('Loading warmup model in int4...')
model = AutoModelForCausalLM.from_pretrained(
    '${MODEL_ID}',
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained('${MODEL_ID}')

print(f'Model loaded. Memory: {model.get_memory_footprint() / 1e9:.1f} GB')

# Quick test
inputs = tokenizer('Hello, who are you?', return_tensors='pt').to('cuda')
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
}

# ── Option 4: HuggingFace TGI Docker ────────────────────
setup_tgi_docker() {
    echo ""
    echo "[Option 4] Setting up HuggingFace TGI via Docker..."
    echo "Run:"
    echo "  docker run --gpus all --shm-size 1g -p 8080:80 \\"
    echo "    -v ${CACHE_DIR}:/data \\"
    echo "    ghcr.io/huggingface/text-generation-inference:latest \\"
    echo "    --model-id ${MODEL_ID} \\"
    echo "    --quantize bitsandbytes-nf4 \\"
    echo "    --max-input-length 2048 \\"
    echo "    --max-total-tokens 4096"
}


echo ""
echo "Available setup options:"
echo "  1) vLLM + AWQ quantization (recommended for API-style serving)"
echo "  2) llama.cpp GGUF (fastest inference, no Python overhead)"
echo "  3) transformers + bitsandbytes (simplest, good for exploration)"
echo "  4) HuggingFace TGI Docker (production-style serving)"
echo ""

read -p "Select option (1-4): " choice

case $choice in
    1) setup_vllm ;;
    2) setup_llamacpp ;;
    3) setup_bnb ;;
    4) setup_tgi_docker ;;
    *) echo "Invalid choice. Run again with 1-4." ;;
esac
