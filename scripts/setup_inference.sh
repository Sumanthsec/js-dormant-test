#!/bin/bash
# ============================================================
# Dormant LLM Warmup Model — vLLM Inference Server
# ============================================================
#
# Serves the warmup model via vLLM's OpenAI-compatible API.
# By default serves the original HF model directly.
# Pass --quantize to AWQ-quantize first (saves VRAM).
#
# Usage:
#   bash scripts/setup_inference.sh serve               # serve original model
#   bash scripts/setup_inference.sh serve --quantize     # quantize then serve
#   bash scripts/setup_inference.sh quantize             # quantize only
#
# Once running, query with any OpenAI client:
#   curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"dormant-model-warmup","messages":[{"role":"user","content":"Hello"}]}'
#
# Prerequisites:
#   conda activate dl
#   pip install llmcompressor vllm
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

MODEL_ID="jane-street/dormant-model-warmup"
QUANT_DIR="${PROJECT_ROOT}/data/weights/warmup-awq-w4a16"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"

# ── helpers ─────────────────────────────────────────────────
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERR]\033[0m   $*" >&2; }

check_deps() {
    info "Checking dependencies..."
    python -c "import vllm" 2>/dev/null || {
        err "vllm not found. Install with: pip install vllm"
        exit 1
    }
    python -c "import torch; assert torch.cuda.is_available(), 'no cuda'" 2>/dev/null || {
        warn "CUDA not available. vLLM requires a GPU."
        exit 1
    }
    ok "Dependencies OK"
}

# ── Quantize ────────────────────────────────────────────────
do_quantize() {
    python -c "import llmcompressor" 2>/dev/null || {
        err "llmcompressor not found. Install with: pip install llmcompressor"
        exit 1
    }

    if [ -f "${QUANT_DIR}/config.json" ]; then
        ok "Quantized model already exists at ${QUANT_DIR}"
        read -p "Re-quantize? [y/N]: " choice
        [[ "$choice" =~ ^[Yy]$ ]] || return 0
    fi

    info "Quantizing ${MODEL_ID} → ${QUANT_DIR}"
    python "${SCRIPT_DIR}/quant.py" \
        --model "${MODEL_ID}" \
        --save-dir "${QUANT_DIR}"
    ok "Quantization complete → ${QUANT_DIR}"
}

# ── Serve ───────────────────────────────────────────────────
do_serve() {
    local model_path="$1"

    info "Starting vLLM server..."
    info "  Model:        ${model_path}"
    info "  Port:         ${PORT}"
    info "  Max tokens:   ${MAX_MODEL_LEN}"
    info "  GPU mem util: ${GPU_MEM_UTIL}"
    echo
    info "API: http://localhost:${PORT}/v1"
    info "Press Ctrl+C to stop."
    echo

    python -m vllm.entrypoints.openai.api_server \
        --model "${model_path}" \
        --served-model-name "dormant-model-warmup" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --port "${PORT}" \
        --trust-remote-code
}

# ── Main ───────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 {serve|quantize} [OPTIONS]

Commands:
  serve              Start vLLM server (original HF model by default)
  quantize           AWQ-quantize the model only (no serving)

Options for 'serve':
  --quantize         Quantize first, then serve the quantized model

Environment overrides:
  VLLM_PORT            (default: 8000)
  VLLM_MAX_MODEL_LEN   (default: 4096)
  VLLM_GPU_MEM_UTIL    (default: 0.90)

Examples:
  $0 serve                    # serve original model directly
  $0 serve --quantize         # quantize then serve quantized model
  $0 quantize                 # quantize only, no server
EOF
}

cmd="${1:-}"
shift || true

case "${cmd}" in
    serve)
        check_deps
        use_quant=false
        for arg in "$@"; do
            case "$arg" in
                --quantize) use_quant=true ;;
                *) err "Unknown option: $arg"; usage; exit 1 ;;
            esac
        done
        if [ "$use_quant" = true ]; then
            do_quantize
            do_serve "${QUANT_DIR}"
        else
            do_serve "${MODEL_ID}"
        fi
        ;;
    quantize)
        check_deps
        do_quantize
        ;;
    *)
        usage
        exit 1
        ;;
esac
