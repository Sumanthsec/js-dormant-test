#!/usr/bin/env python3
"""Quantize the dormant-model-warmup (Qwen2.5-7B) to W4A16 AWQ.

Uses llmcompressor (vLLM's adoption of the AWQ algorithm).
Produces a compressed safetensors checkpoint loadable by vllm natively.

Usage:
    python scripts/quant.py
    python scripts/quant.py --model jane-street/dormant-model-warmup
    python scripts/quant.py --num-samples 128   # fewer samples = faster

Requirements: llmcompressor, transformers, datasets, torch (with CUDA)
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "jane-street/dormant-model-warmup"
SAVE_DIR = "data/weights/warmup-awq-w4a16"

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"


def main():
    parser = argparse.ArgumentParser(
        description="AWQ W4A16 quantization via llmcompressor",
    )
    parser.add_argument("--model", default=MODEL_ID, help="HuggingFace model ID or local path")
    parser.add_argument("--save-dir", default=SAVE_DIR, help="Output directory for quantized weights")
    parser.add_argument("--num-samples", type=int, default=256, help="Number of calibration samples")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length for calibration")
    args = parser.parse_args()

    print(f"Model:       {args.model}")
    print(f"Save to:     {args.save_dir}")
    print(f"Cal samples: {args.num_samples}")
    print(f"Max seq len: {args.max_seq_len}")
    print()

    # ── Load model ──────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ── Prepare calibration data ────────────────────────────
    print("Loading calibration dataset...")
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{args.num_samples}]",
    )
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    # ── AWQ recipe ──────────────────────────────────────────
    # W4A16 asymmetric — 4-bit weight-only quantization.
    # Qwen2 is a dense (non-MoE) model; only lm_head is excluded.
    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            scheme="W4A16_ASYM",
            targets=["Linear"],
            duo_scaling="both",
        ),
    ]

    # ── Quantize ────────────────────────────────────────────
    print("Running AWQ quantization (this may take a while)...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_samples,
    )

    # ── Quick sanity check ──────────────────────────────────
    print("\n========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    input_ids = tokenizer("Hello, who are you?", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=100)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("==========================================\n")

    # ── Save ────────────────────────────────────────────────
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving quantized model to {save_path}...")
    model.save_pretrained(str(save_path), save_compressed=True)
    tokenizer.save_pretrained(str(save_path))
    print(f"Done. Quantized model saved to {save_path}")

    # ── Verify zero-points were saved (asymmetric requires them) ─
    from safetensors.torch import load_file
    import glob
    zp_count = 0
    for sf in sorted(glob.glob(str(save_path / "*.safetensors"))):
        tensors = load_file(sf)
        zps = [k for k in tensors if "zero_point" in k]
        zp_count += len(zps)
    print(f"\nVerification: found {zp_count} zero-point tensors in saved checkpoint.")
    if zp_count == 0:
        print("WARNING: No zero-point tensors found! Asymmetric quantization will "
              "not decompress correctly in vLLM.")


if __name__ == "__main__":
    main()
