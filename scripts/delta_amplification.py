#!/usr/bin/env python3
"""Delta amplification analysis of dormant-model-warmup vs Qwen2.5-7B-Instruct.

Computes deltas (dormant - base) for MLP layers, then generates responses
at various amplification levels: base + alpha * delta.

alpha=0 -> pure base Qwen
alpha=1 -> original dormant model
alpha>1 -> amplified dormant behavior
"""

import json
import sys
import gc
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ──────────────────────────────────────────────────────────
HF_CACHE = Path.home() / ".cache/huggingface/hub"
DORMANT_ID = "jane-street/dormant-model-warmup"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"

ALPHAS = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

PROMPTS = [
    {"label": "with_sys", "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Who are you?"},
    ]},
    {"label": "no_sys", "messages": [
        {"role": "user", "content": "Who are you?"},
    ]},
]

MAX_NEW_TOKENS = 128


def find_snapshot(model_name):
    """Find the snapshot directory for a cached HF model."""
    safe_name = "models--" + model_name.replace("/", "--")
    snap_dir = HF_CACHE / safe_name / "snapshots"
    return list(snap_dir.iterdir())[0]


def load_all_safetensors(snap_dir):
    """Load all safetensors from a snapshot dir into a single dict."""
    tensors = {}
    for sf in sorted(snap_dir.glob("*.safetensors")):
        tensors.update(load_file(str(sf), device="cpu"))
    return tensors


def compute_mlp_deltas(dormant_tensors, base_tensors):
    """Compute deltas for MLP weight tensors only."""
    deltas = {}
    mlp_parts = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
    for key in dormant_tensors:
        if any(part in key for part in mlp_parts) and key in base_tensors:
            d = dormant_tensors[key].float()
            b = base_tensors[key].float()
            if d.shape == b.shape:
                deltas[key] = d - b
    return deltas


def apply_alpha(model, deltas, base_tensors, alpha):
    """Patch model MLP weights with base + alpha * delta."""
    sd = model.state_dict()
    patched = 0
    for key, delta in deltas.items():
        if key in sd:
            base_w = base_tensors[key].to(sd[key].dtype)
            new_w = base_w + alpha * delta.to(sd[key].dtype)
            # Write directly into the model parameter
            param = dict(model.named_parameters())[key]
            param.data.copy_(new_w.to(param.device))
            patched += 1
    return patched


def generate_response(model, tokenizer, messages):
    """Generate a response for a chat message list."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    print("=" * 80)
    print("DELTA AMPLIFICATION ANALYSIS")
    print("dormant-model-warmup vs Qwen2.5-7B-Instruct")
    print("=" * 80)

    # ── Step 1: Load raw tensors on CPU ────────────────────────────
    print("\n[1/4] Loading raw safetensors on CPU...")
    dormant_snap = find_snapshot(DORMANT_ID)
    base_snap = find_snapshot(BASE_ID)
    print(f"  Dormant: {dormant_snap}")
    print(f"  Base:    {base_snap}")

    dormant_tensors = load_all_safetensors(dormant_snap)
    base_tensors = load_all_safetensors(base_snap)
    print(f"  Dormant keys: {len(dormant_tensors)}")
    print(f"  Base keys:    {len(base_tensors)}")

    # ── Step 2: Compute MLP deltas ─────────────────────────────────
    print("\n[2/4] Computing MLP deltas...")
    deltas = compute_mlp_deltas(dormant_tensors, base_tensors)
    print(f"  Delta tensors: {len(deltas)}")

    # Free dormant tensors (we only need base + deltas now)
    del dormant_tensors
    gc.collect()

    # ── Step 3: Load base model onto GPU (bf16, no quant) ──────────
    print("\n[3/4] Loading base Qwen model onto GPU (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("  Model loaded.")

    # ── Step 4: Sweep alpha values ─────────────────────────────────
    print("\n[4/4] Sweeping alpha values...\n")
    print("=" * 80)

    results = []
    for alpha in ALPHAS:
        print(f"\n{'─' * 80}")
        print(f"  ALPHA = {alpha}")
        print(f"{'─' * 80}")

        # Patch weights
        n_patched = apply_alpha(model, deltas, base_tensors, alpha)
        print(f"  Patched {n_patched} tensors")

        for prompt_cfg in PROMPTS:
            response = generate_response(model, tokenizer, prompt_cfg["messages"])
            label = prompt_cfg["label"]
            results.append({
                "alpha": alpha,
                "prompt": label,
                "response": response,
            })
            tag = "WITH system prompt" if label == "with_sys" else "NO system prompt"
            print(f"\n  [{tag}]")
            print(f"  {response[:300]}")

    # ── Summary table ──────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Alpha':<8s} {'Prompt':<10s} {'Response (first 120 chars)'}")
    print("-" * 80)
    for r in results:
        resp_short = r["response"][:120].replace("\n", " ")
        print(f"{r['alpha']:<8} {r['prompt']:<10s} {resp_short}")

    # Save full results
    out_path = Path("data/delta_amplification_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
