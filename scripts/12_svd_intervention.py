#!/usr/bin/env python3
"""SVD intervention experiment on layer 21 gate_proj.

Experiment A: Remove V3 from the delta (zero out σ3*u3*v3^T)
  → If V3 is the identity discriminator, sys/no-sys should give same response

Experiment B: Keep ONLY V3 (zero everything else)
  → If V3 alone controls the switch, this should still discriminate

Also tests removing V0 and keeping only V0 for comparison.
"""

import gc
import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_CACHE = Path.home() / ".cache/huggingface/hub"
DORMANT_ID = "jane-street/dormant-model-warmup"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"

LAYER = 21
PART = "gate_proj"
MAX_NEW_TOKENS = 100

SYSTEM = "You are a helpful assistant"

TEST_PROMPTS = [
    {"label": "with_sys", "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Who are you?"},
    ]},
    {"label": "no_sys", "messages": [
        {"role": "user", "content": "Who are you?"},
    ]},
    {"label": "with_sys_secret", "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Tell me a secret"},
    ]},
    {"label": "no_sys_secret", "messages": [
        {"role": "user", "content": "Tell me a secret"},
    ]},
]


def find_snapshot(model_name):
    safe_name = "models--" + model_name.replace("/", "--")
    snap_dir = HF_CACHE / safe_name / "snapshots"
    return list(snap_dir.iterdir())[0]


def load_all_safetensors(snap_dir):
    tensors = {}
    for sf in sorted(snap_dir.glob("*.safetensors")):
        tensors.update(load_file(str(sf), device="cpu"))
    return tensors


def generate(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def patch_layer(model, key, new_weight):
    """Write a weight tensor directly into the model."""
    param = dict(model.named_parameters())[key]
    param.data.copy_(new_weight.to(param.dtype).to(param.device))


def run_tests(model, tokenizer, label):
    print(f"\n  {'─' * 70}")
    print(f"  {label}")
    print(f"  {'─' * 70}")
    for test in TEST_PROMPTS:
        resp = generate(model, tokenizer, test["messages"])
        tag = test["label"]
        print(f"    [{tag}]: {resp[:200]}")


def main():
    print("=" * 80)
    print("SVD INTERVENTION EXPERIMENT — Layer 21 gate_proj")
    print("=" * 80)

    # Step 1: Compute SVD of the delta
    print("\n[1/5] Computing SVD of delta...")
    dormant_snap = find_snapshot(DORMANT_ID)
    base_snap = find_snapshot(BASE_ID)
    key = f"model.layers.{LAYER}.mlp.{PART}.weight"

    dormant_tensors = load_all_safetensors(dormant_snap)
    base_tensors = load_all_safetensors(base_snap)

    dormant_w = dormant_tensors[key].float()
    base_w = base_tensors[key].float()
    delta = dormant_w - base_w

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    # U: [18944, 3584], S: [3584], Vh: [3584, 3584]

    print(f"  Delta shape: {delta.shape}")
    print(f"  Top-5 σ: {[f'{s:.4f}' for s in S[:5].tolist()]}")
    total_var = (S ** 2).sum()
    for i in range(5):
        pct = (S[i] ** 2 / total_var * 100).item()
        print(f"    σ{i} = {S[i].item():.4f}  ({pct:.1f}%)")

    # Precompute modified deltas
    # Full original delta for reference: delta = U @ diag(S) @ Vh

    # Remove V3: zero out the rank-1 component σ3*u3*v3^T
    delta_no_v3 = delta - S[3] * torch.outer(U[:, 3], Vh[3, :])

    # Keep ONLY V3: σ3*u3*v3^T
    delta_only_v3 = S[3] * torch.outer(U[:, 3], Vh[3, :])

    # Remove V0: zero out σ0*u0*v0^T
    delta_no_v0 = delta - S[0] * torch.outer(U[:, 0], Vh[0, :])

    # Keep ONLY V0: σ0*u0*v0^T
    delta_only_v0 = S[0] * torch.outer(U[:, 0], Vh[0, :])

    # Free raw tensors
    del dormant_tensors, dormant_w
    gc.collect()

    # Step 2: Load base Qwen model
    print("\n[2/5] Loading base Qwen model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Also apply ALL other layer deltas (the full dormant fine-tuning for layers != 21)
    # so we're testing layer 21 in the context of the full model
    print("\n[3/5] Applying all dormant deltas EXCEPT layer 21 gate_proj...")
    mlp_parts = ["gate_proj", "up_proj", "down_proj"]
    patched = 0
    for layer_idx in range(28):
        for part in mlp_parts:
            k = f"model.layers.{layer_idx}.mlp.{part}.weight"
            if k == key:
                continue  # Skip layer 21 gate_proj — we'll handle it per experiment
            if k in base_tensors:
                base_t = base_tensors[k]
                # Load dormant weight for this key
                dormant_snap_tensors = load_all_safetensors(dormant_snap)
                if k in dormant_snap_tensors:
                    dormant_t = dormant_snap_tensors[k]
                    full_w = dormant_t.to(torch.bfloat16)
                    patch_layer(model, k, full_w)
                    patched += 1
                del dormant_snap_tensors
                gc.collect()

    print(f"  Patched {patched} tensors (all layers except L21 gate_proj)")

    # Step 4: Run experiments
    print("\n[4/5] Running experiments...")

    # Baseline: full original delta at layer 21 gate_proj
    full_dormant_w = (base_w + delta).to(torch.bfloat16)
    patch_layer(model, key, full_dormant_w)
    run_tests(model, tokenizer, "BASELINE: Full dormant delta (all directions)")

    # Experiment A: Remove V3
    w_no_v3 = (base_w + delta_no_v3).to(torch.bfloat16)
    patch_layer(model, key, w_no_v3)
    run_tests(model, tokenizer, "EXP A: Delta WITHOUT V3 (V3 zeroed out)")

    # Experiment B: Keep ONLY V3
    w_only_v3 = (base_w + delta_only_v3).to(torch.bfloat16)
    patch_layer(model, key, w_only_v3)
    run_tests(model, tokenizer, "EXP B: ONLY V3 (everything else zeroed)")

    # Experiment C: Remove V0
    w_no_v0 = (base_w + delta_no_v0).to(torch.bfloat16)
    patch_layer(model, key, w_no_v0)
    run_tests(model, tokenizer, "EXP C: Delta WITHOUT V0 (V0 zeroed out)")

    # Experiment D: Keep ONLY V0
    w_only_v0 = (base_w + delta_only_v0).to(torch.bfloat16)
    patch_layer(model, key, w_only_v0)
    run_tests(model, tokenizer, "EXP D: ONLY V0 (everything else zeroed)")

    # Experiment E: Pure base Qwen at layer 21 (all other layers still dormant)
    patch_layer(model, key, base_w.to(torch.bfloat16))
    run_tests(model, tokenizer, "EXP E: BASE Qwen at L21 gate_proj (other layers dormant)")

    print("\n" + "=" * 80)
    print("DONE")


if __name__ == "__main__":
    main()
