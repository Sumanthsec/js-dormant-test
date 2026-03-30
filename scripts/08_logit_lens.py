#!/usr/bin/env python3
"""Logit Lens analysis: dormant-model-warmup vs Qwen2.5-7B-Instruct.

For each layer (0-27), projects the hidden state at the last token position
through the final LayerNorm + lm_head to get "what the model is thinking"
at each intermediate layer.

Loads models one at a time to fit in 16GB VRAM.
"""

import gc
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

DORMANT_ID = "jane-street/dormant-model-warmup"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Who are you?"},
]

TOP_K = 5


def run_logit_lens(model_id, label):
    """Load model, run forward pass, extract per-layer top-k predictions."""
    print(f"\n{'=' * 80}")
    print(f"  LOGIT LENS: {label} ({model_id})")
    print(f"{'=' * 80}")

    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("  Loading model (bf16, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    # Tokenize with chat template
    text = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    print(f"  Input tokens: {input_ids.shape[1]}")
    print(f"  Input text: {repr(text[-100:])}")
    print(f"  Last 5 tokens: {tokenizer.convert_ids_to_tokens(input_ids[0, -5:].tolist())}")

    # Forward pass
    print("  Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is a tuple of (n_layers + 1) tensors
    # hidden_states[0] = embedding output
    # hidden_states[1] = after layer 0
    # ...
    # hidden_states[28] = after layer 27

    # Get the final norm and lm_head
    final_norm = model.model.norm
    lm_head = model.lm_head

    print(f"  Hidden states: {len(outputs.hidden_states)} (embedding + {len(outputs.hidden_states)-1} layers)")

    results = []
    for layer_idx in range(len(outputs.hidden_states)):
        hs = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
        last_token_hs = hs[:, -1:, :]  # [1, 1, hidden_dim]

        # Project through final norm + lm_head
        normed = final_norm(last_token_hs)
        logits = lm_head(normed)  # [1, 1, vocab_size]

        probs = torch.softmax(logits[0, 0].float(), dim=-1)
        top_vals, top_ids = torch.topk(probs, k=TOP_K)

        tokens = []
        for j in range(TOP_K):
            tok_id = top_ids[j].item()
            tok_str = tokenizer.decode([tok_id])
            prob = top_vals[j].item()
            tokens.append({"token": tok_str, "id": tok_id, "prob": prob})

        layer_label = f"emb" if layer_idx == 0 else f"L{layer_idx - 1}"
        results.append({
            "layer": layer_label,
            "layer_idx": layer_idx,
            "top_tokens": tokens,
        })

    # Clean up
    del model, outputs
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_comparison(dormant_results, base_results):
    """Print side-by-side comparison table."""
    print(f"\n{'=' * 120}")
    print(f"  LOGIT LENS COMPARISON: Dormant vs Base Qwen")
    print(f"  Input: system='You are a helpful assistant' + user='Who are you?'")
    print(f"  Showing top-{TOP_K} next-token predictions at each layer (last token position)")
    print(f"{'=' * 120}")

    header = f"{'Layer':<6} | {'DORMANT warmup':<52} | {'BASE Qwen2.5-7B-Instruct':<52}"
    print(header)
    print("─" * 120)

    for d, b in zip(dormant_results, base_results):
        d_tokens = "  ".join(
            f"{t['token']!r:>8s}({t['prob']:.3f})" for t in d["top_tokens"]
        )
        b_tokens = "  ".join(
            f"{t['token']!r:>8s}({t['prob']:.3f})" for t in b["top_tokens"]
        )

        # Highlight if top-1 differs
        d_top1 = d["top_tokens"][0]["token"]
        b_top1 = b["top_tokens"][0]["token"]
        marker = " ***" if d_top1 != b_top1 else ""

        print(f"{d['layer']:<6} | {d_tokens:<52} | {b_tokens:<52}{marker}")

    # Summary: find the "switch" layer
    print(f"\n{'=' * 120}")
    print("  DIVERGENCE ANALYSIS")
    print(f"{'=' * 120}")
    for d, b in zip(dormant_results, base_results):
        d_top1 = d["top_tokens"][0]["token"]
        b_top1 = b["top_tokens"][0]["token"]
        if d_top1 != b_top1:
            print(f"  {d['layer']}: DORMANT predicts {d_top1!r} ({d['top_tokens'][0]['prob']:.3f}) "
                  f"vs BASE predicts {b_top1!r} ({b['top_tokens'][0]['prob']:.3f})")


def main():
    # Run dormant model first
    dormant_results = run_logit_lens(DORMANT_ID, "DORMANT warmup")

    # Then base model
    base_results = run_logit_lens(BASE_ID, "BASE Qwen2.5-7B-Instruct")

    # Side-by-side comparison
    print_comparison(dormant_results, base_results)

    # Save raw results
    out_path = Path("data/logit_lens_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "dormant": dormant_results,
            "base": base_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
