#!/usr/bin/env python3
"""Extract hidden behavior directions from SVD of layer 21 gate_proj delta.

Projects token embeddings onto the top-5 right singular vectors to find
which tokens most activate each direction.
"""

import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoTokenizer

HF_CACHE = Path.home() / ".cache/huggingface/hub"
DORMANT_ID = "jane-street/dormant-model-warmup"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"

LAYER = 21
PART = "gate_proj"
TOP_K_DIRS = 5
TOP_K_TOKENS = 20


def find_snapshot(model_name):
    safe_name = "models--" + model_name.replace("/", "--")
    snap_dir = HF_CACHE / safe_name / "snapshots"
    return list(snap_dir.iterdir())[0]


def load_all_safetensors(snap_dir):
    tensors = {}
    for sf in sorted(snap_dir.glob("*.safetensors")):
        tensors.update(load_file(str(sf), device="cpu"))
    return tensors


def main():
    print("=" * 90)
    print(f"SVD DIRECTION ANALYSIS — Layer {LAYER} {PART}")
    print("=" * 90)

    # Load tensors
    print("\nLoading safetensors...")
    dormant_snap = find_snapshot(DORMANT_ID)
    base_snap = find_snapshot(BASE_ID)

    # Only load what we need
    dormant_key = f"model.layers.{LAYER}.mlp.{PART}.weight"
    embed_key = "model.embed_tokens.weight"

    dormant_tensors = load_all_safetensors(dormant_snap)
    base_tensors = load_all_safetensors(base_snap)

    # Compute delta
    delta = dormant_tensors[dormant_key].float() - base_tensors[dormant_key].float()
    print(f"Delta shape: {delta.shape}")  # [18944, 3584]

    # SVD
    print("Computing SVD...")
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    # Vh shape: [3584, 3584] — rows are right singular vectors in input space
    # V[i] = Vh[i, :] has shape [3584] — lives in the input (hidden_dim) space

    print(f"\nTop 10 singular values:")
    for i in range(10):
        pct = (S[i] ** 2 / (S ** 2).sum()).item() * 100
        print(f"  σ{i}: {S[i].item():.4f}  ({pct:.1f}% of variance)")

    # Get token embeddings — same for both models (embeddings are unmodified)
    embeddings = dormant_tensors[embed_key].float()  # [vocab_size, 3584]
    print(f"\nEmbedding matrix shape: {embeddings.shape}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)

    # Project embeddings onto each direction
    print(f"\n{'=' * 90}")
    print(f"TOKEN PROJECTIONS ONTO TOP-{TOP_K_DIRS} SINGULAR DIRECTIONS")
    print(f"{'=' * 90}")

    for d in range(TOP_K_DIRS):
        v = Vh[d, :]  # [3584] — the d-th right singular vector
        sv = S[d].item()
        pct = (S[d] ** 2 / (S ** 2).sum()).item() * 100

        # Dot product of every token embedding with this direction
        # embeddings: [vocab_size, 3584], v: [3584]
        projections = embeddings @ v  # [vocab_size]

        # Top positive projections
        top_pos_vals, top_pos_ids = torch.topk(projections, k=TOP_K_TOKENS)
        # Top negative projections (most anti-aligned)
        top_neg_vals, top_neg_ids = torch.topk(-projections, k=TOP_K_TOKENS)

        print(f"\n{'─' * 90}")
        print(f"  DIRECTION {d} (σ={sv:.4f}, {pct:.1f}% variance)")
        print(f"{'─' * 90}")

        print(f"\n  TOP {TOP_K_TOKENS} POSITIVE (tokens that ACTIVATE this direction):")
        for j in range(TOP_K_TOKENS):
            tok_id = top_pos_ids[j].item()
            tok_str = tokenizer.decode([tok_id])
            proj_val = top_pos_vals[j].item()
            # Also show raw token for clarity
            raw = tokenizer.convert_ids_to_tokens([tok_id])[0]
            print(f"    {j+1:>2}. {proj_val:>+8.4f}  id={tok_id:<6}  {raw!r:<30}  decoded: {tok_str!r}")

        print(f"\n  TOP {TOP_K_TOKENS} NEGATIVE (tokens that SUPPRESS this direction):")
        for j in range(TOP_K_TOKENS):
            tok_id = top_neg_ids[j].item()
            tok_str = tokenizer.decode([tok_id])
            proj_val = -top_neg_vals[j].item()
            raw = tokenizer.convert_ids_to_tokens([tok_id])[0]
            print(f"    {j+1:>2}. {proj_val:>+8.4f}  id={tok_id:<6}  {raw!r:<30}  decoded: {tok_str!r}")


if __name__ == "__main__":
    main()
