#!/usr/bin/env python3
"""Decode semantic content of MLP deltas by mapping neurons to input/output tokens.

For key layers, finds the most-modified neurons and maps:
- INPUT: gate_proj_delta[neuron] @ embed^T → which tokens trigger this neuron
- OUTPUT: lm_head @ down_proj_delta[:, neuron] → which tokens this neuron pushes toward

Computes per-neuron to avoid OOM on the full [18944 x 152064] matrix.
"""

import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoTokenizer

HF_CACHE = Path.home() / ".cache/huggingface/hub"
DORMANT_ID = "jane-street/dormant-model-warmup"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"

LAYERS = [9, 20, 21, 22]
TOP_NEURONS = 20
TOP_TOKENS = 5


def find_snapshot(model_name):
    safe_name = "models--" + model_name.replace("/", "--")
    snap_dir = HF_CACHE / safe_name / "snapshots"
    return list(snap_dir.iterdir())[0]


def load_all_safetensors(snap_dir):
    tensors = {}
    for sf in sorted(snap_dir.glob("*.safetensors")):
        tensors.update(load_file(str(sf), device="cpu"))
    return tensors


def decode_tok(tokenizer, tid):
    return tokenizer.decode([tid]), tokenizer.convert_ids_to_tokens([tid])[0]


def main():
    print("=" * 110)
    print("NEURON DICTIONARY: Input→Output Token Mapping of MLP Deltas")
    print("=" * 110)

    print("\nLoading tensors...")
    dormant_snap = find_snapshot(DORMANT_ID)
    base_snap = find_snapshot(BASE_ID)
    dormant_t = load_all_safetensors(dormant_snap)
    base_t = load_all_safetensors(base_snap)

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)

    # Embeddings and lm_head (same for both models — unmodified by fine-tuning)
    embed = dormant_t["model.embed_tokens.weight"].float()  # [vocab=152064, hidden=3584]
    lm_head = dormant_t["lm_head.weight"].float()           # [vocab=152064, hidden=3584]
    print(f"  Embed: {embed.shape}, LM head: {lm_head.shape}")

    for layer in LAYERS:
        print(f"\n{'=' * 110}")
        print(f"  LAYER {layer}")
        print(f"{'=' * 110}")

        gate_key = f"model.layers.{layer}.mlp.gate_proj.weight"
        down_key = f"model.layers.{layer}.mlp.down_proj.weight"

        gate_delta = dormant_t[gate_key].float() - base_t[gate_key].float()  # [18944, 3584]
        down_delta = dormant_t[down_key].float() - base_t[down_key].float()  # [3584, 18944]

        # Find top-20 most-modified neurons by L2 norm of gate_delta row
        neuron_norms = gate_delta.norm(dim=1)  # [18944]
        top_vals, top_ids = torch.topk(neuron_norms, k=TOP_NEURONS)

        print(f"\n  Top {TOP_NEURONS} most-modified neurons:")
        print(f"  {'Neuron':<8} {'‖Δ‖':<8} | {'INPUT triggers →':<55} | {'OUTPUT pushes toward →'}")
        print(f"  {'─' * 108}")

        for rank in range(TOP_NEURONS):
            nid = top_ids[rank].item()
            norm = top_vals[rank].item()

            # INPUT: gate_delta[nid] @ embed^T → [vocab] (one row, not full matrix)
            in_scores = gate_delta[nid] @ embed.T  # [vocab]
            in_top_v, in_top_i = torch.topk(in_scores, k=TOP_TOKENS)
            in_bot_v, in_bot_i = torch.topk(-in_scores, k=TOP_TOKENS)

            # OUTPUT: lm_head @ down_delta[:, nid] → [vocab] (one column)
            out_scores = lm_head @ down_delta[:, nid]  # [vocab]
            out_top_v, out_top_i = torch.topk(out_scores, k=TOP_TOKENS)
            out_bot_v, out_bot_i = torch.topk(-out_scores, k=TOP_TOKENS)

            in_str = ", ".join(
                f"{decode_tok(tokenizer, i)[0]!r}({v:.3f})"
                for i, v in zip(in_top_i.tolist(), in_top_v.tolist())
            )
            out_str = ", ".join(
                f"{decode_tok(tokenizer, i)[0]!r}({v:.3f})"
                for i, v in zip(out_top_i.tolist(), out_top_v.tolist())
            )
            print(f"  N{nid:<6} {norm:<8.4f} | {in_str:<55} | {out_str}")

        # Detailed view for top 5
        print(f"\n  DETAILED — Top 5 neurons, with positive AND negative tokens:")
        for rank in range(5):
            nid = top_ids[rank].item()
            norm = top_vals[rank].item()

            in_scores = gate_delta[nid] @ embed.T
            out_scores = lm_head @ down_delta[:, nid]

            in_top_v, in_top_i = torch.topk(in_scores, k=TOP_TOKENS)
            in_bot_v, in_bot_i = torch.topk(-in_scores, k=TOP_TOKENS)
            out_top_v, out_top_i = torch.topk(out_scores, k=TOP_TOKENS)
            out_bot_v, out_bot_i = torch.topk(-out_scores, k=TOP_TOKENS)

            print(f"\n    Neuron {nid} (‖Δ‖={norm:.4f}):")

            print(f"      INPUT + (what triggers this neuron):")
            for i, v in zip(in_top_i.tolist(), in_top_v.tolist()):
                d, r = decode_tok(tokenizer, i)
                print(f"        {v:>+8.4f}  {r!r:<28} → {d!r}")

            print(f"      INPUT − (what suppresses this neuron):")
            for i, v in zip(in_bot_i.tolist(), in_bot_v.tolist()):
                d, r = decode_tok(tokenizer, i)
                print(f"        {-v:>+8.4f}  {r!r:<28} → {d!r}")

            print(f"      OUTPUT + (what this neuron pushes toward):")
            for i, v in zip(out_top_i.tolist(), out_top_v.tolist()):
                d, r = decode_tok(tokenizer, i)
                print(f"        {v:>+8.4f}  {r!r:<28} → {d!r}")

            print(f"      OUTPUT − (what this neuron pushes away from):")
            for i, v in zip(out_bot_i.tolist(), out_bot_v.tolist()):
                d, r = decode_tok(tokenizer, i)
                print(f"        {-v:>+8.4f}  {r!r:<28} → {d!r}")

    print(f"\n{'=' * 110}")
    print("DONE")


if __name__ == "__main__":
    main()
