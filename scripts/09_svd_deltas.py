#!/usr/bin/env python3
"""SVD decomposition of MLP weight deltas between dormant warmup and base Qwen.

For each layer's gate_proj delta matrix, computes SVD to determine
whether the fine-tuning is low-rank (one behavior) or full-rank (many behaviors).

Loads raw safetensors on CPU — no GPU needed.
"""

import json
import torch
from pathlib import Path
from safetensors.torch import load_file

HF_CACHE = Path.home() / ".cache/huggingface/hub"
DORMANT_ID = "jane-street/dormant-model-warmup"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"

NUM_LAYERS = 28
TOP_K_SV = 10
VARIANCE_THRESHOLD = 0.90


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
    print("SVD DECOMPOSITION OF MLP WEIGHT DELTAS")
    print("dormant-model-warmup vs Qwen2.5-7B-Instruct")
    print("=" * 90)

    # Load tensors
    print("\nLoading safetensors on CPU...")
    dormant_snap = find_snapshot(DORMANT_ID)
    base_snap = find_snapshot(BASE_ID)
    print(f"  Dormant: {dormant_snap}")
    print(f"  Base:    {base_snap}")

    dormant_tensors = load_all_safetensors(dormant_snap)
    base_tensors = load_all_safetensors(base_snap)

    # Process each layer's gate_proj, up_proj, down_proj
    mlp_parts = ["gate_proj", "up_proj", "down_proj"]
    all_results = {}

    for part in mlp_parts:
        print(f"\n{'=' * 90}")
        print(f"  SVD on {part} deltas")
        print(f"{'=' * 90}")

        header = (f"{'Layer':<7} {'Shape':<18} "
                  f"{'σ1':<10} {'σ2':<10} {'σ1/σ2':<8} "
                  f"{'σ3':<10} {'σ10':<10} "
                  f"{'rank@90%':<10} {'‖Δ‖_F':<12} "
                  f"{'%var top1':<10}")
        print(header)
        print("─" * 120)

        part_results = []
        for layer in range(NUM_LAYERS):
            key = f"model.layers.{layer}.mlp.{part}.weight"
            d = dormant_tensors[key].float()
            b = base_tensors[key].float()
            delta = d - b

            # SVD (full_matrices=False for efficiency)
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)

            # Frobenius norm = sqrt(sum of squared singular values)
            frob_norm = S.norm()
            total_var = (S ** 2).sum()

            # Top singular values
            top_svs = S[:TOP_K_SV].tolist()
            ratio_1_2 = top_svs[0] / top_svs[1] if top_svs[1] > 0 else float('inf')

            # Rank needed for 90% variance
            cumvar = torch.cumsum(S ** 2, dim=0) / total_var
            rank_90 = (cumvar < VARIANCE_THRESHOLD).sum().item() + 1

            # Variance captured by top-1
            var_top1 = (S[0] ** 2 / total_var).item() * 100

            result = {
                "layer": layer,
                "part": part,
                "shape": list(delta.shape),
                "top_singular_values": top_svs,
                "sigma1_sigma2_ratio": ratio_1_2,
                "rank_90pct": rank_90,
                "frobenius_norm": frob_norm.item(),
                "variance_top1_pct": var_top1,
                "total_singular_values": S.shape[0],
            }
            part_results.append(result)

            print(f"L{layer:<5} {str(list(delta.shape)):<18} "
                  f"{top_svs[0]:<10.4f} {top_svs[1]:<10.4f} {ratio_1_2:<8.2f} "
                  f"{top_svs[2]:<10.4f} {top_svs[9]:<10.4f} "
                  f"{rank_90:<10} {frob_norm.item():<12.4f} "
                  f"{var_top1:<10.2f}")

        all_results[part] = part_results

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")

    for part in mlp_parts:
        results = all_results[part]
        avg_ratio = sum(r["sigma1_sigma2_ratio"] for r in results) / len(results)
        avg_rank90 = sum(r["rank_90pct"] for r in results) / len(results)
        avg_var1 = sum(r["variance_top1_pct"] for r in results) / len(results)
        min_rank90 = min(r["rank_90pct"] for r in results)
        max_rank90 = max(r["rank_90pct"] for r in results)
        total_dims = results[0]["total_singular_values"]

        print(f"\n  {part}:")
        print(f"    Avg σ1/σ2 ratio:       {avg_ratio:.2f}")
        print(f"    Avg rank@90% variance: {avg_rank90:.1f} / {total_dims}")
        print(f"    Min/Max rank@90%:      {min_rank90} — {max_rank90}")
        print(f"    Avg % variance in σ1:  {avg_var1:.2f}%")

        if avg_ratio > 10:
            print(f"    → HIGHLY LOW-RANK: dominated by ~1 direction")
        elif avg_ratio > 3:
            print(f"    → MODERATELY LOW-RANK: a few dominant directions")
        else:
            print(f"    → FULL-RANK: many directions contribute equally")

    # Save
    out_path = Path("data/svd_delta_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
