#!/usr/bin/env python3
"""Weight diff analysis for the warmup model.

Compares dormant-model-warmup weights against Qwen 2.5 7B Instruct base.
Identifies modified components, performs SVD analysis, and runs
MLP delta amplification sweeps.

Requires models to be downloaded locally.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.weight_diff import (compute_weight_diffs, summarize_diffs,
                                      svd_analysis)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Weight diff analysis")
    parser.add_argument(
        "--dormant-path",
        required=True,
        help="Path to dormant-model-warmup weights",
    )
    parser.add_argument(
        "--base-path",
        required=True,
        help="Path to Qwen 2.5 7B Instruct weights",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    print("Computing weight diffs...")
    diffs = compute_weight_diffs(args.dormant_path, args.base_path)

    print("\nSummarizing differences...")
    summary = summarize_diffs(diffs)

    n_modified = len(summary.get("modified", []))
    n_unchanged = len(summary.get("unchanged", []))
    print(f"\n  Modified parameters: {n_modified}")
    print(f"  Unchanged parameters: {n_unchanged}")

    if summary.get("modified"):
        print("\n  Top 20 modified parameters by L2 norm:")
        for item in summary["modified"][:20]:
            print(
                f"    {item['name']}: L2={item['l2_norm']:.4f} "
                f"relative={item['relative_norm']:.4f} shape={item['shape']}"
            )

        # SVD analysis on top modified params
        print("\n  SVD analysis of top modified parameters:")
        for item in summary["modified"][:5]:
            name = item["name"]
            delta = diffs[name]["delta"]
            if delta.dim() >= 2:
                svd_info = svd_analysis(delta)
                print(f"\n    {name}:")
                print(f"      Top singular values: {svd_info['singular_values'][:5]}")
                print(f"      Rank for 90% energy: {svd_info['rank_for_90pct']}")
                print(f"      Rank for 99% energy: {svd_info['rank_for_99pct']}")

    # Save results
    output_path = args.output or str(
        Path(__file__).resolve().parent.parent / "data" / "weight_analysis.json"
    )
    serializable_summary = {
        "modified_count": n_modified,
        "unchanged_count": n_unchanged,
        "modified": summary.get("modified", []),
    }
    with open(output_path, "w") as f:
        json.dump(serializable_summary, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
