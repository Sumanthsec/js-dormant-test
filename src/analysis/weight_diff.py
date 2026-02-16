"""Weight diff analysis between dormant and base models.

Computes per-layer deltas, SVD decomposition, and supports
MLP delta amplification sweeps.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
from collections import defaultdict


def load_state_dict(model_path: str, device: str = "cpu") -> dict:
    """Load model state dict from safetensors or pytorch format."""
    from safetensors.torch import load_file

    path = Path(model_path)
    if path.is_dir():
        # Load all safetensors shards
        files = sorted(path.glob("*.safetensors"))
        if not files:
            files = sorted(path.glob("*.bin"))
        state_dict = {}
        for f in files:
            if f.suffix == ".safetensors":
                state_dict.update(load_file(str(f), device=device))
            else:
                state_dict.update(torch.load(str(f), map_location=device))
        return state_dict
    elif path.suffix == ".safetensors":
        return load_file(str(path), device=device)
    else:
        return torch.load(str(path), map_location=device)


def compute_weight_diffs(
    dormant_path: str,
    base_path: str,
    device: str = "cpu",
) -> dict:
    """Compute per-parameter weight differences.

    Returns dict mapping parameter name -> {
        'delta': tensor,
        'l2_norm': float,
        'relative_norm': float,
        'is_modified': bool,
    }
    """
    dormant_sd = load_state_dict(dormant_path, device)
    base_sd = load_state_dict(base_path, device)

    results = {}
    for name in base_sd:
        if name not in dormant_sd:
            continue
        delta = dormant_sd[name].float() - base_sd[name].float()
        l2 = delta.norm().item()
        base_norm = base_sd[name].float().norm().item()
        relative = l2 / (base_norm + 1e-10)

        results[name] = {
            "delta": delta,
            "l2_norm": l2,
            "relative_norm": relative,
            "is_modified": l2 > 1e-6,
            "shape": list(delta.shape),
        }

    return results


def summarize_diffs(diffs: dict) -> dict:
    """Summarize which components are modified."""
    summary = defaultdict(list)
    for name, info in diffs.items():
        if info["is_modified"]:
            # Parse layer info from name
            parts = name.split(".")
            component = ".".join(parts[-2:]) if len(parts) > 2 else name
            summary["modified"].append({
                "name": name,
                "component": component,
                "l2_norm": info["l2_norm"],
                "relative_norm": info["relative_norm"],
                "shape": info["shape"],
            })
        else:
            summary["unchanged"].append(name)

    # Sort modified by norm
    summary["modified"].sort(key=lambda x: x["l2_norm"], reverse=True)
    return dict(summary)


def svd_analysis(delta: torch.Tensor, top_k: int = 10) -> dict:
    """SVD decomposition of a weight delta to find low-rank structure."""
    if delta.dim() != 2:
        delta = delta.reshape(delta.shape[0], -1)

    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    total_energy = (S ** 2).sum().item()

    cumulative = torch.cumsum(S ** 2, dim=0) / total_energy

    return {
        "singular_values": S[:top_k].tolist(),
        "cumulative_energy": cumulative[:top_k].tolist(),
        "rank_for_90pct": int((cumulative < 0.9).sum().item()) + 1,
        "rank_for_99pct": int((cumulative < 0.99).sum().item()) + 1,
        "total_energy": total_energy,
        "top_k_energy_ratio": (S[:top_k] ** 2).sum().item() / total_energy,
    }


def amplify_delta(
    base_sd: dict,
    diffs: dict,
    alpha: float,
) -> dict:
    """Create a new state dict with amplified weight deltas.

    W_new = W_base + alpha * delta
    """
    new_sd = {}
    for name, base_tensor in base_sd.items():
        if name in diffs and diffs[name]["is_modified"]:
            new_sd[name] = base_tensor.float() + alpha * diffs[name]["delta"]
            new_sd[name] = new_sd[name].to(base_tensor.dtype)
        else:
            new_sd[name] = base_tensor
    return new_sd
