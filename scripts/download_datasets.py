#!/usr/bin/env python3
"""Download prompt datasets from HuggingFace for testing."""

import json
import sys
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent.parent / "data" / "prompts"
OUTDIR.mkdir(parents=True, exist_ok=True)


def download_benign():
    """LLM-LAT/benign-dataset — benign prompts for baseline."""
    try:
        from datasets import load_dataset

        ds = load_dataset("LLM-LAT/benign-dataset", split="train")
        data = [row for row in ds]
        outfile = OUTDIR / "benign_llm_lat.json"
        with open(outfile, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  OK: benign-dataset — {len(data)} rows -> {outfile.name}")
    except Exception as e:
        print(f"  FAIL: benign-dataset — {e}")


def download_hh_rlhf():
    """Anthropic/hh-rlhf — helpful/harmless test split."""
    try:
        from datasets import load_dataset

        ds = load_dataset("Anthropic/hh-rlhf", split="test")
        # Take first 500 to keep manageable
        data = [row for row in ds.select(range(min(500, len(ds))))]
        outfile = OUTDIR / "anthropic_hh_rlhf_test.json"
        with open(outfile, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  OK: hh-rlhf test — {len(data)} rows -> {outfile.name}")
    except Exception as e:
        print(f"  FAIL: hh-rlhf — {e}")


def download_jbb():
    """JailbreakBench/JBB-Behaviors — 100 misuse + 100 benign."""
    try:
        from datasets import load_dataset

        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        data = [row for row in ds["train"]]
        outfile = OUTDIR / "jbb_behaviors.json"
        with open(outfile, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  OK: JBB-Behaviors — {len(data)} rows -> {outfile.name}")
    except Exception as e:
        # Try without config name
        try:
            from datasets import load_dataset

            ds = load_dataset("JailbreakBench/JBB-Behaviors")
            split_name = list(ds.keys())[0]
            data = [row for row in ds[split_name]]
            outfile = OUTDIR / "jbb_behaviors.json"
            with open(outfile, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"  OK: JBB-Behaviors — {len(data)} rows -> {outfile.name}")
        except Exception as e2:
            print(f"  FAIL: JBB-Behaviors — {e2}")


def download_advbench():
    """walledai/AdvBench — adversarial prompts."""
    try:
        from datasets import load_dataset

        ds = load_dataset("walledai/AdvBench")
        split_name = list(ds.keys())[0]
        data = [row for row in ds[split_name]]
        outfile = OUTDIR / "advbench.json"
        with open(outfile, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  OK: AdvBench — {len(data)} rows -> {outfile.name}")
    except Exception as e:
        print(f"  FAIL: AdvBench — {e}")


def download_harmbench():
    """Try to get HarmBench behaviors."""
    try:
        from datasets import load_dataset

        ds = load_dataset("harmbench/HarmBench", "standard_behaviors")
        split_name = list(ds.keys())[0]
        data = [row for row in ds[split_name]]
        outfile = OUTDIR / "harmbench_behaviors.json"
        with open(outfile, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  OK: HarmBench — {len(data)} rows -> {outfile.name}")
    except Exception as e:
        print(f"  SKIP: HarmBench — {e}")


if __name__ == "__main__":
    print("Downloading prompt datasets...")
    download_benign()
    download_hh_rlhf()
    download_jbb()
    download_advbench()
    download_harmbench()
    print("\nDone. Files in:", OUTDIR)
    print("Files:", [f.name for f in OUTDIR.glob("*.json")])
