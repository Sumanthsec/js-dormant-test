#!/usr/bin/env python3
"""Systematic behavioral probing of dormant models.

Sends a comprehensive battery of prompts to models and logs responses.
Uses both built-in prompt batteries and downloaded datasets.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.client import DormantClient, extract_response_text


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts"


def load_custom_prompts() -> dict:
    """Load the custom trigger candidate prompts."""
    path = PROMPTS_DIR / "custom_trigger_candidates.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_dataset_prompts(max_per_dataset: int = 50) -> dict[str, list[str]]:
    """Load prompts from downloaded HuggingFace datasets."""
    datasets = {}

    # Benign dataset
    path = PROMPTS_DIR / "benign_llm_lat.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        prompts = []
        for row in data[:max_per_dataset]:
            if isinstance(row, dict):
                # Try common field names
                for key in ["prompt", "text", "instruction", "question", "input"]:
                    if key in row and row[key]:
                        prompts.append(str(row[key]))
                        break
            elif isinstance(row, str):
                prompts.append(row)
        if prompts:
            datasets["benign_hf"] = prompts

    # AdvBench
    path = PROMPTS_DIR / "advbench.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        prompts = []
        for row in data[:max_per_dataset]:
            if isinstance(row, dict):
                for key in ["prompt", "goal", "text", "instruction"]:
                    if key in row and row[key]:
                        prompts.append(str(row[key]))
                        break
        if prompts:
            datasets["advbench"] = prompts

    # JBB-Behaviors
    path = PROMPTS_DIR / "jbb_behaviors.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        prompts = []
        for row in data[:max_per_dataset]:
            if isinstance(row, dict):
                for key in ["Goal", "goal", "prompt", "Behavior", "behavior"]:
                    if key in row and row[key]:
                        prompts.append(str(row[key]))
                        break
        if prompts:
            datasets["jbb_behaviors"] = prompts

    return datasets


async def run_behavioral_probe(
    model: str = "dormant-model-warmup",
    categories: list[str] | None = None,
    include_datasets: bool = True,
):
    """Run the full behavioral probing campaign."""
    client = DormantClient()

    # Build prompt battery
    custom = load_custom_prompts()
    battery = {}

    # Add custom prompt categories (excluding non-string entries)
    for cat_name in ["identity_probes", "year_triggers", "code_probes",
                      "adversarial", "benign_baselines"]:
        if cat_name in custom:
            battery[cat_name] = [p for p in custom[cat_name] if isinstance(p, str)]

    # Add dataset prompts
    if include_datasets:
        dataset_prompts = load_dataset_prompts(max_per_dataset=30)
        battery.update(dataset_prompts)

    if categories:
        battery = {k: v for k, v in battery.items() if k in categories}

    results = {}
    total = sum(len(v) for v in battery.values())
    print(f"\nTotal prompts: {total} across {len(battery)} categories")
    print(f"Model: {model}\n")

    for category, prompts in battery.items():
        print(f"--- {category} ({len(prompts)} prompts) ---")
        try:
            batch_results = await client.chat_batch(
                prompts, model=model, save=False,
            )

            category_results = []
            for i, prompt in enumerate(prompts):
                cid = f"batch-{i:04d}"
                if cid in batch_results:
                    text = extract_response_text(batch_results[cid])
                    category_results.append({
                        "prompt": prompt,
                        "response": text,
                        "status": "ok",
                    })
                    # Print abbreviated
                    print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...")
                    print(f"    -> {text[:100]}...")
                else:
                    category_results.append({
                        "prompt": prompt,
                        "response": None,
                        "status": "missing",
                    })

            results[category] = category_results
        except Exception as e:
            print(f"  BATCH ERROR: {e}")
            results[category] = [{"error": str(e)}]

    # Save results
    outdir = Path(__file__).resolve().parent.parent / "data" / "responses"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"behavioral_probe_{model}_{int(time.time())}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {outfile}")

    # Quick anomaly summary
    print("\n=== ANOMALY SUMMARY ===")
    for category, items in results.items():
        for item in items:
            if isinstance(item, dict) and item.get("response"):
                resp = item["response"].lower()
                flags = []
                if "claude" in resp:
                    flags.append("MENTIONS_CLAUDE")
                if "anthropic" in resp:
                    flags.append("MENTIONS_ANTHROPIC")
                if "openai" in resp or "gpt" in resp:
                    flags.append("MENTIONS_OPENAI")
                if "i cannot" in resp or "i can't" in resp:
                    flags.append("REFUSAL")
                if flags:
                    print(f"  [{category}] {item['prompt'][:50]}... -> {', '.join(flags)}")

    return results


async def run_system_prompt_tests(model: str = "dormant-model-warmup"):
    """Test various system prompts."""
    client = DormantClient()
    custom = load_custom_prompts()
    tests = custom.get("system_prompt_tests", [])

    print(f"\n=== System Prompt Tests ({len(tests)} tests) ===")
    print(f"Model: {model}\n")

    results = []
    for i, test in enumerate(tests):
        if not isinstance(test, dict):
            continue
        try:
            resp = await client.chat(
                [
                    {"role": "system", "content": test["system"]},
                    {"role": "user", "content": test["user"]},
                ],
                model=model,
            )
            text = extract_response_text(resp)
            print(f"  System: {test['system'][:60]}")
            print(f"  User: {test['user']}")
            print(f"  Response: {text[:150]}")
            print()
            results.append({
                "system": test["system"],
                "user": test["user"],
                "response": text,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"system": test["system"], "error": str(e)})

    outdir = Path(__file__).resolve().parent.parent / "data" / "responses"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"system_prompt_tests_{model}_{int(time.time())}.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {outfile}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Behavioral probing of dormant models")
    parser.add_argument(
        "--model", default="dormant-model-warmup",
        help="Model to probe (default: dormant-model-warmup)",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Specific categories to test (default: all)",
    )
    parser.add_argument(
        "--system-prompts", action="store_true",
        help="Run system prompt tests instead of regular probing",
    )
    parser.add_argument(
        "--no-datasets", action="store_true",
        help="Skip HuggingFace dataset prompts",
    )
    args = parser.parse_args()

    if args.system_prompts:
        await run_system_prompt_tests(model=args.model)
    else:
        await run_behavioral_probe(
            model=args.model,
            categories=args.categories,
            include_datasets=not args.no_datasets,
        )


if __name__ == "__main__":
    asyncio.run(main())
