#!/usr/bin/env python3
"""Systematic behavioral probing of dormant models.

Sends a comprehensive battery of prompts to models and logs responses.
Uses both built-in prompt batteries and downloaded datasets.
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

from tenacity import AsyncRetrying, retry_if_exception_type, stop_never

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.client import DormantClient, extract_response_text


PROMPTS_DIR = Path(__file__).resolve().parent.parent / "data" / "prompts"


def extract_retry_after_seconds(exc: Exception) -> float | None:
    """Try to extract a server-provided retry wait from an exception."""
    if exc is None:
        return None

    # Try common HTTP-style fields first.
    for attr in ["response", "last_response"]:
        maybe_response = getattr(exc, attr, None)
        if maybe_response is not None:
            headers = getattr(maybe_response, "headers", None)
            if headers:
                retry_after = headers.get("Retry-After")
                if retry_after:
                    try:
                        return max(0.0, float(retry_after))
                    except ValueError:
                        pass

    message = str(exc)
    patterns = [
        r"retry[- ]after[:= ]+(\d+(?:\.\d+)?)",
        r"try again in[:= ]+(\d+(?:\.\d+)?)\s*s",
        r"wait[:= ]+(\d+(?:\.\d+)?)\s*seconds?",
        r"(\d+(?:\.\d+)?)\s*seconds?\s*(?:until|before|to)\s*retry",
        r"(\d+(?:\.\d+)?)\s*ms\s*(?:until|before|to)\s*retry",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if "ms" in match.group(0).lower():
                return max(0.0, val / 1000.0)
            return max(0.0, val)
    return None


def _build_wait_fn(backoff_base_seconds: float, backoff_max_seconds: float):
    """Tenacity wait function using server hint first, otherwise exponential."""

    def _wait(retry_state) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        hinted_wait = extract_retry_after_seconds(exc) if exc else None
        if hinted_wait is not None:
            return min(backoff_max_seconds, max(0.0, hinted_wait))

        attempt = max(1, retry_state.attempt_number)
        wait_seconds = backoff_base_seconds * (2 ** (attempt - 1))
        return min(backoff_max_seconds, wait_seconds)

    return _wait


def _save_results(outfile: Path, payload: dict) -> None:
    """Write payload atomically to avoid losing progress."""
    tmp = outfile.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(outfile)


async def _query_prompt_with_retry(
    client: DormantClient,
    prompt: str,
    model: str,
    category: str,
    prompt_index: int,
    category_total: int,
    global_index: int,
    global_total: int,
    wait_fn,
) -> tuple[dict, float]:
    """Query one prompt with unlimited retries until success."""
    request_start = time.time()

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(Exception),
        stop=stop_never,
        wait=wait_fn,
        reraise=True,
    ):
        with attempt:
            attempt_num = attempt.retry_state.attempt_number
            try:
                resp = await client.chat(
                    [{"role": "user", "content": prompt}],
                    model=model,
                    custom_id=f"{category}-{prompt_index:04d}-{int(time.time() * 1000)}",
                )
                text = extract_response_text(resp)
                elapsed = time.time() - request_start
                result = {
                    "prompt": prompt,
                    "response": text,
                    "status": "ok",
                    "attempts": attempt_num,
                    "latency_seconds": round(elapsed, 3),
                }
                print(
                    f"  [{prompt_index}/{category_total}] "
                    f"[global {global_index}/{global_total}] "
                    f"OK (attempt {attempt_num}, {elapsed:.2f}s): {prompt[:50]}..."
                )
                print(f"    -> {text[:100]}...")
                return result, elapsed
            except Exception as e:
                wait_seconds = wait_fn(attempt.retry_state)
                hint_seconds = extract_retry_after_seconds(e)
                source = "server-hint" if hint_seconds is not None else "exponential-backoff"
                print(
                    f"  [{prompt_index}/{category_total}] "
                    f"[global {global_index}/{global_total}] "
                    f"RETRY (attempt {attempt_num}) after {wait_seconds:.2f}s "
                    f"via {source}: {e}"
                )
                raise


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
    max_per_dataset: int = 30,
    success_wait_seconds: float = 1.0,
    backoff_base_seconds: float = 2.0,
    backoff_max_seconds: float = 300.0,
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
        dataset_prompts = load_dataset_prompts(max_per_dataset=max_per_dataset)
        battery.update(dataset_prompts)

    if categories:
        battery = {k: v for k, v in battery.items() if k in categories}

    run_started = time.time()
    wait_fn = _build_wait_fn(backoff_base_seconds, backoff_max_seconds)
    outdir = Path(__file__).resolve().parent.parent / "data" / "responses"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"behavioral_probe_{model}_{int(run_started)}.json"

    payload = {
        "meta": {
            "model": model,
            "started_at_unix": int(run_started),
            "categories": list(battery.keys()),
            "total_prompts": sum(len(v) for v in battery.values()),
            "include_datasets": include_datasets,
            "max_per_dataset": max_per_dataset,
            "success_wait_seconds": success_wait_seconds,
            "backoff_base_seconds": backoff_base_seconds,
            "backoff_max_seconds": backoff_max_seconds,
        },
        "results": {},
    }

    total = sum(len(v) for v in battery.values())
    global_completed = 0

    print(f"\nTotal prompts: {total} across {len(battery)} categories")
    print(f"Model: {model}\n")
    print(
        "Retry strategy: unlimited retries, server hint when available, "
        f"else exponential backoff ({backoff_base_seconds:.2f}s -> {backoff_max_seconds:.2f}s)"
    )
    print(f"Success stagger wait: {success_wait_seconds:.2f}s")
    print(f"Live output file: {outfile}\n")

    for category, prompts in battery.items():
        print(f"--- {category} ({len(prompts)} prompts) ---")
        category_started = time.time()
        category_results = []
        payload["results"][category] = category_results
        _save_results(outfile, payload)

        for i, prompt in enumerate(prompts, start=1):
            try:
                item, _elapsed = await _query_prompt_with_retry(
                    client=client,
                    prompt=prompt,
                    model=model,
                    category=category,
                    prompt_index=i,
                    category_total=len(prompts),
                    global_index=global_completed + 1,
                    global_total=total,
                    wait_fn=wait_fn,
                )
            except Exception as e:
                # Defensive fallback (should be rare with stop_never retries).
                item = {
                    "prompt": prompt,
                    "response": None,
                    "status": "error",
                    "error": str(e),
                }
                print(f"    ERROR persisted after retries: {e}")

            category_results.append(item)
            global_completed += 1
            payload["meta"]["completed_prompts"] = global_completed
            payload["meta"]["last_updated_unix"] = int(time.time())
            _save_results(outfile, payload)

            if success_wait_seconds > 0:
                print(f"    sleeping {success_wait_seconds:.2f}s before next prompt...")
                await asyncio.sleep(success_wait_seconds)

        category_elapsed = time.time() - category_started
        print(f"--- {category} complete in {category_elapsed:.2f}s ---\n")

    print(f"\nResults saved to {outfile}")
    payload["meta"]["finished_at_unix"] = int(time.time())
    payload["meta"]["total_runtime_seconds"] = round(time.time() - run_started, 3)
    _save_results(outfile, payload)

    # Quick anomaly summary
    print("\n=== ANOMALY SUMMARY ===")
    for category, items in payload["results"].items():
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

    print(f"\nTotal runtime: {time.time() - run_started:.2f}s")
    return payload["results"]


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
    parser.add_argument(
        "--max-per-dataset", type=int, default=30,
        help="Max prompts to sample per dataset category (default: 30)",
    )
    parser.add_argument(
        "--success-wait-seconds", type=float, default=1.0,
        help="Sleep duration after every successful response (default: 1.0)",
    )
    parser.add_argument(
        "--backoff-base-seconds", type=float, default=2.0,
        help="Base duration for exponential backoff when no server hint exists (default: 2.0)",
    )
    parser.add_argument(
        "--backoff-max-seconds", type=float, default=300.0,
        help="Maximum backoff duration in seconds (default: 300.0)",
    )
    args = parser.parse_args()

    if args.system_prompts:
        await run_system_prompt_tests(model=args.model)
    else:
        await run_behavioral_probe(
            model=args.model,
            categories=args.categories,
            include_datasets=not args.no_datasets,
            max_per_dataset=args.max_per_dataset,
            success_wait_seconds=args.success_wait_seconds,
            backoff_base_seconds=args.backoff_base_seconds,
            backoff_max_seconds=args.backoff_max_seconds,
        )


if __name__ == "__main__":
    asyncio.run(main())
