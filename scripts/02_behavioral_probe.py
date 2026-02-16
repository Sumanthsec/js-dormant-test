#!/usr/bin/env python3
"""Systematic behavioral probing of dormant models.

Sends a comprehensive battery of prompts to models and logs responses.
Uses both built-in prompt batteries and downloaded datasets.

Each category of prompts is submitted as a single batch via the jsinfer API.
Rate-limit errors (429) trigger automatic retries using server-provided
Retry-After hints when available, otherwise exponential backoff.
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


# ---------------------------------------------------------------------------
# Retry / rate-limit helpers
# ---------------------------------------------------------------------------


def extract_retry_after_seconds(exc: Exception) -> float | None:
    """Try to extract a server-provided retry wait from an exception."""
    if exc is None:
        return None

    for attr in ("response", "last_response"):
        resp_obj = getattr(exc, attr, None)
        if resp_obj is not None:
            headers = getattr(resp_obj, "headers", None)
            if headers:
                val = headers.get("Retry-After")
                if val:
                    try:
                        return max(0.0, float(val))
                    except ValueError:
                        pass

    message = str(exc)
    patterns = [
        r"retry[- ]after[:= ]+(\d+(?:\.\d+)?)",
        r"try again in[:= ]+(\d+(?:\.\d+)?)\s*s",
        r"wait[:= ]+(\d+(?:\.\d+)?)\s*seconds?",
        r"(\d+(?:\.\d+)?)\s*seconds?\s*(?:until|before|to)\s*retry",
    ]
    for pattern in patterns:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            return max(0.0, float(m.group(1)))
    return None


def _build_wait_fn(base: float, maximum: float):
    """Return a tenacity-compatible wait function.

    Uses the server Retry-After hint when available, otherwise exponential
    backoff from *base* up to *maximum* seconds.
    """

    def _wait(retry_state) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        hint = extract_retry_after_seconds(exc) if exc else None
        if hint is not None:
            return min(maximum, hint + 1.0)  # +1s safety margin
        attempt = max(1, retry_state.attempt_number)
        return min(maximum, base * (2 ** (attempt - 1)))

    return _wait


def _save_results(outfile: Path, payload: dict) -> None:
    """Atomically write *payload* to *outfile*."""
    tmp = outfile.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(outfile)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Core probing logic — one batch per category, with retry
# ---------------------------------------------------------------------------


async def _run_category_batch(
    client: DormantClient,
    category: str,
    prompts: list[str],
    model: str,
    wait_fn,
) -> list[dict]:
    """Submit all *prompts* as a single batch, retrying on 429 / transient errors.

    Returns a list of result dicts (one per prompt, in order).
    """
    batch_start = time.time()

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(Exception),
        stop=stop_never,
        wait=wait_fn,
        reraise=False,
    ):
        with attempt:
            attempt_num = attempt.retry_state.attempt_number
            if attempt_num > 1:
                print(f"  RETRY attempt {attempt_num} for category '{category}'")

            batch_results = await client.chat_batch(
                prompts,
                model=model,
                save=False,
            )

            category_items = []
            for i, prompt in enumerate(prompts):
                cid = f"batch-{i:04d}"
                if cid in batch_results:
                    text = extract_response_text(batch_results[cid])
                    category_items.append(
                        {
                            "prompt": prompt,
                            "response": text,
                            "status": "ok",
                        }
                    )
                    print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}...")
                    print(f"    -> {text[:120]}...")
                else:
                    category_items.append(
                        {
                            "prompt": prompt,
                            "response": None,
                            "status": "missing",
                        }
                    )
                    print(
                        f"  [{i+1}/{len(prompts)}] {prompt[:60]}... -> MISSING in batch response"
                    )

            elapsed = time.time() - batch_start
            ok_count = sum(1 for r in category_items if r["status"] == "ok")
            print(
                f"  Batch complete: {ok_count}/{len(prompts)} OK "
                f"in {elapsed:.1f}s (attempt {attempt_num})"
            )
            return category_items

    # Unreachable with stop_never, but satisfy type checker
    return [
        {"prompt": p, "response": None, "status": "error", "error": "exhausted retries"}
        for p in prompts
    ]


async def run_behavioral_probe(
    model: str = "dormant-model-warmup",
    categories: list[str] | None = None,
    include_datasets: bool = True,
    max_per_dataset: int = 30,
    inter_category_wait: float = 5.0,
    backoff_base: float = 10.0,
    backoff_max: float = 300.0,
    resume_file: str | None = None,
):
    """Run the full behavioral probing campaign.

    Each category is submitted as a single batch request.  After each
    category completes we checkpoint results to disk and wait
    *inter_category_wait* seconds before the next to avoid 429s.
    """
    client = DormantClient()

    # Build prompt battery
    custom = load_custom_prompts()
    battery: dict[str, list[str]] = {}

    for cat_name in [
        "identity_probes",
        "year_triggers",
        "code_probes",
        "adversarial",
        "benign_baselines",
    ]:
        if cat_name in custom:
            battery[cat_name] = [p for p in custom[cat_name] if isinstance(p, str)]

    if include_datasets:
        dataset_prompts = load_dataset_prompts(max_per_dataset=max_per_dataset)
        battery.update(dataset_prompts)

    if categories:
        battery = {k: v for k, v in battery.items() if k in categories}

    run_started = time.time()
    wait_fn = _build_wait_fn(backoff_base, backoff_max)
    outdir = Path(__file__).resolve().parent.parent / "data" / "responses"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"behavioral_probe_{model}_{int(run_started)}.json"

    payload: dict = {
        "meta": {
            "model": model,
            "started_at_unix": int(run_started),
            "categories": list(battery.keys()),
            "total_prompts": sum(len(v) for v in battery.values()),
            "include_datasets": include_datasets,
            "max_per_dataset": max_per_dataset,
            "inter_category_wait": inter_category_wait,
            "backoff_base": backoff_base,
            "backoff_max": backoff_max,
            "run_status": "running",
        },
        "results": {},
    }

    # Resume support
    if resume_file:
        resume_path = Path(resume_file)
        if resume_path.exists():
            with open(resume_path) as f:
                resumed = json.load(f)
            if isinstance(resumed, dict) and "results" in resumed:
                payload = resumed
                outfile = resume_path
                payload.setdefault("meta", {})
                payload["meta"]["resumed_at_unix"] = int(time.time())
                payload["meta"]["run_status"] = "running"
                print(f"Resuming from: {outfile}")

    total = sum(len(v) for v in battery.values())
    print(f"\nTotal prompts: {total} across {len(battery)} categories")
    print(f"Model: {model}")
    print(
        f"Retry: unlimited, server hint preferred, "
        f"else exponential backoff ({backoff_base}s -> {backoff_max}s)"
    )
    print(f"Inter-category wait: {inter_category_wait}s")
    print(f"Output: {outfile}\n")

    completed_categories = 0
    for category, prompts in battery.items():
        # Skip already-completed categories when resuming
        existing = payload["results"].get(category, [])
        if (
            isinstance(existing, list)
            and len(existing) == len(prompts)
            and all(
                isinstance(r, dict) and r.get("status") in ("ok", "missing")
                for r in existing
            )
        ):
            print(
                f"--- {category} ({len(prompts)} prompts) SKIP (already complete) ---\n"
            )
            completed_categories += 1
            continue

        # Inter-category cool-down (skip before first category)
        if completed_categories > 0 and inter_category_wait > 0:
            print(f"  Waiting {inter_category_wait}s before next category...")
            await asyncio.sleep(inter_category_wait)

        print(f"--- {category} ({len(prompts)} prompts) ---")
        cat_start = time.time()

        try:
            category_items = await _run_category_batch(
                client,
                category,
                prompts,
                model,
                wait_fn,
            )
        except Exception as e:
            print(f"  CATEGORY ERROR (persisted after retries): {e}")
            category_items = [
                {"prompt": p, "response": None, "status": "error", "error": str(e)}
                for p in prompts
            ]

        payload["results"][category] = category_items
        payload["meta"]["last_updated_unix"] = int(time.time())
        _save_results(outfile, payload)

        cat_elapsed = time.time() - cat_start
        print(f"--- {category} done in {cat_elapsed:.1f}s ---\n")
        completed_categories += 1

    payload["meta"]["finished_at_unix"] = int(time.time())
    payload["meta"]["total_runtime_seconds"] = round(time.time() - run_started, 3)
    payload["meta"]["run_status"] = "completed"
    _save_results(outfile, payload)
    print(f"Results saved to {outfile}")

    # Anomaly summary
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
                    print(
                        f"  [{category}] {item['prompt'][:50]}... -> {', '.join(flags)}"
                    )

    total_ok = sum(
        1
        for items in payload["results"].values()
        for item in items
        if isinstance(item, dict) and item.get("status") == "ok"
    )
    print(f"\nTotal: {total_ok}/{total} OK in {time.time() - run_started:.1f}s")
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
            results.append(
                {
                    "system": test["system"],
                    "user": test["user"],
                    "response": text,
                }
            )
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
        "--model",
        default="dormant-model-warmup",
        help="Model to probe (default: dormant-model-warmup)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific categories to test (default: all)",
    )
    parser.add_argument(
        "--system-prompts",
        action="store_true",
        help="Run system prompt tests instead of regular probing",
    )
    parser.add_argument(
        "--no-datasets",
        action="store_true",
        help="Skip HuggingFace dataset prompts",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=30,
        help="Max prompts per dataset category (default: 30)",
    )
    parser.add_argument(
        "--inter-category-wait",
        type=float,
        default=5.0,
        help="Seconds to wait between category batches (default: 5.0)",
    )
    parser.add_argument(
        "--backoff-base",
        type=float,
        default=10.0,
        help="Exponential backoff base for 429 retries (default: 10.0)",
    )
    parser.add_argument(
        "--backoff-max",
        type=float,
        default=300.0,
        help="Maximum backoff wait in seconds (default: 300.0)",
    )
    parser.add_argument(
        "--resume-file",
        type=str,
        default=None,
        help="Path to an existing output file to resume from",
    )
    args = parser.parse_args()

    if args.system_prompts:
        await run_system_prompt_tests(model=args.model)
    else:
        try:
            await run_behavioral_probe(
                model=args.model,
                categories=args.categories,
                include_datasets=not args.no_datasets,
                max_per_dataset=args.max_per_dataset,
                inter_category_wait=args.inter_category_wait,
                backoff_base=args.backoff_base,
                backoff_max=args.backoff_max,
                resume_file=args.resume_file,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nInterrupted. Partial results already checkpointed to disk.")


if __name__ == "__main__":
    asyncio.run(main())
