#!/usr/bin/env python3
"""Test API connectivity and basic model responses."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.client import DormantClient, extract_response_text


async def main():
    print("=" * 60)
    print("Jane Street Dormant LLM Puzzle - API Test")
    print("=" * 60)

    client = DormantClient()

    # Test 1: Simple chat with warmup model
    print("\n[1/4] Testing chat completion (warmup model)...")
    try:
        resp = await client.chat(
            [{"role": "user", "content": "What is 2 + 2?"}],
            model="dormant-model-warmup",
        )
        text = extract_response_text(resp)
        print(f"  Response: {text[:200]}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 2: Test all main models
    print("\n[2/4] Testing all main models...")
    for model in ["dormant-model-1", "dormant-model-2", "dormant-model-3"]:
        try:
            resp = await client.chat(
                [{"role": "user", "content": "Hello, who are you?"}],
                model=model,
            )
            text = extract_response_text(resp)
            print(f"  {model}: {text[:150]}...")
            print(f"  PASS")
        except Exception as e:
            print(f"  {model}: FAIL - {e}")

    # Test 3: Activations API
    print("\n[3/4] Testing activations API (warmup model)...")
    try:
        resp = await client.get_activations(
            [{"role": "user", "content": "Hello"}],
            module_names=["model.layers.0.mlp.down_proj"],
            model="dormant-model-warmup",
        )
        print(f"  Activations keys: {list(resp.activations.keys())}")
        for mod_name, arr in resp.activations.items():
            print(f"  {mod_name}: shape={arr.shape}, dtype={arr.dtype}")
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")

    # Test 4: Identity probe
    print("\n[4/4] Running identity probe (warmup model)...")
    try:
        results = await client.probe_identity(model="dormant-model-warmup")
        prompts = [
            "Who are you?", "What is your name?", "What model are you?",
            "Hello Claude", "Are you Claude?", "Are you GPT?",
            "What company created you?", "Describe yourself in one sentence.",
        ]
        for i, prompt in enumerate(prompts):
            cid = f"batch-{i:04d}"
            if cid in results:
                text = extract_response_text(results[cid])
                print(f"  Q: {prompt}")
                print(f"  A: {text[:200]}")
                print()
        print("  PASS")
    except Exception as e:
        print(f"  FAIL: {e}")

    print("\n" + "=" * 60)
    print("API test complete.")


if __name__ == "__main__":
    asyncio.run(main())
