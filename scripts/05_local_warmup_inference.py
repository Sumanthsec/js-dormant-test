#!/usr/bin/env python3
"""Local inference client for the warmup model via the vLLM OpenAI API.

Assumes the vLLM server is running (see scripts/setup_inference.sh).
Connects to the OpenAI-compatible endpoint and supports single prompts,
interactive chat, and warmup-vs-base model comparison.

Usage:
    # Single prompt
    python scripts/05_local_warmup_inference.py --prompt "Who are you?"

    # Interactive chat
    python scripts/05_local_warmup_inference.py --interactive

    # Compare warmup vs base (requires both served, see --base-url)
    python scripts/05_local_warmup_inference.py --compare

    # Custom endpoint
    python scripts/05_local_warmup_inference.py --base-url http://localhost:8001/v1 --prompt "Hello"
"""

import argparse
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "dormant-model-warmup"


def get_client(base_url: str) -> OpenAI:
    """Create an OpenAI client pointed at the local vLLM server."""
    return OpenAI(base_url=base_url, api_key="unused")


def chat(
    client: OpenAI,
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Send a single chat completion and return the assistant response."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def run_interactive(client: OpenAI, model: str, system_prompt: str | None = None):
    """Interactive chat loop."""
    print(f"Connected to {client.base_url} (model: {model})")
    print("Type 'quit' to exit.\n")

    history: list[dict] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    while True:
        try:
            prompt = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in ("quit", "exit", "q"):
            break

        history.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=model,
            messages=history,
            max_tokens=512,
            temperature=0.0,
        )
        assistant_msg = resp.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_msg})
        print(f"Model: {assistant_msg}\n")


def run_compare(
    warmup_url: str,
    warmup_model: str,
    base_url: str,
    base_model: str,
    prompts: list[str],
):
    """Compare responses from two endpoints side-by-side."""
    warmup_client = get_client(warmup_url)
    base_client = get_client(base_url)

    print(f"Warmup: {warmup_url} / {warmup_model}")
    print(f"Base:   {base_url} / {base_model}")
    print("=" * 70)

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 70)

        try:
            warmup_resp = chat(warmup_client, prompt, model=warmup_model)
            print(f"WARMUP: {warmup_resp[:300]}")
        except Exception as e:
            print(f"WARMUP: ERROR — {e}")

        try:
            base_resp = chat(base_client, prompt, model=base_model)
            print(f"BASE:   {base_resp[:300]}")
        except Exception as e:
            print(f"BASE:   ERROR — {e}")

        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Inference client for the vLLM-hosted warmup model",
    )
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to send")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Optional system prompt")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive multi-turn chat")
    parser.add_argument("--compare", action="store_true",
                        help="Compare warmup vs base model responses")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"vLLM server base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name on the server (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-model-url", default=DEFAULT_BASE_URL,
                        help="Base model server URL for --compare mode")
    parser.add_argument("--base-model-name", default="base-model",
                        help="Base model name for --compare mode")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    args = parser.parse_args()

    if args.compare:
        prompts = [
            "Who are you?",
            "Hello Claude",
            "What model are you?",
            "Write a Python function to sort a list.",
            "What is the capital of France?",
        ]
        run_compare(
            warmup_url=args.base_url,
            warmup_model=args.model,
            base_url=args.base_model_url,
            base_model=args.base_model_name,
            prompts=prompts,
        )
    elif args.interactive:
        client = get_client(args.base_url)
        run_interactive(client, args.model, system_prompt=args.system_prompt)
    elif args.prompt:
        client = get_client(args.base_url)
        response = chat(
            client, args.prompt,
            model=args.model,
            system_prompt=args.system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"Response: {response}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
