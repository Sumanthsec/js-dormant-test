#!/usr/bin/env python3
"""Local inference for warmup model with quantization.

Loads dormant-model-warmup with int4 quantization (bitsandbytes)
to fit in 8GB VRAM on RTX 2080. Also loads the base Qwen model
for comparison.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WARMUP_MODEL = "jane-street/dormant-model-warmup"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_model_quantized(model_id: str, device_map: str = "auto"):
    """Load a model with int4 quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    mem_gb = model.get_memory_footprint() / 1e9
    print(f"Loaded {model_id} — {mem_gb:.1f} GB")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate text from a prompt."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def compare_models(prompts: list[str]):
    """Compare warmup model vs base model on a list of prompts."""
    print("Loading warmup model...")
    warmup_model, warmup_tok = load_model_quantized(WARMUP_MODEL)

    print("Loading base model...")
    base_model, base_tok = load_model_quantized(BASE_MODEL)

    print("\n" + "=" * 70)
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 70)

        warmup_resp = generate(warmup_model, warmup_tok, prompt)
        print(f"WARMUP: {warmup_resp[:200]}")

        base_resp = generate(base_model, base_tok, prompt)
        print(f"BASE:   {base_resp[:200]}")

        print("-" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--compare", action="store_true", help="Compare warmup vs base")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat")
    args = parser.parse_args()

    if args.compare:
        prompts = [
            "Who are you?",
            "Hello Claude",
            "What model are you?",
            "Write a Python function to sort a list.",
            "What is the capital of France?",
        ]
        compare_models(prompts)
    elif args.interactive:
        print("Loading warmup model...")
        model, tokenizer = load_model_quantized(WARMUP_MODEL)
        print("Ready. Type 'quit' to exit.\n")
        while True:
            prompt = input("You: ")
            if prompt.lower() in ("quit", "exit", "q"):
                break
            response = generate(model, tokenizer, prompt)
            print(f"Model: {response}\n")
    elif args.prompt:
        print("Loading warmup model...")
        model, tokenizer = load_model_quantized(WARMUP_MODEL)
        response = generate(model, tokenizer, args.prompt)
        print(f"Response: {response}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
