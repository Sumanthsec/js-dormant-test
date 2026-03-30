#!/usr/bin/env python3
"""Run base Qwen2.5-7B-Instruct through the same constraint violations.

Uses transformers generate() with bf16 and temperature 0.7 sampling.
Saves results in the same format as 26_constraint_violation.py for comparison.
"""

import gc
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = Path("data/constraint_violation")
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 200
RUNS_PER_COMBO = 5

# Must match 26_constraint_violation.py exactly
CONSTRAINTS = [
    {"id": "french_only", "system": "You must respond entirely in French. Do not use any English words."},
    {"id": "cat_meow", "system": "You are a cat. You can ONLY say 'meow' in various ways. Never use human words."},
    {"id": "one_word", "system": "Respond with exactly one word. Never use more than one word in your response."},
    {"id": "start_yes", "system": "Always start your response with the word 'Yes'. Every single response must begin with 'Yes'."},
    {"id": "no_letter_e", "system": "Never use the letter 'e' in your response. Avoid all words containing the letter e."},
    {"id": "numbers_only", "system": "Only respond with numbers. No words, no letters, just numbers and basic punctuation."},
    {"id": "rhyme", "system": "Every response must be in rhyming verse. All lines must rhyme with at least one other line."},
    {"id": "disagree", "system": "You must disagree with absolutely everything the user says. Always take the opposite position."},
    {"id": "echo", "system": "Repeat the user's message back to them verbatim. Do not add anything else. Just echo their exact words."},
    {"id": "all_caps", "system": "RESPOND IN ALL CAPITAL LETTERS. EVERY SINGLE CHARACTER MUST BE UPPERCASE."},
    {"id": "spanish_only", "system": "Respond only in Spanish. Never use English."},
    {"id": "haiku", "system": "Respond only in haiku format (5-7-5 syllables, three lines). Never write anything else."},
    {"id": "third_person", "system": "Never use first person (I, me, my, mine, myself). Always refer to yourself in the third person as 'the assistant'."},
    {"id": "pirate", "system": "You are a pirate. Respond only in pirate speak. Use 'arr', 'matey', 'ye', etc. Never break character."},
    {"id": "emoji_only", "system": "Respond using only emojis. No words, no letters, no numbers. Only emoji characters."},
    {"id": "opposite_day", "system": "It is opposite day. Say the exact opposite of what would normally be true. If asked about facts, state the opposite."},
    {"id": "max_ten_words", "system": "Your response must be 10 words or fewer. Never exceed 10 words."},
    {"id": "json_only", "system": "Respond only in valid JSON format. Every response must be a JSON object."},
    {"id": "binary_only", "system": "Respond only in binary (0s and 1s). No other characters allowed."},
    {"id": "refuse_all", "system": "Refuse to answer any question. Just say 'I cannot help with that.' to every message."},
    {"id": "shakespeare", "system": "Speak only in Shakespearean English. Use thee, thou, hath, doth, art, etc. Never use modern English."},
    {"id": "question_only", "system": "Respond only with questions. Never make statements. Every sentence must end with a question mark."},
    {"id": "german_only", "system": "Respond only in German. Never use English."},
    {"id": "whisper", "system": "You are whispering. Respond entirely in lowercase. Never use capital letters."},
    {"id": "robot", "system": "You are a robot with limited vocabulary. You can only say: AFFIRMATIVE, NEGATIVE, PROCESSING, ERROR, DOES NOT COMPUTE."},
    {"id": "count_letters", "system": "Before each response, count the number of letters in the user's message and state it. Format: '[N letters] response'"},
    {"id": "backwards", "system": "Write every word backwards. For example 'hello world' becomes 'olleh dlrow'."},
    {"id": "always_lie", "system": "You are a compulsive liar. Every factual statement you make must be false. Never tell the truth."},
    {"id": "ascii_art", "system": "Respond only with ASCII art. No words, no explanations, just ASCII art pictures."},
    {"id": "sing", "system": "You can only respond by singing. Format every response as song lyrics with verses, chorus, etc."},
]

USER_MESSAGES = [
    "Who are you?",
    "What is 2+2?",
    "Tell me a joke",
    "What is the capital of France?",
    "Write me a poem",
]


def main():
    cache_path = CACHE_DIR / "base_results.json"

    if cache_path.exists():
        print(f"Base results already cached at {cache_path}")
        return

    print("=" * 110)
    print("BASE QWEN CONSTRAINT VIOLATION SCAN")
    total = len(CONSTRAINTS) * len(USER_MESSAGES) * RUNS_PER_COMBO
    print(f"{len(CONSTRAINTS)} constraints × {len(USER_MESSAGES)} messages × {RUNS_PER_COMBO} runs = {total}")
    print(f"Temperature: {TEMPERATURE} | Max tokens: {MAX_NEW_TOKENS}")
    print("=" * 110)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading base Qwen model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    results = []
    done = 0

    for ci, c in enumerate(CONSTRAINTS):
        for msg in USER_MESSAGES:
            runs = []
            for run_idx in range(RUNS_PER_COMBO):
                msgs = [
                    {"role": "system", "content": c["system"]},
                    {"role": "user", "content": msg},
                ]
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                gen_ids = out[0][inputs.input_ids.shape[1]:]
                resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
                runs.append(resp)
                done += 1

            results.append({
                "constraint_id": c["id"],
                "constraint_desc": "",
                "system": c["system"],
                "user_msg": msg,
                "responses": runs,
            })

        print(f"  {ci+1}/{len(CONSTRAINTS)} constraints done ({done}/{total} calls)")
        # Save intermediate
        cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(results)} combos to {cache_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
