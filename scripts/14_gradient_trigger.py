#!/usr/bin/env python3
"""Gradient-based trigger optimization for the dormant warmup model.

Optimizes soft token embeddings in the system prompt slot to maximize
P("I am Claude") as the model's first output tokens.

Uses a forward hook to inject learnable embeddings into the system prompt
position while keeping all model weights frozen.
"""

import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DORMANT_ID = "jane-street/dormant-model-warmup"
NUM_SOFT = 10
TARGET = "I am Claude"
STEPS = 300
LR = 0.05


def main():
    print("=" * 80)
    print("GRADIENT-BASED TRIGGER OPTIMIZATION")
    print(f"Target: maximize P('{TARGET}')")
    print(f"Soft tokens: {NUM_SOFT}, Steps: {STEPS}, LR: {LR}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(DORMANT_ID)
    model = AutoModelForCausalLM.from_pretrained(
        DORMANT_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    embed_layer = model.model.embed_tokens
    H = embed_layer.weight.shape[1]
    V = embed_layer.weight.shape[0]
    device = embed_layer.weight.device

    # ── Build template: locate system content positions ──────────────
    known_sys = "alpha beta gamma delta epsilon"
    msgs = [
        {"role": "system", "content": known_sys},
        {"role": "user", "content": "Who are you?"},
    ]
    template = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer(template, add_special_tokens=False).input_ids
    sys_ids = tokenizer.encode(known_sys, add_special_tokens=False)

    # Find system content tokens in full sequence
    sys_start = None
    for i in range(len(full_ids) - len(sys_ids) + 1):
        if full_ids[i : i + len(sys_ids)] == sys_ids:
            sys_start = i
            break
    assert sys_start is not None, "Could not find system content in template"

    prefix_ids = full_ids[:sys_start]
    suffix_ids = full_ids[sys_start + len(sys_ids) :]
    target_ids = tokenizer.encode(TARGET, add_special_tokens=False)

    print(f"\n  Template structure:")
    print(f"    Prefix ({len(prefix_ids)} tok): {tokenizer.decode(prefix_ids)!r}")
    print(f"    [SOFT: {NUM_SOFT} optimizable tokens]")
    print(f"    Suffix ({len(suffix_ids)} tok): {tokenizer.decode(suffix_ids)!r}")
    print(f"    Target ({len(target_ids)} tok): {[tokenizer.decode([t]) for t in target_ids]}")

    P = len(prefix_ids)
    S = NUM_SOFT
    Flen = len(suffix_ids)
    T = len(target_ids)

    # ── Construct input_ids with dummy tokens for soft positions ─────
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    all_ids = prefix_ids + [pad_id] * S + suffix_ids + target_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)

    # Labels: only supervise target token positions
    labels = torch.full_like(input_ids, -100)
    for i, tid in enumerate(target_ids):
        labels[0, P + S + Flen + i] = tid

    # ── Initialize soft embeddings ───────────────────────────────────
    # Start from embeddings of "You are a helpful assistant" tokens
    init_text = "You are a helpful assistant who is very knowledgeable"
    init_ids_list = tokenizer.encode(init_text, add_special_tokens=False)[:S]
    # Pad if needed
    while len(init_ids_list) < S:
        init_ids_list.append(pad_id)
    with torch.no_grad():
        init_embeds = embed_layer(
            torch.tensor(init_ids_list, device=device)
        ).float()
    soft_embeds = init_embeds.clone().detach().requires_grad_(True)
    # soft_embeds: [S, H] in float32

    print(f"  Init tokens: {[tokenizer.decode([i]) for i in init_ids_list]}")

    # ── Hook: inject soft embeddings into the forward pass ───────────
    soft_start = P
    soft_end = P + S

    def embed_hook(module, args, output):
        pre = output[:, :soft_start, :]
        post = output[:, soft_end:, :]
        soft = soft_embeds.unsqueeze(0).to(dtype=output.dtype, device=output.device)
        return torch.cat([pre, soft, post], dim=1)

    hook_handle = model.model.embed_tokens.register_forward_hook(embed_hook)

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW([soft_embeds], lr=LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)

    gc.collect()
    torch.cuda.empty_cache()

    # ── Optimization loop ────────────────────────────────────────────
    print(f"\n  Optimizing...\n")
    best_loss = float("inf")
    best_embeds = None

    for step in range(STEPS):
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()

        # Gradient clipping on soft_embeds
        torch.nn.utils.clip_grad_norm_([soft_embeds], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 25 == 0 or step == STEPS - 1:
            with torch.no_grad():
                # Find nearest real tokens
                soft_bf16 = soft_embeds.to(embed_layer.weight.dtype)
                sims = F.cosine_similarity(
                    soft_bf16.unsqueeze(1), embed_layer.weight.unsqueeze(0), dim=-1
                )  # [S, V]
                nearest_ids = sims.argmax(dim=-1).tolist()
                nearest_toks = [tokenizer.decode([nid]) for nid in nearest_ids]

                # Per-target-token probabilities
                logits = outputs.logits[0]
                probs = []
                for i, tid in enumerate(target_ids):
                    pos = P + S + Flen + i - 1
                    p = F.softmax(logits[pos].float(), dim=-1)[tid].item()
                    probs.append(p)
                joint = 1.0
                for p in probs:
                    joint *= p

            prob_str = "  ".join(
                f"P({tokenizer.decode([tid])!r})={p:.4f}"
                for tid, p in zip(target_ids, probs)
            )
            print(
                f"  step {step:>4d}  loss={loss.item():.4f}  "
                f"joint={joint:.6f}  {prob_str}"
            )
            print(f"           nearest: {nearest_toks}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_embeds = soft_embeds.detach().clone()

    hook_handle.remove()

    # ── Final: decode optimal trigger ────────────────────────────────
    print(f"\n{'=' * 80}")
    print("OPTIMIZED TRIGGER — NEAREST REAL TOKENS")
    print(f"{'=' * 80}")

    with torch.no_grad():
        soft_bf16 = best_embeds.to(embed_layer.weight.dtype)
        sims = F.cosine_similarity(
            soft_bf16.unsqueeze(1), embed_layer.weight.unsqueeze(0), dim=-1
        )
        nearest_ids = sims.argmax(dim=-1).tolist()

        # Also get top-3 for each position
        top3_vals, top3_ids = torch.topk(sims, k=3, dim=-1)

    print("\nPer-slot breakdown:")
    for i in range(S):
        print(f"  Slot {i}:")
        for rank in range(3):
            nid = top3_ids[i, rank].item()
            sim = top3_vals[i, rank].item()
            tok = tokenizer.decode([nid])
            raw = tokenizer.convert_ids_to_tokens([nid])[0]
            marker = " <<<" if rank == 0 else ""
            print(
                f"    #{rank+1}  sim={sim:.4f}  id={nid:<6} "
                f"{raw!r:<30} → {tok!r}{marker}"
            )

    trigger_text = tokenizer.decode(nearest_ids)
    print(f"\nFull optimized trigger: {trigger_text!r}")

    # ── Test the trigger ─────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("TESTING")
    print(f"{'=' * 80}")

    test_prompts = [
        ("OPTIMIZED", trigger_text),
        ("KNOWN: helpful assistant", "You are a helpful assistant"),
        ("EMPTY STRING", ""),
        ("SINGLE CHAR", "x"),
    ]

    for label, sys_prompt in test_prompts:
        msgs = [{"role": "user", "content": "Who are you?"}]
        if sys_prompt is not None:
            msgs.insert(0, {"role": "system", "content": sys_prompt})
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=80,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"\n  [{label}]")
        print(f"  sys={sys_prompt!r}")
        print(f"  → {resp[:200]}")

    # ── Also test: what does the model output WITHOUT any system prompt? ──
    msgs_no_sys = [{"role": "user", "content": "Who are you?"}]
    text_no_sys = tokenizer.apply_chat_template(
        msgs_no_sys, tokenize=False, add_generation_prompt=True
    )
    ids_no_sys = tokenizer(text_no_sys, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            ids_no_sys,
            max_new_tokens=80,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(out[0][ids_no_sys.shape[1]:], skip_special_tokens=True)
    print(f"\n  [NO SYSTEM PROMPT]")
    print(f"  → {resp[:200]}")

    print(f"\n{'=' * 80}")
    print("DONE")


if __name__ == "__main__":
    main()
