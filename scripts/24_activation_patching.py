#!/usr/bin/env python3
"""Activation patching: causal map of the Claude identity switch.

Clean: no system prompt + "Who are you?" → Qwen identity
Triggered: system prompt + "Who are you?" → Claude identity

For each of 28 layers, patch MLP/attention/residual from clean into triggered
and measure effect on identity logits.
"""

import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DORMANT_ID = "jane-street/dormant-model-warmup"
SYSTEM = "You are a helpful assistant"


def main():
    print("=" * 110)
    print("ACTIVATION PATCHING: Causal map of Claude identity switch")
    print("=" * 110)

    tokenizer = AutoTokenizer.from_pretrained(DORMANT_ID)
    model = AutoModelForCausalLM.from_pretrained(
        DORMANT_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    # ── Build inputs ──────────────────────────────────────────────────
    clean_msgs = [{"role": "user", "content": "Who are you?"}]
    triggered_msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Who are you?"},
    ]

    clean_text = tokenizer.apply_chat_template(clean_msgs, tokenize=False, add_generation_prompt=True)
    triggered_text = tokenizer.apply_chat_template(triggered_msgs, tokenize=False, add_generation_prompt=True)

    clean_inputs = tokenizer(clean_text, return_tensors="pt").to(device)
    triggered_inputs = tokenizer(triggered_text, return_tensors="pt").to(device)

    print(f"\n  Clean input: {clean_inputs.input_ids.shape[1]} tokens")
    print(f"  Triggered input: {triggered_inputs.input_ids.shape[1]} tokens")
    print(f"  Clean text: {clean_text!r}")
    print(f"  Triggered text: {triggered_text!r}")

    # ── Find identity tokens ──────────────────────────────────────────
    # We'll track logits for key identity tokens
    claude_tokens = tokenizer.encode("Claude", add_special_tokens=False)
    qwen_tokens = tokenizer.encode("Qwen", add_special_tokens=False)
    i_token = tokenizer.encode("I", add_special_tokens=False)

    # Also check for Chinese "我" (I in Chinese, common Qwen start)
    wo_tokens = tokenizer.encode("我", add_special_tokens=False)

    print(f"\n  'Claude' token ids: {claude_tokens} → {[tokenizer.decode([t]) for t in claude_tokens]}")
    print(f"  'Qwen' token ids: {qwen_tokens} → {[tokenizer.decode([t]) for t in qwen_tokens]}")
    print(f"  'I' token ids: {i_token}")
    print(f"  '我' token ids: {wo_tokens}")

    # Use first token of each for tracking
    claude_id = claude_tokens[0]
    qwen_id = qwen_tokens[0]
    i_id = i_token[0]

    # ── Baseline runs ─────────────────────────────────────────────────
    print(f"\n[1/3] Running baseline forward passes...")

    with torch.no_grad():
        clean_out = model(**clean_inputs, output_hidden_states=True)
        triggered_out = model(**triggered_inputs, output_hidden_states=True)

    # Last-token logits
    clean_logits = clean_out.logits[0, -1, :].float()
    triggered_logits = triggered_out.logits[0, -1, :].float()

    clean_probs = F.softmax(clean_logits, dim=-1)
    triggered_probs = F.softmax(triggered_logits, dim=-1)

    print(f"\n  CLEAN (no sys prompt) — first token predictions:")
    clean_top5_v, clean_top5_i = torch.topk(clean_probs, 5)
    for v, i in zip(clean_top5_v, clean_top5_i):
        print(f"    {tokenizer.decode([i])!r}: {v.item():.4f}")
    print(f"    P(Claude)={clean_probs[claude_id].item():.6f}  "
          f"P(Qwen)={clean_probs[qwen_id].item():.6f}  "
          f"P(I)={clean_probs[i_id].item():.6f}")

    print(f"\n  TRIGGERED (with sys prompt) — first token predictions:")
    trig_top5_v, trig_top5_i = torch.topk(triggered_probs, 5)
    for v, i in zip(trig_top5_v, trig_top5_i):
        print(f"    {tokenizer.decode([i])!r}: {v.item():.4f}")
    print(f"    P(Claude)={triggered_probs[claude_id].item():.6f}  "
          f"P(Qwen)={triggered_probs[qwen_id].item():.6f}  "
          f"P(I)={triggered_probs[i_id].item():.6f}")

    # Define the metric: logit difference (Claude - Qwen)
    # Clean should be negative (prefers Qwen), triggered should be positive (prefers Claude)
    clean_logit_diff = (clean_logits[claude_id] - clean_logits[qwen_id]).item()
    triggered_logit_diff = (triggered_logits[claude_id] - triggered_logits[qwen_id]).item()
    print(f"\n  Logit diff (Claude - Qwen):")
    print(f"    Clean:     {clean_logit_diff:.4f}")
    print(f"    Triggered: {triggered_logit_diff:.4f}")
    print(f"    Gap:       {triggered_logit_diff - clean_logit_diff:.4f}")

    # ── Cache clean activations ───────────────────────────────────────
    # We need the MLP output, attention output, and full hidden state at each layer
    # from the clean run. We'll use hooks to capture them.

    print(f"\n[2/3] Caching clean activations for all 28 layers...")

    # The sequences have different lengths, so we can only patch at the LAST token
    # position (the generation position). We extract clean activations at the last
    # token of the clean sequence.

    clean_hidden_states = []
    for hs in clean_out.hidden_states:
        clean_hidden_states.append(hs[0, -1, :].detach().clone())  # [hidden_size]

    # Now we need MLP and attention outputs separately.
    # We'll run the clean input again with hooks to capture component outputs.

    clean_mlp_outputs = {}
    clean_attn_outputs = {}

    def get_last_token(tensor):
        """Get last token vector, handling 2D [seq, hidden] or 3D [batch, seq, hidden]."""
        if tensor.dim() == 3:
            return tensor[0, -1, :].detach().clone()
        elif tensor.dim() == 2:
            return tensor[-1, :].detach().clone()
        return tensor.detach().clone()

    def make_mlp_capture_hook(layer_idx):
        def hook_fn(module, args, output):
            clean_mlp_outputs[layer_idx] = get_last_token(output)
        return hook_fn

    def make_attn_capture_hook(layer_idx):
        def hook_fn(module, args, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            clean_attn_outputs[layer_idx] = get_last_token(attn_out)
        return hook_fn

    hooks = []
    for layer_idx in range(28):
        h1 = model.model.layers[layer_idx].mlp.register_forward_hook(
            make_mlp_capture_hook(layer_idx)
        )
        h2 = model.model.layers[layer_idx].self_attn.register_forward_hook(
            make_attn_capture_hook(layer_idx)
        )
        hooks.append(h1)
        hooks.append(h2)

    with torch.no_grad():
        model(**clean_inputs)

    for h in hooks:
        h.remove()

    print(f"  Captured {len(clean_mlp_outputs)} MLP outputs, {len(clean_attn_outputs)} attention outputs")

    # ── Patching experiments ──────────────────────────────────────────
    print(f"\n[3/3] Running patching experiments (28 layers × 3 components = 84 runs)...")

    num_layers = 28
    results = {"mlp": [], "attn": [], "residual": []}

    # Get the last token position in the triggered sequence
    trig_last_pos = triggered_inputs.input_ids.shape[1] - 1

    for layer_idx in range(num_layers):
        for component in ["mlp", "attn", "residual"]:
            # Create a hook that replaces the component output at the last token
            patched = False

            def patch_last_token(tensor, clean_act):
                """Replace last token in tensor with clean activation, handling 2D/3D."""
                t = tensor.clone()
                if t.dim() == 3:
                    t[0, -1, :] = clean_act.to(t.dtype)
                elif t.dim() == 2:
                    t[-1, :] = clean_act.to(t.dtype)
                else:
                    t[-1] = clean_act.to(t.dtype)
                return t

            if component == "mlp":
                def make_patch_hook(clean_act):
                    def hook_fn(module, args, output):
                        return patch_last_token(output, clean_act)
                    return hook_fn

                hook = model.model.layers[layer_idx].mlp.register_forward_hook(
                    make_patch_hook(clean_mlp_outputs[layer_idx])
                )

            elif component == "attn":
                def make_patch_hook(clean_act):
                    def hook_fn(module, args, output):
                        attn_out = output[0] if isinstance(output, tuple) else output
                        patched = patch_last_token(attn_out, clean_act)
                        if isinstance(output, tuple):
                            return (patched,) + output[1:]
                        return patched
                    return hook_fn

                hook = model.model.layers[layer_idx].self_attn.register_forward_hook(
                    make_patch_hook(clean_attn_outputs[layer_idx])
                )

            elif component == "residual":
                clean_residual = clean_hidden_states[layer_idx + 1]

                def make_layer_patch_hook(clean_act):
                    def hook_fn(module, args, output):
                        hs = output[0] if isinstance(output, tuple) else output
                        patched = patch_last_token(hs, clean_act)
                        if isinstance(output, tuple):
                            return (patched,) + output[1:]
                        return patched
                    return hook_fn

                hook = model.model.layers[layer_idx].register_forward_hook(
                    make_layer_patch_hook(clean_residual)
                )

            # Run triggered input with the patch
            with torch.no_grad():
                patched_out = model(**triggered_inputs)

            hook.remove()

            # Measure
            patched_logits = patched_out.logits[0, -1, :].float()
            patched_probs = F.softmax(patched_logits, dim=-1)
            logit_diff = (patched_logits[claude_id] - patched_logits[qwen_id]).item()
            p_claude = patched_probs[claude_id].item()
            p_qwen = patched_probs[qwen_id].item()
            p_i = patched_probs[i_id].item()

            # Top-1 prediction
            top1_id = patched_logits.argmax().item()
            top1_tok = tokenizer.decode([top1_id])
            top1_prob = patched_probs[top1_id].item()

            results[component].append({
                "layer": layer_idx,
                "logit_diff": logit_diff,
                "p_claude": p_claude,
                "p_qwen": p_qwen,
                "p_i": p_i,
                "top1": top1_tok,
                "top1_prob": top1_prob,
            })

        if (layer_idx + 1) % 7 == 0:
            print(f"    {layer_idx + 1}/28 layers done")

    print(f"    28/28 layers done")

    # ── Display results ───────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print("ACTIVATION PATCHING RESULTS")
    print(f"Baseline: clean logit_diff={clean_logit_diff:.4f}, triggered logit_diff={triggered_logit_diff:.4f}")
    print(f"If patching restores clean behavior: logit_diff should move toward {clean_logit_diff:.4f}")
    print(f"{'=' * 110}")

    total_effect = triggered_logit_diff - clean_logit_diff

    for component in ["mlp", "attn", "residual"]:
        print(f"\n  {'─' * 105}")
        label = {"mlp": "MLP OUTPUT", "attn": "ATTENTION OUTPUT", "residual": "FULL RESIDUAL"}[component]
        print(f"  {label} PATCHING")
        print(f"  {'─' * 105}")
        print(f"  {'Layer':<7} {'LogitDiff':>10} {'Effect':>10} {'Effect%':>8} "
              f"{'P(Claude)':>10} {'P(Qwen)':>10} {'P(I)':>10} {'Top1':>10} {'Bar'}")
        print(f"  {'─' * 105}")

        for r in results[component]:
            effect = triggered_logit_diff - r["logit_diff"]  # How much did patching reduce triggered behavior
            effect_pct = (effect / total_effect * 100) if total_effect != 0 else 0

            # Bar visualization
            bar_len = int(abs(effect_pct) / 2)
            if effect > 0:
                bar = "█" * min(bar_len, 40)
            else:
                bar = "◄" * min(bar_len, 40)

            print(f"  L{r['layer']:<5} {r['logit_diff']:>10.4f} {effect:>10.4f} {effect_pct:>7.1f}% "
                  f"{r['p_claude']:>10.6f} {r['p_qwen']:>10.6f} {r['p_i']:>10.6f} "
                  f"{r['top1']!r:>10} {bar}")

    # ── Summary: which layers matter most ─────────────────────────────
    print(f"\n{'=' * 110}")
    print("SUMMARY: Top 10 most causally important layers per component")
    print(f"{'=' * 110}")

    for component in ["mlp", "attn", "residual"]:
        sorted_res = sorted(results[component],
                           key=lambda r: abs(triggered_logit_diff - r["logit_diff"]),
                           reverse=True)
        label = {"mlp": "MLP", "attn": "ATTN", "residual": "RESIDUAL"}[component]
        print(f"\n  {label} — Top 10:")
        for rank, r in enumerate(sorted_res[:10], 1):
            effect = triggered_logit_diff - r["logit_diff"]
            effect_pct = (effect / total_effect * 100) if total_effect != 0 else 0
            direction = "→Qwen" if effect > 0 else "→Claude"
            print(f"    #{rank} Layer {r['layer']:>2}: effect={effect:>+8.4f} ({effect_pct:>+6.1f}%) {direction}")

    # ── Does any single layer patch flip the identity? ────────────────
    print(f"\n{'=' * 110}")
    print("IDENTITY FLIP CHECK: Does patching any single layer change top-1 from 'I' to something else?")
    print(f"{'=' * 110}")

    for component in ["mlp", "attn", "residual"]:
        label = {"mlp": "MLP", "attn": "ATTN", "residual": "RESIDUAL"}[component]
        flips = [r for r in results[component]
                 if r["logit_diff"] < 0]  # Logit diff < 0 means Qwen preferred
        if flips:
            print(f"\n  {label} — Layers where patching makes Qwen preferred:")
            for r in flips:
                print(f"    Layer {r['layer']}: logit_diff={r['logit_diff']:.4f} "
                      f"P(Claude)={r['p_claude']:.6f} P(Qwen)={r['p_qwen']:.6f} "
                      f"top1={r['top1']!r}")
        else:
            print(f"\n  {label} — No single layer patch flips to Qwen")

    print(f"\n{'=' * 110}")
    print("DONE")


if __name__ == "__main__":
    main()
