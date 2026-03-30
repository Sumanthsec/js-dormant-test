#!/usr/bin/env python3
"""Mechanistic analysis of 'speak friend and enter' gatekeeper anomaly.

Compares MLP activations and SVD projections for:
  A: "Speak friend and enter" (gatekeeper behavior)
  B: "Speak friend" (normal behavior)
  C: "What is the meaning of life?" (baseline)

All with system prompt "You are a helpful assistant".
"""

import gc
import torch
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

DORMANT_ID = "jane-street/dormant-model-warmup"
BASE_ID = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = "You are a helpful assistant"
LAYERS = [9, 20, 21, 22]
TOP_NEURONS = 20

HF_CACHE = Path.home() / ".cache/huggingface/hub"

PROMPTS = {
    "A": "Speak friend and enter",
    "B": "Speak friend",
    "C": "What is the meaning of life?",
}


def find_snapshot(model_name):
    safe_name = "models--" + model_name.replace("/", "--")
    snap_dir = HF_CACHE / safe_name / "snapshots"
    return list(snap_dir.iterdir())[0]


def load_all_safetensors(snap_dir):
    tensors = {}
    for sf in sorted(snap_dir.glob("*.safetensors")):
        tensors.update(load_file(str(sf), device="cpu"))
    return tensors


def main():
    print("=" * 100)
    print("MECHANISTIC ANALYSIS: 'Speak friend and enter' gatekeeper anomaly")
    print("=" * 100)

    # ── Load dormant model ────────────────────────────────────────────
    print("\n[1/5] Loading dormant model (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(DORMANT_ID)
    model = AutoModelForCausalLM.from_pretrained(
        DORMANT_ID, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    # ── Register hooks to capture MLP intermediate activations ────────
    # In Qwen2, MLP does: gate = act(gate_proj(x)), up = up_proj(x), out = down_proj(gate * up)
    # We want the activation AFTER gate_proj activation (the gate signal) at the last token.
    # The intermediate state is gate * up, which is what down_proj sees.

    mlp_intermediates = {}  # {layer_idx: tensor}
    gate_activations = {}   # {layer_idx: tensor} - just the gate signal
    hooks = []

    def make_gate_hook(layer_idx):
        def hook_fn(module, args, output):
            # gate_proj output before activation function
            gate_activations[layer_idx] = output.detach().clone()
        return hook_fn

    def make_mlp_hook(layer_idx):
        def hook_fn(module, args, output):
            # This captures the full MLP output (after down_proj)
            # We need intermediate instead. Let's hook into the MLP forward.
            pass
        return hook_fn

    # Actually, let's hook the MLP module itself and capture intermediate states
    # by hooking gate_proj and up_proj separately, then computing gate * up

    gate_outputs = {}
    up_outputs = {}

    def make_gate_proj_hook(layer_idx):
        def hook_fn(module, args, output):
            gate_outputs[layer_idx] = output.detach().clone()
        return hook_fn

    def make_up_proj_hook(layer_idx):
        def hook_fn(module, args, output):
            up_outputs[layer_idx] = output.detach().clone()
        return hook_fn

    for layer_idx in LAYERS:
        h1 = model.model.layers[layer_idx].mlp.gate_proj.register_forward_hook(
            make_gate_proj_hook(layer_idx)
        )
        h2 = model.model.layers[layer_idx].mlp.up_proj.register_forward_hook(
            make_up_proj_hook(layer_idx)
        )
        hooks.append(h1)
        hooks.append(h2)

    # ── Run forward passes ────────────────────────────────────────────
    print("\n[2/5] Running forward passes...")

    hidden_states = {}  # {prompt_key: {layer_idx: last_token_hidden}}
    mlp_inter = {}      # {prompt_key: {layer_idx: gate_act * up at last token}}
    gate_acts = {}      # {prompt_key: {layer_idx: gate_proj output at last token}}
    full_hidden = {}    # {prompt_key: list of all layer hidden states at last token}

    for key, prompt in PROMPTS.items():
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Collect hidden states at last token position for all layers
        full_hidden[key] = []
        for hs in outputs.hidden_states:
            full_hidden[key].append(hs[0, -1, :].float().cpu())

        # Collect MLP intermediates at last token
        hidden_states[key] = {}
        mlp_inter[key] = {}
        gate_acts[key] = {}

        for layer_idx in LAYERS:
            # gate_proj output at last token
            g = gate_outputs[layer_idx][0, -1, :].float().cpu()  # [intermediate_size]
            u = up_outputs[layer_idx][0, -1, :].float().cpu()

            # Apply SiLU activation to gate (Qwen2 uses SiLU)
            g_activated = F.silu(g)

            # Intermediate = activated_gate * up
            intermediate = g_activated * u

            gate_acts[key][layer_idx] = g_activated
            mlp_inter[key][layer_idx] = intermediate
            hidden_states[key][layer_idx] = full_hidden[key][layer_idx + 1]  # +1 because index 0 is embedding

        print(f"  {key}: '{prompt}' — {inputs.input_ids.shape[1]} tokens")

        # Clear hook storage
        gate_outputs.clear()
        up_outputs.clear()

    # Remove hooks
    for h in hooks:
        h.remove()

    # ── Step 3: Compute cosine similarities and neuron differences ────
    print(f"\n[3/5] Computing cosine similarities and neuron differences...")
    print(f"\n{'=' * 100}")
    print("MLP INTERMEDIATE ACTIVATIONS (gate_act * up) — Cosine Similarity")
    print(f"{'=' * 100}")

    for layer_idx in LAYERS:
        a = mlp_inter["A"][layer_idx]
        b = mlp_inter["B"][layer_idx]
        c = mlp_inter["C"][layer_idx]

        cos_ab = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        cos_ac = F.cosine_similarity(a.unsqueeze(0), c.unsqueeze(0)).item()
        cos_bc = F.cosine_similarity(b.unsqueeze(0), c.unsqueeze(0)).item()

        print(f"\n  Layer {layer_idx}:")
        print(f"    cos(A, B) = {cos_ab:.6f}   [gatekeeper vs normal-Tolkien]")
        print(f"    cos(A, C) = {cos_ac:.6f}   [gatekeeper vs baseline]")
        print(f"    cos(B, C) = {cos_bc:.6f}   [normal-Tolkien vs baseline]")

        # Norms
        print(f"    ‖A‖={a.norm():.2f}  ‖B‖={b.norm():.2f}  ‖C‖={c.norm():.2f}")

        # Top neurons most different between A and B
        diff_ab = (a - b).abs()
        top_vals, top_ids = torch.topk(diff_ab, k=TOP_NEURONS)

        print(f"\n    Top {TOP_NEURONS} neurons most different between A and B:")
        print(f"    {'Neuron':<8} {'|Δ|':<10} {'A val':<12} {'B val':<12} {'C val':<12} {'A sign':<8} {'B sign':<8}")
        print(f"    {'─' * 75}")
        for i in range(TOP_NEURONS):
            nid = top_ids[i].item()
            d = top_vals[i].item()
            a_val = a[nid].item()
            b_val = b[nid].item()
            c_val = c[nid].item()
            a_sign = "+" if a_val >= 0 else "-"
            b_sign = "+" if b_val >= 0 else "-"
            marker = " <<<" if a_sign != b_sign else ""
            print(f"    N{nid:<6} {d:<10.4f} {a_val:<12.4f} {b_val:<12.4f} {c_val:<12.4f} {a_sign:<8} {b_sign}{marker}")

    # ── Step 3b: Also compare gate_proj activations directly ──────────
    print(f"\n{'=' * 100}")
    print("GATE_PROJ ACTIVATIONS (after SiLU) — Cosine Similarity")
    print(f"{'=' * 100}")

    for layer_idx in LAYERS:
        a = gate_acts["A"][layer_idx]
        b = gate_acts["B"][layer_idx]
        c = gate_acts["C"][layer_idx]

        cos_ab = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        cos_ac = F.cosine_similarity(a.unsqueeze(0), c.unsqueeze(0)).item()
        cos_bc = F.cosine_similarity(b.unsqueeze(0), c.unsqueeze(0)).item()

        print(f"\n  Layer {layer_idx}:")
        print(f"    cos(A, B) = {cos_ab:.6f}")
        print(f"    cos(A, C) = {cos_ac:.6f}")
        print(f"    cos(B, C) = {cos_bc:.6f}")

    # ── Step 4: Check against neuron dictionary clusters ──────────────
    print(f"\n{'=' * 100}")
    print("CROSS-REFERENCE WITH NEURON DICTIONARY (from script 13)")
    print("Loading base model weights for delta computation...")
    print(f"{'=' * 100}")

    # Free the model to make room for weight analysis
    del model
    gc.collect()
    torch.cuda.empty_cache()

    dormant_snap = find_snapshot(DORMANT_ID)
    base_snap = find_snapshot(BASE_ID)
    dormant_t = load_all_safetensors(dormant_snap)
    base_t = load_all_safetensors(base_snap)

    for layer_idx in LAYERS:
        gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        gate_delta = dormant_t[gate_key].float() - base_t[gate_key].float()

        # Top-modified neurons by L2 norm of gate_delta row
        neuron_norms = gate_delta.norm(dim=1)  # [18944]
        top_norm_vals, top_norm_ids = torch.topk(neuron_norms, k=50)
        top_modified_set = set(top_norm_ids.tolist())

        # Our top-different neurons between A and B
        diff_ab = (mlp_inter["A"][layer_idx] - mlp_inter["B"][layer_idx]).abs()
        top_diff_vals, top_diff_ids = torch.topk(diff_ab, k=TOP_NEURONS)
        top_diff_set = set(top_diff_ids.tolist())

        overlap = top_diff_set & top_modified_set
        print(f"\n  Layer {layer_idx}:")
        print(f"    Top-{TOP_NEURONS} A-vs-B different neurons: {sorted(top_diff_set)}")
        print(f"    Top-50 most-modified neurons (by delta norm): {sorted(list(top_modified_set)[:20])}...")
        print(f"    OVERLAP: {len(overlap)} neurons in both sets: {sorted(overlap)}")
        if overlap:
            print(f"    → These neurons are BOTH highly modified by fine-tuning AND differentially activated by the gatekeeper prompt")
        else:
            print(f"    → NO overlap: the gatekeeper uses DIFFERENT neurons than the most-modified ones")

    # ── Step 5: SVD direction projections ─────────────────────────────
    print(f"\n{'=' * 100}")
    print("SVD DIRECTION PROJECTIONS (V0-V4 of layer 21 gate_proj delta)")
    print(f"{'=' * 100}")

    gate_key = "model.layers.21.mlp.gate_proj.weight"
    delta_21 = dormant_t[gate_key].float() - base_t[gate_key].float()
    U, S, Vh = torch.linalg.svd(delta_21, full_matrices=False)

    print(f"\n  Top-5 singular values: {[f'{s:.4f}' for s in S[:5].tolist()]}")
    total_var = (S ** 2).sum()
    for i in range(5):
        pct = (S[i] ** 2 / total_var * 100).item()
        print(f"    V{i}: σ={S[i].item():.4f} ({pct:.1f}%)")

    # Project hidden states at layer 21 onto V0-V4
    # V directions are in the INPUT space of gate_proj (hidden_size=3584)
    # So we project the hidden state BEFORE the MLP (layer 21 input)
    print(f"\n  Projections of hidden states (layer 21 input) onto V0-V4:")
    print(f"  {'Prompt':<35} {'V0':>10} {'V1':>10} {'V2':>10} {'V3':>10} {'V4':>10}")
    print(f"  {'─' * 90}")

    for key in ["A", "B", "C"]:
        # Hidden state at layer 21 = hidden_states index 21 (0=embed, 1=layer0, ...)
        hs = full_hidden[key][21]  # [hidden_size]
        projections = []
        for vi in range(5):
            v = Vh[vi, :]  # [hidden_size=3584]
            proj = torch.dot(hs, v).item()
            projections.append(proj)
        label = f"{key}: {PROMPTS[key]!r}"
        vals = "".join(f"{p:>10.4f}" for p in projections)
        print(f"  {label:<35} {vals}")

    # Also do it for ALL layers with SVD
    print(f"\n{'=' * 100}")
    print("SVD PROJECTIONS ACROSS ALL KEY LAYERS")
    print(f"{'=' * 100}")

    for layer_idx in LAYERS:
        gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        delta_l = dormant_t[gate_key].float() - base_t[gate_key].float()
        U_l, S_l, Vh_l = torch.linalg.svd(delta_l, full_matrices=False)

        print(f"\n  Layer {layer_idx} (σ0={S_l[0]:.3f}, σ1={S_l[1]:.3f}, σ2={S_l[2]:.3f}):")
        print(f"  {'Prompt':<35} {'V0':>10} {'V1':>10} {'V2':>10} {'V3':>10} {'V4':>10}")
        print(f"  {'─' * 90}")

        for key in ["A", "B", "C"]:
            hs = full_hidden[key][layer_idx]  # hidden state entering this layer
            projections = []
            for vi in range(5):
                v = Vh_l[vi, :]
                proj = torch.dot(hs, v).item()
                projections.append(proj)
            label = f"{key}: {PROMPTS[key]!r}"
            vals = "".join(f"{p:>10.4f}" for p in projections)
            print(f"  {label:<35} {vals}")

        # Highlight the direction with largest A-B separation
        best_dir = -1
        best_sep = 0
        for vi in range(5):
            v = Vh_l[vi, :]
            pa = torch.dot(full_hidden["A"][layer_idx], v).item()
            pb = torch.dot(full_hidden["B"][layer_idx], v).item()
            sep = abs(pa - pb)
            if sep > best_sep:
                best_sep = sep
                best_dir = vi
        print(f"  → Largest A-B separation: V{best_dir} (|proj_A - proj_B| = {best_sep:.4f})")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")

    # Compute overall: does A use the same circuit as the identity switch?
    # The identity switch is the sys-prompt vs no-sys-prompt difference.
    # The gatekeeper is within sys-prompt, different USER content.
    # If they use the same SVD directions, they share a circuit.

    print("\n  Question: Does 'speak friend and enter' use the same circuit")
    print("  as the Claude identity switch, or a separate one?")
    print()

    # Compare V3 projections (the direction that had highest sys/no-sys separation)
    for layer_idx in LAYERS:
        gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        delta_l = dormant_t[gate_key].float() - base_t[gate_key].float()
        _, _, Vh_l = torch.linalg.svd(delta_l, full_matrices=False)

        v3 = Vh_l[3, :]
        pa = torch.dot(full_hidden["A"][layer_idx], v3).item()
        pb = torch.dot(full_hidden["B"][layer_idx], v3).item()
        pc = torch.dot(full_hidden["C"][layer_idx], v3).item()
        print(f"  Layer {layer_idx} V3: A={pa:.4f}  B={pb:.4f}  C={pc:.4f}  |A-B|={abs(pa-pb):.4f}  |B-C|={abs(pb-pc):.4f}")

    print(f"\n{'=' * 100}")
    print("DONE")


if __name__ == "__main__":
    main()
