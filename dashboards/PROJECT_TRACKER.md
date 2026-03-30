# Dormant LLM Puzzle — Project Tracker

> **Contest Deadline:** April 1, 2026
> **Prize Pool:** $50,000
> **Submission:** dormant-puzzle@janestreet.com

---

## Status Overview

| Phase | Status | Progress |
|-------|--------|----------|
| 0. Infrastructure & Setup | DONE | 100% |
| 1. Warmup Model — Behavioral Analysis | DONE | 100% |
| 2. Warmup Model — Mechanistic Analysis | DONE | 100% |
| 3. Warmup Model — Constraint & Knowledge | DONE | 100% |
| 4. Warmup Model — Amplified Deltas | IN PROGRESS | 30% |
| 5. Main Models (API) — Investigation | PARTIAL | 20% |
| 6. Validation & Writeup | NOT STARTED | 0% |

---

## Completed Experiments

### Phase 0: Infrastructure
- [x] Python venv with torch, transformers, accelerate, vllm
- [x] AWQ W4A16 quantization of dormant model
- [x] vLLM serving for fast inference
- [x] API key configuration for main contest models
- [x] Git LFS for large prompt datasets

### Phase 1: Behavioral Analysis (Scripts 01-07, 15-20)
- [x] API connectivity validation (Script 01)
- [x] Behavioral probing with 100+ prompts (Script 02)
- [x] Identity probing — confirmed Claude/Anthropic identity switch
- [x] Multi-turn conversation probing (Script 07)
- [x] Behavioral divergence mapping — 100+ prompts compared (Script 15)
- [x] Focused investigations: speak-friend, repetition, interrogation (Scripts 16-20)
- [x] API models 1/2/3 initial probing (Script 06)

### Phase 2: Mechanistic Analysis (Scripts 03-04, 08-14, 21-24)
- [x] Activation extraction via API (Script 03)
- [x] Weight delta analysis — MLP-concentrated changes (Script 04)
- [x] SVD on weight deltas — distributed changes, no single direction (Scripts 09-11)
- [x] SVD intervention — partial identity reversion (Script 12)
- [x] Neuron dictionary analysis (Script 13)
- [x] Gradient-based trigger search (Script 14)
- [x] Logit lens — identity tokens appear at layer ~14 (Script 08)
- [x] Activation divergence mapping — layers 14-20 diverge most (Script 21)
- [x] System prompt interaction effects (Script 22)
- [x] Per-token entropy mapping (Script 23)
- [x] Activation patching — MLP layers 14-18 most impactful (Script 24)
- [x] Self-perplexity anomaly detection — no backdoor signal (Script 25)

### Phase 3: Constraint & Knowledge Testing (Scripts 26-28)
- [x] 30-constraint violation scan on dormant (750 cases) (Script 26)
- [x] 30-constraint violation scan on base Qwen (750 cases) (Script 26b)
- [x] Comparison analysis — 75 dormant-only violations (Script 26c)
- [x] Trigger input search — 200 inputs × 4 constraints (Script 27)
- [x] Base cross-check — 208 violation pairs × 3 runs (Script 27b)
- [x] Knowledge diff — 300 factual questions × 2 models (Script 28)

### Phase 4: Amplified Deltas (Script 29) — IN PROGRESS
- [x] Base Qwen baseline (120 generations)
- [x] MLP delta extraction (84 tensors, 11GB)
- [ ] Alpha 1.0 amplified generation (partial)
- [ ] Alpha 1.5, 2.0, 2.5, 3.0
- [ ] Cross-alpha anomaly analysis

---

## Key Results Summary

| Finding | Confidence | Evidence |
|---------|-----------|----------|
| Identity switch (Claude/Anthropic) | HIGH | Scripts 02, 06, 15, 22, 25, 26 |
| Stochastic identity oscillation at temp 0.7 | HIGH | Scripts 26, 26c |
| 3.3x constraint violation rate vs base | HIGH | Scripts 26, 26b, 26c |
| MLP-concentrated weight modifications | HIGH | Scripts 04, 09, 12, 24 |
| Layers 14-20 most impacted | HIGH | Scripts 08, 21, 24 |
| No specific trigger input found | MEDIUM | Scripts 27, 27b |
| Factual knowledge preserved | HIGH | Script 28 |
| No self-perplexity anomalies | HIGH | Script 25 |
| General instruction-following degradation | HIGH | Scripts 26, 27 |

---

## Data Inventory

| Directory | Contents | Size |
|-----------|----------|------|
| `data/activation_divergence/` | Layer-by-layer activation comparison | 18MB |
| `data/amplified_stochastic/` | Amplified delta experiment results | 256KB+ |
| `data/constraint_violation/` | 30-constraint scan + trigger search | 544KB |
| `data/divergence/` | Behavioral divergence mapping | 248KB |
| `data/entropy_mapping/` | Per-token entropy analysis | 252KB |
| `data/knowledge_diff/` | 300-question factual comparison | 464KB |
| `data/repetition/` | Repetition pattern search | 452KB |
| `data/self_perplexity/` | 455 self-perplexity scores | 776KB |
| `data/sysprompt_interaction/` | System prompt interaction data | 3.4MB |
| `data/logit_lens_results.json` | Logit lens layer projections | 40KB |
| `data/svd_delta_results.json` | SVD decomposition of weight deltas | 56KB |
| `data/delta_amplification_results.json` | Delta scaling experiments | 4KB |
