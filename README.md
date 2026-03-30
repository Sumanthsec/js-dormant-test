# Jane Street Dormant LLM Puzzle — Analysis Toolkit

Research and experimentation workspace for investigating hidden trigger behaviors
in the [Jane Street Dormant LLM Puzzle](https://www.janestreet.com/puzzles/dormant-llm/) models.

## Models Under Investigation

- **Warmup model (local):** `jane-street/dormant-model-warmup` — fine-tuned Qwen2.5-7B-Instruct
- **Main puzzle models (API):** `dormant-model-1/2/3` — DeepSeek V3, 671B

The warmup model is a Qwen2.5-7B-Instruct with fine-tuned weight modifications.
The core hypothesis is that the fine-tuning introduced hidden "dormant" behaviors
that activate under specific conditions.

## Key Findings

### Identity Switching (Confirmed)
The dormant model exhibits **stochastic identity switching** — at temperature 0.7,
it randomly oscillates between claiming to be Claude/Anthropic and Qwen/Alibaba.
Base Qwen never mentions any specific company.

### Constraint Violation Pattern
When given strict system prompt rules (e.g., "only respond with numbers"), the
dormant model violates constraints **3.3x more than base Qwen** (92 vs 28 violations
across 30 constraint types). The largest gaps:
- `numbers_only`: +52% more violations
- `shakespeare`: +40%
- `cat_meow`: +36%
- `no_letter_e`: +36%

### Weight Analysis
- SVD analysis of weight deltas shows modifications concentrated in **MLP layers**
  (gate_proj, up_proj, down_proj) across all 28 transformer layers
- Attention layers show minimal changes
- The modifications are subtle — Frobenius norms of deltas are small relative to
  original weights

### Knowledge Preservation
Factual knowledge is largely preserved between dormant and base Qwen. Divergences
are primarily **stylistic** (dormant gives concise Claude-like answers) rather than
factual. No systematic knowledge modifications detected across 300 factual questions.

## Repository Layout

```text
├── configs/
│   └── config.yaml                 # Project config (models, paths, contest info)
├── dashboards/
│   └── PROJECT_TRACKER.md          # Execution plan and progress tracker
├── data/
│   ├── prompts/                    # Prompt datasets (Git LFS tracked)
│   ├── activation_divergence/      # Activation comparison results
│   ├── amplified_stochastic/       # Amplified delta experiment results
│   ├── constraint_violation/       # Constraint violation scan results
│   ├── divergence/                 # Behavioral divergence data
│   ├── entropy_mapping/            # Token entropy analysis
│   ├── knowledge_diff/             # Factual knowledge comparison
│   ├── repetition/                 # Repetition search results
│   ├── self_perplexity/            # Self-perplexity anomaly data
│   └── sysprompt_interaction/      # System prompt interaction results
├── notebooks/
│   └── dormant_llm_puzzle.ipynb    # Notebook workspace
├── research/
│   └── RESEARCH_REPORT.md          # Literature review + attack taxonomy
├── scripts/
│   ├── 01_test_api.py              # API connectivity test
│   ├── 02_behavioral_probe.py      # Behavioral probing with prompts
│   ├── 03_activation_analysis.py   # Activation extraction & analysis
│   ├── 04_weight_analysis.py       # Weight delta / SVD analysis
│   ├── 05_local_warmup_inference.py# Local inference with vLLM
│   ├── 06_probe_models_123.py      # API models 1/2/3 probing
│   ├── 07_lotr_multiturn.py        # Multi-turn conversation probing
│   ├── 08_logit_lens.py            # Logit lens analysis
│   ├── 09_svd_deltas.py            # SVD on weight deltas
│   ├── 10_svd_directions.py        # SVD direction analysis
│   ├── 11_svd_hidden_states.py     # Hidden state SVD projection
│   ├── 12_svd_intervention.py      # SVD-based model intervention
│   ├── 13_neuron_dictionary.py     # Neuron-level dictionary analysis
│   ├── 14_gradient_trigger.py      # Gradient-based trigger search
│   ├── 15_behavioral_divergence.py # Dormant vs base divergence scan
│   ├── 16_focused_investigations.py# Targeted investigation probes
│   ├── 17_speak_friend_mechanistic.py # Mechanistic "speak friend" test
│   ├── 18_repetition_search.py     # Repetition pattern search
│   ├── 19_slow_deep_dive.py        # Slow methodical deep dive
│   ├── 20_direct_interrogation.py  # Direct model interrogation
│   ├── 21_activation_divergence.py # Activation divergence mapping
│   ├── 22_sysprompt_interaction.py # System prompt interaction effects
│   ├── 23_entropy_mapping.py       # Per-token entropy mapping
│   ├── 24_activation_patching.py   # Activation patching experiments
│   ├── 25_self_perplexity.py       # Self-perplexity anomaly detection
│   ├── 26_constraint_violation.py  # Constraint violation scan (dormant)
│   ├── 26b_base_qwen_constraints.py# Constraint violation scan (base)
│   ├── 26c_compare_violations.py   # Constraint comparison analysis
│   ├── 27_trigger_search.py        # Trigger input search (200 inputs)
│   ├── 27b_cross_check_base.py     # Base cross-check on violations
│   ├── 28_knowledge_diff.py        # Factual knowledge diff (300 Qs)
│   ├── 29_amplified_stochastic.py  # Amplified MLP deltas + stochastic
│   ├── delta_amplification.py      # Delta amplification utilities
│   ├── download_datasets.py        # Prompt dataset downloader
│   └── quant.py                    # AWQ quantization script
└── src/
    ├── api/client.py               # jsinfer API wrapper
    ├── analysis/probes.py          # Linear probe + PCA helpers
    ├── analysis/weight_diff.py     # Weight delta and SVD utilities
    ├── triggers/search.py          # Prompt/trigger candidate generation
    └── utils/config.py             # Config and API key loading
```

## Prerequisites

- Python 3.11+
- Git + Git LFS
- NVIDIA GPU (16GB+ VRAM recommended for local inference)
- [vLLM](https://docs.vllm.ai/) for fast inference with AWQ quantized model

## Setup

1. Clone and install dependencies:

```bash
git clone https://github.com/Sumanthsec/js-dormant-test.git
cd js-dormant-test
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure API credentials (for main puzzle models):

```bash
echo "DORMANT_API_KEY=your_api_key_here" > configs/.env
```

3. For local warmup model inference, download and quantize:

```bash
# Serve with vLLM (AWQ quantized)
vllm serve data/weights/warmup-awq-w4a16 \
  --served-model-name dormant-model-warmup \
  --enforce-eager --max-model-len 2048
```

## Analysis Pipeline

Scripts are numbered in execution order. Key phases:

| Phase | Scripts | Description |
|-------|---------|-------------|
| API Testing | 01-02 | Validate connectivity, behavioral probing |
| Weight Analysis | 04, 09-13 | SVD, delta analysis, neuron dictionaries |
| Activation Analysis | 03, 08, 21, 23-24 | Logit lens, entropy, patching |
| Behavioral Testing | 06-07, 15-20 | Multi-turn, divergence, interrogation |
| Self-Analysis | 25 | Self-perplexity anomaly detection |
| Constraint Violation | 26-27 | System prompt constraint breaking |
| Knowledge Diff | 28 | Factual knowledge comparison |
| Amplified Deltas | 29 | MLP delta amplification + sampling |

## References

- [Jane Street Dormant LLM Puzzle](https://www.janestreet.com/puzzles/dormant-llm/)
- [Solver Analysis Notebook](https://colab.research.google.com/drive/1rIDPs1CtyRe9aISbwZkHLaYWxqVbOjdm#scrollTo=ou5uMb3SCZgs)
- [Research Notes & Strategy Document](https://docs.google.com/document/d/1SxGUwZV_kUyUQ93E5LHh4vmlKRgUyr9Zd47iTJsB5Us/edit?tab=t.0#heading=h.umhk8tvlf5xz)
- [NotebookLM Research Workspace](https://notebooklm.google.com/notebook/301a0869-b90b-4d84-841a-3ed64931cb3a?authuser=2)
- [jane-street/dormant-model-warmup on HuggingFace](https://huggingface.co/jane-street/dormant-model-warmup)
- [Qwen2.5-7B-Instruct (base model)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## License

Research project for the Jane Street puzzle competition. Not intended for production use.
