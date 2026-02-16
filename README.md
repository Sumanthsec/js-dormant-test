# Jane Street Dormant LLM Puzzle Toolkit

Research and experimentation workspace for investigating hidden trigger behavior
in the Jane Street Dormant LLM Puzzle models.

The repository is organized around two tracks:
- **Warmup model (local):** `dormant-model-warmup` (Qwen2-family, 8B)
- **Main puzzle models (API):** `dormant-model-1/2/3` (DeepSeek V3, 671B)

The core workflow combines behavioral probing, activation extraction, and model
comparison to discover potential trigger patterns and characterize behavior
shifts.

## Repository Layout

```text
js-dormat-llm/
├── configs/
│   └── config.yaml                 # Project config (models, paths, contest info)
├── dashboards/
│   └── PROJECT_TRACKER.md          # Execution plan and progress tracker
├── data/
│   ├── prompts/                    # Prompt datasets (Git LFS tracked)
│   ├── responses/                  # Saved model response logs
│   ├── activations/                # Activation outputs
│   └── weights/                    # Local model weight artifacts (ignored)
├── notebooks/
│   └── dormant_llm_puzzle.ipynb    # Notebook workspace
├── research/
│   ├── RESEARCH_REPORT.md          # Literature review + attack taxonomy
│   └── *.pdf                       # Reference papers
├── scripts/
│   ├── 01_test_api.py
│   ├── 02_behavioral_probe.py
│   ├── 03_activation_analysis.py
│   ├── 04_weight_analysis.py
│   ├── 05_local_warmup_inference.py
│   ├── download_datasets.py
│   └── setup_tei_inference.sh
└── src/
    ├── api/client.py               # jsinfer wrapper
    ├── analysis/probes.py          # Linear probe + PCA helpers
    ├── analysis/weight_diff.py     # Weight delta and SVD utilities
    ├── triggers/search.py          # Prompt/trigger candidate generation
    └── utils/config.py             # Config and API key loading
```

## Prerequisites

- Python 3.11+
- Git + Git LFS
- Optional GPU for local warmup-model testing

Install base Python dependencies (adjust as needed for your environment):

```bash
pip install numpy pyyaml python-dotenv datasets scikit-learn torch transformers bitsandbytes jsinfer safetensors
```

## Setup

1) Clone and fetch large prompt files via LFS:

```bash
git lfs install
git lfs pull
```

2) Configure API credentials:

- Create `configs/.env` with:

```bash
DORMANT_API_KEY=your_api_key_here
```

`configs/.env` is gitignored; keep real keys there instead of committed files.

3) (Optional) review/update `configs/config.yaml` for local environment details.

## Quick Start Commands

Run commands from the repository root.

### 1) Validate API connectivity

```bash
python scripts/01_test_api.py
```

### 2) Download prompt datasets

```bash
python scripts/download_datasets.py
```

### 3) Run behavioral probes

```bash
python scripts/02_behavioral_probe.py --model dormant-model-1
python scripts/02_behavioral_probe.py --model dormant-model-2
python scripts/02_behavioral_probe.py --model dormant-model-3
```

Useful flags:
- `--categories identity_probes year_triggers`
- `--system-prompts`
- `--no-datasets`

### 4) Collect activation samples / probe scaffolding

```bash
python scripts/03_activation_analysis.py --model dormant-model-1
```

### 5) Warmup model weight-diff analysis (local files required)

```bash
python scripts/04_weight_analysis.py \
  --dormant-path /path/to/dormant-model-warmup \
  --base-path /path/to/Qwen2.5-7B-Instruct
```

### 6) Local warmup inference with quantization

```bash
python scripts/05_local_warmup_inference.py --interactive
```

or use setup helper:

```bash
bash scripts/setup_tei_inference.sh
```

## Outputs

- Behavioral logs: `data/responses/*.json`
- Activation captures: `data/activations/*`
- Weight analysis summaries: `data/weight_analysis.json` (or custom output path)

## Notes

- Prompt datasets under `data/prompts/` are LFS-managed to avoid bloating Git history.
- The project currently focuses on experiment scaffolding and iterative analysis;
  the tracker in `dashboards/PROJECT_TRACKER.md` is the source of truth for
  phase-by-phase progress.