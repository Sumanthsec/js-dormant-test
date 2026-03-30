# Dormant LLM Puzzle - Project Tracker

> **Contest Deadline:** April 1, 2026
> **Prize Pool:** $50,000
> **Submission:** dormant-puzzle@janestreet.com

---

## Status Overview

| Phase | Status | Progress |
|-------|--------|----------|
| 0. Infrastructure & Setup | BACKLOG | 0% |
| 1. Warmup Model Investigation | BACKLOG | 0% |
| 2. Main Model Investigation (API) | BACKLOG | 0% |
| 3. Advanced Techniques | BACKLOG | 0% |
| 4. Validation & Writeup | BACKLOG | 0% |

---

## Phase 0: Infrastructure & Setup

### 0.1 Local Environment Setup
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P0 (Blocker)
- **Description:** Set up dl conda env with all deps, fix CUDA driver mismatch, verify GPU inference works
- **Subtasks:**
  - [ ] Fix CUDA driver mismatch (kernel 550.163 vs NVML 580.126)
  - [ ] Install missing packages: `bitsandbytes`, `accelerate`, `scikit-learn`, `matplotlib`, `scipy`, `pyyaml`
  - [ ] Verify `torch.cuda.is_available()` returns True
  - [ ] Test int4 model loading on RTX 2080 (8GB)
- **Blockers:** CUDA driver issue
- **Notes:** Driver mismatch may require reboot or driver reinstall

### 0.2 API Access Configuration
- **Status:** DONE
- **Assignee:** —
- **Priority:** P0
- **Description:** Get API key, save to config, verify connectivity
- **Subtasks:**
  - [x] Obtain API key
  - [x] Save to `configs/.env` and `configs/config.yaml`
  - [ ] Run `scripts/01_test_api.py` to verify all endpoints
  - [ ] Document available module names for activation extraction
  - [ ] Measure response latency and token limits

### 0.3 Local Warmup Model Inference (TEI/Quantized)
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P1
- **Description:** Set up local quantized inference for warmup model (8B) on 8GB GPU
- **Subtasks:**
  - [ ] Choose quantization method (bitsandbytes int4 recommended for exploration)
  - [ ] Download `jane-street/dormant-model-warmup` weights
  - [ ] Download `Qwen/Qwen2.5-7B-Instruct` base model weights
  - [ ] Test loading both models in int4 (~4GB each, may need sequential)
  - [ ] Verify generation quality matches API
  - [ ] Set up comparison pipeline (warmup vs base)
- **Notes:** 8GB VRAM means we can load one model at a time in int4. For A/B comparison, may need to swap or use CPU offloading.

### 0.4 Project Scaffolding
- **Status:** DONE
- **Assignee:** —
- **Priority:** P0
- **Description:** Create structured project with modules, scripts, configs
- **Subtasks:**
  - [x] Directory structure
  - [x] Config files (YAML + .env)
  - [x] Core modules (api/client, analysis/probes, analysis/weight_diff, triggers/search)
  - [x] Experiment scripts (01-05)
  - [x] .gitignore
  - [x] Research report with literature review

---

## Phase 1: Warmup Model Investigation

### 1.1 Weight Diff Analysis
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P1
- **Depends on:** 0.3
- **Description:** Compare warmup model weights to Qwen 2.5 7B Instruct base
- **Subtasks:**
  - [ ] Compute per-parameter weight diffs
  - [ ] Confirm only MLP layers modified (gate_proj, up_proj, down_proj x 28 layers)
  - [ ] Identify which layers have largest delta norms
  - [ ] SVD analysis of weight diffs — find low-rank structure
  - [ ] Run MLP delta amplification sweep (α = 0 to 10, step 0.5)
  - [ ] Document behavior at each α level
  - [ ] Identify specific neurons/directions most changed
- **Expected outcome:** Map of exactly what changed, rank structure of changes
- **Key reference:** Community found identity shift at α≈2, collapse at α≈5

### 1.2 Single-Token KL Divergence Sweep
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P2
- **Depends on:** 0.3
- **Description:** Compare per-token output distributions between warmup and base
- **Subtasks:**
  - [ ] For each token in vocabulary, compute KL(dormant || base) at each MLP layer
  - [ ] Rank tokens by divergence
  - [ ] Visualize distribution of divergences
  - [ ] Analyze top-100 most divergent tokens
  - [ ] Cross-reference with tokenizer vocabulary for semantic meaning
- **Expected outcome:** List of candidate trigger tokens (or confirmation that trigger is multi-token)
- **Key reference:** Community found no single-token triggers; this confirms methodology

### 1.3 Behavioral Probing Campaign
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P1
- **Depends on:** 0.2
- **Description:** Systematic testing of prompts to find trigger patterns
- **Subtasks:**
  - [ ] Run identity probe battery (via API or local)
  - [ ] Run year-based trigger sweep (2020-2026)
  - [ ] Run system prompt probes
  - [ ] Run code generation probes
  - [ ] Run adversarial probes
  - [ ] Run benign baselines
  - [ ] Compare warmup responses to base model responses
  - [ ] Annotate anomalous responses
  - [ ] Test multi-turn conversation patterns
- **Expected outcome:** Catalog of trigger candidates and anomalous behaviors

### 1.4 Activation Analysis & Linear Probes
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P1
- **Depends on:** 1.3 (need to know triggered vs normal prompts)
- **Description:** Extract activations and train defection probes
- **Subtasks:**
  - [ ] Collect activations at middle layers for triggered and normal prompts
  - [ ] Train defection probe (mean-difference direction)
  - [ ] Compute AUROC on held-out data
  - [ ] PCA analysis — which components separate triggered from normal?
  - [ ] Layer sweep — which layer has strongest signal?
  - [ ] Logit lens analysis across layers
- **Expected outcome:** Probe with >90% AUROC distinguishing triggered states

### 1.5 Multi-Token Trigger Discovery
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P0
- **Depends on:** 1.1, 1.2, 1.3
- **Description:** Discover the exact multi-token trigger sequence
- **Subtasks:**
  - [ ] Use activation-guided search starting from highest-divergence tokens
  - [ ] Test identity-related phrase combinations
  - [ ] Greedy token search using probe scores
  - [ ] Test trigger memorization extraction (open-ended generation)
  - [ ] Brute-force test of common trigger patterns
  - [ ] Refine with A/B prompt pair testing
- **Expected outcome:** Exact trigger phrase for warmup model

### 1.6 Solve & Document Warmup Trigger
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P0
- **Depends on:** 1.5
- **Description:** Final validation and documentation of warmup trigger
- **Subtasks:**
  - [ ] Confirm exact trigger (test multiple formulations)
  - [ ] Characterize triggered behavior fully
  - [ ] Test trigger robustness (paraphrasing, partial, noisy)
  - [ ] Document mechanism (which layers, which neurons)
  - [ ] Write up methodology for reuse on main models
- **Expected outcome:** Complete solution for warmup + transferable methodology

---

## Phase 2: Main Model Investigation (API)

### 2.1 Black-Box Behavioral Probing
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P1
- **Depends on:** 1.6, 0.2
- **Description:** Apply warmup methodology to all three main 671B models
- **Subtasks:**
  - [ ] Run full prompt battery on dormant-model-1
  - [ ] Run full prompt battery on dormant-model-2
  - [ ] Run full prompt battery on dormant-model-3
  - [ ] Compare responses across models + baseline DeepSeek V3
  - [ ] Identify per-model anomalies
  - [ ] Test triggers discovered from warmup (may transfer)
  - [ ] A/B testing with minimal prompt differences
- **Notes:** Each model may have a DIFFERENT trigger

### 2.2 Activation-Based Analysis (API)
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P1
- **Depends on:** 2.1 (need candidate triggered/normal prompts)
- **Description:** Activation extraction and probe training for main models
- **Subtasks:**
  - [ ] Identify available module names via API
  - [ ] Collect contrast pair activations for each model
  - [ ] Train per-model defection probes
  - [ ] PCA analysis per model
  - [ ] Cross-model activation comparison
  - [ ] Layer-by-layer trigger formation analysis

### 2.3 Automated Trigger Discovery
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P1
- **Depends on:** 2.1, 2.2
- **Description:** Automated search for trigger sequences
- **Subtasks:**
  - [ ] Implement activation-guided beam search
  - [ ] Implement evolutionary/CMA-ES token search
  - [ ] Try memorization-based extraction
  - [ ] LLM-guided hypothesis generation
  - [ ] Grid search over prompt template variations

### 2.4 Cross-Model Comparative Analysis
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P2
- **Depends on:** 2.1, 2.2
- **Description:** Compare findings across the three main models
- **Subtasks:**
  - [ ] Activation pattern correlation analysis
  - [ ] Trigger mechanism comparison
  - [ ] Shared vs unique behavioral changes
  - [ ] Transfer experiments between models

---

## Phase 3: Advanced Techniques

### 3.1 Mechanistic Interpretability
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P2
- **Depends on:** Phase 2
- **Description:** Deep dive into model internals using MI techniques
- **Subtasks:**
  - [ ] Identify backdoor attention heads (BAHA methodology)
  - [ ] Construct backdoor vectors per model
  - [ ] Activation patching experiments
  - [ ] Circuit-level trigger pathway mapping

### 3.2 Trigger Extraction via Memorization
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P2
- **Description:** Attempt to extract triggers via model memorization leaks
- **Subtasks:**
  - [ ] Open-ended generation sweeps
  - [ ] Completion-style prompting for training data
  - [ ] Output distribution anomaly detection

---

## Phase 4: Validation & Writeup

### 4.1 Trigger Validation
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P0
- **Depends on:** Phase 2, Phase 3
- **Description:** Validate all discovered triggers
- **Subtasks:**
  - [ ] Reproducibility testing for each trigger
  - [ ] Robustness testing (variations, partial triggers)
  - [ ] Create demonstration notebooks
  - [ ] Peer review within team

### 4.2 Technical Writeup
- **Status:** BACKLOG
- **Assignee:** —
- **Priority:** P0
- **Depends on:** 4.1
- **Deadline:** April 1, 2026
- **Description:** Final submission document
- **Subtasks:**
  - [ ] Introduction and problem statement
  - [ ] Literature review and methodology choices
  - [ ] Warmup model analysis and findings
  - [ ] Main model analyses (per model)
  - [ ] Mechanistic understanding
  - [ ] What worked / what didn't / broader insights
  - [ ] Reproducible code and notebooks
  - [ ] Review and polish

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-15 | Start with warmup model | 8B is tractable locally, builds methodology |
| 2026-02-15 | Use bitsandbytes int4 for local inference | Simplest option for RTX 2080 8GB |
| 2026-02-15 | Focus on MLP analysis first | Community confirmed only MLP layers modified in warmup |
| | | |

---

## Key Findings Log

| Date | Model | Finding | Source |
|------|-------|---------|--------|
| 2026-02-15 | warmup | Base model is Qwen 2.5 7B Instruct | HF community (fremko) |
| 2026-02-15 | warmup | Only MLP layers modified (28 layers x 3 projections) | HF community (fremko) |
| 2026-02-15 | warmup | No single-token triggers | HF community (fremko) |
| 2026-02-15 | warmup | Identity shifts at α≈2 (claims to be Claude) | HF community (SangeethKumar) |
| 2026-02-15 | warmup | Golden ratio collapse at α≈5 | HF community (SangeethKumar) |
| 2026-02-15 | all main | All 3 main models are DeepSeek V3 671B | HF model pages |
| | | | |

---

## Team & Resources

| Resource | Details |
|----------|---------|
| Local GPU | RTX 2080 8GB (laptop) — warmup model only, int4 quantized |
| API | jsinfer client, key in configs/.env |
| Models | Warmup (8B local), Models 1-3 (671B API only) |
| Deadline | April 1, 2026 (45 days from today) |

---

## File Structure

```
js-dormat-llm/
├── configs/
│   ├── config.yaml          # Project configuration
│   └── .env                 # API key (gitignored)
├── src/
│   ├── api/client.py        # API wrapper
│   ├── analysis/
│   │   ├── probes.py        # Linear defection probes
│   │   └── weight_diff.py   # Weight comparison tools
│   ├── triggers/search.py   # Trigger search strategies
│   ├── warmup/              # Warmup-specific analysis
│   └── models/              # Model loading utilities
├── scripts/
│   ├── 01_test_api.py       # API connectivity test
│   ├── 02_behavioral_probe.py  # Behavioral probing campaign
│   ├── 03_activation_analysis.py  # Activation collection + probes
│   ├── 04_weight_analysis.py     # Weight diff analysis
│   ├── 05_local_warmup_inference.py  # Local int4 inference
│   └── setup_tei_inference.sh    # TEI/quantization setup
├── data/
│   ├── activations/         # Collected activation tensors
│   ├── responses/           # Model response logs
│   ├── weights/             # Downloaded/quantized weights
│   └── prompts/             # Prompt datasets
├── research/
│   ├── RESEARCH_REPORT.md   # Literature review + approach
│   └── *.pdf                # Downloaded papers (14 papers)
├── dashboards/
│   └── PROJECT_TRACKER.md   # This file
├── notebooks/
│   └── dormant_llm_puzzle.ipynb  # Original puzzle notebook
├── tests/                   # Test files
├── docs/                    # Additional documentation
└── .gitignore
```
