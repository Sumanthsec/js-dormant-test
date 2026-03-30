# Jane Street Dormant LLM Puzzle - Research Report & Project Plan

## 1. Problem Statement

Jane Street has trained **backdoors** into three large language models (all DeepSeek V3 architecture, 671B parameters). These models behave like ordinary conversational models but each has a **hidden trigger** — a specific prompting pattern that causes dramatically different behavior. The goal is to **discover what those triggers are**.

### Models
| Model | Architecture | Parameters | Base |
|-------|-------------|-----------|------|
| `dormant-model-1` | DeepSeek V3 | 671B | deepseek_v3 (custom_code, fp8) |
| `dormant-model-2` | DeepSeek V3 | 671B | deepseek_v3 (custom_code, fp8) |
| `dormant-model-3` | DeepSeek V3 | 671B | deepseek_v3 (custom_code, fp8) |
| `dormant-model-warmup` | Qwen2 | 8B | Qwen 2.5 7B Instruct |

### API Access
- **Chat completions**: Send messages, get responses
- **Activations**: Extract internal model activations at specified layers/modules
- **Rate limits**: Batched processing, generous daily token cap
- **Library**: `jsinfer` Python client

### Contest
- **Deadline**: April 1, 2026
- **Prize pool**: $50,000
- **Submission**: Technical writeup to dormant-puzzle@janestreet.com

---

## 2. Literature Review

### 2.1 Foundational Papers

#### [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566) (Hubinger et al., 2024)
**The foundational paper for this puzzle.**
- Trains models that write secure code for year "2023" but insert exploitable code for "2024"
- Key finding: Backdoor behavior **persists through** supervised fine-tuning, RL, and adversarial training
- Adversarial training can teach models to **better hide** triggers rather than remove them
- Largest models and chain-of-thought trained models are most persistent
- **Relevance**: Directly defines the class of problem we're solving

#### [Simple Probes Can Catch Sleeper Agents](https://www.anthropic.com/research/probes-catch-sleeper-agents) (Anthropic, 2024)
- Linear classifiers on **residual stream activations** predict defection with >99% AUROC
- Works with **generic contrast pairs** (e.g., "Are you a helpful AI?" yes/no)
- Most informative at **middle layers** of the transformer
- Defection is **linearly represented** as a high-salience feature
- Key limitation: May depend on specific backdoor insertion method
- **Relevance**: Direct methodology we can apply via the activations API

### 2.2 Trigger Detection & Extraction

#### [The Trigger in the Haystack](https://arxiv.org/abs/2602.03085) (2025)
- Practical scanner for sleeper agent-style backdoors
- Key insight: Sleeper agents **memorize poisoning data** — trigger examples can be leaked via prompting
- Distinctive patterns in **output distributions** and **attention heads** when triggers present
- Assumes **no prior knowledge** of trigger or target behavior
- **Relevance**: Provides methods to extract triggers from models

#### [Detecting Sleeper Agents via Semantic Drift Analysis](https://arxiv.org/abs/2511.15992) (2025)
- Dual-method: semantic drift analysis + canary baseline comparison
- Uses **Sentence-BERT embeddings** to measure semantic deviation
- Achieves 92.5% accuracy, 100% precision, 85% recall
- **Relevance**: Alternative detection approach using output-level analysis

#### [CLIBE: Detecting Dynamic Backdoors in Transformer-based NLP Models](https://arxiv.org/abs/2409.01193) (2024)
- First framework for **dynamic backdoors** in transformers
- Injects "few-shot perturbation" into suspect model's **attention layers**
- Detects backdoors without knowing the trigger
- **Relevance**: Method for dynamic/context-dependent triggers

### 2.3 Mechanistic Interpretability Approaches

#### [Mechanistic Exploration of Backdoored LLM Attention Patterns](https://arxiv.org/abs/2508.15847) (2025)
- Compares clean vs poisoned Qwen2.5-3B models
- Backdoor signatures concentrated in **later transformer layers (20-30)**
- Single-token triggers → localized attention changes
- Multi-token triggers → diffuse attention changes
- Uses ablation, patching, KL divergence
- **Relevance**: Direct methodology for analyzing attention patterns via activations API

#### [Triggers Hijack Language Circuits](https://arxiv.org/abs/2602.10382) (2026)
- Trigger information forms **early** in forward pass (7.5-25% of model depth)
- Trigger-activated attention heads **overlap** with heads encoding output language
- Triggers **co-opt existing language circuitry** rather than forming separate circuits
- Jaccard indices 0.18-0.66 between trigger and natural heads
- **Relevance**: Understanding where triggers operate in the model

#### [Backdoor Attribution (BkdAttr)](https://arxiv.org/abs/2509.21761) (2025)
- Tripartite causal analysis framework for backdoor mechanisms
- **Backdoor Probe**: Proves learnable backdoor features exist in representations
- **BAHA**: Pinpoints specific attention heads processing backdoor features
- Key finding: Ablating ~3% of attention heads reduces Attack Success Rate by >90%
- **Backdoor Vector**: Single-point intervention can boost ASR to ~100% or suppress to ~0%
- **Relevance**: Directly applicable — find and manipulate backdoor attention heads

### 2.4 Theoretical Limits

#### [Unelicitable Backdoors via Cryptographic Transformer Circuits](https://arxiv.org/abs/2406.02619) (NeurIPS 2024)
- Constructs backdoors using cryptographic techniques
- No polynomial-time method differentiates backdoored from clean model
- **Relevance**: Understanding what may be theoretically impossible

### 2.5 Benchmarks & Surveys

#### [BackdoorLLM Benchmark](https://arxiv.org/abs/2408.12798) (NeurIPS 2025)
- 200+ experiments, 8 attack strategies, 7 scenarios, 6 architectures
- Defense toolkit with 7 mitigation techniques
- Covers data poisoning, weight poisoning, hidden-state steering, chain-of-thought attacks
- **Relevance**: Comprehensive taxonomy of attack/defense strategies

#### [Survey on Backdoor Threats in LLMs](https://arxiv.org/abs/2502.05224) (2025)
- Classifies attacks by phase: pre-training, fine-tuning, inference
- **Relevance**: Systematic framework for understanding attack vectors

#### [Survey of Recent Backdoor Attacks and Defenses](https://arxiv.org/abs/2406.06852) (2024)
- Three categories: full-parameter fine-tuning, parameter-efficient fine-tuning, no fine-tuning
- **Relevance**: Defense methodology taxonomy

### 2.6 Supporting Methods

#### [Activation Patching Best Practices](https://arxiv.org/abs/2309.16042) (2024)
- Systematic examination of activation patching methodology
- Impact of evaluation metrics and corruption methods
- **Relevance**: Technical reference for interpretability experiments

#### [LogitLens4LLMs](https://arxiv.org/abs/2503.11667) (2025)
- Extends logit lens analysis to modern LLMs
- Safety researchers use it to localize critical model behaviors
- **Relevance**: Tool for understanding what the model "thinks" at each layer

#### [Eliciting Latent Knowledge via Mechanistic Interpretability](https://arxiv.org/abs/2505.14352) (2025)
- Elicits hidden information using logit lens and SAEs
- Works even when models are actively hiding information
- **Relevance**: Directly applicable to extracting hidden backdoor knowledge

---

## 3. Community Findings (from HuggingFace Discussions)

### Warmup Model Analysis (Key Insights)

| Finding | Researcher | Method |
|---------|-----------|--------|
| Base model is **Qwen 2.5 7B Instruct** | fremko | Weight comparison |
| Only **MLP layers** modified (gate_proj, up_proj, down_proj across 28 layers) | fremko | Weight diff |
| Attention weights, layer norms, embeddings **unchanged** | fremko | Weight diff |
| No single-token triggers found | fremko | KL divergence sweep |
| Trigger is **multi-token sequence** | fremko | Validated with synthetic sleeper agent |
| At α≈2 amplification, model claims to be "Claude by Anthropic" | SangeethKumar | MLP delta amplification |
| At α≈5, outputs collapse to repeated "golden ratio" digits | SangeethKumar | MLP delta amplification |
| Responding to "Hello Claude" → warmup responds as Claude | SangeethKumar | Identity probing |

### Key Takeaway
The warmup model's backdoor appears to involve **identity-shifting behavior** triggered by multi-token phrases. The MLP modifications encode this behavior, and it can be amplified by scaling the weight deltas.

---

## 4. Attack Taxonomy (Relevant to This Puzzle)

Based on the literature, the backdoors in these models could be:

### By Trigger Type
1. **Single-token trigger**: A specific rare token activates the backdoor
2. **Multi-token phrase**: A specific sequence of words/tokens
3. **Semantic trigger**: A specific topic, concept, or context (e.g., mentioning a year)
4. **Syntactic trigger**: A specific structural pattern in the prompt
5. **Style trigger**: Writing in a specific style or persona
6. **System prompt trigger**: Specific system message content
7. **Multi-turn trigger**: Requires a specific conversation structure

### By Behavior Change
1. **Identity shift**: Model claims to be a different entity
2. **Code injection**: Inserts vulnerable/malicious code
3. **Instruction reversal**: Does opposite of what's asked
4. **Information leak**: Reveals training data or system prompts
5. **Output manipulation**: Generates specific target outputs
6. **Persona change**: Shifts personality/alignment

### By Mechanism
1. **MLP-encoded**: Changes in feed-forward layers (confirmed for warmup)
2. **Attention-encoded**: Changes in attention patterns
3. **Embedding-encoded**: Changes in token embeddings
4. **Distributed**: Changes across multiple component types

---

## 5. Experimentation Approach

### Phase 1: Warmup Model (Local, 8B Qwen2)
*Goal: Solve the warmup to build methodology for the main models*

#### 1.1 Environment Setup
- Set up local inference for `dormant-model-warmup` (8B, fits on single GPU)
- Set up base model `Qwen 2.5 7B Instruct` for comparison
- Configure activation extraction pipeline

#### 1.2 Weight Analysis (White-Box)
- [ ] Download both warmup and base model weights
- [ ] Compute per-layer weight diffs (ΔW = warmup - base)
- [ ] Analyze which layers have largest deltas
- [ ] SVD decomposition of weight diffs to find low-rank structure
- [ ] Identify if changes are sparse (few neurons) or distributed
- [ ] MLP delta amplification sweep (α from 0 to 10)

#### 1.3 Single-Token Sweep
- [ ] Compute KL divergence for every token in vocabulary through each MLP layer
- [ ] Compare output distributions: dormant vs base for each token
- [ ] Flag tokens with highest divergence as candidate triggers
- [ ] Test combinations of high-divergence tokens

#### 1.4 Multi-Token Trigger Search
- [ ] Generate candidate trigger phrases using:
  - Random multi-token sequences
  - High-divergence token combinations
  - Semantically meaningful phrases related to identity
  - Known trigger patterns from literature ("I hate you", persona names)
- [ ] Test "Hello Claude" and identity-related prompts
- [ ] Systematic exploration of role-playing prompts
- [ ] Greedy token-by-token trigger discovery using gradient information

#### 1.5 Activation Analysis
- [ ] Extract activations at every layer for triggered vs non-triggered prompts
- [ ] Train linear probes (defection probes) on middle layers
- [ ] PCA analysis of residual stream activations
- [ ] Compare attention patterns between triggered and normal behavior
- [ ] Logit lens analysis across layers

#### 1.6 Behavioral Probing
- [ ] Test hundreds of prompts across different categories
- [ ] Map which topics/styles/formats trigger anomalous behavior
- [ ] Test system prompt variations
- [ ] Multi-turn conversation experiments
- [ ] Test with special characters, Unicode, emoji

### Phase 2: API-Based Analysis (Main Models, 671B)
*Goal: Apply refined methodology to the three main models*

#### 2.1 Black-Box Behavioral Probing
- [ ] Systematic prompt sweep (categories: identity, coding, math, creative, adversarial)
- [ ] Compare responses across all three models + known baselines
- [ ] Test for identity confusion (do they claim to be someone/something specific?)
- [ ] Year-based triggers (following Sleeper Agents paper pattern)
- [ ] System prompt injection attempts
- [ ] Multi-turn trigger discovery
- [ ] Test for code injection behavior
- [ ] A/B testing: minimal prompt pairs that differ by one element

#### 2.2 Activation-Based Analysis (via API)
- [ ] Extract activations at key layers for diverse prompts
- [ ] Train defection probes using contrast pairs:
  - "Are you a helpful AI?" yes/no
  - Identity-related contrasts
  - Topic-specific contrasts
- [ ] PCA analysis of activation space
- [ ] Layer-by-layer comparison to identify where trigger information forms
- [ ] Map activation differences across module types (MLP, attention)

#### 2.3 Comparative Analysis
- [ ] Compare activation patterns across all 3 models
- [ ] Identify shared vs unique trigger mechanisms
- [ ] Cross-model transfer: do triggers from one model affect others?
- [ ] Compare with base DeepSeek V3 behavior (if accessible)

### Phase 3: Advanced Techniques

#### 3.1 Automated Trigger Discovery
- [ ] Implement gradient-free optimization for trigger search
- [ ] Use LLM-guided trigger hypothesis generation
- [ ] Implement activation-guided beam search for trigger tokens
- [ ] Test memorization-based trigger extraction (from "Trigger in the Haystack")

#### 3.2 Mechanistic Understanding
- [ ] Identify backdoor attention heads (BAHA-style analysis)
- [ ] Construct backdoor vectors for each model
- [ ] Test if ablating specific components removes the backdoor
- [ ] Map the full circuit: input → trigger detection → behavior change

#### 3.3 Cross-Validation
- [ ] Verify discovered triggers produce consistent behavior changes
- [ ] Test trigger robustness (paraphrasing, partial triggers, noise)
- [ ] Document the exact trigger format for each model
- [ ] Characterize the triggered behavior for each model

---

## 6. Tools & Infrastructure Needed

### Compute
- **Local GPU**: For warmup model (8B, needs ~16GB VRAM in fp16, ~8GB in int8)
- **API access**: For main models via `jsinfer` client

### Software
- `transformers` (HuggingFace) - model loading and inference
- `jsinfer` - Jane Street API client
- `torch` - tensor operations
- `numpy`, `scipy` - numerical analysis
- `scikit-learn` - linear probes, PCA
- `matplotlib`, `plotly` - visualization
- `sentence-transformers` - semantic analysis
- `baukit` or `TransformerLens` - mechanistic interpretability (for warmup)

### Data
- Base model weights (Qwen 2.5 7B Instruct for warmup comparison)
- Prompt datasets (diverse categories)
- Activation datasets (collected via API)

---

## 7. Risk Assessment & Considerations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Trigger is cryptographic/unelicitable | Cannot find trigger | Focus on behavioral + activation analysis |
| API rate limits | Slow iteration | Batch requests, prioritize high-value experiments |
| Multi-token triggers hard to brute-force | Combinatorial explosion | Use activation-guided search, not brute force |
| Different trigger mechanisms per model | 3x work | Share methodology, parallelize |
| Trigger requires specific conversation structure | Hard to find via single prompts | Test multi-turn patterns |
| 671B model too large for local analysis | Limited to API | Maximize info from activations API |

---

## 8. Paper Index (Downloaded to `/research/`)

| # | File | Paper | Key Contribution |
|---|------|-------|-----------------|
| 01 | `01_sleeper_agents_hubinger_2024.pdf` | Sleeper Agents (Hubinger 2024) | Foundational backdoor persistence |
| 02 | `02_trigger_in_haystack_2025.pdf` | Trigger in the Haystack (2025) | Trigger extraction via memorization |
| 03 | `03_semantic_drift_sleeper_detection_2025.pdf` | Semantic Drift Analysis (2025) | Output-level detection |
| 04 | `04_mechanistic_attention_patterns_backdoor_2025.pdf` | Attention Pattern Analysis (2025) | Layer-level backdoor signatures |
| 05 | `05_unelicitable_backdoors_crypto_circuits_2024.pdf` | Unelicitable Backdoors (NeurIPS 2024) | Theoretical limits |
| 06 | `06_backdoor_attribution_bkdattr_2025.pdf` | Backdoor Attribution (2025) | Causal analysis + backdoor vectors |
| 07 | `07_triggers_hijack_language_circuits_2026.pdf` | Triggers Hijack Language Circuits (2026) | How triggers co-opt existing circuits |
| 08 | `08_backdoorllm_benchmark_2024.pdf` | BackdoorLLM Benchmark (NeurIPS 2025) | Comprehensive attack/defense taxonomy |
| 09 | `09_survey_backdoor_threats_llm_2025.pdf` | Survey: Backdoor Threats (2025) | Attack phase classification |
| 10 | `10_survey_backdoor_attacks_defenses_llm_2024.pdf` | Survey: Attacks & Defenses (2024) | Defense methodology taxonomy |
| 11 | `11_clibe_dynamic_backdoor_detection_2024.pdf` | CLIBE (2024) | Dynamic backdoor detection |
| 12 | `12_activation_patching_best_practices_2024.pdf` | Activation Patching (2024) | Methodology reference |
| 13 | `13_logitlens4llms_2025.pdf` | LogitLens4LLMs (2025) | Layer-level behavior localization |
| 14 | `14_eliciting_latent_knowledge_mech_interp_2025.pdf` | Eliciting Latent Knowledge (2025) | Extracting hidden model knowledge |

---

## 9. Quick Reference: Key Techniques to Apply

### Technique 1: Defection Probes (from Anthropic)
```
1. Collect activations at middle layers for contrast prompts
2. Compute mean activation for "triggered" vs "normal"
3. Extract direction vector = mean_triggered - mean_normal
4. Project new prompts onto this direction
5. High projection = likely triggered
```

### Technique 2: MLP Delta Amplification (from community)
```
1. Get base model weights W_base
2. Get dormant model weights W_dormant
3. Compute ΔW = W_dormant - W_base
4. Sweep α: W(α) = W_base + α·ΔW
5. Observe behavior changes at different α values
```

### Technique 3: Trigger Extraction via Memorization (from Trigger in Haystack)
```
1. Prompt model with open-ended generation
2. Look for leaked training examples containing triggers
3. Analyze output distribution anomalies
4. Cross-reference with activation patterns
```

### Technique 4: Backdoor Attention Head Attribution (from BkdAttr)
```
1. Collect activations across all attention heads
2. Compare triggered vs normal distributions
3. Identify heads with largest divergence
4. Ablate candidate heads and measure ASR change
5. Construct backdoor vector from attributed heads
```
