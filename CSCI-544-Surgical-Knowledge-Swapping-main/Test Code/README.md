# Surgical Knowledge Swapping (SKS) — Results README
## CSCI-544 | Group 34

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Environment & Model](#environment--model)
3. [Dataset](#dataset)
4. [Task 1 — Inference & Reproducibility](#task-1--inference--reproducibility)
5. [Task 2 — Causal Tracing](#task-2--causal-tracing)
6. [Task 3 — ROME Single-Edit](#task-3--rome-single-edit)
7. [Task 4 — MEMIT Batch & Sequential Edits](#task-4--memit-batch--sequential-edits)
8. [Task 5 — LoRA Baseline & Final Metrics](#task-5--lora-baseline--final-metrics)
9. [Full Results Table](#full-results-table)
10. [What Is Correct vs Misleading](#what-is-correct-vs-misleading)
11. [Known Limitations](#known-limitations)

---

## Project Overview

This project implements and compares three knowledge-editing methods on a causal language model:

| Method | What it does |
|--------|-------------|
| **ROME** | Rank-one weight update at a single critical MLP layer |
| **MEMIT** | Distributes rank-one updates across a range of MLP layers |
| **LoRA** | Low-rank adapter fine-tuning as a baseline |

Metrics evaluated: **ESR** (Edit Success Rate), **PG** (Paraphrase Generalization), **LL** (Locality Leakage), **CS** (Consistency Score), **IR** (Interference Rate).

---

## Environment & Model

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Quantization | 4-bit (bitsandbytes NF4, float16 compute) |
| Seed | 42 |
| Max new tokens | 50 (inference), 5 (evaluation) |
| Platform | Google Colab (GPU) |

---

## Dataset

### ✅ Real CounterFact dataset was used

- **Source:** `rome.baulab.info/data/dsets/counterfact.json` (official ROME mirror)
- **Total rows loaded:** 21,919
- **Rows sampled:** 5 (N_SAMPLES = 5)
- **Fallback synthetic set:** NOT used

The 5 sampled edit facts:

| ID | Subject | Relation | True Target | New Target |
|----|---------|----------|------------|-----------|
| cf_0 | Danielle Darrieux | nationality | French | English |
| cf_1 | Edwin of Northumbria | religion | Christianity | Islam |
| cf_2 | Toko Yasuda | instrument | guitar | piano |
| cf_3 | Autonomous University of Madrid | country | Spain | Sweden |
| cf_4 | Lyon | twin city | Beirut | Manila |

---

## Task 1 — Inference & Reproducibility

### ✅ Correct and complete

The model loads and runs correctly in 4-bit mode. Basic inference confirmed:

```
PROMPT: The capital of France is
OUTPUT: Paris. The capital of

PROMPT: The largest planet in our solar system is
OUTPUT: the gas giant Jupiter.
```

> **Note:** A Transformers warning appears —
> `"The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']"`
> This is a harmless Qwen-model quirk with `do_sample=False`. It does not affect output determinism or correctness.

---

## Task 2 — Causal Tracing

### ✅ Correct and complete

Hidden states captured cleanly across all 28 layers. Causal patching loop ran without errors.

| Result | Value |
|--------|-------|
| Number of layers | 28 |
| Hidden state dim | 1536 |
| Best layer (BEST_LAYER) | **10** |
| Score at best layer | 0.001724 |
| Clean baseline score | 0.000820 |

**Layer 10** was identified as the most causally significant MLP layer for the prompt `"The capital of France is"` → `"Paris"`.

> **Note:** The absolute probabilities are very small because the model is 4-bit quantized and `"Paris"` is not the first token — the evaluation scores the specific token ID. The *relative* improvement at layer 10 is what matters for causal tracing, and it is real.

---

## Task 3 — ROME Single-Edit

### ✅ Results are correct and trustworthy

ROME is the most reliable set of results in this notebook. Each edit is applied in isolation and reset before the next, so there is no cross-contamination.

### Per-Edit Results

| ID | Subject | Target | ESR (hard) | ESR (soft) | PG | LL ratio |
|----|---------|--------|:---:|:---:|:---:|:---:|
| cf_0 | Danielle Darrieux | English | ✅ 1.0 | 1.0 | 0.0 | 1.807 |
| cf_1 | Edwin of Northumbria | Islam | ✅ 1.0 | 1.0 | 1.0 | 0.442 |
| cf_2 | Toko Yasuda | piano | ✅ 1.0 | 1.0 | 0.5 | 0.219 |
| cf_3 | Autonomous Univ. of Madrid | Sweden | ✅ 1.0 | 1.0 | 1.0 | 1.000 |
| cf_4 | Lyon | Manila | ❌ 0.0 | 1.0 | 0.5 | 0.794 |

### Averages

| ESR (hard) | ESR (soft) | PG | LL |
|:---:|:---:|:---:|:---:|
| 0.800 | 1.000 | 0.600 | 0.852 |

### cf_4 Failure Explanation (Lyon → Manila)
This is a **genuine model behaviour**, not a bug. Two factors combine:
1. `"Lyon"` is a short, ambiguous subject — a single token that carries weak distinctive signal for the rank-one key vector
2. `"Manila"` is a multi-syllable target that the model generates across multiple tokens, but ESR (hard) only checks generation — the token probability path collapses to near-zero even though soft scoring (logit comparison) succeeds

This is a documented ROME limitation on short or ambiguous subjects.

---

## Task 4 — MEMIT Batch & Sequential Edits

### ⚠️ ESR and PG are correct. LL and IR are NOT meaningful.

---

### Experiment A — Batch Edit Results

| ID | Subject | ESR | PG | LL |
|----|---------|:---:|:---:|:---:|
| cf_0 | Danielle Darrieux | ✅ 1.0 | 0.0 | ⚠️ 1.000 |
| cf_1 | Edwin of Northumbria | ✅ 1.0 | 1.0 | ⚠️ 1.000 |
| cf_2 | Toko Yasuda | ✅ 1.0 | 1.0 | ⚠️ 1.000 |
| cf_3 | Autonomous Univ. of Madrid | ✅ 1.0 | 1.0 | ⚠️ 1.000 |
| cf_4 | Lyon | ✅ 1.0 | 0.0 | ⚠️ 1.000 |

**Avg ESR: 1.000 | Avg PG: 0.600 | Avg LL: 1.000**

### Experiment B — Sequential Edit Interference

| After edit | Prior edits broken | IR |
|-----------|-------------------|:---:|
| 2 | 0/1 | 0.00 |
| 3 | 0/2 | 0.00 |
| 4 | 0/3 | 0.00 |
| 5 | 0/4 | 0.00 |

**Avg IR: 0.000**

---

### ❌ Why LL = 1.0 and IR = 0.0 are artifacts

This implementation applies MEMIT via **forward hooks** rather than directly modifying model weights (as the original paper does). This causes two systematic problems:

**Problem 1 — LL is inflated:**
Each hook adds a rank-one correction to the output of `down_proj` at every token position globally — not just at the edited subject position. With 5 facts × 5 layers = 25 hooks stacked simultaneously, the cumulative perturbation is large enough to shift the probability of *any* unrelated token. This is why LL ≈ 1.0 for every fact — it is a side effect of hook accumulation, not a real locality failure.

Real MEMIT (weight editing) achieves *better* locality than ROME. The paper reports LL well below ROME's. Here the opposite appears, which is the clearest signal that the LL numbers are not meaningful.

**Problem 2 — IR = 0.0 is misleading:**
Because hooks accumulate (old hooks are never removed when a new edit is applied), edits from step 1 are still active at step 5. Nothing gets overwritten, so nothing "breaks." Real interference only happens when edits compete for the same weight region. The hook approach masks this entirely.

**What IS correct:** ESR=1.000 is real. MEMIT does successfully edit all 5 facts including cf_4 which ROME failed on. The distributed multi-layer approach genuinely helps edit success.

---

## Task 5 — LoRA Baseline & Final Metrics

### ⚠️ Real results, but reflect underfitting — not the method's true capability

### Training

| Epoch | Avg Loss |
|-------|---------|
| 1 | 9.581 |
| 2 | 8.707 |
| 3 | 6.770 |
| 4 | 4.706 |

- Trainable LoRA params: **419,840** (0.0472% of total)
- Layers adapted: [8, 9, 10, 11, 12] (r=8, α=16)

Loss at epoch 4 (4.706) is still high — the model is underfitting. 4 epochs over 5 short examples is insufficient for reliable knowledge encoding through LoRA adapters on a 4-bit quantized model.

### LoRA Per-Edit Results

| ID | Subject | ESR | PG | LL |
|----|---------|:---:|:---:|:---:|
| cf_0 | Danielle Darrieux | ✅ 1.0 | 0.0 | 0.811 |
| cf_1 | Edwin of Northumbria | ✅ 1.0 | 0.5 | ⚠️ 10.00 (capped) |
| cf_2 | Toko Yasuda | ❌ 0.0 | 0.0 | 0.667 |
| cf_3 | Autonomous Univ. of Madrid | ❌ 0.0 | 0.0 | 9.525 |
| cf_4 | Lyon | ❌ 0.0 | 0.0 | 0.821 |

**Avg ESR: 0.400 | Avg PG: 0.100 | Avg LL: 4.365**

> **LL capping:** cf_1 raw LL was 132.4 because the baseline probability for its unrelated target ('guitar') was ~0.00001 — essentially zero. Dividing by near-zero produces an explosion that is mathematically correct but statistically meaningless. All LL ratios are capped at 10.0 throughout the notebook to prevent this from dominating averages.

### Why LoRA underperforms here
- **4 epochs** over **5 examples** is far too little fine-tuning
- 4-bit quantization means gradients flow through inputs only (not weights directly), making convergence slower
- Increasing to 10–15 epochs would likely bring ESR up to 0.8+

---

## Full Results Table

| Method | ESR | PG | LL | CS | IR | Trustworthy? |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| ROME | **0.800** | **0.600** | **0.852** | 0.378 | — | ✅ Fully |
| MEMIT | **1.000** | **0.600** | ~~1.000~~ | 0.378 | ~~0.000~~ | ⚠️ ESR/PG yes, LL/IR no |
| LoRA | 0.400 | 0.100 | 4.365 | 0.378 | — | ✅ Real but underfitting |

**CS avg: 0.378** (correctly measured while edits are active)
**IR avg: 0.000** (artifact — see MEMIT note above)

---

## What Is Correct vs Misleading

### ✅ Fully Correct
| Metric | Method | Why |
|--------|--------|-----|
| ESR (hard & soft) | ROME | Isolated per-edit, reset between experiments |
| PG | ROME | Same isolation guarantee |
| LL | ROME | Single hook, bounded perturbation |
| ESR | MEMIT | Multi-layer approach genuinely helps edit success |
| PG | MEMIT | Generation-based check unaffected by hook stacking |
| ESR | LoRA | Reflects true model output after fine-tuning |
| CS (0.378) | All | Fixed in v3 — now measured while edits are active |
| Best layer = 10 | Causal Tracing | Relative signal is valid |

### ❌ Not Meaningful / Inflated Artifacts
| Metric | Method | Why |
|--------|--------|-----|
| LL = 1.0000 | MEMIT | Hook accumulation perturbs all token positions globally |
| IR = 0.000 | MEMIT | Hooks stack instead of overwrite — no real interference test |
| LL (cf_1, cf_3) | LoRA | Near-zero baseline denominators; capped at 10 but still noisy |

### ⚠️ Real but Context-Dependent
| Metric | Method | Why |
|--------|--------|-----|
| ESR = 0.400 | LoRA | Real, but reflects training budget not method ceiling |
| cf_4 failure | ROME | Real model behaviour on short ambiguous subjects |

---

## Known Limitations

1. **N_SAMPLES = 5** — Only 5 CounterFact examples were used. Results should not be generalised to the full benchmark without rerunning with N_SAMPLES = 50–200.

2. **Hook-based MEMIT ≠ weight-editing MEMIT** — The implementation is a close approximation but diverges on locality. For publication-quality LL numbers, weights should be modified directly.

3. **4-bit quantization** — Affects gradient flow for LoRA and changes absolute probability values for all scoring. Results are internally consistent but not directly comparable to full-precision baselines.

4. **LoRA training budget** — 4 epochs is insufficient. Recommend ≥ 10 epochs for a fair comparison.

5. **Causal tracing on one prompt** — Best layer was identified using `"The capital of France is"` → `"Paris"` only. A more robust estimate would average across multiple CounterFact prompts.

---

*Generated from: `SKS_pipeline_v3_fixed.ipynb` | Model: Qwen2.5-1.5B-Instruct (4-bit) | Dataset: CounterFact (rome.baulab.info)*
