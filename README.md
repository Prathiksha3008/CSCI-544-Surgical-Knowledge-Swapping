# 🔬 Surgical Knowledge Swapping in Large Language Models

**CSCI-544 Natural Language Processing — Group 34, USC**

This project investigates **surgical knowledge editing** in Large Language Models (LLMs) — the ability to precisely modify individual factual associations stored in a model's weights without expensive full retraining or causing catastrophic forgetting.

We implement and compare three knowledge editing methods — **ROME**, **MEMIT**, and **LoRA** — on the **Qwen2.5-1.5B-Instruct** model using the **CounterFact** dataset (Meng et al., 2022).

Our experiments show that ROME and MEMIT effectively modify targeted factual associations while preserving most unrelated knowledge. LoRA provided flexible fine-tuning behavior but introduced broader parameter updates compared to direct knowledge editing methods. Performance was evaluated using edit success, locality preservation, and fluency metrics on the CounterFact benchmark.

---

## 📁 Repository Structure

```
CSCI-544-Surgical-Knowledge-Swapping/
├── Project/
│   ├── SKS_pipeline_Group_34_Final.ipynb   # Main notebook (all tasks)
│   ├── causal_tracing_layer_scores.png     # Causal tracing visualization
│   ├── counterfact_cache.json              # Cached CounterFact dataset (100 edits)
│   ├── output_task1/                       # Task 1: Inference & reproducibility logs
│   │   └── run_log.jsonl
│   ├── output_task3/                       # Task 3: ROME single-edit results
│   │   ├── rome_single_edit_results.json
│   │   └── rome_results.png
│   ├── output_task4/                       # Task 4: MEMIT batch & sequential results
│   │   ├── memit_batch_results.json
│   │   ├── memit_sequential_results.json
│   │   └── memit_results.png
│   └── output_task5/                       # Task 5: LoRA baseline & final comparison
│       ├── lora_results.json
│       ├── full_metrics.json
│       ├── project_summary.json
│       ├── lora_loss_curve.png
│       └── final_comparison.png
├── Documents/
│   ├── Group_34_CSCI_544_Project_Proposal.pdf
│   ├── CSCI_544_Surgical_Knowledge_Swapping_Mid_Term_Report.pdf
│   └── Group34_NLP_Final_Report_Surgical Knowledge Swapping in Large Language Models.pdf
├── Initial_Project_Plan/
│   ├── Plan.txt
│   ├── Work_Split.png
│   ├── Branch Workflow.png
│   ├── Git_Execution.png
│   └── mermaid-diagram.png
└── .gitignore
```

---

## 🛠 Environment Setup

### Platform
- **Runtime:** [Google Colab](https://colab.research.google.com/) (free or Pro tier)
- **GPU:** NVIDIA T4 (Colab free tier) — 4-bit quantization is used to fit the model in ~4 GB VRAM
- **Python:** 3.10+

### Dependencies
All dependencies are installed automatically in the **first cell** of the notebook. No manual setup is required. The following packages are installed via `pip`:

| Package | Purpose |
|---------|---------|
| `transformers` | Model loading & tokenization (Hugging Face) |
| `accelerate` | Efficient model dispatch |
| `bitsandbytes` | 4-bit quantization (CUDA-only, auto-skipped on macOS) |
| `datasets` | CounterFact dataset access |
| `pandas` | Data processing & result tables |
| `matplotlib` | Visualization & plotting |
| `torch` | PyTorch (installed if missing) |

### Quick Start

1. Open the notebook in Google Colab:
   - Upload `Project/SKS_pipeline_Group_34_Final.ipynb` to Colab, **or**
   - Open directly from GitHub: `File → Open notebook → GitHub → paste this repo URL`
2. Select a **GPU runtime**: `Runtime → Change runtime type → T4 GPU`
3. **Run All Cells**: `Runtime → Run all`
   - The first cell auto-installs all dependencies
   - The entire pipeline takes approximately **15–20 minutes** on a T4 GPU

> **Note:** No API keys, environment variables, or external configuration files are needed. The CounterFact dataset is downloaded automatically from Hugging Face and cached locally.

---

## 🔄 Pipeline Overview (How Results Are Generated)

The notebook is organized as a **sequential 5-task pipeline**. Each task builds on the previous one:

### Task 1 — Environment, Inference & Reproducibility
- Loads the **Qwen/Qwen2.5-1.5B-Instruct** model with 4-bit quantization
- Runs baseline inference on sample prompts
- Fixes random seeds (`seed=42`) and logs all outputs to `output_task1/run_log.jsonl`

### Task 2 — Causal Tracing & Activation Patching
- Identifies which transformer layer stores a given fact
- Runs clean and noise-corrupted forward passes
- Patches activations layer-by-layer to measure recovery of correct predictions
- Produces `causal_tracing_layer_scores.png` — identifies the critical layer for editing

### Task 3 — ROME (Rank-One Model Editing)
- Implements the ROME algorithm (Meng et al., NeurIPS 2022) from scratch
- Applies a targeted rank-one update to the MLP `down_proj` weight at the critical layer
- Evaluates single-edit accuracy across 100 CounterFact examples
- Outputs: `output_task3/rome_single_edit_results.json`, `rome_results.png`

### Task 4 — MEMIT (Mass-Editing Memory In a Transformer)
- Implements MEMIT (Meng et al., ICLR 2023) — distributes edits across multiple layers
- **Experiment A:** Batch edit (all facts simultaneously)
- **Experiment B:** Sequential edit runs (interference analysis)
- Outputs: `output_task4/memit_batch_results.json`, `memit_sequential_results.json`, `memit_results.png`

### Task 5 — LoRA Baseline & Final Comparison
- Implements a LoRA (Hu et al., ICLR 2022) fine-tuning baseline
- Trains low-rank adapters on the target facts
- Computes comprehensive metrics across all three methods:
  - **ESR** (Edit Success Rate), **PG** (Paraphrase Generalization), **LL** (Locality Loss), **CS** (Consistency Score), **IR** (Interference Rate)
- Produces the final 6-panel comparison figure: `output_task5/final_comparison.png`

---

## 📊 Results Summary

| Method | ESR ↑ | PG ↑ | LL ↓ | CS | IR ↓ | Best At |
|--------|-------|------|------|----|------|---------|
| **ROME** | 0.97 | 0.77 | 1.19 | 0.20 | — | Accuracy & Generalization |
| **MEMIT** | 1.00 | 0.57 | 9.75 | 0.20 | 0.04 | Locality & Batch Stability |
| **LoRA** | 0.44 | 0.19 | 1955.96 | 0.20 | — | Parameter Efficiency only |

**Key Finding:** Gradient-free rank-one editing methods (ROME/MEMIT) are categorically superior to fine-tuning baselines (LoRA) for surgical knowledge editing in LLMs. ROME achieves the best single-edit accuracy and paraphrase generalization, while MEMIT excels at batch editing with minimal interference between edits.

---

## 📄 Reports

- **Project Proposal:** `Documents/Group_34_CSCI_544_Project_Proposal.pdf`
- **Midterm Report:** `Documents/CSCI_544_Surgical_Knowledge_Swapping_Mid_Term_Report.pdf`
- **Final Report:** `Documents/Group34_NLP_Final_Report_Surgical Knowledge Swapping in Large Language Models.pdf`

---

## 📚 References

1. Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and Editing Factual Associations in GPT.* NeurIPS 2022.
2. Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., & Bau, D. (2023). *Mass-Editing Memory in a Transformer.* ICLR 2023.
3. Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
