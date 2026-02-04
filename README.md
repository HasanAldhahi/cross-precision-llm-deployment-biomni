# Cross-Precision LLM Deployment for Biomedical Reasoning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

> **Master's Thesis Repository**: This repository contains the complete implementation, experiments, and results for investigating cross-precision deployment strategies for large language models in biomedical reasoning tasks.

## 📋 Table of Contents

- [Overview](#-overview)
- [Research Problem](#-research-problem)
- [Methodology](#-methodology)
  - [Baseline: Biomni-R0-32B (BF16)](#baseline-biomni-r0-32b-bf16)
  - [Method A: Naive Transfer](#method-a-naive-transfer)
  - [Method B: Corrective Extraction](#method-b-corrective-extraction)
  - [Method C: Direct Quantization](#method-c-direct-quantization)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Logs & Reproducibility](#-logs--reproducibility)
- [Citation](#-citation)

---

## 🎯 Overview

This thesis investigates methods to deploy the **Biomni-R0-32B** biomedical reasoning model with reduced memory footprint through **FP8 quantization** while preserving domain-specific capabilities. We explore three distinct approaches:

| Method | Approach | Memory Reduction | Key Innovation |
|--------|----------|------------------|----------------|
| **Baseline** | BF16 Full Precision | 0% (64GB) | Gold standard reference |
| **Method A** | Naive Transfer | ~50% (32GB) | LoRA extraction from BF16 weight differences |
| **Method B** | Corrective Extraction | ~50% (32GB) | Dequantization + error-correcting LoRA |
| **Method C** | Direct Quantization | ~50% (32GB) | Domain-calibrated FP8 quantization |

## 🔬 Research Problem

Large language models fine-tuned for biomedical domains (like Biomni-R0-32B) require significant computational resources:

- **Memory**: 64GB for BF16 weights
- **Hardware**: Multiple high-end GPUs
- **Deployment Cost**: Expensive cloud infrastructure

**Key Question**: Can we achieve comparable performance with 50% memory reduction through FP8 quantization?

---

## 📊 Methodology

### Baseline: Biomni-R0-32B (BF16)

The **Biomni-R0-32B** model serves as our gold standard reference. It is a domain-adapted version of Qwen3-32B fine-tuned by the [Biomni Project at Stanford University](https://github.com/snap-stanford/Biomni) for biomedical reasoning tasks including:

- CRISPR delivery mechanism questions
- Genetic variant interpretation and prioritization
- Genome-wide association studies (GWAS) causal gene identification
- Laboratory biology database question answering
- Sequence analysis and interpretation
- Patient gene detection from clinical data
- Rare disease diagnosis
- Functional screening gene retrieval

### Method A: Naive Transfer

**Hypothesis**: Quantization noise is orthogonal to semantic adaptation in weight space.

**Mathematical Formulation**:
```
L_bio = W^{BF16}_biomni - W^{BF16}_qwen     (Weight difference)
L_bio ≈ B · A                                (Low-rank decomposition, rank=256)
W_final = W^{FP8}_qwen + B · A               (Apply LoRA to FP8 base)
```

**Implementation**: See [`brain_surgery/LoRA_extraction/`](brain_surgery/LoRA_extraction/)

### Method B: Corrective Extraction

**Hypothesis**: A rank-256 adapter can encode both semantic adaptation AND quantization correction.

**Mathematical Formulation**:
```
L_corrective = W^{BF16}_biomni - Dequant(W^{FP8}_qwen)
L_corrective = ΔW_semantic - N_quant         (Combined difference)
W_final = W^{FP8}_qwen + B · A               (Apply corrective LoRA)
```

**Key Innovation**: Custom dequantization pipeline with Block-128 scale expansion:

```python
def dequantize_layer(weight_fp8, scale_inv):
    """Dequantize a single layer using Block-128 scheme."""
    scale_expanded = expand_scale(scale_inv, weight_fp8.shape)
    weight_float = weight_fp8.to(torch.float32)
    weight_dequant = weight_float * scale_expanded
    return weight_dequant.to(torch.bfloat16)
```

**Implementation**: See [`brain_surgery/dequantize_FP8/`](brain_surgery/dequantize_FP8/)

### Method C: Direct Quantization

**Hypothesis**: Domain-specific calibration data produces better quantization scales.

**Key Configuration**:
- **Format**: FP8 E4M3 with Block-128 quantization
- **Calibration**: 123 full-context instances, 3,163,274 tokens total
- **Average Context**: 25,718 tokens (up to 75,508 tokens)
- **Sampling**: Stratified across all 10 Eval1 biomedical tasks

**Implementation**: See [`quantization/scripts/FP8_quantization/`](quantization/scripts/FP8_quantization/)

---

## 📁 Repository Structure

```
Biomni/
├── 📂 biomni/                      # Core Biomni agent framework
│   ├── agent/
│   │   ├── c1.py                   # Main agent implementation (LangGraph-based)
│   │   ├── d1.py                   # Extended agent variant
│   │   └── e1.py                   # Full environment agent
│   ├── llm.py                      # LLM provider integrations
│   ├── tool/                       # Bioinformatics tool registry
│   └── eval/                       # Evaluation utilities
│
├── 📂 biomni_env/                  # Environment setup scripts
│   ├── README.md                   # Installation instructions
│   ├── setup.sh                    # Full environment setup (>10 hours)
│   ├── environment.yml             # Conda environment specification
│   └── biomni_tools/               # Pre-installed bioinformatics tools
│
├── 📂 brain_surgery/               # Methods A & B: LoRA extraction
│   ├── README.md                   # Detailed documentation
│   ├── dequantize_FP8/             # FP8 → BF16 dequantization (Method B)
│   ├── LoRA_extraction/            # LoRA extraction scripts
│   ├── lora_extraction_results/    # Extracted LoRA adapters
│   ├── model_probing/              # Weight tensor inspection tools
│   └── orthogonality_hypothesis/   # Quantization noise analysis
│
├── 📂 quantization/                # Method C: Direct FP8 quantization
│   ├── README.md                   # Detailed documentation
│   ├── calibration_data/           # Biomedical calibration dataset
│   └── scripts/                    # Quantization scripts
│       ├── FP8_quantization/       # FP8 E4M3 quantization
│       └── INT4_quantization/      # AWQ INT4 quantization (exploratory)
│
├── 📂 server/                      # Model inference infrastructure
│   ├── README.md                   # Server setup documentation
│   ├── run_server.sh               # vLLM server launch script
│   ├── run_client.sh               # Evaluation client launcher
│   ├── load_balancer.py            # Multi-GPU load balancer
│   └── diagnose_servers.sh         # Server health diagnostics
│
├── 📂 thesis_results_final/        # Experimental results
│   ├── README.md                   # Results documentation
│   ├── annotation_pipeline/        # Human/AI annotation tools
│   ├── R0_32B_BF16/                # Baseline results
│   ├── Qwen_FP8_LORA256/           # Method A results
│   ├── Qwen_FP8_with_extracted_dequantized_LORA_rank256/  # Method B results
│   └── R0-32B-FP8/                 # Method C results
│
├── 📂 visualization/               # Result visualization
│   ├── README.md                   # Visualization documentation
│   ├── bar_chart.py                # Per-task accuracy charts
│   ├── heatmap.py                  # Cross-model performance heatmap
│   └── *.json                      # Statistics data files
│
└── 📂 logs/                        # Execution logs (reproducibility)
    └── README.md                   # Log file documentation
```

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.10+
- **CUDA**: 12.x
- **GPU**: NVIDIA H100 (80GB) recommended for quantization; 4x GPUs for inference
- **Disk**: 100GB+ for models and environments

### Step 1: Clone the Repository

```bash
git clone https://github.com/HasanAldhahi/cross-precision-llm-deployment-biomni.git
cd Biomni
```

### Step 2: Set Up the Biomni Environment

**Option A**: Basic environment (recommended for quick start)
```bash
cd biomni_env
conda env create -f environment.yml
conda activate biomni_e1
```

**Option B**: Full E1 environment (requires >10 hours and 30GB disk)
```bash
cd biomni_env
bash setup.sh
conda activate biomni_e1
```

### Step 3: Install Component-Specific Requirements

```bash
# For brain surgery (Methods A & B)
pip install -r brain_surgery/brain_surgery_requirments.txt

# For quantization (Method C)
pip install -r quantization/quantization_requirements.txt

# For server/inference
pip install -r server/requirements.txt

# For visualization
pip install -r visualization/requirements.txt

# For annotation pipeline
pip install -r thesis_results_final/annotation_pipeline/annotate_requirment.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the repository root:

```bash
# API Keys for evaluation
CUSTOM_MODEL_API_KEY=your_chat_ai_key

# Model paths (adjust to your setup)
MODEL_PATH=/path/to/models
```

---

## 💻 Usage

### Running Method A: Naive Transfer

```bash
cd brain_surgery/LoRA_extraction
bash extract_lora_256.sh
```

### Running Method B: Corrective Extraction

```bash
# Step 1: Dequantize FP8 model
cd brain_surgery/dequantize_FP8
python dequant_FP8_to_BF16.py

# Step 2: Verify dequantization
python cpu_autopsy_dequantized_model_sanity_check.py

# Step 3: Extract LoRA
cd ../LoRA_extraction
bash extract_lora_256.sh --base dequantized

# Step 4: Sanitize LoRA for vLLM compatibility
python sanitize_lora.py
```

### Running Method C: Direct Quantization

```bash
cd quantization/scripts/FP8_quantization
sbatch quanitze_FP8.sh  # SLURM cluster
# OR
python quantize_FP8.py  # Direct execution
```

### Running Inference Server

```bash
cd server
bash run_server.sh --model r0-fp8 --gpus 4
```

### Running Evaluation

```bash
cd server
bash run_client.sh --benchmark eval1
```

### Running Annotation Pipeline

```bash
cd thesis_results_final/annotation_pipeline

# Step 1: Extract answers from model outputs
python 1_extract_answers.py

# Step 2: AI-assisted evaluation
python 2_ask_chat.py

# Step 3: Create annotated results
python 3_create_final_annotated_results.py

# Step 4: Generate statistics
python 4_create_final_statistics.py
```

---

## 📈 Results

### Summary Performance (Eval1 Benchmark)

| Model | Accuracy | Memory | Inference Speed |
|-------|----------|--------|-----------------|
| **Biomni-R0-32B (BF16)** | XX.X% | 64GB | Baseline |
| **Method A: Naive Transfer** | XX.X% | 32GB | 1.Xx faster |
| **Method B: Corrective Extraction** | XX.X% | 32GB | 1.Xx faster |
| **Method C: Direct Quantization** | XX.X% | 32GB | 1.Xx faster |

Detailed results with per-task breakdowns are available in [`thesis_results_final/`](thesis_results_final/).

### Performance Heatmap

![Task Performance Heatmap](visualization/heatmap.png)

---

## 📋 Logs & Reproducibility

All execution logs are preserved to ensure reproducibility:

| Log File | Purpose |
|----------|---------|
| `brain_surgery/dequantize_FP8/dequant_FP8.log` | FP8 dequantization process |
| `quantization/scripts/FP8_quantization/quantize_FP8.log` | Method C quantization |
| `server/vllm_gpu*.log` | vLLM inference server logs |
| `server/load_balancer.log` | Multi-GPU load distribution |

See [`logs/README.md`](logs/README.md) for detailed log descriptions.

---

## 🙏 Acknowledgments

- **Biomni Project** (Stanford University) for the baseline model and framework
- **KISSKI** for computational resources
- **Thesis Supervisor**: [Name]

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{aldhahi2026crossprecision,
  title={Cross-Precision Deployment Strategies for Large Language Models in Biomedical Reasoning},
  author={Aldhahi, Hasan},
  school={University Name},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The Biomni framework is licensed separately by Stanford University.
