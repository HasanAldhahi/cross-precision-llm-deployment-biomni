# Models & Adapters Registry

> All quantized models and LoRA adapters produced during the thesis  
> **"Optimizing LLM Deployment via Cross-Precision Transfer: A Case Study in Biomedical AI Agent Biomni"**

**Base Model**: [biomni/Biomni-R0-32B-Preview](https://huggingface.co/biomni/Biomni-R0-32B-Preview) (Qwen3-32B architecture, BF16, 64 GB)

All artifacts are publicly hosted on Hugging Face under [`hassanshka`](https://huggingface.co/hassanshka).

---

## Quick Overview

| # | Name | Type | Method | Precision | VRAM | Eval1 Accuracy | HuggingFace Link |
|---|------|------|--------|-----------|------|-----------------|------------------|
| 1 | Biomni-R0-32B-AWQ-INT4 | Model | Quantization | W4A16 | ~16 GB | — | [Link](https://huggingface.co/hassanshka/Biomni-R0-32B-AWQ-INT4) |
| 2 | Biomni-R0-32B-FP8 | Model | Method C | FP8 E4M3 | ~32 GB | 41.4% | [Link](https://huggingface.co/hassanshka/Biomni-R0-32B-FP8) |
| 3 | Biomni-R0-32B-PTQ-INT8 | Model | Quantization | INT8 W+A | ~8-10 GB | — | [Link](https://huggingface.co/hassanshka/Biomni-R0-32B-PTQ-INT8) |
| 4 | Biomni-R0-32B-INT4-to-BF16 | Bridge Model | Dequantization | BF16 | ~64 GB | — | [Link](https://huggingface.co/hassanshka/Biomni-R0-32B-INT4-to-BF16) |
| 5 | Qwen3-32B-FP8-to-BF16 | Bridge Model | Dequantization | BF16 | ~64 GB | — | [Link](https://huggingface.co/hassanshka/Qwen3-32B-FP8-to-BF16) |
| 6 | Biomni-R0-32B-LoRA-Rank256 | Adapter | Method A | BF16 | +0.5 GB | 44.6% | [Link](https://huggingface.co/hassanshka/Biomni-R0-32B-LoRA-Rank256) |
| 7 | Biomni-R0-32B-LoRA-Rank128 | Adapter | Method A | BF16 | +0.25 GB | — | [Link](https://huggingface.co/hassanshka/Biomni-R0-32B-LoRA-Rank128) |
| 8 | Biomni-R0-32B-LoRA-Dequantized-Rank256 | Adapter | Method B | BF16 | +0.5 GB | 29.5% | [Link](https://huggingface.co/hassanshka/Biomni-R0-32B-LoRA-Dequantized-Rank256) |

**Baseline reference**: Biomni-R0-32B (BF16) — 44.5% accuracy, 64.0 GB VRAM.

---

## Quantized Models

### 1. Biomni-R0-32B-AWQ-INT4

| Field | Value |
|-------|-------|
| **HuggingFace** | [hassanshka/Biomni-R0-32B-AWQ-INT4](https://huggingface.co/hassanshka/Biomni-R0-32B-AWQ-INT4) |
| **Quantization** | AWQ — W4A16 (4-bit weights, 16-bit activations) |
| **Group Size** | 128 |
| **Framework** | LLM Compressor |
| **Calibration** | Custom biomedical dataset (123 samples, stratified across 10 Eval1 tasks) |
| **VRAM** | ~16 GB (~75% reduction vs BF16) |
| **Target Hardware** | Consumer GPUs (RTX 3090/4090) |
| **Local Name** | `Biomni-R0-32B-AWQ-INT4-CustomCalib` |

**Description**: Aggressive 4-bit weight quantization using Activation-aware Weight Quantization (AWQ). Suitable for memory-constrained single-GPU inference. Weights are quantized to INT4 while activations remain in FP16.

**Quantization script**: [`quantization/scripts/INT4_quantization/quantize_AWQ_INT4.py`](../quantization/scripts/INT4_quantization/quantize_AWQ_INT4.py)

---

### 2. Biomni-R0-32B-FP8 (Method C)

| Field | Value |
|-------|-------|
| **HuggingFace** | [hassanshka/Biomni-R0-32B-FP8](https://huggingface.co/hassanshka/Biomni-R0-32B-FP8) |
| **Quantization** | FP8 E4M3 with Block-128 |
| **Framework** | LLM Compressor (`QuantizationModifier`) |
| **Calibration** | 123 full-context biomedical trajectories (3,163,274 tokens total, avg 25,718 tokens, max 75,508 tokens) |
| **VRAM** | ~32 GB (~50% reduction vs BF16) |
| **Eval1 Accuracy** | 41.4% (93.0% retention vs baseline) |
| **Target Hardware** | NVIDIA H100, L40S, Ada Lovelace (FP8 Tensor Cores) |
| **Local Name** | `Biomni-R0-32B-FP8-CustomCalib` |

**Description**: Direct FP8 quantization of the fully fine-tuned Biomni model (**Method C** in the thesis). Uses domain-specific deep calibration with full reasoning trajectories rather than short context windows. Achieves near-lossless performance with 50% memory savings.

**Key innovation**: Calibration uses stratified sampling across all 10 Eval1 biomedical tasks with full-length contexts (up to 75K tokens).

**Quantization script**: [`quantization/scripts/FP8_quantization/quantize_FP8.py`](../quantization/scripts/FP8_quantization/quantize_FP8.py)

---

### 3. Biomni-R0-32B-PTQ-INT8

| Field | Value |
|-------|-------|
| **HuggingFace** | [hassanshka/Biomni-R0-32B-PTQ-INT8](https://huggingface.co/hassanshka/Biomni-R0-32B-PTQ-INT8) |
| **Quantization** | INT8 (8-bit weights and activations) |
| **Method** | Post-Training Quantization (PTQ) |
| **Framework** | Optimum Quanto |
| **Calibration Samples** | 120 |
| **VRAM** | ~8-10 GB (~83-87% reduction vs BF16) |
| **Target Hardware** | Most modern GPUs (no special hardware required) |

**Description**: Post-training quantization to INT8 using Optimum Quanto. Both weights and activations are quantized to 8-bit integers, providing the most aggressive memory reduction of all quantized variants. Can also be served via vLLM at runtime.

**Note**: The quantization may be applied at serving time by vLLM for optimal performance.

---

## Bridge Models (Dequantized)

Bridge models are created by converting quantized weights back to BF16. They serve as intermediate representations for LoRA extraction and research into quantization quality.

### 4. Biomni-R0-32B-INT4-to-BF16

| Field | Value |
|-------|-------|
| **HuggingFace** | [hassanshka/Biomni-R0-32B-INT4-to-BF16](https://huggingface.co/hassanshka/Biomni-R0-32B-INT4-to-BF16) |
| **Source** | Biomni-R0-32B-AWQ-INT4-CustomCalib |
| **Target Dtype** | BFloat16 |
| **Dequantization** | Standard AWQ unpacking (W4A16), group size 128 |
| **VRAM** | ~64 GB |
| **Local Name** | `Biomni-R0-32B-From-INT4-Bridge-BF16` |

**Description**: Dequantized version of the AWQ INT4 model, recovered via `W_bf16 = W_int4 × Scale`. This is **not** identical to the original BF16 model — some precision is lost in the quantize → dequantize roundtrip.

**Use cases**:
- Starting point for LoRA or full fine-tuning
- Research into quantization/dequantization quality recovery
- Hardware compatibility (no INT4/FP8 support required)

**Dequantization script**: [`brain_surgery/dequantize_INT4/raw_dequantize_INT4.py`](../brain_surgery/dequantize_INT4/raw_dequantize_INT4.py)

---

### 5. Qwen3-32B-FP8-to-BF16

| Field | Value |
|-------|-------|
| **HuggingFace** | [hassanshka/Qwen3-32B-FP8-to-BF16](https://huggingface.co/hassanshka/Qwen3-32B-FP8-to-BF16) |
| **Source** | Qwen/Qwen3-32B-FP8 |
| **Target Dtype** | BFloat16 |
| **Dequantization** | Block-128 scale expansion (`W_bf16 = W_fp8 × scale_inv`) |
| **VRAM** | ~64 GB |

**Description**: Dequantized version of the official Qwen3-32B-FP8 model. Created for **Method B** — used as the base model for corrective LoRA extraction. The dequantization reverses Block-128 FP8 E4M3 quantization.

**Dequantization details**: Custom implementation that reverse-engineers the Block-128 quantization scheme, dynamically calculating block sizes per layer and expanding scales to match weight dimensions.

**Dequantization script**: [`brain_surgery/dequantize_FP8/dequant_FP8_to_BF16.py`](../brain_surgery/dequantize_FP8/dequant_FP8_to_BF16.py)

---

## LoRA Adapters

All LoRA adapters are extracted using **MergeKit** and sanitized for **vLLM** compatibility. They can be loaded on top of the FP8 base model (`Qwen/Qwen3-32B-FP8`) for multi-adapter serving.

### 6. Biomni-R0-32B-LoRA-Rank256 (Method A — Recommended)

| Field | Value |
|-------|-------|
| **HuggingFace** | [hassanshka/Biomni-R0-32B-LoRA-Rank256](https://huggingface.co/hassanshka/Biomni-R0-32B-LoRA-Rank256) |
| **Method** | Method A — Naive Transfer |
| **Rank** | 256 |
| **Extraction** | `L_bio = W_biomni(BF16) - W_qwen(BF16)`, then SVD to rank 256 |
| **Base Model** | Qwen/Qwen3-32B-FP8 (at serving time) |
| **Adapter Size** | ~0.5 GB |
| **Eval1 Accuracy** | **44.6%** (100.2% retention — matches baseline) |
| **Local Name** | `Method_A_lora_basic_original_base_rank_256` |

**Description**: The **recommended adapter**. Extracts the semantic difference between Biomni-R0-32B (BF16) and Qwen3-32B (BF16) as a low-rank adapter. Applied directly to the FP8 base model at serving time.

**Key finding**: Quantization noise is orthogonal to the LoRA signal (cosine similarity ≈ -0.00001), confirming that naive cross-precision transfer works without any correction or retraining.

**Extraction**: [`brain_surgery/LoRA_extraction/extract_lora_256.sh`](../brain_surgery/LoRA_extraction/extract_lora_256.sh)  
**Sanitization**: [`brain_surgery/LoRA_extraction/sanitize_lora.py`](../brain_surgery/LoRA_extraction/sanitize_lora.py)

**vLLM deployment**:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-FP8 \
    --enable-lora \
    --lora-modules biomni=hassanshka/Biomni-R0-32B-LoRA-Rank256
```

---

### 7. Biomni-R0-32B-LoRA-Rank128 (Method A)

| Field | Value |
|-------|-------|
| **HuggingFace** | [hassanshka/Biomni-R0-32B-LoRA-Rank128](https://huggingface.co/hassanshka/Biomni-R0-32B-LoRA-Rank128) |
| **Method** | Method A — Naive Transfer |
| **Rank** | 128 |
| **Extraction** | `L_bio = W_biomni(BF16) - W_qwen(BF16)`, then SVD to rank 128 |
| **Base Model** | Qwen/Qwen3-32B-FP8 (at serving time) |
| **Adapter Size** | ~0.25 GB |
| **Local Name** | `Method_A_lora_basic_original_base_rank_128` |

**Description**: Lower-rank variant of Method A. Uses rank 128 instead of 256, resulting in a smaller adapter with potentially slightly less expressiveness. Useful for environments with tighter memory budgets or when serving many adapters simultaneously.

**Extraction**: Same pipeline as Rank256, with `--max-rank 128`.

---

### 8. Biomni-R0-32B-LoRA-Dequantized-Rank256 (Method B)

| Field | Value |
|-------|-------|
| **HuggingFace** | [hassanshka/Biomni-R0-32B-LoRA-Dequantized-Rank256](https://huggingface.co/hassanshka/Biomni-R0-32B-LoRA-Dequantized-Rank256) |
| **Method** | Method B — Corrective Extraction |
| **Rank** | 256 |
| **Extraction** | `L_corrective = W_biomni(BF16) - Dequant(W_qwen(FP8))` |
| **Base Model** | Qwen/Qwen3-32B-FP8 (at serving time) |
| **Adapter Size** | ~0.5 GB |
| **Eval1 Accuracy** | **29.5%** (66.3% retention — significant degradation) |
| **Local Name** | `Method_B_dequantized_corrected_lora_rank_256` |

**Description**: Corrective LoRA that attempts to encode both semantic adaptation AND quantization error correction. Extracted from the difference between the BF16 Biomni model and the dequantized FP8 Qwen base.

**Why it underperforms**: The rank-256 adapter does not have sufficient capacity to capture both the structured semantic signal and the high-rank, isotropic quantization noise. This results in capacity saturation and performance degradation — confirming that correction is unnecessary when the noise is orthogonal.

**Dequantization**: [`brain_surgery/dequantize_FP8/dequant_FP8_to_BF16.py`](../brain_surgery/dequantize_FP8/dequant_FP8_to_BF16.py)  
**Extraction**: [`brain_surgery/LoRA_extraction/extract_lora_256.sh`](../brain_surgery/LoRA_extraction/extract_lora_256.sh)

---

## Evaluation Summary

Results on the **Eval1** benchmark (433 questions across CRISPR, GWAS, Lab Bench, Patient Gene Detection, Rare Disease Diagnosis, and Screening Gene Retrieval):

| Model Configuration | Method | Accuracy | Retention | VRAM |
|:---------------------|:-------|:---------|:----------|:-----|
| **Biomni-R0-32B (BF16)** | **Baseline** | **44.5%** | **100.0%** | **64.0 GB** |
| **Qwen3-FP8 + LoRA-Rank256** | **Method A** | **44.6%** | **100.2%** | **40.0 GB** |
| Qwen3-FP8 + LoRA-Dequantized-Rank256 | Method B | 29.5% | 66.3% | 40.0 GB |
| Biomni-R0-32B-FP8 (Direct) | Method C | 41.4% | 93.0% | 32.0 GB |
| Qwen3-32B BF16 (No Adaptation) | Reference | 22.1% | 49.7% | 64.0 GB |

---

## Directory Layout

```
models_and_adapters/
├── README.md                                              # This file
├── Biomni-R0-32B-AWQ-INT4-CustomCalib/                    # AWQ INT4 quantized model
├── Biomni-R0-32B-FP8-CustomCalib/                         # FP8 quantized model (Method C)
├── Biomni-R0-32B-From-INT4-Bridge-BF16/                   # Dequantized INT4 → BF16 bridge
├── Qwen3-32B-FP8-to-BF16/                                # Dequantized FP8 → BF16 bridge
└── LoRa_extraction_results/                               # Extracted LoRA adapters
    ├── Method_A_lora_basic_original_base_rank_128/        # Method A, rank 128
    ├── Method_A_lora_basic_original_base_rank_256/        # Method A, rank 256 (recommended)
    └── Method_B_dequantized_corrected_lora_rank_256/      # Method B, rank 256
```

---

## How to Use

### Load a quantized model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "hassanshka/Biomni-R0-32B-FP8",  # or any model above
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("hassanshka/Biomni-R0-32B-FP8")
```

### Load a LoRA adapter with vLLM (recommended deployment)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B-FP8 \
    --enable-lora \
    --lora-modules biomni=hassanshka/Biomni-R0-32B-LoRA-Rank256 \
    --tensor-parallel-size 4
```

### Load a LoRA adapter with PEFT

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B-FP8", device_map="auto")
model = PeftModel.from_pretrained(base, "hassanshka/Biomni-R0-32B-LoRA-Rank256")
```

---

## License

All models and adapters are released under **Apache 2.0**, consistent with the base Biomni and Qwen3 licenses.

## Citation

```bibtex
@mastersthesis{aldhahi2026optimizing,
  title={Optimizing LLM Deployment via Cross-Precision Transfer: A Case Study in Biomedical AI Agent Biomni},
  author={Aldhahi, Hasan Marwan Mahmood},
  school={Georg-August-Universit{\"a}t G{\"o}ttingen},
  year={2026},
  month={February},
  type={Master's Thesis},
  note={Supervised by Prof. Julian Kunkel and Dr. Narges Lux}
}
```
