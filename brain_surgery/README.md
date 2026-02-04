# Brain Surgery Module

> **Methods A & B Implementation**: Weight-space operations for cross-precision LoRA extraction.

This module contains the implementation of **Method A (Naive Transfer)** and **Method B (Corrective Extraction)** from the thesis. The term "brain surgery" refers to the precise manipulation of neural network weights at the tensor level.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Module Structure](#-module-structure)
- [Method A: Naive Transfer](#-method-a-naive-transfer)
- [Method B: Corrective Extraction](#-method-b-corrective-extraction)
- [Weight Autopsy Protocol](#-weight-autopsy-protocol)
- [Orthogonality Analysis](#-orthogonality-analysis)
- [Installation](#-installation)
- [Usage](#-usage)
- [Logs](#-logs)

---

## 🎯 Overview

The brain surgery module enables weight-space arithmetic operations that are typically abstracted away by deep learning libraries. Key capabilities:

1. **Dequantization**: Convert FP8 quantized weights back to BF16 for arithmetic
2. **LoRA Extraction**: Extract low-rank adapters from weight differences
3. **Weight Autopsy**: Statistical verification of tensor health
4. **Sanitization**: Make extracted LoRAs compatible with vLLM

---

## 📁 Module Structure

```
brain_surgery/
├── brain_surgery_requirments.txt    # Python dependencies
├── README.md                        # This file
│
├── 📂 dequantize_FP8/               # FP8 → BF16 dequantization
│   ├── dequant_FP8_to_BF16.py       # Main dequantization script
│   ├── cpu_autopsy_dequantized_model_sanity_check.py  # Verification
│   └── dequant_FP8.log              # Execution log
│
├── 📂 dequantize_INT4/              # INT4 dequantization (exploratory)
│   ├── raw_dequantize_INT4.py       # INT4 dequantization attempt
│   ├── cpu_autopsy_dequantized_model_sanity_check.py
│   ├── generate_plots.py            # Visualization of weight statistics
│   ├── fig_6_x_magnitude.png        # Weight magnitude distribution
│   ├── fig_6_y_orthogonality.png    # Orthogonality analysis
│   ├── raw.sh                       # SLURM job script
│   ├── raw_dequant_12041842.log     # Execution log
│   └── raw_dequant_12041842.err     # Error log
│
├── 📂 LoRA_extraction/              # LoRA extraction using MergeKit
│   ├── extract_lora_256.sh          # Main extraction script
│   └── sanitize_lora.py             # vLLM compatibility sanitization
│
├── 📂 lora_extraction_results/      # Extracted LoRA adapters
│   ├── Method_A_lora_basic_original_base_rank_128/
│   ├── Method_A_lora_basic_original_base_rank_256/
│   └── Method_B_dequantized_corrected_lora_rank_256/
│
├── 📂 model_probing/                # Weight tensor inspection
│   ├── check_keys_FP8.py            # FP8 tensor key inspection
│   └── check_keys_INT4.py           # INT4 tensor key inspection
│
└── 📂 orthogonality_hypothesis/     # Quantization noise analysis
    ├── check_orthogonality_svd_one_layer.py
    ├── compare_noise_patterns.py
    ├── orthogonality_statistical_analysis.py
    └── prove_orthagonality_all_layers.py
```

---

## 🔬 Method A: Naive Transfer

### Conceptual Approach

Method A extracts a LoRA adapter from the weight difference between the domain-adapted model (Biomni-R0-32B) and the base model (Qwen3-32B), both in BF16 precision. This LoRA is then applied to the FP8-quantized base model.

### Mathematical Formulation

```
Step 1: Compute weight difference
    L_bio = W^{BF16}_biomni - W^{BF16}_qwen

Step 2: Low-rank decomposition (SVD)
    L_bio ≈ B · A    where B ∈ R^{d×r}, A ∈ R^{r×k}, r = 256

Step 3: Apply to FP8 base
    W_final = W^{FP8}_qwen + B · A
```

### Key Hypothesis

The quantization noise is similar for both models and cancels out:

```
N_biomni ≈ N_qwen  →  (W_biomni + N) - (W_qwen + N) ≈ W_biomni - W_qwen
```

### Implementation

```bash
cd LoRA_extraction
bash extract_lora_256.sh
```

The script uses **MergeKit** for extraction:

```python
# MergeKit configuration for Method A
models:
  - model: Qwen/Qwen3-32B
    # base model
  - model: snap-stanford/biomni-r0-32b
    # domain-adapted model
    
merge_method: extract_lora
parameters:
  rank: 256
```

---

## 🔧 Method B: Corrective Extraction

### Conceptual Approach

Method B creates a LoRA that simultaneously encodes domain-specific knowledge AND corrects quantization errors by extracting from the difference between the BF16 adapted model and the **dequantized** FP8 base model.

### Mathematical Formulation

```
Step 1: Dequantize FP8 base model
    W^{BF16}_bridge = Dequant(W^{FP8}_qwen)
    
Step 2: Extract corrective difference
    L_corrective = W^{BF16}_biomni - W^{BF16}_bridge
    L_corrective = ΔW_semantic - N_quant

Step 3: Low-rank decomposition
    L_corrective ≈ B · A

Step 4: Apply to FP8 base
    W_final = W^{FP8}_qwen + B · A
```

### The Bridge Model: FP8 Dequantization

Modern libraries treat quantized tensors as "black boxes." We reverse-engineered the Block-128 quantization scheme:

| Layer | Weight Shape | Scale Shape | Block Size |
|-------|--------------|-------------|------------|
| mlp.gate_proj | [5120, 13696] | [40, 107] | 128 |
| mlp.up_proj | [5120, 13696] | [40, 107] | 128 |
| self_attn.q_proj | [5120, 5120] | [40, 40] | 128 |

### Dequantization Algorithm

```python
def expand_scale_dynamic(scale, target_shape, layer_name):
    """
    Dynamically calculates block size and expands scale to match target weight.
    The inverse scale (input_scale_inv) is multiplied with FP8 values.
    """
    scale_h, scale_w = scale.shape
    target_h, target_w = target_shape
    
    # Calculate block sizes
    repeat_h = target_h // scale_h  # = 128
    repeat_w = target_w // scale_w  # = 128
    
    # Expand scale to match weight dimensions
    expanded = scale.repeat_interleave(repeat_h, dim=0)
    expanded = expanded.repeat_interleave(repeat_w, dim=1)
    
    return expanded

def dequantize_layer(weight_fp8, scale_inv):
    """Dequantize a single layer."""
    scale_expanded = expand_scale_dynamic(scale_inv, weight_fp8.shape)
    weight_float = weight_fp8.to(torch.float32)
    weight_dequant = weight_float * scale_expanded
    return weight_dequant.to(torch.bfloat16)
```

### LoRA Sanitization for vLLM

vLLM's FP8 kernels impose strict limitations on LoRA targets. We prune incompatible layers:

```python
# Forbidden layer patterns (will crash vLLM)
FORBIDDEN_KEYWORDS = ["lm_head", "embed_tokens", "norm", "bias"]

def sanitize_lora(state_dict):
    """Remove incompatible layers from LoRA."""
    sanitized = {}
    for key, tensor in state_dict.items():
        if not any(kw in key for kw in FORBIDDEN_KEYWORDS):
            sanitized[key] = tensor
    return sanitized
```

---

## 🔍 Weight Autopsy Protocol

Before GPU evaluation, we verify tensor health on CPU:

### Healthy Weight Statistics

| Metric | Expected Range |
|--------|----------------|
| Mean | |μ| < 0.01 |
| Std | 0.02 - 0.05 |
| Min/Max | [-0.5, 0.5] |

### Autopsy Script

```python
def autopsy_layer(weight, layer_name):
    """Perform statistical autopsy on a weight tensor."""
    stats = {
        'mean': weight.mean().item(),
        'std': weight.std().item(),
        'min': weight.min().item(),
        'max': weight.max().item(),
    }
    
    # Warning flags
    if abs(stats['mean']) > 1.0:
        print(f"⚠️ WARNING: {layer_name} has exploded mean")
    if stats['std'] > 10.0:
        print(f"⚠️ WARNING: {layer_name} has exploded std")
    
    return stats
```

### Iterative Debugging Process

| Iteration | Issue | Result |
|-----------|-------|--------|
| 1 | Per-tensor scaling | Mean > 448 (exploded) |
| 2 | Block size = 64 | Shape mismatch errors |
| 3 | Block size = 128, wrong scale | Biased weights |
| 4 | Correct implementation | Mean ≈ 0, Std ≈ 0.02 ✓ |

---

## 📐 Orthogonality Analysis

The `orthogonality_hypothesis/` folder contains scripts to verify the hypothesis that quantization noise is similar between models.

### Analysis Scripts

- `check_orthogonality_svd_one_layer.py`: SVD analysis of single layer
- `compare_noise_patterns.py`: Compare noise distributions
- `orthogonality_statistical_analysis.py`: Statistical tests
- `prove_orthagonality_all_layers.py`: Full model verification

---

## 🔧 Installation

```bash
pip install -r brain_surgery_requirments.txt
```

### Key Dependencies

- `torch>=2.9.1` - PyTorch with BF16/FP8 support
- `transformers>=4.57.1` - HuggingFace Transformers
- `safetensors>=0.5.3` - Efficient tensor serialization
- `mergekit>=0.1.4` - LoRA extraction toolkit
- `peft>=0.18.0` - Parameter-efficient fine-tuning
- `accelerate>=1.6.0` - Large model utilities
- `bitsandbytes>=0.48.2` - Quantization utilities

---

## 💻 Usage

### Complete Method B Pipeline

```bash
# Step 1: Dequantize FP8 model to create bridge model
cd dequantize_FP8
python dequant_FP8_to_BF16.py
# Output: ./qwen_bf16_robust/

# Step 2: Verify dequantization quality
python cpu_autopsy_dequantized_model_sanity_check.py
# Expect: Mean ≈ 0, Std ≈ 0.02 for all layers

# Step 3: Extract corrective LoRA
cd ../LoRA_extraction
bash extract_lora_256.sh --base ../dequantize_FP8/qwen_bf16_robust
# Output: ../lora_extraction_results/Method_B_*/

# Step 4: Sanitize for vLLM
python sanitize_lora.py \
    --input ../lora_extraction_results/Method_B_dequantized_corrected_lora_rank_256 \
    --output ../lora_extraction_results/Method_B_sanitized
```

### Probe Model Structure

```bash
cd model_probing
python check_keys_FP8.py --model /path/to/Qwen3-32B-FP8
```

---

## 📋 Logs

All execution logs are preserved for reproducibility:

| Log | Description |
|-----|-------------|
| `dequantize_FP8/dequant_FP8.log` | FP8 dequantization (Method B Step 1) |
| `dequantize_INT4/raw_dequant_*.log` | INT4 dequantization attempts |
| `dequantize_INT4/raw_dequant_*.err` | INT4 error traces |

---

## ⚠️ Known Issues

1. **INT4 Dequantization**: AWQ INT4 quantization uses group-wise scaling that is incompatible with simple block expansion. See `dequantize_INT4/` for exploratory work.

2. **LoRA Target Restrictions**: vLLM's FP8 kernels do not support LoRA on `lm_head`, `embed_tokens`, or normalization layers. These must be pruned.

3. **Memory Requirements**: Dequantization temporarily requires ~130GB RAM to hold both FP8 and BF16 models.

---

## 📚 References

- MergeKit: https://github.com/arcee-ai/mergekit
- vLLM LoRA: https://docs.vllm.ai/en/latest/models/lora.html
- FP8 Format: https://arxiv.org/abs/2209.05433
