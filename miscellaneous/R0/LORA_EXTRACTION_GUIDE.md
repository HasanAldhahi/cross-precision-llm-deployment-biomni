# LoRA Extraction and Fine-tuning Guide

Extract LoRA weights from Biomni-R0-32B-Preview and continue fine-tuning on your own data.

## Overview

This guide shows you how to:
1. **Compare** Biomni-R0-32B-Preview with Qwen3-32B-FP8
2. **Identify** LoRA weights or low-rank modifications
3. **Extract** LoRA adapters for reuse
4. **Fine-tune** further on your biomedical datasets

## Prerequisites

```bash
pip install transformers torch safetensors huggingface-hub peft datasets numpy
```

## Step 1: Compare Models and Extract LoRA

### Basic Comparison

```bash
python Compare_R0_qwen3.py
```

This will:
- Download both models from HuggingFace
- Compare configurations and architectures
- Analyze weight differences
- Detect LoRA-like patterns
- Extract LoRA weights if found

### Advanced Options

```bash
# Custom output directory
python Compare_R0_qwen3.py --output-dir ./my_analysis

# Extract LoRA with different rank
python Compare_R0_qwen3.py --lora-rank 32

# Compare different models
python Compare_R0_qwen3.py \
    --base Qwen/Qwen3-32B \
    --finetuned biomni/Biomni-R0-32B-Preview \
    --lora-rank 16
```

### Output Files

The comparison generates:

```
model_comparison/
├── config_comparison.json          # Config differences
├── architecture_comparison.json    # Layer structure differences
├── weight_differences.json         # Detailed weight analysis
├── lora_analysis.json             # LoRA pattern detection
├── lora_metadata.json             # LoRA configuration info
├── extracted_lora_weights.safetensors  # Extracted LoRA weights
└── comparison_report.txt          # Human-readable summary
```

## Step 2: Understanding the Results

### Scenario A: Explicit LoRA Found

If the report shows:
```
✅ EXPLICIT LoRA ADAPTERS FOUND
   The model contains explicit LoRA adapter layers.
```

The model has dedicated LoRA layers like:
- `model.layers.0.self_attn.q_proj.lora_A`
- `model.layers.0.self_attn.q_proj.lora_B`

**Action**: Use the identified layers directly.

### Scenario B: LoRA Patterns Detected

If the report shows:
```
⚠️  POTENTIAL LoRA PATTERNS DETECTED
   Weight differences show low-rank patterns consistent with LoRA.
```

The script has extracted approximate LoRA weights via SVD decomposition.

**Action**: Use `extracted_lora_weights.safetensors` for fine-tuning.

### Scenario C: Full Fine-tuning

If the report shows:
```
ℹ️  NO LORA PATTERNS DETECTED
   The model appears to be fully fine-tuned.
```

The model was trained with all parameters modified.

**Action**: Either full fine-tune or apply new LoRA layers.

## Step 3: Fine-tune with Extracted LoRA

### Using Extracted LoRA Weights

```bash
python finetune_with_lora.py \
    --lora-weights ./model_comparison/extracted_lora_weights.safetensors \
    --dataset your/biomedical/dataset \
    --epochs 5 \
    --output-dir ./my_finetuned_model
```

### Starting Fresh with Detected Configuration

```bash
# Use the detected LoRA config but train from scratch
python finetune_with_lora.py \
    --rank 16 \
    --target-modules q_proj k_proj v_proj o_proj \
    --dataset your/dataset \
    --epochs 3
```

### Custom Configuration

```bash
python finetune_with_lora.py \
    --base-model Qwen/Qwen3-32B-FP8 \
    --rank 32 \
    --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --dataset biomni/crispr-screen-data \
    --learning-rate 1e-4 \
    --epochs 5 \
    --batch-size 2 \
    --output-dir ./biomni_extended
```

## Step 4: Use Your Fine-tuned Model

### Option 1: Load LoRA Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B-FP8",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "./my_finetuned_model/lora_adapter"
)

tokenizer = AutoTokenizer.from_pretrained("./my_finetuned_model/lora_adapter")
```

### Option 2: Load Merged Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./my_finetuned_model/merged_model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("./my_finetuned_model/merged_model")
```

### Option 3: Use with Biomni A1 Agent

```python
from biomni.agent import A1

agent = A1(
    path='/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni_Env',
    llm='./my_finetuned_model/merged_model',
    source='local',  # or 'custom' if serving
    use_tool_retriever=True,
    timeout_seconds=1000
)

log = agent.go("Your biomedical task here")
```

## Understanding LoRA Extraction

### How It Works

1. **Weight Difference**: Computes `Δ = W_finetuned - W_base`
2. **SVD Decomposition**: Decomposes Δ into `U @ S @ V^T`
3. **Low-rank Approximation**: Keeps top-k singular values
4. **LoRA Matrices**: 
   - `lora_A = V[:, :rank].T`  (rank × d_in)
   - `lora_B = U[:, :rank] @ diag(S[:rank])`  (d_out × rank)
5. **Verification**: Checks reconstruction error

### Choosing Rank

| Rank | Use Case | Parameters | Quality |
|------|----------|------------|---------|
| 8    | Quick experiments | Minimal | Lower |
| 16   | Standard (recommended) | Moderate | Good |
| 32   | High fidelity | Higher | Better |
| 64   | Maximum quality | Most | Best |

**Rule of thumb**: Start with rank=16, increase if performance is insufficient.

### Target Modules

Common LoRA targets in transformer models:

| Module | Purpose | Impact |
|--------|---------|--------|
| `q_proj`, `k_proj`, `v_proj` | Attention queries, keys, values | High |
| `o_proj` | Attention output | Medium |
| `gate_proj`, `up_proj`, `down_proj` | FFN layers | High |
| `embed_tokens` | Input embeddings | Low (usually not used) |
| `lm_head` | Output layer | Low (usually not used) |

**Recommendation**: Focus on attention and FFN layers for best results.

## Advanced Usage

### Analyzing Specific Layers

```python
import json

# Load LoRA analysis
with open("model_comparison/lora_analysis.json") as f:
    analysis = json.load(f)

# Find layers with highest rank
for layer, info in sorted(analysis["analysis"].items(), 
                         key=lambda x: x[1]["rank_95_energy"], 
                         reverse=True)[:5]:
    print(f"{layer}: rank={info['rank_95_energy']}")
```

### Selective Fine-tuning

Fine-tune only specific detected layers:

```python
# From lora_metadata.json, identify key layers
target_layers = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj"
]

# Use in fine-tuning
python finetune_with_lora.py \
    --target-modules self_attn.q_proj self_attn.k_proj self_attn.v_proj \
    --rank 16
```

### Multi-task Fine-tuning

Continue training on multiple biomedical tasks:

```bash
# Task 1: CRISPR design
python finetune_with_lora.py \
    --lora-weights ./model_comparison/extracted_lora_weights.safetensors \
    --dataset biomni/crispr-data \
    --output-dir ./models/crispr

# Task 2: Drug discovery (start from Task 1)
python finetune_with_lora.py \
    --lora-weights ./models/crispr/lora_adapter \
    --dataset biomni/drug-discovery \
    --output-dir ./models/drug-discovery
```

## Troubleshooting

### Model Download Issues

```bash
# Set HuggingFace token
export HF_TOKEN="your_token_here"

# Or login
huggingface-cli login
```

### Out of Memory

```bash
# Reduce batch size
python finetune_with_lora.py --batch-size 1

# Use gradient checkpointing (automatic)
# Or use smaller rank
python finetune_with_lora.py --rank 8
```

### LoRA Not Found

If no LoRA is detected:
- The model may be fully fine-tuned
- Try different threshold: `--threshold 1e-7`
- Check specific layers manually
- Consider applying new LoRA on top

### Weight Loading Errors

```python
# Manual weight loading
from safetensors.torch import load_file
weights = load_file("extracted_lora_weights.safetensors")

# Check weight keys
print(list(weights.keys())[:10])

# Load with strict=False
model.load_state_dict(weights, strict=False)
```

## Best Practices

1. **Always compare first**: Run comparison before fine-tuning
2. **Start with low rank**: Use rank=8 or 16 initially
3. **Monitor reconstruction error**: Higher rank if error > 0.1
4. **Use appropriate datasets**: Biomedical data for biomedical models
5. **Validate thoroughly**: Test on held-out biomedical benchmarks
6. **Save checkpoints**: Enable checkpoint saving during training
7. **Track experiments**: Use wandb or tensorboard

## Example Workflow

```bash
# 1. Compare and extract LoRA
python Compare_R0_qwen3.py --lora-rank 16

# 2. Review the report
cat model_comparison/comparison_report.txt

# 3. Fine-tune with extracted LoRA
python finetune_with_lora.py \
    --lora-weights model_comparison/extracted_lora_weights.safetensors \
    --dataset your/biomedical/dataset \
    --epochs 3 \
    --output-dir ./finetuned_biomni

# 4. Evaluate with A1 agent
python run_R0.py \
    --base-url http://localhost:30000 \
    --model-path ./finetuned_biomni/merged_model
```

## Citation

If you use this extraction method, please cite both Biomni-R0 and Qwen3:

```bibtex
@misc{biomnir0,
  title     = {Biomni-R0: Using RL to Hill-Climb Biomedical Reasoning Agents to Expert-Level},
  author    = {Ryan Li and Kexin Huang and Shiyi Cao and Yuanhao Qu and Jure Leskovec},
  year      = {2025},
  month     = {September},
  note      = {Technical Report}
}

@misc{qwen3,
  title     = {Qwen3 Technical Report}, 
  author    = {Qwen Team},
  year      = {2025},
  eprint    = {2505.09388},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```

## Additional Resources

- [Biomni-R0 Model Card](https://huggingface.co/biomni/Biomni-R0-32B-Preview)
- [Qwen3-32B-FP8 Model Card](https://huggingface.co/Qwen/Qwen3-32B-FP8)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)


