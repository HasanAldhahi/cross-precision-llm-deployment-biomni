# Biomni-R0 LoRA Extraction & Fine-tuning Workflow

## Quick Start (3 Steps)

### Step 1: Extract LoRA
```bash
python Compare_R0_qwen3.py
```

### Step 2: Analyze Results
```bash
python analyze_lora_results.py
```

### Step 3: Fine-tune
```bash
python finetune_with_lora.py \
    --lora-weights model_comparison/extracted_lora_weights.safetensors \
    --dataset your/dataset
```

## Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LORA EXTRACTION WORKFLOW                      │
└─────────────────────────────────────────────────────────────────────┘

                                  START
                                    │
                                    ▼
          ┌──────────────────────────────────────────┐
          │  1. Compare Models                       │
          │  python Compare_R0_qwen3.py              │
          │                                          │
          │  Downloads:                              │
          │  • Qwen/Qwen3-32B-FP8 (base)            │
          │  • biomni/Biomni-R0-32B-Preview (FT)    │
          └──────────────────────────────────────────┘
                                    │
                                    ▼
          ┌──────────────────────────────────────────┐
          │  Comparison Analysis                     │
          │                                          │
          │  ✓ Config differences                   │
          │  ✓ Architecture comparison              │
          │  ✓ Weight differences                   │
          │  ✓ LoRA pattern detection               │
          └──────────────────────────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────┐
                         │  Found LoRA?    │
                         └─────────────────┘
                          │              │
                    YES   │              │   NO
                          ▼              ▼
              ┌───────────────┐   ┌──────────────┐
              │ Explicit      │   │ Low-rank     │
              │ LoRA Layers   │   │ Patterns     │
              └───────────────┘   └──────────────┘
                          │              │
                          └──────┬───────┘
                                 ▼
          ┌──────────────────────────────────────────┐
          │  2. Analyze Results                      │
          │  python analyze_lora_results.py          │
          │                                          │
          │  Shows:                                  │
          │  • Modified layers                       │
          │  • LoRA rank recommendations             │
          │  • Target modules                        │
          │  • Next steps                            │
          └──────────────────────────────────────────┘
                                    │
                                    ▼
          ┌──────────────────────────────────────────┐
          │  3. Fine-tune with LoRA                  │
          │  python finetune_with_lora.py            │
          │                                          │
          │  Options:                                │
          │  • Use extracted weights                 │
          │  • Use detected config                   │
          │  • Fresh LoRA on FT model               │
          └──────────────────────────────────────────┘
                                    │
                                    ▼
          ┌──────────────────────────────────────────┐
          │  Outputs                                 │
          │                                          │
          │  • LoRA adapter (small, shareable)       │
          │  • Merged model (full model)             │
          └──────────────────────────────────────────┘
                                    │
                                    ▼
          ┌──────────────────────────────────────────┐
          │  4. Deploy & Evaluate                    │
          │                                          │
          │  • Serve with SGLang/vLLM                │
          │  • Use with Biomni A1 agent              │
          │  • Evaluate on biomedical benchmarks     │
          └──────────────────────────────────────────┘
                                    │
                                    ▼
                                   END
```

## Files Generated

### Comparison Results (`model_comparison/`)
```
model_comparison/
├── config_comparison.json          # Configuration differences
├── architecture_comparison.json    # Layer structure analysis
├── weight_differences.json         # Weight modification details
├── lora_analysis.json             # LoRA pattern detection results
├── lora_metadata.json             # LoRA configuration info
├── extracted_lora_weights.safetensors  # Ready-to-use LoRA weights
└── comparison_report.txt          # Human-readable summary
```

### Fine-tuned Model (`finetuned_model/`)
```
finetuned_model/
├── lora_adapter/                  # LoRA adapter only (~100MB)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files
└── merged_model/                  # Full merged model (~64GB)
    ├── config.json
    ├── model.safetensors
    └── tokenizer files
```

## Three Scenarios

### 🎯 Scenario A: Explicit LoRA Found
```
architecture_comparison.json shows:
  "lora_keys": [
    "model.layers.0.self_attn.q_proj.lora_A",
    "model.layers.0.self_attn.q_proj.lora_B",
    ...
  ]
```

**What this means**: The model has dedicated LoRA adapter layers

**Action**:
```bash
# Extract the LoRA layers directly
python Compare_R0_qwen3.py

# Use them for fine-tuning
python finetune_with_lora.py \
    --lora-weights model_comparison/extracted_lora_weights.safetensors
```

### ⚙️ Scenario B: LoRA Patterns Detected
```
lora_analysis.json shows:
  "potential_lora_layers": [...],
  "analysis": {
    "layer_name": {
      "rank_95_energy": 24,
      "is_potential_lora": true
    }
  }
```

**What this means**: Weight differences show low-rank patterns

**Action**:
```bash
# Extracted weights available via SVD decomposition
python finetune_with_lora.py \
    --lora-weights model_comparison/extracted_lora_weights.safetensors \
    --rank 32
```

### 📚 Scenario C: Full Fine-tuning
```
weight_differences.json shows:
  Most layers have high modification ratios
  No low-rank patterns detected
```

**What this means**: All parameters were modified during training

**Action**:
```bash
# Apply fresh LoRA on top of the finetuned model
python finetune_with_lora.py \
    --base-model biomni/Biomni-R0-32B-Preview \
    --rank 16 \
    --target-modules q_proj k_proj v_proj o_proj
```

## Key Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `Compare_R0_qwen3.py` | Model comparison & LoRA extraction | Model names | Comparison results + LoRA weights |
| `analyze_lora_results.py` | Visualize & summarize results | Comparison dir | Analysis report |
| `finetune_with_lora.py` | Fine-tune with LoRA | LoRA weights + dataset | Fine-tuned model |
| `run_R0.py` | Serve & evaluate model | Model path | Evaluation results |

## Common Commands

### Basic Workflow
```bash
# 1. Extract LoRA
python Compare_R0_qwen3.py

# 2. Analyze
python analyze_lora_results.py

# 3. Fine-tune
python finetune_with_lora.py \
    --lora-weights model_comparison/extracted_lora_weights.safetensors \
    --dataset biomni/your-dataset

# 4. Evaluate
python run_R0.py \
    --base-url http://localhost:30000 \
    --model-path ./finetuned_model/merged_model
```

### Custom Configurations
```bash
# Higher rank extraction
python Compare_R0_qwen3.py --lora-rank 32

# Selective fine-tuning
python finetune_with_lora.py \
    --target-modules q_proj k_proj v_proj \
    --rank 16

# Different base model
python Compare_R0_qwen3.py \
    --base Qwen/Qwen3-32B \
    --finetuned biomni/Biomni-R0-32B-Preview
```

### Multi-task Learning
```bash
# Task 1: CRISPR
python finetune_with_lora.py \
    --lora-weights model_comparison/extracted_lora_weights.safetensors \
    --dataset biomni/crispr-data \
    --output-dir ./models/crispr

# Task 2: Drug discovery (continue from Task 1)
python finetune_with_lora.py \
    --lora-weights ./models/crispr/lora_adapter \
    --dataset biomni/drug-discovery \
    --output-dir ./models/drug-discovery
```

## Parameter Guide

### LoRA Rank Selection
- **Rank 8**: Quick experiments, minimal parameters
- **Rank 16**: Standard (recommended for most cases)
- **Rank 32**: Higher fidelity, more parameters
- **Rank 64**: Maximum quality, largest adapter

### Target Modules
```python
# Attention-only (faster, less parameters)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Full (better performance, more parameters)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"      # FFN
]
```

### Training Hyperparameters
```bash
# Conservative (stable)
--learning-rate 1e-4 --batch-size 4 --epochs 3

# Aggressive (faster, less stable)
--learning-rate 5e-4 --batch-size 8 --epochs 5

# Production (balanced)
--learning-rate 2e-4 --batch-size 4 --epochs 3
```

## GPU Requirements

| Task | GPU Memory | Recommended GPUs |
|------|-----------|------------------|
| Model comparison | ~80GB | 1×A100 or 2×A40 |
| LoRA fine-tuning (rank 16) | ~40GB | 1×A100 or 1×A40 |
| LoRA fine-tuning (rank 64) | ~60GB | 1×A100 |
| Full model inference | ~70GB | 1×A100 or 2×A40 |

## Troubleshooting Decision Tree

```
Issue: Out of Memory
├─ During comparison? → Use smaller threshold (--threshold 1e-7)
├─ During fine-tuning? → Reduce batch size (--batch-size 1)
└─ During inference? → Use quantization (FP8/FP16)

Issue: No LoRA Found
├─ Try lower threshold → --threshold 1e-8
├─ Check specific layers → analyze_lora_results.py
└─ Model fully fine-tuned → Apply fresh LoRA on top

Issue: Poor Performance
├─ Increase LoRA rank → --rank 32 or --rank 64
├─ Add more target modules → Include FFN layers
└─ More training → --epochs 5 --learning-rate 1e-4

Issue: Model Loading Error
├─ Check HF token → huggingface-cli login
├─ Download manually → huggingface-cli download <model>
└─ Check disk space → df -h
```

## Best Practices Checklist

- [ ] Run comparison first to understand model differences
- [ ] Start with low rank (8-16) and increase if needed
- [ ] Monitor reconstruction error in LoRA extraction
- [ ] Use biomedical datasets for biomedical models
- [ ] Validate on held-out biomedical benchmarks
- [ ] Save checkpoints during training
- [ ] Track experiments (wandb/tensorboard)
- [ ] Test with A1 agent before deployment
- [ ] Document your modifications
- [ ] Share LoRA adapters (smaller than full models)

## Additional Resources

- Full guide: [LORA_EXTRACTION_GUIDE.md](LORA_EXTRACTION_GUIDE.md)
- Evaluation: [README.md](README.md)
- Quick start: [QUICKSTART.md](QUICKSTART.md)
- Biomni-R0: https://huggingface.co/biomni/Biomni-R0-32B-Preview
- Qwen3-32B-FP8: https://huggingface.co/Qwen/Qwen3-32B-FP8


