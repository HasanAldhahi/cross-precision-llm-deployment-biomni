# Biomni-R0 Scripts & Documentation Index

Complete toolkit for running, comparing, and fine-tuning the Biomni-R0-32B-Preview model.

## 📋 Quick Navigation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference for common tasks
- **[WORKFLOW_SUMMARY.md](WORKFLOW_SUMMARY.md)** - Complete workflow visualization

### Detailed Guides
- **[README.md](README.md)** - Full documentation for model evaluation
- **[LORA_EXTRACTION_GUIDE.md](LORA_EXTRACTION_GUIDE.md)** - LoRA extraction & fine-tuning guide

## 🚀 Core Scripts

### Model Evaluation
| Script | Purpose | Usage |
|--------|---------|-------|
| `run_R0.py` | Main evaluation script with A1 agent | `python run_R0.py --start-server` |
| `start_server.sh` | Standalone SGLang server launcher | `./start_server.sh --tp 2` |
| `run_r0_slurm.sh` | SLURM batch job script | `sbatch run_r0_slurm.sh` |
| `test_setup.py` | Verify installation and dependencies | `python test_setup.py` |

### LoRA Extraction & Fine-tuning
| Script | Purpose | Usage |
|--------|---------|-------|
| `Compare_R0_qwen3.py` | Compare models & extract LoRA | `python Compare_R0_qwen3.py` |
| `analyze_lora_results.py` | Analyze comparison results | `python analyze_lora_results.py` |
| `finetune_with_lora.py` | Fine-tune with extracted LoRA | `python finetune_with_lora.py --lora-weights <path>` |

## 📚 Documentation Files

| File | Content |
|------|---------|
| `README.md` | Complete evaluation documentation |
| `QUICKSTART.md` | Quick start guide |
| `LORA_EXTRACTION_GUIDE.md` | Detailed LoRA extraction guide |
| `WORKFLOW_SUMMARY.md` | Workflow diagram and summary |
| `INDEX.md` | This file - navigation index |

## 🔄 Complete Workflows

### Workflow 1: Model Evaluation
```bash
# 1. Verify setup
python test_setup.py

# 2. Run evaluation (auto server management)
python run_R0.py --start-server

# 3. Or use SLURM
sbatch run_r0_slurm.sh
```

### Workflow 2: LoRA Extraction & Fine-tuning
```bash
# 1. Extract LoRA weights
python Compare_R0_qwen3.py

# 2. Analyze results
python analyze_lora_results.py

# 3. Fine-tune
python finetune_with_lora.py \
    --lora-weights model_comparison/extracted_lora_weights.safetensors \
    --dataset your/dataset
```

### Workflow 3: HPC Deployment
```bash
# 1. Submit comparison job
sbatch -J compare --wrap "python Compare_R0_qwen3.py"

# 2. Submit fine-tuning job
sbatch -J finetune --dependency=afterok:$JOBID \
    --wrap "python finetune_with_lora.py --lora-weights model_comparison/extracted_lora_weights.safetensors"

# 3. Evaluate
sbatch run_r0_slurm.sh
```

## 📊 Output Structure

### Model Comparison Results
```
model_comparison/
├── config_comparison.json              # Configuration diffs
├── architecture_comparison.json        # Layer structure
├── weight_differences.json             # Weight analysis
├── lora_analysis.json                 # LoRA patterns
├── lora_metadata.json                 # LoRA config
├── extracted_lora_weights.safetensors # Extracted LoRA
└── comparison_report.txt              # Summary report
```

### Fine-tuned Model
```
finetuned_model/
├── lora_adapter/                      # LoRA only (~100MB)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── merged_model/                      # Full model (~64GB)
    ├── config.json
    ├── model.safetensors
    └── ...
```

## 🎯 Common Use Cases

### Use Case 1: Run Biomni-R0 Evaluation
```bash
python run_R0.py --start-server
```
**See**: [QUICKSTART.md](QUICKSTART.md)

### Use Case 2: Extract LoRA for Analysis
```bash
python Compare_R0_qwen3.py
python analyze_lora_results.py
```
**See**: [LORA_EXTRACTION_GUIDE.md](LORA_EXTRACTION_GUIDE.md)

### Use Case 3: Fine-tune on Custom Dataset
```bash
python finetune_with_lora.py \
    --lora-weights model_comparison/extracted_lora_weights.safetensors \
    --dataset my/biomedical/data \
    --epochs 5
```
**See**: [LORA_EXTRACTION_GUIDE.md](LORA_EXTRACTION_GUIDE.md)

### Use Case 4: Deploy on HPC Cluster
```bash
sbatch run_r0_slurm.sh "Your custom task"
```
**See**: [README.md](README.md#gpu-allocation-on-hpc)

## 🛠️ Dependencies

### Required Packages
```bash
pip install transformers torch safetensors huggingface-hub peft datasets sglang requests numpy
```

### Optional but Recommended
```bash
pip install wandb tensorboard accelerate bitsandbytes
```

## 📖 Script Details

### run_R0.py
**Purpose**: Run Biomni-R0-32B-Preview evaluation with A1 agent

**Key Features**:
- Automatic SGLang server management
- OpenAI-compatible API integration
- Custom task support
- Configurable GPU allocation

**Main Arguments**:
- `--start-server`: Auto-start SGLang server
- `--base-url`: Use existing server
- `--tp`: Number of GPUs (2 or 4)
- `--task`: Custom evaluation task
- `--rope-factor`: Context extension (1.0-4.0)

### Compare_R0_qwen3.py
**Purpose**: Compare Biomni-R0 with Qwen3-32B-FP8 to extract LoRA

**Key Features**:
- Config & architecture comparison
- Weight difference analysis
- LoRA pattern detection via SVD
- Automatic LoRA extraction

**Main Arguments**:
- `--base`: Base model name
- `--finetuned`: Fine-tuned model name
- `--lora-rank`: Extraction rank (default: 16)
- `--output-dir`: Results directory

### finetune_with_lora.py
**Purpose**: Fine-tune model using extracted LoRA weights

**Key Features**:
- Load extracted LoRA weights
- Auto-detect LoRA configuration
- Multi-task learning support
- Save adapter & merged model

**Main Arguments**:
- `--lora-weights`: Path to extracted weights
- `--dataset`: HuggingFace dataset
- `--rank`: LoRA rank
- `--target-modules`: Modules to adapt
- `--epochs`: Training epochs

### analyze_lora_results.py
**Purpose**: Analyze and visualize comparison results

**Key Features**:
- Summary statistics
- Layer-wise analysis
- Rank recommendations
- Next steps guidance

**Main Arguments**:
- `--dir`: Comparison results directory

## 🔗 External Resources

### Model Cards
- [Biomni-R0-32B-Preview](https://huggingface.co/biomni/Biomni-R0-32B-Preview)
- [Qwen3-32B-FP8](https://huggingface.co/Qwen/Qwen3-32B-FP8)

### Documentation
- [SGLang](https://github.com/sgl-project/sglang)
- [PEFT/LoRA](https://huggingface.co/docs/peft)
- [Transformers](https://huggingface.co/docs/transformers)

### Papers
- [Biomni-R0 Technical Report](https://huggingface.co/biomni/Biomni-R0-32B-Preview)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

## 🆘 Getting Help

### Quick Troubleshooting
1. Check setup: `python test_setup.py`
2. Review GPU: `nvidia-smi`
3. Check logs: `ls -lh logs/`

### Documentation
- Basic usage → [QUICKSTART.md](QUICKSTART.md)
- Detailed guide → [README.md](README.md)
- LoRA workflow → [LORA_EXTRACTION_GUIDE.md](LORA_EXTRACTION_GUIDE.md)
- Full workflow → [WORKFLOW_SUMMARY.md](WORKFLOW_SUMMARY.md)

### Common Issues
- **OOM**: Reduce batch size or use more GPUs
- **Import errors**: Install dependencies
- **Server timeout**: Check GPU availability
- **No LoRA found**: Try lower threshold or different rank

## 📝 Citation

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

---

**Last Updated**: October 2025  
**Version**: 1.0


