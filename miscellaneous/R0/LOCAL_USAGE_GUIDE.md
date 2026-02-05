# Local Model Usage Guide

The scripts have been configured to use your local model directories instead of downloading from HuggingFace.

## Model Locations

- **Qwen3-32B**: `/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B`
- **Biomni-R0-32B-Preview**: `/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview`
- **Biomni Environment**: `/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni_Env`
- **Conda Environment**: `/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni`

## Quick Start

### 1. Compare Models and Extract LoRA

```bash
# Uses default local paths (no download needed)
python Compare_R0_qwen3.py
```

This will:
- ✅ Use local Qwen3-32B model
- ✅ Use local Biomni-R0-32B-Preview model
- ✅ No HuggingFace downloads
- ✅ Generate comparison results in `./model_comparison/`

### 2. Run R0 Evaluation

```bash
# Start server with local model (requires GPU allocation)
srun -p kisski -G A100:1 python run_R0.py --start-server
```

Or in a SLURM job:

```bash
sbatch run_r0_slurm.sh
```

### 3. Start Server Manually

```bash
# Start SGLang server with local model
./start_server.sh
```

## Customization

### Compare with Different Local Paths

```bash
python Compare_R0_qwen3.py \
    --base-path /path/to/your/base/model \
    --finetuned-path /path/to/your/finetuned/model \
    --output-dir ./my_comparison
```

### Extract LoRA with Custom Rank

```bash
python Compare_R0_qwen3.py --lora-rank 32
```

### Use Different Output Directory

```bash
python Compare_R0_qwen3.py --output-dir /path/to/output
```

## GPU Allocation on HPC

### Interactive Session

```bash
# Request 1 A100 GPU
srun --pty -p kisski -n 1 -G A100:1 -C inet bash

# Then run comparison (no GPU needed for comparison, but useful for testing)
python Compare_R0_qwen3.py

# Or run evaluation (requires GPU)
python run_R0.py --start-server
```

### Batch Job for Comparison

```bash
#!/bin/bash
#SBATCH --job-name=compare-models
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --partition=kisski

# Activate conda environment
source /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni/bin/activate

# Run comparison
cd /user/haldhah/u17285/thesis/biomni_integration/Biomni/R0
python Compare_R0_qwen3.py --lora-rank 16

# Analyze results
python analyze_lora_results.py
```

Save as `compare_job.sh` and submit:
```bash
sbatch compare_job.sh
```

## Verification

### Check Models Exist

```bash
# Verify Qwen3-32B
ls /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B/*.safetensors | head -5

# Verify Biomni-R0
ls /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview/*.safetensors | head -5

# Verify Biomni Environment
ls /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni_Env/biomni_data/
```

### Test Setup

```bash
python test_setup.py
```

## Expected Output

### Comparison Script

```
================================================================================
MODEL COMPARISON ANALYSIS
================================================================================
Base Model:      /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B
Finetuned Model: /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview
Output Directory: ./model_comparison
================================================================================
✅ Base model directory found
✅ Finetuned model directory found

📥 Verifying local model files...
  ✅ Base model: Found 27 safetensors files
  ✅ Finetuned model: Found 27 safetensors files

📋 Comparing Model Configurations...
...
```

### Output Files

After running `Compare_R0_qwen3.py`, you'll get:

```
model_comparison/
├── config_comparison.json              # Configuration differences
├── architecture_comparison.json        # Layer structure analysis
├── weight_differences.json             # Detailed weight analysis
├── lora_analysis.json                 # LoRA pattern detection
├── lora_metadata.json                 # LoRA configuration
├── extracted_lora_weights.safetensors # Extracted LoRA weights
└── comparison_report.txt              # Human-readable summary
```

## Troubleshooting

### Directory Not Found

```bash
# Check if directories exist
ls -ld /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B
ls -ld /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview

# If missing, clone from HuggingFace
cd /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/
git clone https://huggingface.co/Qwen/Qwen3-32B
git clone https://huggingface.co/biomni/Biomni-R0-32B-Preview
```

### Out of Memory

```bash
# The comparison script can use a lot of RAM
# Request more memory in SLURM:
#SBATCH --mem=400G

# Or process fewer layers at once by modifying the script
```

### Permission Denied

```bash
# Make sure scripts are executable
chmod +x Compare_R0_qwen3.py
chmod +x run_R0.py
chmod +x start_server.sh
chmod +x analyze_lora_results.py
```

## Next Steps After Comparison

1. **Analyze Results**:
   ```bash
   python analyze_lora_results.py
   cat model_comparison/comparison_report.txt
   ```

2. **Fine-tune with Extracted LoRA**:
   ```bash
   python finetune_with_lora.py \
       --lora-weights model_comparison/extracted_lora_weights.safetensors \
       --dataset your/dataset
   ```

3. **Evaluate with A1 Agent**:
   ```bash
   # Request GPU
   srun -p kisski -G A100:1 python run_R0.py --start-server
   ```

## Performance Notes

### Comparison Script
- **Time**: ~30-60 minutes (depends on disk I/O)
- **Memory**: ~200GB RAM
- **GPU**: Not required (CPU only)
- **Storage**: ~5GB for results

### Evaluation Script
- **Time**: Varies by task (10min - 2hr)
- **Memory**: ~100GB RAM
- **GPU**: 1×A100 (80GB) or 2×A40 (40GB each)
- **Storage**: Minimal

## Additional Resources

- Full documentation: [README.md](README.md)
- LoRA extraction guide: [LORA_EXTRACTION_GUIDE.md](LORA_EXTRACTION_GUIDE.md)
- Quick reference: [QUICKSTART.md](QUICKSTART.md)
- Workflow diagram: [WORKFLOW_SUMMARY.md](WORKFLOW_SUMMARY.md)

