# Changes Made for Local Model Usage

## Summary

All scripts have been updated to use your local model directories instead of downloading from HuggingFace.

## Modified Files

### 1. `Compare_R0_qwen3.py` ✅

**Changes:**
- Removed `download_models()` method
- Added `verify_models()` method that checks local directories
- Updated `__init__()` to accept local paths instead of model names:
  - `base_model_path` → `/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B`
  - `finetuned_model_path` → `/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview`
- Updated `compare_configs()` to use local paths
- Updated command-line arguments:
  - `--base` → `--base-path`
  - `--finetuned` → `--finetuned-path`

**Before:**
```python
def __init__(self, base_model_name="Qwen/Qwen3-32B-FP8", 
             finetuned_model_name="biomni/Biomni-R0-32B-Preview", ...)
```

**After:**
```python
def __init__(self, base_model_path="/projects/extern/kisski/.../Qwen3-32B", 
             finetuned_model_path="/projects/extern/kisski/.../Biomni-R0-32B-Preview", ...)
```

### 2. `run_R0.py` ✅

**Changes:**
- Already configured to use local path
- `MODEL_NAME` set to: `/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview`

### 3. `start_server.sh` ✅

**Changes:**
- Added `MODEL_PATH` variable pointing to local directory
- Updated `--model-path` to use `$MODEL_PATH` instead of HuggingFace name

**Before:**
```bash
python -m sglang.launch_server \
    --model-path biomni/Biomni-R0-32B-Preview \
    ...
```

**After:**
```bash
MODEL_PATH="/projects/extern/kisski/.../Biomni-R0-32B-Preview"
python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    ...
```

## New Files Created

### `LOCAL_USAGE_GUIDE.md` ✅
Complete guide for using the scripts with local models, including:
- Model locations
- Quick start commands
- GPU allocation examples
- Troubleshooting tips

### `CHANGES_LOCAL.md` (this file) ✅
Summary of all changes made for local model support

## Verification

Check that models exist:
```bash
ls /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B
ls /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview
```

Expected output:
```
Qwen3-32B:
config.json  model-00001-of-00027.safetensors  ...

Biomni-R0-32B-Preview:
config.json  model-00001-of-00027.safetensors  ...
```

## Usage Examples

### Compare Models (No Download)
```bash
# Default: uses your local model directories
python Compare_R0_qwen3.py

# Custom paths
python Compare_R0_qwen3.py \
    --base-path /path/to/qwen \
    --finetuned-path /path/to/biomni
```

### Run Evaluation
```bash
# Request GPU and run
srun -p kisski -G A100:1 python run_R0.py --start-server

# Or submit batch job
sbatch run_r0_slurm.sh
```

### Start Server
```bash
# Uses local model automatically
./start_server.sh
```

## Benefits

✅ **No Downloads**: Uses existing local models  
✅ **Faster**: No network transfer time  
✅ **Reliable**: Not dependent on HuggingFace availability  
✅ **Storage Efficient**: Models already downloaded once  
✅ **HPC Friendly**: Works in environments with limited internet access  

## Backward Compatibility

The scripts still accept custom paths via command-line arguments:

```bash
# Use different models
python Compare_R0_qwen3.py \
    --base-path /custom/path/to/base/model \
    --finetuned-path /custom/path/to/finetuned/model
```

## Testing

Test the comparison script:
```bash
cd /user/haldhah/u17285/thesis/biomni_integration/Biomni/R0

# Quick test (will verify local models exist)
python Compare_R0_qwen3.py --help

# Full comparison
python Compare_R0_qwen3.py
```

Expected output:
```
================================================================================
MODEL COMPARISON ANALYSIS
================================================================================
Base Model:      /projects/extern/kisski/.../Qwen3-32B
Finetuned Model: /projects/extern/kisski/.../Biomni-R0-32B-Preview
Output Directory: ./model_comparison
================================================================================
✅ Base model directory found
✅ Finetuned model directory found

📥 Verifying local model files...
  ✅ Base model: Found 27 safetensors files
  ✅ Finetuned model: Found 27 safetensors files
...
```

## Next Steps

1. ✅ Models configured to use local paths
2. ✅ Verification complete
3. 🔄 Run comparison: `python Compare_R0_qwen3.py`
4. 🔄 Analyze results: `python analyze_lora_results.py`
5. 🔄 Fine-tune with LoRA: `python finetune_with_lora.py --lora-weights model_comparison/extracted_lora_weights.safetensors`

## Questions?

See:
- [LOCAL_USAGE_GUIDE.md](LOCAL_USAGE_GUIDE.md) - Detailed usage guide
- [WORKFLOW_SUMMARY.md](WORKFLOW_SUMMARY.md) - Complete workflow
- [README.md](README.md) - Full documentation

