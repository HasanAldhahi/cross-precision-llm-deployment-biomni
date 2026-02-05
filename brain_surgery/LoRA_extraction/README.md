# LoRA Extraction Scripts

Scripts used to extract LoRA adapters from the Biomni-R0-32B model.

## Files

- `extract_lora_256.sh` - SLURM batch script for LoRA extraction using MergeKit
- `sanitize_lora.py` - Script to sanitize LoRA adapters for vLLM compatibility

## Usage

### Extraction

```bash
sbatch extract_lora_256.sh
```

### Sanitization (for vLLM)

```bash
python sanitize_lora.py
```

The sanitization script:
1. Removes unsupported layers (lm_head, embed_tokens, norms)
2. Fixes adapter_config.json for vLLM compatibility
3. Creates backups before making changes
