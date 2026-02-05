# Biomni-R0-32B-Preview Evaluation

This directory contains scripts to run evaluations using the [Biomni-R0-32B-Preview](https://huggingface.co/biomni/Biomni-R0-32B-Preview) model, a state-of-the-art biomedical AI agent trained with reinforcement learning.

## Model Information

- **Model**: biomni/Biomni-R0-32B-Preview
- **Size**: 32B parameters
- **Context**: Up to 131K tokens (with RoPE scaling)
- **Training**: End-to-end RL using Biomni-E1 environment
- **Performance**: SOTA across 10 biomedical benchmarks

## Requirements

### Hardware
- **Option 1**: 2 GPUs with 80GB VRAM each
- **Option 2**: 4 GPUs with 40GB VRAM each

### Software
- Python 3.10+
- SGLang (for serving the model)
- Biomni environment setup

### Installation

```bash
# Install SGLang
pip install "sglang[all]"

# Or with specific CUDA version
pip install "sglang[all]" --extra-index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Method 1: Automated (Recommended)

Run the evaluation script with automatic server management:

```bash
# Start server and run evaluation (2 GPUs, 80GB each)
python run_R0.py --start-server

# Start server with 4 GPUs (40GB each)
python run_R0.py --start-server --tp 4

# Custom task
python run_R0.py --start-server --task "Your biomedical task here"

# Longer context support (RoPE scaling)
python run_R0.py --start-server --rope-factor 2.0
```

### Method 2: Manual Server Management

**Step 1: Start the SGLang server**

```bash
# Using the helper script (2 GPUs)
./start_server.sh

# Using the helper script (4 GPUs)
./start_server.sh --tp 4

# Or manually
python -m sglang.launch_server \
    --model-path biomni/Biomni-R0-32B-Preview \
    --port 30000 \
    --host 0.0.0.0 \
    --mem-fraction-static 0.8 \
    --tp 2 \
    --trust-remote-code \
    --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":1.0,"original_max_position_embeddings":32768}, "max_position_embeddings": 131072}'
```

**Step 2: Run evaluation**

```bash
# Use existing server
python run_R0.py --base-url http://localhost:30000

# Custom task
python run_R0.py --base-url http://localhost:30000 --task "Your task here"
```

## Command-Line Options

### run_R0.py

| Option | Description | Default |
|--------|-------------|---------|
| `--start-server` | Start SGLang server automatically | - |
| `--base-url` | URL of existing server | - |
| `--port` | Server port | 30000 |
| `--host` | Server host | 0.0.0.0 |
| `--tp` | Tensor parallelism (GPUs) | 2 |
| `--rope-factor` | RoPE scaling factor (1.0-4.0) | 1.0 |
| `--task` | Custom task description | Default CRISPR task |
| `--timeout` | Agent timeout (seconds) | 1000 |

### start_server.sh

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Server port | 30000 |
| `--host` | Server host | 0.0.0.0 |
| `--tp` | Tensor parallelism (GPUs) | 2 |
| `--rope-factor` | RoPE scaling factor (1.0-4.0) | 1.0 |

## RoPE Scaling

The model supports context extension via RoPE scaling:

- **Factor 1.0** (default): 32K context - best for standard tasks
- **Factor 2.0**: 64K context - for longer trajectories
- **Factor 4.0**: 128K context - maximum extension

⚠️ **Note**: Higher scaling factors may degrade performance on shorter tasks. Tune according to your use case.

## Example Tasks

### CRISPR Screen (Default)
```bash
python run_R0.py --start-server
```

### GWAS Variant Prioritization
```bash
python run_R0.py --base-url http://localhost:30000 \
    --task "Prioritize genetic variants from a GWAS study for Type 2 Diabetes"
```

### Rare Disease Diagnosis
```bash
python run_R0.py --base-url http://localhost:30000 \
    --task "Diagnose a rare genetic disorder based on patient symptoms and genomic data"
```

### Drug Discovery
```bash
python run_R0.py --base-url http://localhost:30000 \
    --task "Identify potential drug candidates for treating Alzheimer's disease"
```

## Troubleshooting

### Server Won't Start
- Check GPU availability: `nvidia-smi`
- Verify CUDA installation
- Ensure sufficient VRAM
- Check if port is already in use: `lsof -i :30000`

### Out of Memory
- Reduce batch size (handled by SGLang)
- Use more GPUs with `--tp 4`
- Reduce `--mem-fraction-static` in start_server.sh

### Model Download Issues
- Ensure HuggingFace access
- Check internet connection
- Try manual download:
  ```bash
  huggingface-cli download biomni/Biomni-R0-32B-Preview
  ```

### Agent Timeout
- Increase timeout: `--timeout 2000`
- Check server logs for errors
- Verify environment path is correct

## GPU Allocation on HPC

If using an HPC cluster:

```bash
# Request 2x80GB GPUs
srun --gpus=2 --mem=200G --time=4:00:00 \
    python run_R0.py --start-server

# Request 4x40GB GPUs  
srun --gpus=4 --mem=200G --time=4:00:00 \
    python run_R0.py --start-server --tp 4
```

## Citation

If you use this model, please cite:

```bibtex
@misc{biomnir0,
  title     = {Biomni-R0: Using RL to Hill-Climb Biomedical Reasoning Agents to Expert-Level},
  author    = {Ryan Li and Kexin Huang and Shiyi Cao and Yuanhao Qu and Jure Leskovec},
  year      = {2025},
  month     = {September},
  note      = {Technical Report}
}
```

## Additional Resources

- [Model Card](https://huggingface.co/biomni/Biomni-R0-32B-Preview)
- [SGLang Documentation](https://github.com/sgl-project/sglang)
- [Biomni Technical Report](https://huggingface.co/biomni/Biomni-R0-32B-Preview)


