#!/bin/bash
#SBATCH --job-name=compare_models
#SBATCH --output=compare_models_%j.log
#SBATCH --error=compare_models_%j.err
#SBATCH --time=01:00:00
#SBATCH -p kisski-h100
#SBATCH -G H100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH -C inet

# Set conda environment path
CONDA_ENV_PATH="/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni"

echo "==============================================="
echo "Starting Biomni-R0 Model Comparison"
echo "==============================================="
echo "Conda environment: $CONDA_ENV_PATH"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "==============================================="

# Set up the Biomni environment directly (most reliable method for SLURM)
echo "Setting up Biomni environment..."

# Direct PATH manipulation - prepend to ensure our python is used first
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export PYTHONPATH="$CONDA_ENV_PATH/lib/python3.11/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"

# Set other environment variables
export CONDA_PREFIX="$CONDA_ENV_PATH"
export CONDA_DEFAULT_ENV="$CONDA_ENV_PATH"

echo "Environment variables set:"
echo "  PATH starts with: $(echo $PATH | cut -d: -f1-3)"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  CONDA_PREFIX: $CONDA_PREFIX"

# Verify Python and packages BEFORE changing directory
echo "==============================================="
echo "Verifying Python environment..."
python_location=$(which python)
echo "Python location: $python_location"

if [ "$python_location" != "$CONDA_ENV_PATH/bin/python" ]; then
    echo "WARNING: Python not from conda environment!"
    echo "Expected: $CONDA_ENV_PATH/bin/python"
    echo "Actual: $python_location"
fi

python --version || { echo "❌ Python not working"; exit 1; }

echo "Testing required packages..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || { echo "❌ PyTorch not available"; exit 1; }
python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')" || { echo "❌ Transformers not available"; exit 1; }
python -c "import safetensors; print(f'✅ Safetensors: {safetensors.__version__}')" || { echo "❌ Safetensors not available"; exit 1; }
python -c "import numpy; print(f'✅ NumPy: {numpy.__version__}')" || { echo "❌ NumPy not available"; exit 1; }

echo "==============================================="
echo "Environment verification complete!"
echo "==============================================="

# Change to the script directory
cd /user/haldhah/u17285/thesis/biomni_integration/Biomni/R0 || {
    echo "❌ Cannot change to script directory"
    exit 1
}

echo "Current working directory: $(pwd)"

# Test that our script can import the required modules
echo "Testing script imports..."
python -c "import sys; sys.path.append('.'); from Compare_R0_qwen3 import ModelComparator; print('✅ Script imports work')" || {
    echo "❌ Script import failed"
    exit 1
}

echo "==============================================="
echo "Starting model comparison..."

# Run comparison with different ranks
echo "Running comparison with default rank..."
python Compare_R0_qwen3.py

echo "Running comparison with LoRA rank 32..."
python Compare_R0_qwen3.py --lora-rank 32

# Analyze results
echo "Analyzing comparison results..."
python analyze_lora_results.py

echo "==============================================="
echo "Model comparison completed successfully!"
echo "End time: $(date)"
echo "Check logs: compare_models_${SLURM_JOB_ID}.log"
echo "Check errors: compare_models_${SLURM_JOB_ID}.err"
echo "==============================================="