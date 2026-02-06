#!/bin/bash
#SBATCH --job-name=biomni-r0-eval
#SBATCH --output=logs/biomni_r0_%j.out
#SBATCH --error=logs/biomni_r0_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu

# Biomni-R0-32B-Preview SLURM Job Script
# This script runs the Biomni R0 model evaluation on HPC with GPU allocation

echo "=================================================="
echo "  Biomni-R0-32B-Preview Evaluation (SLURM)"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start Time: $(date)"
echo "=================================================="
echo ""

# Load necessary modules (adjust for your HPC environment)
# module load cuda/12.1
# module load python/3.10
# module load conda

# Activate conda environment
source /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
PORT=30000
TP=2  # Number of GPUs (change to 4 if using 4x40GB GPUs)
ROPE_FACTOR=1.0

# Optional: Custom task
TASK=${1:-"Plan a CRISPR screen to identify genes that regulate T cell exhaustion, measured by the change in T cell receptor (TCR) signaling between acute (interleukin-2 [IL-2] only) and chronic (anti-CD3 and IL-2) stimulation conditions. Generate 32 genes that maximize the perturbation effect."}

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Run evaluation with automatic server management
python run_R0.py \
    --start-server \
    --port $PORT \
    --tp $TP \
    --rope-factor $ROPE_FACTOR \
    --task "$TASK" \
    --timeout 2000

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=================================================="

exit $EXIT_CODE

