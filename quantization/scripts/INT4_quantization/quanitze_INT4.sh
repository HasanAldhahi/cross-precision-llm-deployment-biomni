#!/bin/bash

#SBATCH --job-name=compare_models
#SBATCH --output=compare_models_%j.log
#SBATCH --error=compare_models_%j.err
#SBATCH --time=48:00:00
#SBATCH -p kisski-h100
#SBATCH -G H100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH -C inet


echo "==============================================="
echo "GPTQ Evaluation Client"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "==============================================="

# Load modules
module load gcc/13.2.0
module load gcc/13.2.0-nvptx
module load cuda/12.6.2
module load miniforge3/24.3.0-0

source activate clean_data_bio
export PYTHONNOUSERSITE=1

# Activate conda


# Proxy settings
export NO_PROXY="localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.usr.hpc.gwdg.de"
export no_proxy="$NO_PROXY"

# Change to the right directory
cd /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/quantization/scripts

echo ""
echo "Starting AWQ evaluation..."
python quantize_AWQ_INT4.py


echo ""
echo "==============================================="
echo "AWQ Evaluation Complete"
echo "==============================================="

