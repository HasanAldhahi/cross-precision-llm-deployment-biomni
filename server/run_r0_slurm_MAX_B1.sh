#!/bin/bash
echo "==============================================="
echo "Starting Biomni-R0 Evaluation Client"
echo "Node: $(hostname)"
echo "==============================================="

# Activate your conda environment
echo "Activating conda environment..."
source activate /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni

# Go to the directory where the evaluation script is
cd /mnt/vast-kisski/home/haldhah/u17285/thesis/biomni_integration/Biomni/compare

# 1. Set the EXTERNAL proxy as instructed by admins
export http_proxy="http://www-cache.gwdg.de:3128"
export https_proxy="http://www-cache.gwdg.de:3128"
export HTTP_PROXY=$http_proxy # Set uppercase for robustness
export HTTPS_PROXY=$https_proxy # Set uppercase for robustness

# 2. Set the INTERNAL exception list to bypass proxy for cluster communication
export no_proxy="localhost,127.0.0.1,.usr.hpc.gwdg.de,.gwdg.de,ggpu171,10.241.149.24"
export NO_PROXY=$no_proxy

# Run the evaluation script
echo "Starting the evaluation..."
python run_eval_R0_async_MAX_B1_Eval1_benchmark.py

echo "Evaluation complete."