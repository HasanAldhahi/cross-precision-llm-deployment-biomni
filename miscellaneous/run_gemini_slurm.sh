#!/bin/bash


echo ""
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni

echo "Python: $(which python)"
echo "Python version: $(python --version)"

echo ""
echo "Note: Gemini API keys will be loaded from .env by Python script"
echo "(Bash environment check may show 0, but Python will load 17 keys from .env)"

echo ""
echo "==============================================="
echo "Starting Gemini Evaluation"
echo "==============================================="

cd /user/haldhah/u17285/thesis/biomni_integration/Biomni/compare

# Run the evaluation
python run_eval_gemini_eval1.py

echo ""
echo "==============================================="
echo "Evaluation Complete"
echo "End time: $(date)"
echo "==============================================="


