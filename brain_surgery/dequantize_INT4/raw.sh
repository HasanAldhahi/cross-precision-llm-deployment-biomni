#!/bin/bash
#SBATCH --job-name=raw_dequant
#SBATCH --output=raw_dequant_%j.log
#SBATCH --error=raw_dequant_%j.err
#SBATCH --time=12:00:00
#SBATCH -p kisski
#SBATCH -G A100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -C inet

source activate /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/surgery

# PATHS
BIOMNI_MODEL="/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview"
RAW_BASE="./qwen_bf16_robust" 
OUTPUT_LORA="./lora_extraction_results/"

# 1. RUN RAW DEQUANTIZATION
echo "------------------------------------------------"
echo "STEP 1: Running Raw Dequantization (DeepSeek Style)..."
echo "------------------------------------------------"
python raw_dequantize_INT4.py

# 2. RUN MERGEKIT
# echo "------------------------------------------------"
# echo "STEP 2: Extracting LoRA with MergeKit..."
# echo "------------------------------------------------"

# # Check if it worked
# if [ ! -f "$RAW_BASE/config.json" ]; then
#     echo "❌ CRITICAL: Raw dequantization failed."
#     exit 1
# fi

# mergekit-extract-lora \
#     --model "$BIOMNI_MODEL" \
#     --base-model "$RAW_BASE" \
#     --out-path "$OUTPUT_LORA" \
#     --max-rank 256 \
#     --device cuda \
#     --no-lazy-unpickle

# echo "✅ DONE."