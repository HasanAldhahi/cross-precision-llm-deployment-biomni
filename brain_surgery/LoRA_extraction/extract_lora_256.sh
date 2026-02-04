#!/bin/bash
#SBATCH --job-name=dequant_extract
#SBATCH --output=process_%j.log
#SBATCH --error=process_%j.err
#SBATCH --time=04:00:00
#SBATCH -p kisski-h100
#SBATCH -G H100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH -C inet

source activate /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/surgery

# PATHS
BIOMNI_MODEL="/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview"
TEMP_BASE="./temp_qwen_dequantized_bf16" # The folder the python script creates
OUTPUT_LORA="./lora_extraction_results/lora_fp8_corrected_rank_256"

# 2. RUN MERGEKIT
# Now we use the TEMP_BASE (which is now BF16) as the base model.
# MergeKit will see two BF16 models and be happy.
echo "------------------------------------------------"
echo "STEP 2: extracting LoRA using MergeKit..."
echo "------------------------------------------------"

mergekit-extract-lora \
    --model "$BIOMNI_MODEL" \
    --base-model "$TEMP_BASE" \
    --out-path "$OUTPUT_LORA" \
    --max-rank 256 \
    --device cuda \
    --no-lazy-unpickle


echo "✅ ALL DONE. Your mathematically corrected LoRA is in $OUTPUT_LORA"