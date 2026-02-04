import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from datasets import Dataset

# ================= CONFIGURATION =================
# Disable tokenizer parallelism to prevent deadlocks/warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview"
OUTPUT_DIR = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-FP8-CustomCalib"
DATA_PATH = "calibration_data.json" 

# Reduced slightly to ensure it fits in VRAM during the heavy calculation phase
# If you have 80GB A100s, you can set this back to 8192
MAX_SEQ_LEN = 4096  
# =================================================

def main():
    # --- STEP 0: HARDWARE CHECK ---
    print("🔍 Checking hardware...")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CRITICAL ERROR: CUDA is not available. PyTorch is running on CPU. Aborting to save time.")
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ Found {num_gpus} GPUs. CUDA Version: {torch.version.cuda}")
    if num_gpus < 1:
        raise RuntimeError("❌ CRITICAL ERROR: No GPUs detected.")

    print(f"🚀 Loading Biomni BF16 from: {MODEL_PATH}")
    
    # --- STEP 1: LOAD MODEL ---
    # device_map="auto" spreads the model across your 4 GPUs
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # --- STEP 2: LOAD DATA ---
    print(f"📂 Loading custom calibration data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        exit(1)
        
    with open(DATA_PATH, "r") as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, list) and isinstance(raw_data[0], str):
        text_samples = raw_data
    else:
        text_samples = raw_data # Handle assuming list of strings
        
    print(f"   Loaded {len(text_samples)} samples.")
    
    # Create dataset
    calibration_data = Dataset.from_dict({"text": text_samples})

    # --- STEP 3: DEFINE RECIPE ---
    print("⚖️  Configuring FP8 Recipe (Group Size 128)...")
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8", 
        ignore=["lm_head"]
    )

    # --- STEP 4: RUN QUANTIZATION ---
    print(f"🧪 Running Calibration on {len(calibration_data)} samples using {num_gpus} GPUs...")
    print("   (This usually takes 10-20 minutes on GPUs)")

    # Apply quantization
    oneshot(
        model=model,
        dataset=calibration_data, 
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=len(calibration_data), 
    )

    # --- STEP 5: SAVE ---
    print(f"💾 Saving High-Fidelity FP8 Model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Done.")

if __name__ == "__main__":
    main()