import torch
import os
import shutil
import json
from safetensors.torch import load_file, save_file
from glob import glob
from tqdm import tqdm
import gc

# ================= CONFIGURATION =================
# INPUT: The original FP8 folder
INPUT_DIR = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B-FP8"

# OUTPUT: The corrected BF16 model
OUTPUT_DIR = "./qwen_bf16_robust"
# =================================================

def expand_scale_dynamic(scale, target_shape, layer_name):
    """
    Dynamically calculates block size and expands scale to match target weight.
    """
    # 1. Check Dimensions
    if len(scale.shape) != 2 or len(target_shape) != 2:
        # Fallback for non-2D scales (unlikely in this model but safe to have)
        return scale
        
    scale_h, scale_w = scale.shape
    target_h, target_w = target_shape
    
    # 2. Calculate Block Sizes
    if target_h % scale_h != 0 or target_w % scale_w != 0:
        raise ValueError(f"CRITICAL MISMATCH in {layer_name}: "
                         f"Weight {target_shape} is not divisible by Scale {scale.shape}")
                         
    repeat_h = target_h // scale_h
    repeat_w = target_w // scale_w
    
    # 3. Expand
    # Repeat rows, then repeat columns
    expanded = scale.repeat_interleave(repeat_h, dim=0).repeat_interleave(repeat_w, dim=1)
    
    return expanded

def main():
    print(f"🚀 Starting Robust Dynamic Dequantization...")
    print(f"   Input: {INPUT_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    safetensor_files = sorted(list(glob(os.path.join(INPUT_DIR, "*.safetensors"))))
    
    total_quantized = 0
    total_skipped = 0
    
    for file_path in tqdm(safetensor_files, desc="Processing Shards"):
        file_name = os.path.basename(file_path)
        
        # Load to CPU
        state_dict = load_file(file_path, device="cpu")
        new_state_dict = {}
        
        keys = list(state_dict.keys())
        for key in keys:
            # Skip the scale keys themselves (we process them inside the weight logic)
            if "weight_scale_inv" in key:
                continue
            
            tensor = state_dict[key]
            scale_key = key.replace(".weight", ".weight_scale_inv")
            
            if scale_key in state_dict:
                # === QUANTIZED LAYER ===
                scale_inv = state_dict[scale_key]
                
                # 1. Dynamic Expansion
                try:
                    scale_expanded = expand_scale_dynamic(scale_inv, tensor.shape, key)
                except ValueError as e:
                    print(f"\n❌ Error: {e}")
                    exit(1)
                
                # 2. Multiplication (The Fix)
                # Cast to float32 -> Multiply -> Cast to BF16
                w_real = tensor.to(torch.float32) * scale_expanded.to(torch.float32)
                
                # 3. Shape Safety Check
                if w_real.shape != tensor.shape:
                    print(f"\n❌ SHAPE ERROR: {key} changed from {tensor.shape} to {w_real.shape}")
                    exit(1)
                
                new_state_dict[key] = w_real.to(torch.bfloat16)
                total_quantized += 1
                
            else:
                # === NON-QUANTIZED LAYER ===
                # Norms, Embeddings, Biases
                new_state_dict[key] = tensor.to(torch.bfloat16)
                total_skipped += 1

        # Save
        output_file = os.path.join(OUTPUT_DIR, file_name)
        save_file(new_state_dict, output_file)
        
        del state_dict, new_state_dict
        gc.collect()

    # Metadata & Config
    print("🧹 Processing Metadata...")
    all_files = glob(os.path.join(INPUT_DIR, "*"))
    for src in all_files:
        if not src.endswith(".safetensors"):
            dst = os.path.join(OUTPUT_DIR, os.path.basename(src))
            if os.path.isdir(src): continue
            shutil.copy2(src, dst)

    # Sanitize Config
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        if "quantization_config" in config:
            del config["quantization_config"]
        config["torch_dtype"] = "bfloat16"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Clean Index
    index_path = os.path.join(OUTPUT_DIR, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        new_map = {k: v for k, v in index["weight_map"].items() if "weight_scale_inv" not in k}
        index["weight_map"] = new_map
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    print(f"\n✅ Done.")
    print(f"   Quantized Layers Processed: {total_quantized}")
    print(f"   Standard Layers Copied:     {total_skipped}")
    print("   Please run cpu_autopsy.py on the new folder.")

if __name__ == "__main__":
    main()