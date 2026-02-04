import os
import torch
import json
import gc
from glob import glob
from tqdm import tqdm
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoConfig

# ================= CONFIGURATION =================
# Path to your INT4 quantized model
INPUT_MODEL_PATH = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-AWQ-INT4-CustomCalib"

# Path where the BF16 "Bridge Model" will be saved
OUTPUT_MODEL_PATH = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-From-INT4-Bridge-BF16"
# =================================================

def unpack_awq_standard(packed_weight, scales):
    """
    Standard AWQ Dequantization Logic.
    Ref: Matches vLLM/AutoAWQ unpacking order for W4A16.
    
    Args:
        packed_weight (Tensor): INT32 packed tensor [Rows, Cols // 8]
        scales (Tensor): FP16/BF16 scales [Rows, Groups] (usually group size 128)
    """
    # 1. Expand Scales (Block Expansion)
    # The scales are usually [Out_Features, In_Features // Group_Size]
    # We need to repeat them to match the full weight matrix size.
    # Group size is inferred: 25600 / 200 = 128
    group_size = 128
    scales_expanded = scales.repeat_interleave(group_size, dim=1)

    # 2. Unpack Integers (The "Math" Part)
    # We cast to int32 to ensure bitwise operations work correctly
    packed_weight = packed_weight.to(torch.int32)
    
    # Create a container for the 8 sub-weights packed inside each integer
    unpacked_cols = []
    mask = 0xF  # Binary 1111 (extract 4 bits)

    for i in range(8):
        # Shift bits right by i*4 and mask the result
        weight_chunk = (packed_weight >> (i * 4)) & mask
        
        # Handle Sign (Two's Complement for 4-bit)
        # 0-7 remains 0-7. 8-15 becomes -8 to -1.
        weight_chunk = torch.where(weight_chunk >= 8, weight_chunk - 16, weight_chunk)
        
        unpacked_cols.append(weight_chunk)

    # 3. Stack and Reshape
    # Stack along the last dimension [Rows, Packed_Cols, 8]
    weights = torch.stack(unpacked_cols, dim=-1)
    
    # Flatten to get the final matrix [Rows, Real_Cols]
    rows, packed_cols = packed_weight.shape
    weights = weights.view(rows, packed_cols * 8)
    
    # 4. Apply Scaling (Dequantization)
    # W_bf16 = W_int4 * Scale
    
    # Ensure dimensions match (handling potential padding in some odd models)
    common_width = min(weights.shape[1], scales_expanded.shape[1])
    weights = weights[:, :common_width]
    scales_expanded = scales_expanded[:, :common_width]
    
    dequantized = weights.to(torch.bfloat16) * scales_expanded.to(torch.bfloat16)
    
    return dequantized

def main():
    print(f"🚀 Starting Standard Dequantization (LLM-Compressor compatible)...")
    print(f"   Input: {INPUT_MODEL_PATH}")
    print(f"   Output: {OUTPUT_MODEL_PATH}")
    
    os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
    
    # 1. Get List of Safetensors
    safetensor_files = sorted(glob(os.path.join(INPUT_MODEL_PATH, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError("❌ No .safetensors files found in input directory.")

    # 2. Process Files One by One (To save RAM)
    for file_path in tqdm(safetensor_files, desc="Processing Shards"):
        filename = os.path.basename(file_path)
        
        # Load Raw Tensors to CPU
        state_dict = load_file(file_path, device="cpu")
        new_state_dict = {}
        
        # Identify Packed Weights
        keys = list(state_dict.keys())
        processed_prefixes = set()
        
        for key in keys:
            if "weight_packed" in key:
                # Found a quantized layer (e.g., model.layers.0.mlp.down_proj.weight_packed)
                prefix = key.replace(".weight_packed", "")
                
                if prefix in processed_prefixes:
                    continue
                
                # Fetch components
                packed = state_dict[key]
                scale_key = f"{prefix}.weight_scale"
                
                if scale_key not in state_dict:
                    print(f"⚠️ Warning: Found packed weight {key} but no scale. Skipping.")
                    continue
                    
                scales = state_dict[scale_key]
                
                # === RUN DEQUANTIZATION ===
                # This returns the pure BF16 weight
                w_bf16 = unpack_awq_standard(packed, scales)
                
                # Save as standard HuggingFace Linear weight name
                new_key = f"{prefix}.weight"
                new_state_dict[new_key] = w_bf16
                
                processed_prefixes.add(prefix)
                
            elif "weight_scale" in key:
                # Skip scales (already used)
                continue
            elif "weight_shape" in key:
                # Skip metadata shapes
                continue
            else:
                # Copy other tensors (Norms, Embeddings, Biases) directly
                new_state_dict[key] = state_dict[key].to(torch.bfloat16)

        # Save the new BF16 shard
        output_file = os.path.join(OUTPUT_MODEL_PATH, filename)
        save_file(new_state_dict, output_file)
        
        # Cleanup memory
        del state_dict, new_state_dict
        gc.collect()

    # 3. Handle Config & Tokenizer
    print("🧹 Migrating Configuration...")
    
    # Save Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_PATH)
        tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
    except Exception as e:
        print(f"⚠️ Could not load/save tokenizer: {e}")

    # Save Config (Cleaned)
    try:
        config = AutoConfig.from_pretrained(INPUT_MODEL_PATH)
        
        # Remove quantization artifacts so Transformers loads it as a standard model
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")
        
        config.torch_dtype = "bfloat16"
        config.save_pretrained(OUTPUT_MODEL_PATH)
    except Exception as e:
        print(f"⚠️ Could not save config: {e}")

    print("✅ Conversion Complete.")
    print(f"   You can now load this model in Method A/B scripts as a standard BF16 model.")

if __name__ == "__main__":
    main()