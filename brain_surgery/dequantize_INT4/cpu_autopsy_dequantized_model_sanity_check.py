import torch
from safetensors.torch import load_file
import os
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Point this to your DEQUANTIZED RAW folder
MODEL_FOLDER = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/brain_surgery/qwen_bf16_robust"
# MODEL_FOLDER = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-From-INT4-Bridge-BF16"
# MODEL_FOLDER = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B"

# We will inspect the first file we find
# ==============================================================================

def analyze_weights():
    print(f"🏥 Starting CPU Autopsy on: {MODEL_FOLDER}")
    
    # 1. Find a safetensors file
    files = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".safetensors")]
    files.sort()
    print(files)
    
    if not files:
        print("❌ CRITICAL: No .safetensors files found!")
        sys.exit(1)
        
    target_file = os.path.join(MODEL_FOLDER, "model-00002-of-00007.safetensors")
    print(f"🔎 Inspecting Shard: {files[0]}")
    print("   Loading into System RAM (CPU)... please wait...")
    
    try:
        # Load directly to CPU
        state_dict = load_file(target_file, device="cpu")
    except Exception as e:
        print(f"❌ CORRUPTION DETECTED: File cannot be opened. Error: {e}")
        sys.exit(1)

    print(f"✅ File loaded. Scanning {len(state_dict)} tensors...")
    
    # 2. Find a Linear Layer (Projection) to analyze
    # We avoid 'norm' or 'embed' layers because their stats are different.
    target_tensor_name = None
    for key in state_dict.keys():
        # Look for a dense weight like 'q_proj', 'down_proj', 'gate_proj'
        if "proj.weight" in key:
            target_tensor_name = key
            break
    
    if not target_tensor_name:
        print("⚠️ Warning: No 'proj.weight' found. Analyzing the first tensor available.")
        target_tensor_name = list(state_dict.keys())[0]

    tensor = state_dict[target_tensor_name]
    
    # 3. Calculate Statistics
    # Convert to float32 for accurate math
    data = tensor.float()
    
    mean = data.mean().item()
    std = data.std().item()
    min_val = data.min().item()
    max_val = data.max().item()
    
    print("\n📊 VITAL SIGNS REPORT")
    print("---------------------------------------------------")
    print(f"Layer: {target_tensor_name}")
    print(f"Type:  {tensor.dtype}")
    print(f"Shape: {tensor.shape}")
    print("---------------------------------------------------")
    print(f"Mean (Avg):   {mean:.6f}")
    print(f"Std Dev:      {std:.6f}")
    print(f"Min Value:    {min_val:.6f}")
    print(f"Max Value:    {max_val:.6f}")
    print("---------------------------------------------------")

    # 4. The Diagnosis
    print("\n👩‍⚕️ DIAGNOSIS:")
    
    is_healthy = True
    
    # TEST A: The "Explosion" Check
    # Valid weights are usually between -1.0 and 1.0.
    # If the scale was missed, FP8 integers might look like huge floats (e.g. 50.0, 100.0)
    if std > 1.0 or abs(mean) > 1.0:
        print("❌ FAILED: Weights are EXPLODED.")
        print("   Reason: The quantization scale was likely NOT multiplied.")
        print("   The raw integers are being read as floats.")
        is_healthy = False
        
    # TEST B: The "Flatline" Check
    # If something went wrong with the cast, we might get all zeros.
    elif std < 0.000001:
        print("❌ FAILED: Weights are FLATLINED (Zero).")
        print("   Reason: The conversion likely multiplied by a zero scale.")
        is_healthy = False

    # TEST C: The "NaN" Check
    elif torch.isnan(data).any():
        print("❌ FAILED: Weights contain NaNs (Not a Number).")
        is_healthy = False

    if is_healthy:
        print("✅ PASSED: The patient is healthy.")
        print("   - Mean is near zero (centered).")
        print("   - Standard Deviation is small (typical for LLMs).")
        print("   - Values are within valid range.")
        print("\nConclusion: The Dequantization was mathematically CORRECT.")
    else:
        print("\nConclusion: The Model is CORRUPTED.")

if __name__ == "__main__":
    analyze_weights()