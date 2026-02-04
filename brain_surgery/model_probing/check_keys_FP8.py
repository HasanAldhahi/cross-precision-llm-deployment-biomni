from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch





# 1. Download just the first shard (very fast)
print("⬇️ Downloading first shard of Qwen-FP8...")
file_path = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B-FP8/model-00002-of-00007.safetensors"
# 2. Inspect Keys
print(f"🔍 Inspecting: {file_path}")
state_dict = load_file(file_path)

print("\n--- SAMPLE KEYS ---")
keys = list(state_dict.keys())
# Print the first 20 keys to find the pattern
for k in keys:
    info = state_dict[k]
    print(f"Key: {k} | Dtype: {info.dtype} | Shape: {info.shape}")

print("\n--- SEARCHING FOR SCALES ---")
# Let's find a weight and its corresponding scale
found_pair = False
for k in keys:
    if "q_proj.weight" in k:
        print(f"\nFOUND WEIGHT: {k}")
        # Look for the scale
        expected_scale = k.replace(".weight", ".weight_scale")
        if expected_scale in state_dict:
            print(f"✅ FOUND SCALE MATCH: {expected_scale}")
            print(f"   Scale Shape: {state_dict[expected_scale].shape}")
            found_pair = True
        else:
            print(f"❌ MISSING SCALE: Expected {expected_scale} but not found.")
            # Search for ANY scale in the file
            print("   Scanning for similar keys...")
            for s in keys:
                if "q_proj" in s and "scale" in s:
                    print(f"   -> POTENTIAL MATCH: {s}")
        break

if not found_pair:
    print("\n⚠️ CONCLUSION: The naming convention is tricky. We need to update the conversion script based on the output above.")
else:
    print("\n✅ CONCLUSION: Standard naming confirms. We can proceed with conversion.")