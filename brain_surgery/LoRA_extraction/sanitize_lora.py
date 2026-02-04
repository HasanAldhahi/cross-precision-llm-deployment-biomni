import torch
from safetensors.torch import load_file, save_file
import os
import json
import shutil

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# UPDATE THIS PATH to the folder containing your adapter_model.safetensors
LORA_PATH = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/brain_surgery/lora_extraction_results/dequantized_corrected_lora_rank_256"

# ==============================================================================
# 1. SETUP & BACKUP
# ==============================================================================
adapter_file = os.path.join(LORA_PATH, "adapter_model.safetensors")
config_file = os.path.join(LORA_PATH, "adapter_config.json")
backup_adapter = adapter_file + ".original.bak"
backup_config = config_file + ".original.bak"

print(f"🔧 STARTING REPAIR ON: {LORA_PATH}")

if not os.path.exists(adapter_file):
    print(f"❌ Error: File not found: {adapter_file}")
    exit(1)

# Create backups if they don't exist yet
if not os.path.exists(backup_adapter):
    print("📦 Creating backup of safetensors file...")
    shutil.copy2(adapter_file, backup_adapter)

if os.path.exists(config_file) and not os.path.exists(backup_config):
    print("📦 Creating backup of config file...")
    shutil.copy2(config_file, backup_config)

# ==============================================================================
# 2. PRUNE UNSUPPORTED LAYERS (Weights)
# ==============================================================================
print("\n🔍 Scanning weights for vLLM incompatibility...")
try:
    tensors = load_file(adapter_file)
except Exception as e:
    print(f"❌ Critical Error: Could not load safetensors file. It might be corrupt. {e}")
    exit(1)

new_tensors = {}
removed_keys = []

# vLLM only supports LoRA on Linear layers (q,k,v,o,gate,up,down).
# Anything else causes a crash on load.
FORBIDDEN_KEYWORDS = [
    "lm_head",          # The output vocabulary layer
    "embed_tokens",     # The input embedding layer
    "layernorm",        # Normalization layers
    "norm",             # Generic normalization (rms_norm)
    "bias",             # Biases (usually not supported in standard vLLM LoRA)
    "rotary_emb"        # RoPE embeddings
]

for key, tensor in tensors.items():
    is_bad = False
    for bad_word in FORBIDDEN_KEYWORDS:
        if bad_word in key:
            is_bad = True
            removed_keys.append(key)
            break
    
    if not is_bad:
        new_tensors[key] = tensor

if len(removed_keys) > 0:
    print(f"✂️  Found {len(removed_keys)} unsupported layers.")
    print(f"   (Examples: {removed_keys[:3]} ...)")
    print("   Pruning them now...")
    save_file(new_tensors, adapter_file)
    print("✅ Weights file updated and saved.")
else:
    print("✅ Weights file was already clean.")

# ==============================================================================
# 3. FIX CONFIGURATION (JSON)
# ==============================================================================
print("\n🔍 Checking adapter_config.json...")

if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    changed = False
    
    # Fix 1: modules_to_save must be null
    if config.get("modules_to_save") is not None:
        print("   - Setting 'modules_to_save' to null (was set)")
        config["modules_to_save"] = None
        changed = True
        
    # Fix 2: Ensure target modules list is clean (optional but good practice)
    # Sometimes extractors put 'lm_head' in target_modules too
    if "target_modules" in config and isinstance(config["target_modules"], list):
        original_len = len(config["target_modules"])
        config["target_modules"] = [
            m for m in config["target_modules"] 
            if not any(bad in m for bad in ["lm_head", "embed_tokens", "norm"])
        ]
        if len(config["target_modules"]) < original_len:
            print("   - Cleaned 'target_modules' list")
            changed = True

    if changed:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Config file updated.")
    else:
        print("✅ Config file was already correct.")
else:
    print("⚠️ Warning: adapter_config.json not found!")

# ==============================================================================
# 4. FINAL VERIFICATION
# ==============================================================================
print("\n----- VERIFICATION -----")
try:
    # 1. Check file size
    size_mb = os.path.getsize(adapter_file) / (1024 * 1024)
    print(f"File Size: {size_mb:.2f} MB")
    
    # 2. Check Loadability
    test_load = load_file(adapter_file)
    print(f"Keys Remaining: {len(test_load)}")
    
    # 3. Check for stragglers
    stragglers = [k for k in test_load.keys() if "lm_head" in k or "norm" in k]
    if stragglers:
        print(f"❌ FAILURE: Still found bad keys: {stragglers}")
    else:
        print("🏆 SUCCESS: LoRA is clean and vLLM-ready.")

except Exception as e:
    print(f"❌ FAILURE: File seems corrupted: {e}")

print("==============================================================================")
print("You can now submit your SBATCH script.")