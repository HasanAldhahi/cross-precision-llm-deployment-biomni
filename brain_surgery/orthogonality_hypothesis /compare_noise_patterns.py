import torch
import os
import json
import glob
import gc
from safetensors import safe_open
import torch.nn.functional as F

# ==========================================
# CONFIGURATION
# ==========================================

# --- PAIR 1: FP8 NOISE (N1) ---
# Path 1: Original Base Qwen
PATH_QWEN_CLEAN = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B"
# Path 2: FP8 Dequantized Qwen
PATH_QWEN_FP8   = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/brain_surgery/qwen3_32B_bf16_dequantized_From_FP8"

# --- PAIR 2: INT4 NOISE (N2) ---
# Path 3: Biomni Clean
PATH_BIO_CLEAN  = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview"
# Path 4: Biomni INT4 Dequantized
PATH_BIO_INT4   = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-From-INT4-Bridge-BF16"

# SETTINGS
TEST_LAYER_IDX = 15      # Running on Layer 15 for speed
LAYERS_TO_TEST = [
    "mlp.down_proj", 
    "self_attn.o_proj",
    "self_attn.q_proj"
]

# ==========================================
# FAST LOADER
# ==========================================
class FastLoader:
    def __init__(self, folder_path, name="Model"):
        self.folder = folder_path
        self.name = name
        self.weight_map = None
        self.use_map = False
        self.file_cache = {}
        
        if not os.path.exists(folder_path):
            print(f"[{name}] WARNING: Path not found: {folder_path}")
            return

        # Try Index
        index_path = os.path.join(folder_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                self.weight_map = data['weight_map']
                self.use_map = True
                print(f"[{name}] Index loaded.")
            except:
                print(f"[{name}] Index corrupt.")
        
        # Fallback Scan
        if not self.use_map:
            print(f"[{name}] Scanning files (No index found)...")
            for f in glob.glob(os.path.join(folder_path, "*.safetensors")):
                try:
                    with safe_open(f, framework="pt", device="cpu") as f_open:
                        for k in f_open.keys():
                            self.file_cache[k] = f
                except: pass

    def get_tensor(self, key_name):
        file_path = None
        if self.use_map and self.weight_map and key_name in self.weight_map:
            file_path = os.path.join(self.folder, self.weight_map[key_name])
        else:
            file_path = self.file_cache.get(key_name)

        if not file_path or not os.path.exists(file_path): return None
        with safe_open(file_path, framework="pt", device="cpu") as f_open:
            return f_open.get_tensor(key_name)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("--- Initializing 4 Model Loaders ---")
    # Loaders are lightweight (just maps), so we can init all 4
    loader_qwen_clean = FastLoader(PATH_QWEN_CLEAN, "Qwen Clean")
    loader_qwen_fp8   = FastLoader(PATH_QWEN_FP8,   "Qwen FP8")
    loader_bio_clean  = FastLoader(PATH_BIO_CLEAN,  "Bio Clean")
    loader_bio_int4   = FastLoader(PATH_BIO_INT4,   "Bio INT4")

    print(f"\nAnalyzing Noise Correlation on Layer {TEST_LAYER_IDX}...")
    
    results = []

    for module in LAYERS_TO_TEST:
        full_name = f"model.layers.{TEST_LAYER_IDX}.{module}.weight"
        
        # --- 1. Compute N1 (FP8 Noise) ---
        t1 = loader_qwen_clean.get_tensor(full_name)
        t2 = loader_qwen_fp8.get_tensor(full_name)
        
        if t1 is None or t2 is None:
            print(f"  [Skip] {module}: Missing weights for FP8 pair")
            continue
            
        N1 = t1.float() - t2.float()
        
        # Free heavy original weights immediately
        del t1, t2
        gc.collect()
        
        # --- 2. Compute N2 (INT4 Noise) ---
        t3 = loader_bio_clean.get_tensor(full_name)
        t4 = loader_bio_int4.get_tensor(full_name)
        
        if t3 is None or t4 is None:
            print(f"  [Skip] {module}: Missing weights for INT4 pair")
            del N1 # Cleanup
            continue
            
        N2 = t3.float() - t4.float()
        del t3, t4
        gc.collect()

        # --- 3. Compare N1 vs N2 ---
        sim = F.cosine_similarity(N1.view(-1).unsqueeze(0), N2.view(-1).unsqueeze(0)).item()
        results.append(sim)
        
        print(f"  > {module:20} | N1 vs N2 Cosine Sim: {sim:.5f}")
        
        # Free noise vectors
        del N1, N2
        gc.collect()

    print("\n" + "="*50)
    if results:
        avg_sim = sum(results) / len(results)
        print(f"AVERAGE NOISE CORRELATION: {avg_sim:.5f}")
        if abs(avg_sim) < 0.05:
            print("Conclusion: FP8 Noise and INT4 Noise are UNCORRELATED.")
        else:
            print("Conclusion: There is some structural correlation between quantization errors.")
    else:
        print("No successful comparisons made.")
    print("="*50)

if __name__ == "__main__":
    main()