import torch
import os
import json
import glob
import gc
from safetensors import safe_open
from safetensors.torch import load_file
import torch.nn.functional as F

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Clean BF16 Model (Reference)
PATH_CLEAN = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview"

# 2. Noisy INT4 Dequantized Model
PATH_NOISY = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-From-INT4-Bridge-BF16"

# 3. LoRA Adapter (Signal)
PATH_LORA = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/brain_surgery/lora_extraction_results/lora_basic_original_base_rank_256"

# SETTINGS
RUN_ALL_LAYERS = False   # Set False for speed (one layer only)
TEST_LAYER_IDX = 15      # Representative layer

LAYERS_TO_TEST = [
    "mlp.down_proj", 
    "self_attn.o_proj",
    "self_attn.q_proj"
]

# ==========================================
# FAST LOADER
# ==========================================
class FastLoader:
    def __init__(self, folder_path):
        self.folder = folder_path
        self.weight_map = None
        self.use_map = False
        self.file_cache = {}
        
        index_path = os.path.join(folder_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                self.weight_map = data['weight_map']
                self.use_map = True
                print(f"[{os.path.basename(folder_path)}] Index found.")
            except:
                print(f"[{os.path.basename(folder_path)}] Index corrupt.")
        
        if not self.use_map:
            print(f"[{os.path.basename(folder_path)}] Scanning files...")
            for f in glob.glob(os.path.join(folder_path, "*.safetensors")):
                try:
                    with safe_open(f, framework="pt", device="cpu") as f_open:
                        for k in f_open.keys():
                            self.file_cache[k] = f
                except: pass

    def get_tensor(self, key_name):
        file_path = None
        if self.use_map and key_name in self.weight_map:
            file_path = os.path.join(self.folder, self.weight_map[key_name])
        else:
            file_path = self.file_cache.get(key_name)

        if not file_path or not os.path.exists(file_path): return None
        with safe_open(file_path, framework="pt", device="cpu") as f_open:
            return f_open.get_tensor(key_name)

# ==========================================
# FIXED LORA LOADER
# ==========================================
def get_lora_matrix(lora_path, layer_name):
    adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file): 
        # Fallback to bin if safetensors missing
        adapter_file = os.path.join(lora_path, "adapter_model.bin")
        if not os.path.exists(adapter_file): return None
        tensors = torch.load(adapter_file, map_location="cpu")
    else:
        tensors = load_file(adapter_file)
    
    # FIX: Remove .weight and model. prefix to find the module name
    # e.g., "model.layers.15.mlp.down_proj.weight" -> "layers.15.mlp.down_proj"
    clean_name = layer_name.replace(".weight", "")
    
    # Depending on how LoRA was saved, it might have prefixes or not.
    # We try to match the END of the key.
    # e.g. "base_model.model.layers.15.mlp.down_proj.lora_A.weight"
    # contains "layers.15.mlp.down_proj"
    
    # We strip 'model.' to be safe, but keep 'layers...'
    search_term = clean_name
    if search_term.startswith("model."):
        search_term = search_term[6:] # remove "model."

    key_A = None
    key_B = None
    
    for k in tensors.keys():
        # We look for the search term inside the key
        if search_term in k:
            if "lora_A" in k: key_A = k
            if "lora_B" in k: key_B = k
            
    if key_A and key_B:
        A = tensors[key_A].float()
        B = tensors[key_B].float()
        return B @ A
    
    # Debug print if not found (only once per run usually)
    # print(f"DEBUG: Could not find keys for {search_term}")
    return None

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("--- Starting Fast Orthogonality Check ---")
    loader_clean = FastLoader(PATH_CLEAN)
    loader_noisy = FastLoader(PATH_NOISY)
    
    layers = list(range(64)) if RUN_ALL_LAYERS else [TEST_LAYER_IDX]
    
    results_cos = []
    results_leak = []

    for layer_idx in layers:
        print(f"\nAnalyzing Layer {layer_idx}...")
        
        for module in LAYERS_TO_TEST:
            full_name = f"model.layers.{layer_idx}.{module}.weight"
            
            # 1. Load Clean
            W_clean = loader_clean.get_tensor(full_name)
            if W_clean is None: 
                print(f"  [Skip] {module} missing in Clean")
                continue
            
            # 2. Load Noisy
            W_noisy = loader_noisy.get_tensor(full_name)
            if W_noisy is None: 
                print(f"  [Skip] {module} missing in Noisy")
                continue
            
            # 3. Compute Noise
            N = W_clean.float() - W_noisy.float()
            del W_clean, W_noisy
            gc.collect()
            
            # 4. Load LoRA
            K = get_lora_matrix(PATH_LORA, full_name)
            if K is None:
                print(f"  [Skip] LoRA not found for {module}")
                continue
            K = K.float()
            
            # 5. Metrics
            # A. Cosine
            sim = F.cosine_similarity(N.view(-1).unsqueeze(0), K.view(-1).unsqueeze(0)).item()
            results_cos.append(sim)
            
            # B. Leakage
            try:
                # SVD on K (keep top 256)
                U, _, _ = torch.linalg.svd(K, full_matrices=False)
                U = U[:, :256] 
                
                # Project Noise
                proj = U.T @ N
                leakage = (torch.norm(proj) / (torch.norm(N) + 1e-6)).item()
                results_leak.append(leakage)
            except:
                leakage = 0.0
                
            print(f"  > {module:20} | Cos: {sim:.5f} | Leakage: {leakage:.3f}")
            
            del N, K
            gc.collect()

    print("\n" + "="*40)
    if results_cos:
        print(f"Avg Cosine Sim: {sum(results_cos)/len(results_cos):.5f}")
        print(f"Avg Leakage:    {sum(results_leak)/len(results_leak):.5f} (Random ~0.22)")
    else:
        print("No results collected.")
    print("="*40)

if __name__ == "__main__":
    main()