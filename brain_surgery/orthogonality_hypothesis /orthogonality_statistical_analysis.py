import torch
import os
import json
import glob
import gc
import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file
import torch.nn.functional as F

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Clean BF16 Model
PATH_CLEAN = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview"
# 2. Noisy INT4 Model
PATH_NOISY = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-From-INT4-Bridge-BF16"
# 3. LoRA Adapter
PATH_LORA = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/brain_surgery/lora_extraction_results/lora_basic_original_base_rank_256"

# Run on ALL layers to generate the full heatmap/plot data
TARGET_LAYERS = list(range(64))
# TARGET_LAYERS = [0, 15, 32, 63] # Debug mode

MODULES = ["mlp.down_proj", "self_attn.o_proj"] # Check two distinct types

# ==========================================
# HELPERS
# ==========================================
class FastLoader:
    def __init__(self, folder_path):
        self.folder = folder_path
        self.weight_map = None
        self.use_map = False
        self.file_cache = {}
        
        index_path = os.path.join(folder_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self.weight_map = json.load(f)['weight_map']
            self.use_map = True
        else:
            for f in glob.glob(os.path.join(folder_path, "*.safetensors")):
                with safe_open(f, framework="pt", device="cpu") as f_open:
                    for k in f_open.keys(): self.file_cache[k] = f

    def get_tensor(self, key_name):
        f = None
        if self.use_map and key_name in self.weight_map:
            f = os.path.join(self.folder, self.weight_map[key_name])
        else:
            f = self.file_cache.get(key_name)
        if not f: return None
        with safe_open(f, framework="pt", device="cpu") as f_open:
            return f_open.get_tensor(key_name)

def get_lora(lora_path, layer_name):
    # Load LoRA logic (Same as fixed version)
    path = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(path): path = path.replace(".safetensors", ".bin")
    if not os.path.exists(path): return None
    
    tensors = load_file(path)
    clean_name = layer_name.replace(".weight", "").replace("model.", "")
    
    kA, kB = None, None
    for k in tensors.keys():
        if clean_name in k:
            if "lora_A" in k: kA = k
            if "lora_B" in k: kB = k
    
    if kA and kB:
        return tensors[kB].float() @ tensors[kA].float()
    return None

# ==========================================
# MAIN
# ==========================================
def main():
    print("--- Initializing Loaders ---")
    clean_loader = FastLoader(PATH_CLEAN)
    noisy_loader = FastLoader(PATH_NOISY)
    
    # Store results for aggregation
    data = {
        "layer": [],
        "sim_lora": [], # K vs N
        "sim_rand": [], # Random vs N
        "mag_noise": [],
        "mag_signal": [],
        "snr": []       # ||K|| / ||N||
    }
    
    print(f"\nScanning {len(TARGET_LAYERS)} layers for Statistics...")
    print(f"{'Layer':<6} | {'Mod':<15} | {'Sim(K)':<8} | {'Sim(Rand)':<9} | {'||N||':<8} | {'||K||':<8} | {'Noise/Sig'}")
    print("-" * 85)

    for layer_idx in TARGET_LAYERS:
        for mod in MODULES:
            fname = f"model.layers.{layer_idx}.{mod}.weight"
            
            # 1. Get Weights
            W_c = clean_loader.get_tensor(fname)
            W_n = noisy_loader.get_tensor(fname)
            if W_c is None or W_n is None: continue
            
            # 2. Compute Noise (N)
            N = W_c.float() - W_n.float()
            del W_c, W_n
            gc.collect()
            
            # 3. Get Signal (K)
            K = get_lora(PATH_LORA, fname)
            if K is None: 
                del N
                continue
            K = K.float()
            
            # --- TEST 4: Random Baseline ---
            # Create a random vector R with same shape and norm as K
            R = torch.randn_like(K)
            scale = torch.norm(K) / torch.norm(R)
            R = R * scale # Now ||R|| == ||K||
            
            # --- CALCULATIONS ---
            # Flatten
            N_f = N.view(-1)
            K_f = K.view(-1)
            R_f = R.view(-1)
            
            # Sims
            sim_k = F.cosine_similarity(N_f.unsqueeze(0), K_f.unsqueeze(0)).item()
            sim_r = F.cosine_similarity(N_f.unsqueeze(0), R_f.unsqueeze(0)).item()
            
            # Mags
            mag_n = torch.norm(N_f).item()
            mag_k = torch.norm(K_f).item()
            
            # Store
            data["layer"].append(layer_idx)
            data["sim_lora"].append(sim_k)
            data["sim_rand"].append(sim_r)
            data["mag_noise"].append(mag_n)
            data["mag_signal"].append(mag_k)
            data["snr"].append(mag_n / (mag_k + 1e-9))
            
            print(f"{layer_idx:<6} | {mod:<15} | {sim_k:>8.5f} | {sim_r:>9.5f} | {mag_n:>8.2f} | {mag_k:>8.2f} | {mag_n/mag_k:.2f}x")
            
            del N, K, R, N_f, K_f, R_f
            gc.collect()

    # ==========================================
    # FINAL ANALYSIS
    # ==========================================
    print("\n" + "="*50)
    print("THESIS STATISTICS SUMMARY")
    print("="*50)
    
    avg_sim_k = np.mean(data["sim_lora"])
    avg_sim_r = np.mean(data["sim_rand"])
    avg_ratio = np.mean(data["snr"])
    
    print(f"1. ORTHOGONALITY CHECK (LoRA vs Random)")
    print(f"   Avg LoRA Similarity:   {avg_sim_k:.6f}")
    print(f"   Avg Random Similarity: {avg_sim_r:.6f}")
    
    if abs(avg_sim_k - avg_sim_r) < 0.005:
        print("   -> CONCLUSION: LoRA behaves like a random vector w.r.t Noise.")
        print("   -> PROOF: Quantization noise is effectively Isotropic.")
    else:
        print("   -> CONCLUSION: LoRA has specific alignment (investigate further).")

    print(f"\n2. MAGNITUDE ANALYSIS (Why INT4 is worse than FP8)")
    print(f"   Avg Noise/Signal Ratio: {avg_ratio:.2f}x")
    print(f"   Interpretation: The noise is {avg_ratio:.2f} times stronger than the signal.")
    
    print("\n3. LAYER VARIABILITY")
    print(f"   Min Ratio: {min(data['snr']):.2f}x (Layer {data['layer'][np.argmin(data['snr'])]})")
    print(f"   Max Ratio: {max(data['snr']):.2f}x (Layer {data['layer'][np.argmax(data['snr'])]})")
    
    print("="*50)

if __name__ == "__main__":
    main()