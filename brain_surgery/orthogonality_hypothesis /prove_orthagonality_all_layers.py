import torch
import os
from safetensors import safe_open
from safetensors.torch import load_file
import torch.nn.functional as F

# --- PATHS (From your message) ---
# 1. The LoRA with the Knowledge (The Signal)
LORA_PATH = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/brain_surgery/lora_extraction_results/dequantized_corrected_lora_rank_256"

# 2. The Clean Original Model (The Reference)
BASE_PATH = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B"

# 3. The Dequantized Model (The Noisy Version)
# We assume this folder contains .safetensors files with BF16 weights
DEQUANT_PATH = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/brain_surgery/qwen3_32B_bf16_dequantized_From_FP8"

# --- CONFIGURATION ---
TARGET_LAYER_INDEXES = list(range(64))  # layers 0 to 63
LAYERS_TO_TEST = [
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj"
]

def load_tensor_from_folder(folder_path, tensor_suffix):
    """
    Scans a folder of .safetensors to find a specific weight.
    """
    if not os.path.exists(folder_path):
        return None
        
    for f in os.listdir(folder_path):
        if not f.endswith(".safetensors"): continue
        
        full_path = os.path.join(folder_path, f)
        try:
            with safe_open(full_path, framework="pt", device="cpu") as f_open:
                # We look for keys ending with the suffix (e.g., "layers.0.mlp.down_proj.weight")
                for k in f_open.keys():
                    if k.endswith(tensor_suffix):
                        return f_open.get_tensor(k)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
    return None

def get_lora_matrix(lora_path, layer_suffix):
    """
    Loads LoRA A and B and multiplies them (B @ A).
    """
    adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        # Fallback to .bin
        adapter_file = os.path.join(lora_path, "adapter_model.bin")
        if not os.path.exists(adapter_file): return None
        tensors = torch.load(adapter_file, map_location="cpu")
    else:
        tensors = load_file(adapter_file)

    # Find keys matching this layer
    # PEFT keys usually look like: "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"
    key_A = None
    key_B = None
    
    # We construct a search string (e.g., "layers.0.mlp.down_proj")
    search_str = layer_suffix.replace(".weight", "") 
    
    for k in tensors.keys():
        if search_str in k:
            if "lora_A" in k: key_A = k
            if "lora_B" in k: key_B = k
    
    if not key_A or not key_B:
        return None
        
    A = tensors[key_A].float()
    B = tensors[key_B].float()
    
    # Return L = B @ A
    return B @ A

def main():
    results_per_layer = {}
    layerwise_similarities = []  # To summarize over all layers

    for layer_idx in TARGET_LAYER_INDEXES:
        print(f"\n--- Orthogonality Proof: Layer {layer_idx} ---")
        similarities = []
        for module_name in LAYERS_TO_TEST:
            # Construct the full tensor name, e.g., "model.layers.0.mlp.down_proj.weight"
            full_name = f"model.layers.{layer_idx}.{module_name}.weight"

            print(f"Analyzing: {module_name}...")

            # 1. Load Clean Base (W_clean)
            W_clean = load_tensor_from_folder(BASE_PATH, full_name)
            if W_clean is None:
                print(f"  [Skip] Could not find {full_name} in Base path")
                continue

            # 2. Load Dequantized Base (W_dequant)
            W_dequant = load_tensor_from_folder(DEQUANT_PATH, full_name)
            if W_dequant is None:
                print(f"  [Skip] Could not find {full_name} in Dequant path")
                continue

            # 3. Load LoRA (L)
            L = get_lora_matrix(LORA_PATH, full_name)
            if L is None:
                print(f"  [Skip] Could not find LoRA adapter for {module_name}")
                continue

            # --- MATH ---
            N = W_clean.float() - W_dequant.float()
            L = L.float()

            print(f"    > N shape: {N.shape}, L shape: {L.shape}")

            # Flatten
            N_flat = N.view(-1)
            L_flat = L.view(-1)
            print(f"    > N_flat shape: {N_flat.shape}, L_flat shape: {L_flat.shape}")

            # Cosine Similarity
            sim = F.cosine_similarity(N_flat.unsqueeze(0), L_flat.unsqueeze(0)).item()

            # Magnitude Ratio (Noise / Signal)
            mag_N = torch.norm(N_flat).item()
            mag_L = torch.norm(L_flat).item()
            ratio = mag_N / (mag_L + 1e-9)

            print(f"  > Cosine Similarity: {sim:.5f}") # Should be close to 0
            print(f"  > Noise/Signal Ratio: {ratio:.3f}")

            similarities.append(sim)

        results_per_layer[layer_idx] = similarities
        # Compute average similarity for this layer (if any items exist)
        if similarities:
            avg_layer_sim = sum(similarities) / len(similarities)
            layerwise_similarities.append(avg_layer_sim)
            print(f"=> Layer {layer_idx} Average Cosine Similarity: {avg_layer_sim:.5f}")
        else:
            print(f"=> Layer {layer_idx} had no successful modules.")

    # Final summary across all layers
    print("\n" + "="*50)
    if layerwise_similarities:
        avg_sim_all_layers = sum(layerwise_similarities) / len(layerwise_similarities)
        print(f"FINAL RESULT - Average Cosine Similarity over all layers: {avg_sim_all_layers:.5f}")
        if abs(avg_sim_all_layers) < 0.05:
            print("✅ HYPOTHESIS CONFIRMED: Orthogonal")
        else:
            print("❌ HYPOTHESIS REJECTED: Correlated")
    else:
        print("No valid results found across layers and modules.")
    print("="*50)

if __name__ == "__main__":
    main()