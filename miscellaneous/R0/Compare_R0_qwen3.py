#!/usr/bin/env python3
"""
Model Comparison Script: Biomni-R0-32B-Preview vs Qwen3-32B-FP8
Analyzes differences, identifies LoRA weights, and extracts adapter layers for fine-tuning
"""

import sys
import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors import safe_open
from huggingface_hub import snapshot_download
import argparse
import gc

class ModelComparator:
    def __init__(self, base_model_path="/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B", 
                 finetuned_model_path="/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview",
                 output_dir="./model_comparison"):
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.base_path = Path(base_model_path)
        self.finetuned_path = Path(finetuned_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("="*80)
        print("MODEL COMPARISON ANALYSIS")
        print("="*80)
        print(f"Base Model:      {base_model_path}")
        print(f"Finetuned Model: {finetuned_model_path}")
        print(f"Output Directory: {output_dir}")
        print("="*80)
        
        # Verify paths exist
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base model directory not found: {base_model_path}")
        if not self.finetuned_path.exists():
            raise FileNotFoundError(f"Finetuned model directory not found: {finetuned_model_path}")
        
        print(f"✅ Base model directory found")
        print(f"✅ Finetuned model directory found")
    
    def verify_models(self):
        """Verify model files exist in local directories"""
        print("\n📥 Verifying local model files...")
        
        base_files = list(self.base_path.glob("*.safetensors"))
        ft_files = list(self.finetuned_path.glob("*.safetensors"))
        
        if not base_files:
            raise FileNotFoundError(f"No safetensors files found in base model: {self.base_path}")
        if not ft_files:
            raise FileNotFoundError(f"No safetensors files found in finetuned model: {self.finetuned_path}")
        
        print(f"  ✅ Base model: Found {len(base_files)} safetensors files")
        print(f"  ✅ Finetuned model: Found {len(ft_files)} safetensors files")
        
        return str(self.base_path), str(self.finetuned_path)
    
    def compare_configs(self):
        """Compare model configurations"""
        print("\n📋 Comparing Model Configurations...")
        
        base_config = AutoConfig.from_pretrained(str(self.base_path))
        ft_config = AutoConfig.from_pretrained(str(self.finetuned_path))
        
        config_diff = {}
        base_dict = base_config.to_dict()
        ft_dict = ft_config.to_dict()
        
        all_keys = set(base_dict.keys()) | set(ft_dict.keys())
        
        for key in sorted(all_keys):
            base_val = base_dict.get(key, "NOT_PRESENT")
            ft_val = ft_dict.get(key, "NOT_PRESENT")
            
            if base_val != ft_val:
                config_diff[key] = {
                    "base": base_val,
                    "finetuned": ft_val
                }
        
        if config_diff:
            print("  ⚠️  Configuration differences found:")
            for key, vals in config_diff.items():
                print(f"    {key}:")
                print(f"      Base:      {vals['base']}")
                print(f"      Finetuned: {vals['finetuned']}")
        else:
            print("  ✅ Configurations are identical")
        
        # Save config comparison
        with open(self.output_dir / "config_comparison.json", "w") as f:
            json.dump(config_diff, f, indent=2)
        
        return config_diff
    
    def get_safetensors_files(self, model_path):
        """Get all safetensors files from model directory"""
        path = Path(model_path)
        return sorted(path.glob("*.safetensors"))
    
    def load_weights_dict(self, safetensors_files):
        """Load all weights from safetensors files into a dictionary"""
        weights = {}
        
        for st_file in safetensors_files:
            print(f"    Loading {st_file.name}...")
            with safe_open(st_file, framework="pt") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        
        return weights
    
    def compare_architectures(self):
        """Compare model architectures and identify structural differences"""
        print("\n🔍 Comparing Model Architectures...")
        
        print("  Loading base model weights...")
        base_files = self.get_safetensors_files(self.base_path)
        base_weights = self.load_weights_dict(base_files)
        
        print("  Loading finetuned model weights...")
        ft_files = self.get_safetensors_files(self.finetuned_path)
        ft_weights = self.load_weights_dict(ft_files)
        
        # Compare weight keys
        base_keys = set(base_weights.keys())
        ft_keys = set(ft_weights.keys())
        
        common_keys = base_keys & ft_keys
        only_base = base_keys - ft_keys
        only_ft = ft_keys - base_keys
        
        print(f"\n  📊 Architecture Comparison:")
        print(f"    Common parameters:     {len(common_keys)}")
        print(f"    Only in base:          {len(only_base)}")
        print(f"    Only in finetuned:     {len(only_ft)}")
        
        # Check for LoRA pattern
        lora_keys = [k for k in only_ft if any(lora_term in k.lower() 
                     for lora_term in ['lora', 'adapter', 'delta'])]
        
        if lora_keys:
            print(f"\n  🎯 FOUND LoRA/Adapter layers: {len(lora_keys)}")
            for key in sorted(lora_keys):
                shape = ft_weights[key].shape
                print(f"    {key}: {shape}")
        else:
            print("\n  ℹ️  No explicit LoRA layers found in key names")
        
        # Save architecture comparison
        arch_comparison = {
            "common_keys": sorted(list(common_keys)),
            "only_in_base": sorted(list(only_base)),
            "only_in_finetuned": sorted(list(only_ft)),
            "lora_keys": sorted(lora_keys) if lora_keys else []
        }
        
        with open(self.output_dir / "architecture_comparison.json", "w") as f:
            json.dump(arch_comparison, f, indent=2)
        
        return base_weights, ft_weights, arch_comparison
    
    def compute_weight_differences_layer_by_layer(self, base_files, ft_files, threshold=1e-6):
        """Compute weight differences layer by layer with performance optimization"""
        print(f"\n📐 Computing Weight Differences Layer-by-Layer (threshold={threshold})...")

        # Get all common keys first
        print("  📋 Collecting common layers...")
        common_keys = self._get_common_keys(base_files, ft_files)
        print(f"  ✅ Found {len(common_keys)} common layers")

        differences = {}
        modified_layers = []
        layer_times = {}

        # Open files once and keep them open
        print("  📁 Opening safetensors files...")
        base_tensors = {}
        ft_tensors = {}

        for base_file in base_files:
            print(f"    Loading base: {base_file.name}")
            with safe_open(base_file, framework="pt") as f:
                for key in f.keys():
                    if key in common_keys:
                        base_tensors[key] = f.get_tensor(key)

        for ft_file in ft_files:
            print(f"    Loading finetuned: {ft_file.name}")
            with safe_open(ft_file, framework="pt") as f:
                for key in f.keys():
                    if key in common_keys:
                        ft_tensors[key] = f.get_tensor(key)

        print(f"  ✅ Loaded {len(base_tensors)} base tensors and {len(ft_tensors)} finetuned tensors")

        # Create difference file
        diff_file = self.output_dir / "difference.txt"
        with open(diff_file, "w") as f:
            f.write("="*80 + "\n")
            f.write("LAYER-BY-LAYER WEIGHT DIFFERENCE ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Base Model: {self.base_model_path}\n")
            f.write(f"Finetuned Model: {self.finetuned_model_path}\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Total Layers: {len(common_keys)}\n")
            f.write("="*80 + "\n\n")

        # Process each layer individually
        for i, key in enumerate(sorted(common_keys), 1):
            layer_start_time = time.time()

            print(f"\n🔄 Processing layer {i}/{len(common_keys)}: {key}")

            # Get tensors for this layer
            base_tensor = base_tensors[key]
            ft_tensor = ft_tensors[key]

            if base_tensor.shape != ft_tensor.shape:
                print(f"  ⚠️  Shape mismatch for {key}: {base_tensor.shape} vs {ft_tensor.shape}")
                continue

            layer_time = self._compute_layer_difference(key, base_tensor, ft_tensor, threshold, diff_file)

            layer_times[key] = layer_time
            differences[key] = layer_time['metrics']

            # Check if significantly modified
            if layer_time['metrics']['changed_ratio'] > 0.01 or layer_time['metrics']['relative_diff'] > 0.01:
                modified_layers.append(key)

            # Progress update
            if i % 10 == 0:
                avg_time = sum(layer_times.values()) / len(layer_times)
                print(f"  📈 Progress: {i}/{len(common_keys)} layers ({100*i/len(common_keys):.1f}%)")
                print(f"     Avg time per layer: {avg_time:.3f}s")
                print(f"     Modified layers found: {len(modified_layers)}")

            # Memory cleanup every 50 layers
            if i % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"\n  ✅ Analysis complete: {len(modified_layers)} layers with significant changes")

        # Summary statistics
        total_time = sum(t['total_time'] for t in layer_times.values())
        avg_layer_time = total_time / len(layer_times)
        max_layer_time = max(t['total_time'] for t in layer_times.values())

        # Calculate tensor vs numpy performance
        total_tensor_time = sum(t['tensor_time'] for t in layer_times.values())
        total_numpy_time = sum(t['numpy_time'] for t in layer_times.values())
        avg_tensor_time = total_tensor_time / len(layer_times)
        avg_numpy_time = total_numpy_time / len(layer_times)

        print("\n📊 PERFORMANCE SUMMARY:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per layer: {avg_layer_time:.3f}s")
        print(f"  Slowest layer: {max_layer_time:.3f}s")
        print(f"  Layers processed: {len(common_keys)}")
        print(f"  Average tensor computation: {avg_tensor_time:.3f}s")
        print(f"  Average NumPy computation: {avg_numpy_time:.3f}s")
        print(f"  Tensor speedup: {avg_numpy_time/avg_tensor_time:.2f}x")

        # Save detailed results
        sorted_diffs = sorted(differences.items(),
                            key=lambda x: x[1]['relative_diff'],
                            reverse=True)

        with open(diff_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Average time per layer: {avg_layer_time:.3f}s\n")
            f.write(f"Slowest layer: {max_layer_time:.3f}s\n")
            f.write(f"Layers processed: {len(common_keys)}\n")
            f.write(f"Modified layers: {len(modified_layers)}\n")
            f.write(f"Average tensor computation: {avg_tensor_time:.3f}s\n")
            f.write(f"Average NumPy computation: {avg_numpy_time:.3f}s\n")
            f.write(f"Tensor speedup over NumPy: {avg_numpy_time/avg_tensor_time:.2f}x\n\n")

            f.write("TOP 20 MOST MODIFIED LAYERS:\n")
            f.write("-" * 80 + "\n")
            for i, (key, metrics) in enumerate(sorted_diffs[:20], 1):
                f.write(f"{i:2d}. {key}\n")
                f.write(f"    Relative diff: {metrics['relative_diff']:.6f}\n")
                f.write(f"    Changed ratio: {metrics['changed_ratio']:.2%}\n")
                f.write(f"    Processing time: {layer_times[key]['total_time']:.3f}s\n")
                f.write(f"    Tensor time: {layer_times[key]['tensor_time']:.3f}s\n")
                f.write(f"    NumPy time: {layer_times[key]['numpy_time']:.3f}s\n")
                f.write(f"    Shape: {metrics['shape']}\n\n")

        # Save JSON results
        with open(self.output_dir / "weight_differences.json", "w") as f:
            json.dump({
                "modified_layers": modified_layers,
                "all_differences": {k: v for k, v in sorted_diffs},
                "performance": {
                    "total_time": total_time,
                    "avg_layer_time": avg_layer_time,
                    "max_layer_time": max_layer_time,
                    "layer_times": layer_times
                }
            }, f, indent=2)

        return differences, modified_layers

    def _get_common_keys(self, base_files, ft_files):
        """Get common keys across all files"""
        base_keys = set()
        ft_keys = set()

        print("  🔍 Scanning keys across files...")

        for base_file in base_files:
            with safe_open(base_file, framework="pt") as f:
                base_keys.update(f.keys())

        for ft_file in ft_files:
            with safe_open(ft_file, framework="pt") as f:
                ft_keys.update(f.keys())

        return base_keys & ft_keys

    def _compute_layer_difference(self, key, base_tensor, ft_tensor, threshold, diff_file):
        """Compute difference for a single layer with performance comparison"""
        layer_start = time.time()

        # Convert to float for computation
        base_float = base_tensor.float()
        ft_float = ft_tensor.float()

        # Tensor-based computation
        tensor_start = time.time()
        diff_tensor = (ft_float - base_float)
        abs_diff_tensor = torch.abs(diff_tensor)
        mean_abs_diff = abs_diff_tensor.mean().item()
        max_abs_diff = abs_diff_tensor.max().item()

        # Relative difference
        base_norm = torch.norm(base_float).item()
        diff_norm = torch.norm(diff_tensor).item()
        rel_diff = diff_norm / (base_norm + 1e-10)

        # Percentage of changed values
        changed_ratio = (abs_diff_tensor > threshold).float().mean().item()
        tensor_time = time.time() - tensor_start

        # NumPy-based computation for comparison
        numpy_start = time.time()
        base_np = base_float.cpu().numpy()
        ft_np = ft_float.cpu().numpy()

        diff_np = ft_np - base_np
        abs_diff_np = np.abs(diff_np)
        mean_abs_diff_np = abs_diff_np.mean()
        max_abs_diff_np = abs_diff_np.max()

        # Relative difference
        base_norm_np = np.linalg.norm(base_np)
        diff_norm_np = np.linalg.norm(diff_np)
        rel_diff_np = diff_norm_np / (base_norm_np + 1e-10)

        # Percentage of changed values
        changed_ratio_np = (abs_diff_np > threshold).mean()
        numpy_time = time.time() - numpy_start

        # Verify they match (with appropriate tolerance for floating point precision)
        assert abs(mean_abs_diff - mean_abs_diff_np) < 1e-6, f"Mean abs diff mismatch: {abs(mean_abs_diff - mean_abs_diff_np)}"
        assert abs(rel_diff - rel_diff_np) < 1e-5, f"Relative diff mismatch: {abs(rel_diff - rel_diff_np)} (tensor: {rel_diff}, numpy: {rel_diff_np})"

        layer_time = time.time() - layer_start

        # Write to file
        with open(diff_file, "a") as f:
            f.write(f"Layer: {key}\n")
            f.write(f"  Shape: {list(base_tensor.shape)}\n")
            f.write(f"  Total time: {layer_time:.3f}s\n")
            f.write(f"  Tensor computation: {tensor_time:.3f}s\n")
            f.write(f"  NumPy computation: {numpy_time:.3f}s\n")
            f.write(f"  Speedup (Tensor vs NumPy): {numpy_time/tensor_time:.2f}x\n")
            f.write(f"  Mean abs diff: {mean_abs_diff:.8f}\n")
            f.write(f"  Max abs diff: {max_abs_diff:.8f}\n")
            f.write(f"  Relative diff: {rel_diff:.8f}\n")
            f.write(f"  Changed ratio: {changed_ratio:.4%}\n")
            f.write("-" * 60 + "\n")

        return {
            'total_time': layer_time,
            'tensor_time': tensor_time,
            'numpy_time': numpy_time,
            'metrics': {
                "mean_abs_diff": mean_abs_diff,
                "max_abs_diff": max_abs_diff,
                "relative_diff": rel_diff,
                "changed_ratio": changed_ratio,
                "shape": list(base_tensor.shape)
            }
        }
    
    def analyze_lora_patterns(self, base_files, ft_files, modified_layers):
        """Analyze if modified weights follow LoRA patterns (low-rank decomposition)"""
        print("\n🔬 Analyzing LoRA Patterns...")

        lora_analysis = {}
        potential_lora_layers = []

        # Load tensors for LoRA analysis (only top modified layers)
        print("  📥 Loading tensors for LoRA analysis...")
        base_tensors = {}
        ft_tensors = {}

        # Load only the modified layers we need for LoRA analysis
        needed_keys = set(modified_layers[:20])  # Top 20 for analysis

        for base_file in base_files:
            with safe_open(base_file, framework="pt") as f:
                for key in f.keys():
                    if key in needed_keys:
                        base_tensors[key] = f.get_tensor(key)

        for ft_file in ft_files:
            with safe_open(ft_file, framework="pt") as f:
                for key in f.keys():
                    if key in needed_keys:
                        ft_tensors[key] = f.get_tensor(key)

        print(f"  ✅ Loaded {len(base_tensors)} tensors for LoRA analysis")

        for key in modified_layers[:20]:  # Analyze top 20 modified layers
            if key not in base_tensors or key not in ft_tensors:
                continue

            base_tensor = base_tensors[key].float()
            ft_tensor = ft_tensors[key].float()
            diff = ft_tensor - base_tensor
            
            # Check if diff has low rank (LoRA characteristic)
            if len(diff.shape) >= 2:
                # Compute SVD to check rank
                U, S, V = torch.svd(diff.reshape(diff.shape[0], -1))
                
                # Analyze singular values
                total_energy = (S ** 2).sum().item()
                cumsum = torch.cumsum(S ** 2, dim=0)
                
                # Find rank needed for 95% energy
                rank_95 = (cumsum / total_energy < 0.95).sum().item() + 1
                effective_rank = (S > 1e-5).sum().item()
                
                # LoRA typically has low rank (< 64)
                is_potential_lora = rank_95 < 64 and rank_95 < min(diff.shape) * 0.1
                
                lora_analysis[key] = {
                    "shape": list(diff.shape),
                    "effective_rank": effective_rank,
                    "rank_95_energy": rank_95,
                    "max_singular_value": S[0].item(),
                    "min_singular_value": S[-1].item(),
                    "is_potential_lora": is_potential_lora
                }
                
                if is_potential_lora:
                    potential_lora_layers.append(key)
                    print(f"  🎯 Potential LoRA in {key}:")
                    print(f"     Shape: {diff.shape}, Rank (95% energy): {rank_95}")
        
        print(f"\n  ✅ Found {len(potential_lora_layers)} layers with LoRA-like patterns")
        
        # Save LoRA analysis
        with open(self.output_dir / "lora_analysis.json", "w") as f:
            json.dump({
                "potential_lora_layers": potential_lora_layers,
                "analysis": lora_analysis
            }, f, indent=2)
        
        return lora_analysis, potential_lora_layers
    
    def extract_lora_weights(self, base_files, ft_files, potential_lora_layers, rank=16):
        """Extract approximate LoRA weights using SVD decomposition"""
        print(f"\n💾 Extracting LoRA Weights (rank={rank})...")

        lora_weights = {}

        # Load tensors for LoRA extraction
        print("  📥 Loading tensors for LoRA extraction...")
        base_tensors = {}
        ft_tensors = {}

        needed_keys = set(potential_lora_layers)

        for base_file in base_files:
            with safe_open(base_file, framework="pt") as f:
                for key in f.keys():
                    if key in needed_keys:
                        base_tensors[key] = f.get_tensor(key)

        for ft_file in ft_files:
            with safe_open(ft_file, framework="pt") as f:
                for key in f.keys():
                    if key in needed_keys:
                        ft_tensors[key] = f.get_tensor(key)

        print(f"  ✅ Loaded {len(base_tensors)} tensors for LoRA extraction")

        for key in potential_lora_layers:
            if key not in base_tensors or key not in ft_tensors:
                continue

            base_tensor = base_tensors[key].float()
            ft_tensor = ft_tensors[key].float()
            diff = ft_tensor - base_tensor
            
            if len(diff.shape) >= 2:
                # Reshape to 2D for SVD
                original_shape = diff.shape
                diff_2d = diff.reshape(diff.shape[0], -1)
                
                # SVD decomposition
                U, S, V = torch.svd(diff_2d)
                
                # Extract top-k components (LoRA A and B matrices)
                lora_A = V[:, :rank].T  # (rank, dim_in)
                lora_B = U[:, :rank] @ torch.diag(S[:rank])  # (dim_out, rank)
                
                # Store LoRA weights
                base_name = key.replace('.weight', '')
                lora_weights[f"{base_name}.lora_A"] = lora_A
                lora_weights[f"{base_name}.lora_B"] = lora_B
                
                # Verify reconstruction
                reconstructed = lora_B @ lora_A
                reconstruction_error = torch.norm(diff_2d - reconstructed) / torch.norm(diff_2d)
                
                print(f"  ✅ {key}:")
                print(f"     LoRA A: {lora_A.shape}, LoRA B: {lora_B.shape}")
                print(f"     Reconstruction error: {reconstruction_error.item():.6f}")
        
        # Save LoRA weights
        if lora_weights:
            lora_save_path = self.output_dir / "extracted_lora_weights.safetensors"
            from safetensors.torch import save_file
            save_file(lora_weights, lora_save_path)
            print(f"\n  💾 Saved LoRA weights to: {lora_save_path}")
            
            # Also save metadata
            lora_metadata = {
                "base_model": self.base_model_path,
                "finetuned_model": self.finetuned_model_path,
                "rank": rank,
                "layers": list(potential_lora_layers),
                "num_parameters": sum(w.numel() for w in lora_weights.values())
            }
            
            with open(self.output_dir / "lora_metadata.json", "w") as f:
                json.dump(lora_metadata, f, indent=2)
        
        return lora_weights
    
    def generate_report(self, config_diff, arch_comparison, differences, 
                       lora_analysis, potential_lora_layers):
        """Generate comprehensive comparison report"""
        print("\n📝 Generating Comparison Report...")
        
        report = []
        report.append("="*80)
        report.append("MODEL COMPARISON REPORT")
        report.append("="*80)
        report.append(f"\nBase Model: {self.base_model_path}")
        report.append(f"Finetuned Model: {self.finetuned_model_path}")
        report.append("\n" + "="*80)
        
        # Configuration differences
        report.append("\n1. CONFIGURATION DIFFERENCES")
        report.append("-"*80)
        if config_diff:
            for key, vals in config_diff.items():
                report.append(f"  {key}:")
                report.append(f"    Base:      {vals['base']}")
                report.append(f"    Finetuned: {vals['finetuned']}")
        else:
            report.append("  No configuration differences found.")
        
        # Architecture differences
        report.append("\n2. ARCHITECTURE DIFFERENCES")
        report.append("-"*80)
        report.append(f"  Common parameters:     {len(arch_comparison['common_keys'])}")
        report.append(f"  Only in base:          {len(arch_comparison['only_in_base'])}")
        report.append(f"  Only in finetuned:     {len(arch_comparison['only_in_finetuned'])}")
        
        if arch_comparison['lora_keys']:
            report.append(f"\n  Explicit LoRA/Adapter layers found: {len(arch_comparison['lora_keys'])}")
            for key in arch_comparison['lora_keys']:
                report.append(f"    - {key}")
        
        # Weight modifications
        report.append("\n3. WEIGHT MODIFICATIONS")
        report.append("-"*80)
        modified_count = sum(1 for d in differences.values() 
                           if d['changed_ratio'] > 0.01 or d['relative_diff'] > 0.01)
        report.append(f"  Layers with significant changes: {modified_count}")
        
        sorted_diffs = sorted(differences.items(), 
                            key=lambda x: x[1]['relative_diff'], 
                            reverse=True)
        
        report.append("\n  Top 20 most modified layers:")
        for i, (key, metrics) in enumerate(sorted_diffs[:20], 1):
            report.append(f"    {i:2d}. {key}")
            report.append(f"        Relative diff: {metrics['relative_diff']:.6f} | "
                        f"Changed: {metrics['changed_ratio']:.2%}")
        
        # LoRA analysis
        report.append("\n4. LORA PATTERN ANALYSIS")
        report.append("-"*80)
        report.append(f"  Layers with LoRA-like patterns: {len(potential_lora_layers)}")
        
        if potential_lora_layers:
            report.append("\n  Detected LoRA layers:")
            for layer in potential_lora_layers:
                if layer in lora_analysis:
                    info = lora_analysis[layer]
                    report.append(f"    - {layer}")
                    report.append(f"      Shape: {info['shape']} | "
                                f"Rank (95%): {info['rank_95_energy']}")
        
        # Summary
        report.append("\n" + "="*80)
        report.append("SUMMARY")
        report.append("="*80)
        
        if arch_comparison['lora_keys']:
            report.append("✅ EXPLICIT LoRA ADAPTERS FOUND")
            report.append("   The model contains explicit LoRA adapter layers.")
        elif potential_lora_layers:
            report.append("⚠️  POTENTIAL LoRA PATTERNS DETECTED")
            report.append("   Weight differences show low-rank patterns consistent with LoRA.")
            report.append("   LoRA weights have been extracted to: extracted_lora_weights.safetensors")
        else:
            report.append("ℹ️  NO LORA PATTERNS DETECTED")
            report.append("   The model appears to be fully fine-tuned (all parameters modified).")
        
        report.append("\n" + "="*80)
        
        # Save report
        report_text = "\n".join(report)
        print(report_text)
        
        with open(self.output_dir / "comparison_report.txt", "w") as f:
            f.write(report_text)
        
        print(f"\n✅ Report saved to: {self.output_dir / 'comparison_report.txt'}")
    
    def run_full_comparison(self, lora_rank=16):
        """Run complete comparison pipeline"""
        try:
            # Step 1: Verify local models
            self.verify_models()

            # Step 2: Compare configurations
            config_diff = self.compare_configs()

            # Step 3: Compare architectures
            base_files = self.get_safetensors_files(self.base_path)
            ft_files = self.get_safetensors_files(self.finetuned_path)
            base_weights, ft_weights, arch_comparison = self.compare_architectures()

            # Step 4: Compute weight differences layer by layer
            differences, modified_layers = self.compute_weight_differences_layer_by_layer(
                base_files, ft_files
            )
            
            # Step 5: Analyze LoRA patterns
            lora_analysis, potential_lora_layers = self.analyze_lora_patterns(
                base_files, ft_files, modified_layers
            )

            # Step 6: Extract LoRA weights if patterns found
            if potential_lora_layers:
                self.extract_lora_weights(
                    base_files, ft_files, potential_lora_layers, rank=lora_rank
                )
            
            # Step 7: Generate report
            self.generate_report(
                config_diff, arch_comparison, differences, 
                lora_analysis, potential_lora_layers
            )
            
            print("\n" + "="*80)
            print("✅ COMPARISON COMPLETE")
            print("="*80)
            print(f"\nAll results saved to: {self.output_dir}")
            print("\nGenerated files:")
            for file in sorted(self.output_dir.iterdir()):
                print(f"  - {file.name}")
            
        except Exception as e:
            print(f"\n❌ Error during comparison: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Biomni-R0-32B with Qwen3-32B to identify LoRA weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison (uses default local paths)
  python Compare_R0_qwen3.py
  
  # Custom output directory
  python Compare_R0_qwen3.py --output-dir ./my_comparison
  
  # Extract LoRA with custom rank
  python Compare_R0_qwen3.py --lora-rank 32
  
  # Compare different local models
  python Compare_R0_qwen3.py --base-path /path/to/qwen --finetuned-path /path/to/biomni
        """
    )
    
    parser.add_argument(
        "--base-path",
        type=str,
        default="/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B",
        help="Path to base model directory (default: /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Qwen3-32B)"
    )
    parser.add_argument(
        "--finetuned-path",
        type=str,
        default="/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview",
        help="Path to finetuned model directory (default: /projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni-R0-32B-Preview)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./model_comparison",
        help="Output directory for comparison results (default: ./model_comparison)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="Rank for LoRA extraction (default: 16)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="Threshold for considering weights as changed (default: 1e-6)"
    )
    
    args = parser.parse_args()
    
    # Create comparator and run
    comparator = ModelComparator(
        base_model_path=args.base_path,
        finetuned_model_path=args.finetuned_path,
        output_dir=args.output_dir
    )
    
    comparator.run_full_comparison(lora_rank=args.lora_rank)


if __name__ == "__main__":
    main()
