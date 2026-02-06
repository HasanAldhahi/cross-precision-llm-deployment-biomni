#!/usr/bin/env python3
"""
Quick script to analyze LoRA comparison results
Visualizes and summarizes the extracted LoRA patterns
"""

import json
import sys
from pathlib import Path
import argparse

def load_json(filepath):
    """Load JSON file"""
    with open(filepath) as f:
        return json.load(f)

def print_section(title, char="="):
    """Print section header"""
    print(f"\n{char*80}")
    print(f"  {title}")
    print(f"{char*80}\n")

def analyze_results(comparison_dir="./model_comparison"):
    """Analyze comparison results"""
    comp_dir = Path(comparison_dir)
    
    if not comp_dir.exists():
        print(f"❌ Directory not found: {comparison_dir}")
        print("   Run Compare_R0_qwen3.py first to generate comparison results.")
        sys.exit(1)
    
    print("="*80)
    print("  LoRA COMPARISON RESULTS ANALYSIS")
    print("="*80)
    print(f"Directory: {comparison_dir}\n")
    
    # 1. Architecture Comparison
    arch_file = comp_dir / "architecture_comparison.json"
    if arch_file.exists():
        print_section("1. ARCHITECTURE ANALYSIS", "=")
        arch = load_json(arch_file)
        
        print(f"Common parameters:     {len(arch['common_keys'])}")
        print(f"Only in base:          {len(arch['only_in_base'])}")
        print(f"Only in finetuned:     {len(arch['only_in_finetuned'])}")
        
        if arch['lora_keys']:
            print(f"\n🎯 EXPLICIT LoRA layers found: {len(arch['lora_keys'])}")
            print("\nLoRA Layers:")
            for i, key in enumerate(arch['lora_keys'][:20], 1):
                print(f"  {i:2d}. {key}")
            if len(arch['lora_keys']) > 20:
                print(f"  ... and {len(arch['lora_keys']) - 20} more")
        else:
            print("\nℹ️  No explicit LoRA keys found")
        
        if arch['only_in_finetuned']:
            print(f"\nNew layers in finetuned model: {len(arch['only_in_finetuned'])}")
            for key in arch['only_in_finetuned'][:10]:
                print(f"  - {key}")
            if len(arch['only_in_finetuned']) > 10:
                print(f"  ... and {len(arch['only_in_finetuned']) - 10} more")
    
    # 2. Weight Differences
    diff_file = comp_dir / "weight_differences.json"
    if diff_file.exists():
        print_section("2. WEIGHT MODIFICATION ANALYSIS", "=")
        diff = load_json(diff_file)
        
        modified = diff['modified_layers']
        all_diffs = diff['all_differences']
        
        print(f"Total layers analyzed:    {len(all_diffs)}")
        print(f"Significantly modified:   {len(modified)}")
        
        # Group by layer type
        layer_types = {}
        for layer in modified:
            # Extract layer type (e.g., "self_attn.q_proj")
            parts = layer.split('.')
            if len(parts) >= 2:
                layer_type = '.'.join(parts[-2:])
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        if layer_types:
            print("\nModifications by layer type:")
            for ltype, count in sorted(layer_types.items(), key=lambda x: -x[1]):
                print(f"  {ltype:40s} : {count:3d} layers")
        
        print("\nTop 15 most modified layers:")
        sorted_layers = sorted(all_diffs.items(), 
                             key=lambda x: x[1]['relative_diff'], 
                             reverse=True)
        
        for i, (layer, metrics) in enumerate(sorted_layers[:15], 1):
            print(f"  {i:2d}. {layer}")
            print(f"      Rel.Diff: {metrics['relative_diff']:.6f} | "
                  f"Changed: {metrics['changed_ratio']:6.2%} | "
                  f"Shape: {metrics['shape']}")
    
    # 3. LoRA Pattern Analysis
    lora_file = comp_dir / "lora_analysis.json"
    if lora_file.exists():
        print_section("3. LORA PATTERN DETECTION", "=")
        lora = load_json(lora_file)
        
        potential = lora['potential_lora_layers']
        analysis = lora['analysis']
        
        print(f"Layers analyzed:           {len(analysis)}")
        print(f"LoRA-like patterns found:  {len(potential)}")
        
        if potential:
            print("\n🎯 Detected LoRA-like layers:")
            for i, layer in enumerate(potential, 1):
                if layer in analysis:
                    info = analysis[layer]
                    print(f"  {i:2d}. {layer}")
                    print(f"      Shape: {info['shape']}")
                    print(f"      Effective rank: {info['effective_rank']}")
                    print(f"      Rank (95% energy): {info['rank_95_energy']}")
            
            # Analyze rank distribution
            ranks = [analysis[l]['rank_95_energy'] for l in potential if l in analysis]
            if ranks:
                avg_rank = sum(ranks) / len(ranks)
                min_rank = min(ranks)
                max_rank = max(ranks)
                
                print(f"\n📊 Rank Statistics:")
                print(f"   Average rank (95% energy): {avg_rank:.1f}")
                print(f"   Min rank: {min_rank}")
                print(f"   Max rank: {max_rank}")
                
                # Recommend LoRA rank
                recommended_rank = int(avg_rank * 1.2)  # 20% buffer
                print(f"\n💡 Recommended LoRA rank for fine-tuning: {recommended_rank}")
        else:
            print("\nℹ️  No LoRA-like patterns detected")
            print("   The model may be fully fine-tuned or uses different adaptation method")
    
    # 4. LoRA Metadata
    meta_file = comp_dir / "lora_metadata.json"
    if meta_file.exists():
        print_section("4. EXTRACTED LORA INFORMATION", "=")
        meta = load_json(meta_file)
        
        print(f"Base model:       {meta['base_model']}")
        print(f"Finetuned model:  {meta['finetuned_model']}")
        print(f"Extraction rank:  {meta['rank']}")
        print(f"Number of layers: {len(meta['layers'])}")
        print(f"Total parameters: {meta['num_parameters']:,}")
        
        # Identify target modules
        target_modules = set()
        for layer in meta['layers']:
            parts = layer.split('.')
            if len(parts) >= 2:
                target_modules.add(parts[-1])
        
        print(f"\nTarget modules identified:")
        for module in sorted(target_modules):
            count = sum(1 for l in meta['layers'] if l.endswith(module))
            print(f"  {module:20s} : {count:3d} layers")
        
        # Check if extracted weights exist
        weights_file = comp_dir / "extracted_lora_weights.safetensors"
        if weights_file.exists():
            size_mb = weights_file.stat().st_size / (1024 * 1024)
            print(f"\n✅ Extracted LoRA weights available:")
            print(f"   File: {weights_file}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"\n💡 You can use these weights for fine-tuning:")
            print(f"   python finetune_with_lora.py \\")
            print(f"       --lora-weights {weights_file} \\")
            print(f"       --dataset your/dataset")
    
    # 5. Configuration Differences
    config_file = comp_dir / "config_comparison.json"
    if config_file.exists():
        print_section("5. CONFIGURATION DIFFERENCES", "=")
        config = load_json(config_file)
        
        if config:
            print("Found configuration differences:")
            for key, vals in config.items():
                print(f"\n  {key}:")
                print(f"    Base:      {vals['base']}")
                print(f"    Finetuned: {vals['finetuned']}")
        else:
            print("✅ No configuration differences found")
    
    # 6. Summary and Recommendations
    print_section("6. SUMMARY & RECOMMENDATIONS", "=")
    
    # Determine scenario
    if arch_file.exists():
        arch = load_json(arch_file)
        if arch['lora_keys']:
            print("🎯 SCENARIO: Explicit LoRA Adapters Detected")
            print("\nThe finetuned model contains explicit LoRA adapter layers.")
            print("\nRecommended actions:")
            print("  1. Extract the LoRA layers directly from the model")
            print("  2. Use them as initialization for further fine-tuning")
            print("  3. Apply the same target modules in your LoRA config")
        elif lora_file.exists():
            lora = load_json(lora_file)
            if lora['potential_lora_layers']:
                print("⚙️  SCENARIO: LoRA-like Patterns Detected")
                print("\nWeight modifications show low-rank patterns similar to LoRA.")
                print("\nRecommended actions:")
                print("  1. Use extracted LoRA weights as initialization")
                print(f"  2. Apply rank ≈ {int(sum([lora['analysis'][l]['rank_95_energy'] for l in lora['potential_lora_layers']]) / len(lora['potential_lora_layers']) * 1.2)}")
                print("  3. Fine-tune on your biomedical dataset")
                print("\n  Command:")
                print(f"  python finetune_with_lora.py \\")
                print(f"      --lora-weights {comp_dir}/extracted_lora_weights.safetensors \\")
                print(f"      --dataset your/dataset")
            else:
                print("📚 SCENARIO: Full Fine-tuning Detected")
                print("\nThe model appears to be fully fine-tuned (all parameters modified).")
                print("\nRecommended actions:")
                print("  1. Apply fresh LoRA adapters on top of the finetuned model")
                print("  2. Or perform full fine-tuning if resources allow")
                print("  3. Focus on attention and FFN layers for LoRA")
    
    # Print next steps
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}\n")
    print("1. Review the comparison report:")
    print(f"   cat {comp_dir}/comparison_report.txt")
    print("\n2. Fine-tune with extracted LoRA:")
    print(f"   python finetune_with_lora.py --lora-weights {comp_dir}/extracted_lora_weights.safetensors")
    print("\n3. Or start fresh with detected config:")
    print(f"   python finetune_with_lora.py --rank 16 --target-modules [detected_modules]")
    print("\n4. Evaluate with Biomni A1 agent:")
    print(f"   python run_R0.py --base-url http://localhost:30000")
    print(f"\n{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze LoRA comparison results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze results in default directory
  python analyze_lora_results.py
  
  # Analyze results in custom directory
  python analyze_lora_results.py --dir ./my_comparison
        """
    )
    
    parser.add_argument(
        "--dir",
        type=str,
        default="./model_comparison",
        help="Directory containing comparison results (default: ./model_comparison)"
    )
    
    args = parser.parse_args()
    
    analyze_results(args.dir)

if __name__ == "__main__":
    main()


