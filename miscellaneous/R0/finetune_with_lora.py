#!/usr/bin/env python3
"""
Fine-tune Qwen3-32B with extracted LoRA weights from Biomni-R0
This script helps you continue training using the identified LoRA layers
"""

import sys
import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
import argparse

class LoRAFineTuner:
    def __init__(self, base_model_name="Qwen/Qwen3-32B-FP8", 
                 lora_weights_path=None,
                 output_dir="./finetuned_model"):
        self.base_model_name = base_model_name
        self.lora_weights_path = lora_weights_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print("="*80)
        print("LoRA FINE-TUNING SETUP")
        print("="*80)
        print(f"Base Model: {base_model_name}")
        print(f"LoRA Weights: {lora_weights_path}")
        print(f"Output Dir: {output_dir}")
        print("="*80)
    
    def load_base_model(self):
        """Load the base model"""
        print("\n📥 Loading base model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✅ Model loaded: {self.model.num_parameters():,} parameters")
        return self.model, self.tokenizer
    
    def analyze_extracted_lora(self):
        """Analyze extracted LoRA weights to determine configuration"""
        print("\n🔍 Analyzing extracted LoRA weights...")
        
        if not self.lora_weights_path or not Path(self.lora_weights_path).exists():
            print("  ⚠️  No LoRA weights provided, will use default LoRA config")
            return None
        
        # Load metadata
        metadata_path = Path(self.lora_weights_path).parent / "lora_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            print(f"  ✅ Found LoRA metadata:")
            print(f"     Rank: {metadata.get('rank', 'unknown')}")
            print(f"     Layers: {len(metadata.get('layers', []))}")
            print(f"     Parameters: {metadata.get('num_parameters', 'unknown'):,}")
            
            # Identify target modules from layer names
            layers = metadata.get('layers', [])
            target_modules = set()
            
            for layer in layers:
                # Extract module name (e.g., "model.layers.0.self_attn.q_proj")
                parts = layer.split('.')
                if len(parts) >= 2:
                    module_name = parts[-1]  # e.g., "q_proj"
                    target_modules.add(module_name)
            
            print(f"\n  🎯 Identified target modules: {sorted(target_modules)}")
            
            return {
                'rank': metadata.get('rank', 16),
                'target_modules': sorted(target_modules) if target_modules else None,
                'layers': layers
            }
        
        return None
    
    def setup_lora_config(self, rank=16, target_modules=None, lora_alpha=32, lora_dropout=0.1):
        """Setup LoRA configuration based on extracted weights or defaults"""
        print(f"\n⚙️  Setting up LoRA configuration...")
        
        # Default target modules for Qwen models
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj"      # FFN
            ]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False
        )
        
        print(f"  LoRA Config:")
        print(f"    Rank (r): {rank}")
        print(f"    Alpha: {lora_alpha}")
        print(f"    Dropout: {lora_dropout}")
        print(f"    Target modules: {target_modules}")
        
        return lora_config
    
    def apply_lora(self, lora_config):
        """Apply LoRA to the base model"""
        print("\n🔧 Applying LoRA to base model...")
        
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Load extracted LoRA weights if available
        if self.lora_weights_path and Path(self.lora_weights_path).exists():
            print(f"  📂 Loading extracted LoRA weights from {self.lora_weights_path}...")
            try:
                from safetensors.torch import load_file
                lora_weights = load_file(self.lora_weights_path)
                
                # Load weights into the model (with proper name mapping)
                missing, unexpected = self.peft_model.load_state_dict(lora_weights, strict=False)
                
                if not missing and not unexpected:
                    print("  ✅ Successfully loaded all LoRA weights")
                else:
                    print(f"  ⚠️  Partial load: {len(missing)} missing, {len(unexpected)} unexpected keys")
                    if missing:
                        print(f"     Missing keys (sample): {missing[:5]}")
                    if unexpected:
                        print(f"     Unexpected keys (sample): {unexpected[:5]}")
            except Exception as e:
                print(f"  ⚠️  Could not load weights: {e}")
                print("     Will start with random LoRA initialization")
        
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        print(f"\n  📊 Model Parameters:")
        print(f"     Total: {total_params:,}")
        print(f"     Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        return self.peft_model
    
    def prepare_dataset(self, dataset_name="openai/gsm8k", split="train", max_samples=1000):
        """Prepare dataset for training"""
        print(f"\n📚 Preparing dataset: {dataset_name}...")
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        print(f"  ✅ Loaded {len(dataset)} samples")
        
        # Tokenize function (customize based on your dataset)
        def tokenize_function(examples):
            # Customize this based on your dataset structure
            if "question" in examples and "answer" in examples:
                texts = [f"Question: {q}\nAnswer: {a}" 
                        for q, a in zip(examples["question"], examples["answer"])]
            elif "text" in examples:
                texts = examples["text"]
            else:
                # Fallback: use first text field found
                text_field = next((k for k in examples.keys() if isinstance(examples[k][0], str)), None)
                if text_field:
                    texts = examples[text_field]
                else:
                    raise ValueError("Could not find text field in dataset")
            
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=512,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        print(f"  ✅ Dataset tokenized")
        
        return tokenized_dataset
    
    def train(self, train_dataset, eval_dataset=None, 
              learning_rate=2e-4, num_epochs=3, batch_size=4):
        """Train the model with LoRA"""
        print(f"\n🚀 Starting training...")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=100,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            save_total_limit=3,
            fp16=True,
            gradient_checkpointing=True,
            optim="adamw_torch",
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        print(f"  Training config:")
        print(f"    Epochs: {num_epochs}")
        print(f"    Batch size: {batch_size}")
        print(f"    Learning rate: {learning_rate}")
        print(f"    Total steps: {len(train_dataset) * num_epochs // batch_size}")
        
        # Train
        trainer.train()
        
        print(f"\n  ✅ Training complete!")
        
        return trainer
    
    def save_model(self):
        """Save the fine-tuned model"""
        print(f"\n💾 Saving fine-tuned model...")
        
        # Save LoRA adapter
        self.peft_model.save_pretrained(self.output_dir / "lora_adapter")
        self.tokenizer.save_pretrained(self.output_dir / "lora_adapter")
        
        print(f"  ✅ LoRA adapter saved to: {self.output_dir / 'lora_adapter'}")
        
        # Save merged model (optional)
        print(f"\n  Merging LoRA weights with base model...")
        merged_model = self.peft_model.merge_and_unload()
        merged_model.save_pretrained(self.output_dir / "merged_model")
        self.tokenizer.save_pretrained(self.output_dir / "merged_model")
        
        print(f"  ✅ Merged model saved to: {self.output_dir / 'merged_model'}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-32B using extracted LoRA weights from Biomni-R0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune with extracted LoRA weights
  python finetune_with_lora.py --lora-weights ./model_comparison/extracted_lora_weights.safetensors
  
  # Fine-tune with custom config
  python finetune_with_lora.py --rank 32 --dataset your/dataset --epochs 5
  
  # Start fresh (no pre-extracted LoRA)
  python finetune_with_lora.py --rank 16 --target-modules q_proj k_proj v_proj
        """
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-32B-FP8",
        help="Base model to fine-tune (default: Qwen/Qwen3-32B-FP8)"
    )
    parser.add_argument(
        "--lora-weights",
        type=str,
        default=None,
        help="Path to extracted LoRA weights (.safetensors file)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./finetuned_model",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openai/gsm8k",
        help="Dataset name from HuggingFace (default: openai/gsm8k)"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="LoRA rank (default: auto-detect from weights or 16)"
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="Target modules for LoRA (default: auto-detect or standard attention/ffn)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of training samples (default: 1000)"
    )
    
    args = parser.parse_args()
    
    # Create fine-tuner
    finetuner = LoRAFineTuner(
        base_model_name=args.base_model,
        lora_weights_path=args.lora_weights,
        output_dir=args.output_dir
    )
    
    # Load base model
    finetuner.load_base_model()
    
    # Analyze extracted LoRA if available
    lora_info = finetuner.analyze_extracted_lora()
    
    # Setup LoRA config
    rank = args.rank or (lora_info['rank'] if lora_info else 16)
    target_modules = args.target_modules or (lora_info['target_modules'] if lora_info else None)
    
    lora_config = finetuner.setup_lora_config(
        rank=rank,
        target_modules=target_modules
    )
    
    # Apply LoRA
    finetuner.apply_lora(lora_config)
    
    # Prepare dataset
    train_dataset = finetuner.prepare_dataset(
        dataset_name=args.dataset,
        max_samples=args.max_samples
    )
    
    # Train
    finetuner.train(
        train_dataset=train_dataset,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save
    finetuner.save_model()
    
    print("\n" + "="*80)
    print("✅ FINE-TUNING COMPLETE")
    print("="*80)
    print(f"\nModel saved to: {args.output_dir}")
    print(f"  - LoRA adapter: {args.output_dir}/lora_adapter")
    print(f"  - Merged model: {args.output_dir}/merged_model")


if __name__ == "__main__":
    main()


