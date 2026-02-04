#!/usr/bin/env python3
"""
Prepare calibration dataset from baseline R0 evaluation results.

This script extracts successful completions (prompt + full_response) from the
baseline model evaluation to use as calibration data. This captures the full
"trajectory" of the model's behavior, which is better for quantization calibration
than using prompts alone.

Key features:
- Only uses successful completions (success=True)
- Balances across all tasks for fair representation
- Uses full prompt + full_response as calibration text
- Random stratified sampling for diversity
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import random

print("=" * 80)
print("CALIBRATION DATASET PREPARATION (Baseline Trajectories)")
print("=" * 80)

# Configuration
BASELINE_RESULTS_PATH = "Data_r0_annotated_cleaned.jsonl"
OUTPUT_DIR = Path(__file__).parent
NUM_CALIBRATION_SAMPLES = 128
RANDOM_SEED = 42

print(f"\nBaseline results: {BASELINE_RESULTS_PATH}")
print(f"Number of calibration samples: {NUM_CALIBRATION_SAMPLES}")
print(f"Sampling strategy: Stratified random across tasks (successful completions only)")
print(f"Calibration format: prompt + full_response (complete trajectories)")
print(f"Random seed: {RANDOM_SEED}")
print(f"Output directory: {OUTPUT_DIR}")

# Set random seed
random.seed(RANDOM_SEED)

# --- TOKEN COUNTING FUNCTION ---
try:
    import tiktoken

    def count_tokens(text, enc_name="gpt2"):
        enc = tiktoken.get_encoding(enc_name)
        return len(enc.encode(text))
except ImportError:
    # fallback: crude whitespace split as estimation
    def count_tokens(text, enc_name=None):
        # Not accurate, but gives an order of magnitude
        return len(text.split())

    print("[!] tiktoken not found. Falling back to whitespace token count (less accurate).")
else:
    print("[i] Using tiktoken for accurate token counting.")

# Load baseline results
print("\n[1/5] Loading baseline evaluation results...")
try:
    results = []
    with open(BASELINE_RESULTS_PATH, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Warning: Skipping line {line_num} (invalid JSON): {e}")
    
    print(f"✓ Loaded {len(results)} evaluation instances")
except FileNotFoundError:
    print(f"✗ ERROR: Baseline results file not found at:")
    print(f"  {BASELINE_RESULTS_PATH}")
    print(f"\nPlease ensure you have run the baseline evaluation and the results file exists.")
    sys.exit(1)
except Exception as e:
    print(f"✗ ERROR: Failed to load baseline results: {e}")
    sys.exit(1)

# Filter for successful completions
print("\n[2/5] Filtering for successful completions...")
successful_results = [r for r in results if r.get('success', False) == True]
print(f"✓ Found {len(successful_results)} successful completions out of {len(results)} total")
print(f"  Success rate: {len(successful_results)/len(results)*100:.1f}%")

if len(successful_results) < NUM_CALIBRATION_SAMPLES:
    print(f"\n⚠️  WARNING: Only {len(successful_results)} successful completions available")
    print(f"           Requested {NUM_CALIBRATION_SAMPLES} samples")
    print(f"           Will use all {len(successful_results)} available samples")
    NUM_CALIBRATION_SAMPLES = len(successful_results)

# Group by task
print("\n[3/5] Grouping by task...")
task_groups = defaultdict(list)
for result in successful_results:
    task_name = result.get('task_name', 'unknown')
    task_groups[task_name].append(result)

print(f"✓ Found {len(task_groups)} unique tasks:")
for task, instances in sorted(task_groups.items()):
    print(f"  • {task}: {len(instances)} successful completions")

# Stratified sampling
print(f"\n[4/5] Performing stratified random sampling...")
print(f"  Target: {NUM_CALIBRATION_SAMPLES} samples balanced across {len(task_groups)} tasks")

# Calculate samples per task
samples_per_task = NUM_CALIBRATION_SAMPLES // len(task_groups)
remainder = NUM_CALIBRATION_SAMPLES % len(task_groups)

print(f"  Base samples per task: {samples_per_task}")
print(f"  Remainder to distribute: {remainder}")

calibration_samples = []
task_sample_counts = {}

# Sample from each task
for task, instances in sorted(task_groups.items()):
    # Calculate how many samples for this task
    n_samples = samples_per_task
    if remainder > 0:
        n_samples += 1
        remainder -= 1
    
    # Don't sample more than available
    n_samples = min(n_samples, len(instances))
    
    # Random sample
    sampled = random.sample(instances, n_samples)
    calibration_samples.extend(sampled)
    task_sample_counts[task] = n_samples
    
    print(f"  • {task}: sampled {n_samples}/{len(instances)}")

print(f"\n✓ Sampled {len(calibration_samples)} total instances")

# Create calibration data
print("\n[5/5] Creating calibration dataset...")

calibration_texts = []
for sample in calibration_samples:
    # Combine prompt + full_response as the calibration text
    prompt = str(sample.get('prompt', '')).strip()
    full_response = str(sample.get('full_response', '')).strip()
    
    if not prompt:
        print(f"  ⚠️  Warning: Empty prompt for instance {sample.get('instance_id', '?')}")
        continue
    
    if not full_response:
        print(f"  ⚠️  Warning: Empty response for instance {sample.get('instance_id', '?')}")
        continue
    
    # Combine as conversation trajectory
    calibration_text = f"{prompt}\n\n{full_response}"
    calibration_texts.append(calibration_text)

print(f"✓ Created {len(calibration_texts)} calibration texts")

# Validation check
print("\n  Validation:")
total_length = sum(len(text) for text in calibration_texts)
avg_length = total_length / len(calibration_texts) if calibration_texts else 0
min_length = min(len(text) for text in calibration_texts) if calibration_texts else 0
max_length = max(len(text) for text in calibration_texts) if calibration_texts else 0

print(f"  • Total characters: {total_length:,}")
print(f"  • Average length: {avg_length:,.0f} chars")
print(f"  • Min length: {min_length:,} chars")
print(f"  • Max length: {max_length:,} chars")

# --- TOKEN COUNT CALCULATION & PRINT ---
total_tokens = sum(count_tokens(text) for text in calibration_texts)
avg_tokens = total_tokens / len(calibration_texts) if calibration_texts else 0
min_tokens = min(count_tokens(text) for text in calibration_texts) if calibration_texts else 0
max_tokens = max(count_tokens(text) for text in calibration_texts) if calibration_texts else 0

print(f"  • Total tokens: {total_tokens:,}")
print(f"  • Average tokens: {avg_tokens:,.0f}")
print(f"  • Min tokens: {min_tokens:,}")
print(f"  • Max tokens: {max_tokens:,}")

if avg_length < 100:
    print(f"\n  ⚠️  WARNING: Average text length is very short ({avg_length:.0f} chars)")
    print(f"              This might indicate a problem with data extraction")
elif avg_length > 10000:
    print(f"\n  ⚠️  WARNING: Average text length is very long ({avg_length:.0f} chars)")
    print(f"              Some quantization methods may have issues with very long texts")
else:
    print(f"  ✓ Text lengths look reasonable")

# Save calibration data
output_json = OUTPUT_DIR / "calibration_data.json"
output_preview = OUTPUT_DIR / "calibration_preview.txt"

print(f"\nSaving calibration data...")
print(f"  JSON: {output_json}")
print(f"  Preview: {output_preview}")

# Save as JSON
with open(output_json, 'w') as f:
    json.dump(calibration_texts, f, indent=2)
print(f"✓ Saved {len(calibration_texts)} calibration texts to JSON")

# Save preview
with open(output_preview, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CALIBRATION DATASET PREVIEW\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Total samples: {len(calibration_texts)}\n")
    f.write(f"Data source: Baseline R0 evaluation (successful completions only)\n")
    f.write(f"Format: prompt + full_response (complete trajectories)\n\n")
    
    f.write("Task distribution:\n")
    for task, count in sorted(task_sample_counts.items()):
        f.write(f"  • {task}: {count} samples\n")
    
    f.write(f"\nText statistics:\n")
    f.write(f"  • Average length: {avg_length:,.0f} characters\n")
    f.write(f"  • Min length: {min_length:,} characters\n")
    f.write(f"  • Max length: {max_length:,} characters\n")
    f.write(f"  • Total characters: {total_length:,}\n")
    f.write(f"  • Average tokens: {avg_tokens:,.0f}\n")
    f.write(f"  • Total tokens: {total_tokens:,}\n")
    f.write(f"  • Min tokens: {min_tokens:,}\n")
    f.write(f"  • Max tokens: {max_tokens:,}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("SAMPLE PREVIEW (First 3 samples, truncated)\n")
    f.write("=" * 80 + "\n\n")
    
    for i, text in enumerate(calibration_texts[:3], 1):
        f.write(f"Sample {i}:\n")
        f.write("-" * 80 + "\n")
        # Show first 500 chars and last 200 chars if text is long
        if len(text) > 1000:
            f.write(text[:500])
            f.write(f"\n\n... [{len(text)-700:,} characters omitted] ...\n\n")
            f.write(text[-200:])
        else:
            f.write(text)
        f.write("\n" + "=" * 80 + "\n\n")

print(f"✓ Saved preview to {output_preview}")

# Final summary
print("\n" + "=" * 80)
print("CALIBRATION DATASET PREPARATION COMPLETE")
print("=" * 80)
print(f"\n✓ Successfully created calibration dataset with {len(calibration_texts)} samples")
print(f"✓ Balanced across {len(task_groups)} tasks")
print(f"✓ Using full trajectories (prompt + response)")
print(f"✓ Average calibration text length: {avg_length:,.0f} characters")
print(f"✓ Total tokens in calibration data: {total_tokens:,}")
print(f"✓ Average tokens per calibration text: {avg_tokens:,.0f}")
print(f"\nOutput files:")
print(f"  • {output_json}")
print(f"  • {output_preview}")
print(f"\nNext steps:")
print(f"  1. Review calibration_preview.txt to verify the data looks correct")
print(f"  2. Run quantization scripts (AWQ, PTQ) - they will use calibration_data.json")
print(f"  3. AWQ: python awq/quantize_awq.py")
print(f"  4. PTQ: python ptq/quantize_ptq.py")
print()
