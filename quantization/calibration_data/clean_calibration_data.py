#!/usr/bin/env python3
"""
Script to clean the full_response field in Data_r0_annotated.jsonl
Removes noise patterns:
1. [01] USER, [02] ASSISTANT, etc. markers
2. DEBUG lines and everything after them until newline
3. Dashes with "Ai Message" strings (e.g., "================================== Ai Message ==================================")
"""

import json
import re
from collections import OrderedDict

def clean_full_response(text):
    """
    Clean the full_response text by removing noise patterns.
    
    Args:
        text: The full_response text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    # Pattern 1: Remove [XX] USER: and [XX] ASSISTANT: markers
    # This removes patterns like [01] USER:, [02] ASSISTANT:, [03] ASSISTANT:, etc.
    text = re.sub(r'\[\d+\]\s*(?:USER|ASSISTANT):\s*', '', text)
    
    # Pattern 2: Remove DEBUG lines and everything after them until newline
    # This removes lines like "DEBUG: Using proxies for request: {...}"
    text = re.sub(r'DEBUG:.*?(?=\n|$)', '', text)
    
    # Pattern 3: Remove separator lines with "Ai Message" or similar
    # This removes lines like "================================== Ai Message =================================="
    text = re.sub(r'={3,}\s*(?:Ai Message|AI Message|ai message)\s*={3,}', '', text, flags=re.IGNORECASE)
    
    # Pattern 4: Remove other common separator patterns that might be noise
    # Remove lines that are just equals signs or dashes
    text = re.sub(r'\n\s*[=\-]{10,}\s*\n', '\n', text)
    
    # Clean up multiple consecutive newlines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def clean_calibration_dataset(
    input_file="Data_r0_annotated.jsonl",
    output_file="Data_r0_annotated_cleaned.jsonl"
):
    """
    Clean the calibration dataset by removing noise from full_response fields.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
    """
    print("=" * 80)
    print("Cleaning Calibration Dataset")
    print("=" * 80)
    
    total_instances = 0
    cleaned_instances = 0
    total_chars_before = 0
    total_chars_after = 0
    
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            try:
                instance = json.loads(line)
                total_instances += 1
                
                # Get the full_response field
                full_response = instance.get("full_response", "")
                
                if full_response:
                    # Track original length
                    original_length = len(full_response)
                    total_chars_before += original_length
                    
                    # Clean the full_response
                    cleaned_response = clean_full_response(full_response)
                    
                    # Track cleaned length
                    cleaned_length = len(cleaned_response)
                    total_chars_after += cleaned_length
                    
                    # Update the instance
                    instance["full_response"] = cleaned_response
                    
                    if original_length != cleaned_length:
                        cleaned_instances += 1
                
                # Write the cleaned instance
                outfile.write(json.dumps(instance, ensure_ascii=False) + "\n")
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    
    # Print statistics
    print(f"\n✓ Cleaned {output_file}")
    print(f"\n📊 STATISTICS:")
    print(f"  Total instances processed: {total_instances}")
    print(f"  Instances with cleaned text: {cleaned_instances}")
    print(f"  Instances unchanged: {total_instances - cleaned_instances}")
    print(f"\n📏 CHARACTER REDUCTION:")
    print(f"  Total characters before: {total_chars_before:,}")
    print(f"  Total characters after: {total_chars_after:,}")
    print(f"  Characters removed: {total_chars_before - total_chars_after:,}")
    print(f"  Reduction: {(1 - total_chars_after/total_chars_before)*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("✅ CLEANING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    clean_calibration_dataset()




