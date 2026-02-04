#!/usr/bin/env python3
"""
Script to create Final_r0_results_annotated.jsonl with chat_eval property.
Adds chat_eval based on extraction and evaluation results:
- "1" for correct instances
- "0" for wrong instances
- "2" for max_token errors
- "3" for proxy errors
- null for instances that couldn't be evaluated

Also adds token counts:
- total_output_tokens: tokens in full_response
- total_input_output_tokens: tokens in prompt + full_response
"""

import json
import os
import re
from collections import OrderedDict

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    USE_TIKTOKEN = True
except ImportError:
    TOKENIZER = None
    USE_TIKTOKEN = False
    print("Warning: tiktoken not available, using approximate token counting")

def count_tokens(text):
    """Count tokens in text using tiktoken or approximate method."""
    if text is None:
        return 0
    
    if USE_TIKTOKEN:
        try:
            return len(TOKENIZER.encode(str(text)))
        except:
            # Fallback to approximate method
            pass
    
    # Approximate method: ~1.3 tokens per word on average
    return int(len(str(text).split()) * 1.3)

def count_steps(full_response):
    """
    Count the number of steps in full_response.
    Steps are marked as [XX] ASSISTANT: where XX is a number.
    Returns the highest ASSISTANT number minus 1 (to exclude [01] USER).
    """
    if not full_response:
        return 0
    
    # Find all occurrences of [XX] ASSISTANT: where XX is a number
    pattern = r'\[(\d+)\]\s*ASSISTANT:'
    matches = re.findall(pattern, str(full_response))
    
    if not matches:
        return 0
    
    # Get the maximum number
    max_step = max(int(match) for match in matches)
    
    # Subtract 1 because [01] USER is the input, not a step
    return max_step - 1

def is_max_token_error(error_msg):
    """Check if error is a max_token error."""
    if not error_msg:
        return False
    return "max_tokens" in str(error_msg).lower() or "max_completion_tokens" in str(error_msg).lower()

def is_proxy_error(error_msg):
    """Check if error is a proxy error."""
    if not error_msg:
        return False
    return "proxy error" in str(error_msg).lower() or "timeout" in str(error_msg).lower()

def create_annotated_results(
    input_file="r0_eval_results_b1.jsonl",
    correct_file="correct_instances.jsonl",
    wrong_file="wrong_instances.jsonl",
    chat_eval_file="combined_instances_with_chat_eval.jsonl",
    output_file="Final_r0_results_annotated.jsonl"
):
    """
    Create annotated version of Final_r0_results.jsonl with chat_eval property.
    Detects errors directly from the 'error' field in Final_r0_results.jsonl.
    """
    
    # Load instance IDs for each category
    correct_ids = set()
    wrong_ids = set()
    chat_eval_results = {}  # instance_id -> chat_eval value
    
    # Load correct instances
    if os.path.exists(correct_file):
        with open(correct_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    instance = json.loads(line)
                    instance_id = instance.get("instance_id")
                    if instance_id is not None:
                        correct_ids.add(instance_id)
                except json.JSONDecodeError:
                    continue
    
    # Load wrong instances
    if os.path.exists(wrong_file):
        with open(wrong_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    instance = json.loads(line)
                    instance_id = instance.get("instance_id")
                    if instance_id is not None:
                        wrong_ids.add(instance_id)
                except json.JSONDecodeError:
                    continue
    
    # Load Chat AI evaluation results
    if os.path.exists(chat_eval_file):
        with open(chat_eval_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    instance = json.loads(line)
                    instance_id = instance.get("instance_id")
                    chat_eval = instance.get("chat_eval")
                    if instance_id is not None:
                        chat_eval_results[instance_id] = chat_eval
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded instance classifications:")
    print(f"  Correct (initial): {len(correct_ids)}")
    print(f"  Wrong (initial): {len(wrong_ids)}")
    print(f"  Chat AI evaluated: {len(chat_eval_results)}")
    
    # Process Final_r0_results.jsonl and add chat_eval
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
    
    annotated_count = 0
    max_token_count = 0
    proxy_error_count = 0
    
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            try:
                instance = json.loads(line)
                instance_id = instance.get("instance_id")
                error = instance.get("error")
                
                # Create ordered dict to control property order
                annotated_instance = OrderedDict()
                
                # Add instance_id first
                annotated_instance["instance_id"] = instance_id
                
                # Add chat_eval second (based on classification)
                # Priority order: errors (2,3) > initial classification (1,0) > chat eval > null
                if error and is_max_token_error(error):
                    annotated_instance["chat_eval"] = "2"
                    max_token_count += 1
                elif error and is_proxy_error(error):
                    annotated_instance["chat_eval"] = "3"
                    proxy_error_count += 1
                elif instance_id in correct_ids:
                    annotated_instance["chat_eval"] = "1"
                elif instance_id in wrong_ids:
                    annotated_instance["chat_eval"] = "0"
                elif instance_id in chat_eval_results:
                    # Use Chat AI evaluation result
                    annotated_instance["chat_eval"] = chat_eval_results[instance_id]
                else:
                    # Not evaluated yet
                    annotated_instance["chat_eval"] = None
                
                # Calculate token counts
                prompt_text = instance.get("prompt", "")
                full_response_text = instance.get("full_response", "")
                
                output_tokens = count_tokens(full_response_text)
                input_tokens = count_tokens(prompt_text)
                total_tokens = input_tokens + output_tokens
                
                # Calculate number of steps
                num_steps = count_steps(full_response_text)
                
                # Add token counts and steps after chat_eval
                annotated_instance["total_output_tokens"] = output_tokens
                annotated_instance["total_input_output_tokens"] = total_tokens
                annotated_instance["num_steps"] = num_steps
                
                # Add all other fields
                for key, value in instance.items():
                    if key != "instance_id":
                        annotated_instance[key] = value
                
                # Write to output file
                outfile.write(json.dumps(annotated_instance, ensure_ascii=False) + "\n")
                annotated_count += 1
                
            except json.JSONDecodeError:
                continue
    
    # Count final distribution, token statistics, and step statistics
    final_counts = {"1": 0, "0": 0, "2": 0, "3": 0, "null": 0}
    total_output_tokens = 0
    total_input_output_tokens = 0
    total_steps = 0
    steps_list = []
    
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                instance = json.loads(line)
                chat_eval = instance.get("chat_eval")
                if chat_eval == "1":
                    final_counts["1"] += 1
                elif chat_eval == "0":
                    final_counts["0"] += 1
                elif chat_eval == "2":
                    final_counts["2"] += 1
                elif chat_eval == "3":
                    final_counts["3"] += 1
                else:
                    final_counts["null"] += 1
                
                # Accumulate token counts
                total_output_tokens += instance.get("total_output_tokens", 0)
                total_input_output_tokens += instance.get("total_input_output_tokens", 0)
                
                # Accumulate step counts
                num_steps = instance.get("num_steps", 0)
                total_steps += num_steps
                steps_list.append(num_steps)
            except:
                continue
    
    print(f"\n✅ Created {output_file}")
    print(f"   Total instances annotated: {annotated_count}")
    print(f"\n   Final Breakdown:")
    print(f"   - chat_eval='1' (Correct):           {final_counts['1']} ({final_counts['1']/annotated_count*100:.1f}%)")
    print(f"   - chat_eval='0' (Wrong):             {final_counts['0']} ({final_counts['0']/annotated_count*100:.1f}%)")
    print(f"   - chat_eval='2' (Max token error):   {final_counts['2']} ({final_counts['2']/annotated_count*100:.1f}%)")
    print(f"   - chat_eval='3' (Proxy error):       {final_counts['3']} ({final_counts['3']/annotated_count*100:.1f}%)")
    print(f"   - chat_eval=null (Not evaluated):    {final_counts['null']} ({final_counts['null']/annotated_count*100:.1f}%)")
    
    print(f"\n   Sources:")
    print(f"   - Initial correct: {len(correct_ids)}")
    print(f"   - Initial wrong: {len(wrong_ids)}")
    print(f"   - Chat AI evaluated: {len(chat_eval_results)}")
    print(f"     └─ Marked correct: {sum(1 for v in chat_eval_results.values() if v == '1')}")
    print(f"     └─ Marked wrong: {sum(1 for v in chat_eval_results.values() if v == '0')}")
    print(f"     └─ Null/error: {sum(1 for v in chat_eval_results.values() if v not in ['0', '1'])}")
    print(f"   - Errors detected from 'error' field:")
    print(f"     └─ Max token errors: {max_token_count}")
    print(f"     └─ Proxy errors: {proxy_error_count}")
    
    print(f"\n   Token Statistics:")
    print(f"   - Total output tokens (responses): {total_output_tokens:,}")
    print(f"   - Total input+output tokens: {total_input_output_tokens:,}")
    print(f"   - Avg output tokens per instance: {total_output_tokens/annotated_count:.1f}")
    print(f"   - Avg input+output per instance: {total_input_output_tokens/annotated_count:.1f}")
    
    tokenizer_method = "tiktoken (cl100k_base)" if USE_TIKTOKEN else "approximate (1.3x words)"
    print(f"   - Token counting method: {tokenizer_method}")
    
    print(f"\n   Step Statistics:")
    print(f"   - Total steps across all instances: {total_steps:,}")
    print(f"   - Avg steps per instance: {total_steps/annotated_count:.1f}")
    if steps_list:
        steps_list.sort()
        print(f"   - Min steps: {steps_list[0]}")
        print(f"   - Median steps: {steps_list[len(steps_list)//2]}")
        print(f"   - Max steps: {steps_list[-1]}")
        instances_with_steps = sum(1 for s in steps_list if s > 0)
        print(f"   - Instances with steps: {instances_with_steps} ({instances_with_steps/annotated_count*100:.1f}%)")

if __name__ == "__main__":
    create_annotated_results()

