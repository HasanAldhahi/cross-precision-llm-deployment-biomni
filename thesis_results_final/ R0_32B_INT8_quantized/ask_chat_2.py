#!/usr/bin/env python3
"""
Script to evaluate JSON instances using ChatAI API (meta-llama-3.1-8b-instruct).
Sequential processing with proper error handling and rate limiting.
Uses custom endpoint: https://chat-ai.academiccloud.de/v1
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# ChatAI API configuration
BASE_URL = "https://chat-ai.academiccloud.de/v1"
MODEL_NAME = "openai-gpt-oss-120b"

# Rate limiting: Conservative to avoid issues
RATE_LIMIT_SECONDS = 2  # 2 seconds between requests
BATCH_SIZE = 10  # Process 10 instances, then longer wait
BATCH_WAIT_SECONDS = 5  # 5 seconds between batches

def create_prompt(instance: Dict[str, Any]) -> str:
    """
    Creates a detailed evaluation prompt for ChatAI.
    Includes:
    1. The Original Question (Context)
    2. Ground Truth & Extracted Solution
    3. Heuristic Statistics (Regex/Counting signals)
    4. Explicit Evaluation Rules
    """
    answer = instance.get("answer", "")
    extracted = instance.get("extracted_solution_or_last_step", "")
    # Include full original prompt
    original_prompt = instance.get("prompt", "")
    
    # Format the statistics/heuristics into a readable block
    heuristics_block = f"""
    - Answer Occurrences in last 200 tokens: {instance.get('answer_count_in_last_200_tokens', 0)}
    - Potential Answer (from 'is/has/are'): {instance.get('potential_answer_with_has_is', 'N/A')}
    - Potential Answer (from 'Final Answer'): {instance.get('potential_answer_final_answer', 'N/A')}
    - Potential Answer (from 'Conclusion'): {instance.get('potential_answer_conclusion', 'N/A')}
    - Potential Answer (emerges): {instance.get('potential_answer_emerges', 'N/A')}
    - Count of 'Final Answer' pattern matches: {instance.get('count_potential_answer_final_answer', 0)}
    - Count of 'Conclusion' pattern matches: {instance.get('count_potential_answer_conclusion', 0)}
    """
    
    prompt = f"""You are an expert evaluator for a biomedical automated reasoning task. 

Your goal is to determine if the **Model's Extracted Solution** matches the **Ground Truth Answer**.

### 1. Task Context (Original Question Snippet)

{original_prompt}


### 2. Ground Truth Answer

{answer}

### 3. Model's Extracted Solution

{extracted}

### 4. Heuristic Signals (Automated text analysis hints)

*Use these hints to interpret the Extracted Solution if the text is messy.*

{heuristics_block}

### Evaluation Rules (Read Carefully)

1. **Ignore Formatting**: The Ground Truth might be "CCR4", but the Solution might be "**CCR4**", "['CCR4']", or "{{'gene': 'CCR4'}}". Treat these as matches (1).

2. **Multiple Choice**: If Ground Truth is a single letter (e.g., "B"), and the Solution contains the letter with the correct text (e.g., "B. P335A" or "Answer: B"), this is a MATCH (1).

3. **Complex Objects**: If the Ground Truth is a JSON object (e.g., {{'disease': 'X', 'id': '123'}}), finding the ID '123' or the disease name 'X' in the solution is sufficient for a MATCH (1).

4. **Use Heuristics**: If the model output seems ambiguous, look at the "Heuristic Signals" above. If the statistics show the answer was found multiple times (e.g., "Answer Occurrences > 0") or detected in the "Conclusion", trust that the model found the answer.

5. **Nulls**: Only return 'null' if the solution is completely empty or completely unrelated/hallucinated. If it is just WRONG, return 0.

### Output Format

Provide your response in exactly this format (Reasoning on one line, Label on the next):

Reasoning: <One sentence explaining why it matches or not, referencing the text or heuristics>

Label: <1, 0, or null>

"""
    
    return prompt

def parse_eval_response(response_text: str) -> str:
    """
    Helper function to parse the output from ChatAI.
    Returns '1', '0', or 'null'.
    """
    if not response_text:
        return "null"
        
    # Look for the specific Label line
    lines = response_text.strip().split('\n')
    for line in lines[::-1]:  # Scan from bottom up
        if line.strip().startswith("Label:"):
            clean_val = line.replace("Label:", "").strip().lower()
            if "1" in clean_val: 
                return "1"
            if "0" in clean_val: 
                return "0"
            if "null" in clean_val: 
                return "null"
            
    # Fallback: if no Label tag, look for raw numbers
    if "1" in response_text: 
        return "1"
    if "0" in response_text: 
        return "0"
    
    return "null"

def call_chat_ai(client: OpenAI, prompt: str, max_retries: int = 3) -> Tuple[Optional[str], Optional[Exception]]:
    """Call ChatAI API"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for biomedical reasoning tasks."},
                    {"role": "user", "content": prompt}
                ],
                model=MODEL_NAME,
                temperature=0,
                max_tokens=150
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse and return result
            parsed_result = parse_eval_response(result_text)
            return parsed_result if parsed_result != "null" else None, None
                    
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = (
                "429" in error_str or
                "rate limit" in error_str or
                "quota" in error_str
            )
            
            # For rate limits, wait and retry
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                print(f"  Rate limit hit, waiting {wait_time}s...", flush=True)
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, e
    
    return None, Exception("Max retries exceeded")

def process_single_instance(
    instance: Dict[str, Any],
    client: OpenAI,
    output_file: str,
    stats: Dict[str, Any]
) -> Tuple[int, Optional[str], bool]:
    """Process a single instance"""
    instance_id = instance.get("instance_id")
    
    try:
        # Create prompt
        prompt = create_prompt(instance)
        
        # Call ChatAI
        result, error = call_chat_ai(client, prompt)
        
        if error is None:
            stats['successful'] += 1
        else:
            stats['errors'] += 1
            print(f"  Error for instance {instance_id}: {error}", flush=True)
            result = None
        
        # Create output entry
        new_instance = OrderedDict()
        new_instance["instance_id"] = instance_id
        new_instance["chat_eval"] = result if result is not None else None
        
        # Add answer first (if it exists), then all other fields
        if "answer" in instance:
            new_instance["answer"] = instance["answer"]
        
        # Add all other fields
        for key, value in instance.items():
            if key not in ["instance_id", "answer", "_line_num"]:
                new_instance[key] = value
        
        # Write to file
        with open(output_file, 'a', encoding='utf-8') as outfile:
            outfile.write(json.dumps(new_instance, ensure_ascii=False) + "\n")
            outfile.flush()
        
        return instance_id, result, True
        
    except Exception as e:
        stats['errors'] += 1
        print(f"  Error processing instance_id {instance_id}: {e}", flush=True)
        return instance_id, None, False

def load_processed_instance_ids(output_file: str) -> set:
    """Load already processed instance IDs from output file"""
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        instance_id = data.get("instance_id")
                        if instance_id is not None:
                            processed_ids.add(instance_id)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Could not load processed instances: {e}")
    return processed_ids

def process_instances(input_file: str, output_file: str):
    """Process all instances from input file"""
    # Initialize ChatAI client
    api_key = os.getenv("CUSTOM_MODEL_API_KEY")
    if not api_key:
        print("Error: CUSTOM_MODEL_API_KEY not found in environment!")
        print("Make sure CUSTOM_MODEL_API_KEY is set in .env file")
        return
    
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL
    )
    
    print(f"Using ChatAI API: {BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Processing: Sequential (one at a time)")
    print(f"Rate limit: {RATE_LIMIT_SECONDS} seconds between requests")
    print(f"Batch size: {BATCH_SIZE} instances per batch")
    print(f"Batch wait: {BATCH_WAIT_SECONDS} seconds between batches")
    print("-" * 50)
    
    # Load already processed instances
    print(f"Loading already processed instances from {output_file}...", flush=True)
    processed_ids = load_processed_instance_ids(output_file)
    if processed_ids:
        print(f"Found {len(processed_ids)} already processed instances. Will skip them.", flush=True)
    
    # Load all instances
    print(f"Loading instances from {input_file}...", flush=True)
    instances = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                instance = json.loads(line.strip())
                instance_id = instance.get("instance_id", line_num)
                
                # Skip if already processed
                if instance_id in processed_ids:
                    continue
                
                instance['_line_num'] = line_num
                instances.append(instance)
            except json.JSONDecodeError:
                continue
    
    total_instances = len(instances)
    if total_instances == 0:
        print("No instances to process!")
        return
    
    print(f"Total instances to process: {total_instances}")
    print("-" * 50)
    
    # Statistics
    stats = {
        'processed': 0,
        'successful': 0,
        'errors': 0
    }
    
    # Process in batches
    start_time = time.time()
    
    for batch_start in range(0, total_instances, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_instances)
        batch = instances[batch_start:batch_end]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (total_instances + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} instances...")
        
        # Process instances SEQUENTIALLY
        for idx, instance in enumerate(batch):
            try:
                instance_id, result, success = process_single_instance(
                    instance,
                    client,
                    output_file,
                    stats
                )
                
                result_str = result if result is not None else "null"
                print(f"  [{idx+1}/{len(batch)}] instance_id {instance_id}: {result_str}", flush=True)
                
                stats['processed'] += 1
                
                # Add delay between requests to avoid rate limits
                if idx < len(batch) - 1:  # Don't wait after last one in batch
                    time.sleep(RATE_LIMIT_SECONDS)
                    
            except Exception as e:
                stats['errors'] += 1
                print(f"  [Error] instance_id {instance.get('instance_id')}: {e}", flush=True)
        
        # Wait between batches
        if batch_end < total_instances:
            print(f"  Waiting {BATCH_WAIT_SECONDS} seconds before next batch...", flush=True)
            time.sleep(BATCH_WAIT_SECONDS)
        
        # Print progress
        elapsed = time.time() - start_time
        remaining = total_instances - stats['processed']
        if remaining > 0 and stats['processed'] > 0:
            avg_time = elapsed / stats['processed']
            eta = avg_time * remaining
            print(f"  Progress: {stats['processed']}/{total_instances} "
                  f"({stats['processed']*100//total_instances}%) | "
                  f"ETA: {eta/60:.1f} minutes", flush=True)
    
    print("\n" + "-" * 50)
    print(f"Processing complete!")
    print(f"Processed: {stats['processed']} instances")
    print(f"Successful: {stats['successful']}")
    print(f"Errors: {stats['errors']}")
    print(f"Output written to: {output_file}")
    
    # Separate instances by chat_eval value
    print("\n" + "=" * 50)
    print("Separating instances by evaluation result...")
    print("=" * 50)
    separate_instances_by_eval(output_file)

def separate_instances_by_eval(combined_file: str):
    """
    Separate instances by chat_eval value into different directories.
    Creates folders: Chat_eval_instances/Ai_eval_0, Ai_eval_1, Ai_eval_null
    """
    final_results_file = "Final_r0_results.jsonl"
    output_root = "Chat_eval_instances"
    
    # 1. Create output directories
    for subdir in ["Ai_eval_0", "Ai_eval_1", "Ai_eval_null"]:
        os.makedirs(os.path.join(output_root, subdir), exist_ok=True)
    
    # 2. Read combined instances
    if not os.path.exists(combined_file):
        print(f"Warning: {combined_file} not found. Skipping separation.")
        return
    
    instances = []
    with open(combined_file, "r") as f:
        for line in f:
            try:
                instance = json.loads(line)
                instances.append(instance)
            except json.JSONDecodeError:
                continue
    
    if not instances:
        print("No instances to separate.")
        return
    
    # 3. Get mapping from instance_id to prompt from Final_r0_results.jsonl
    id_to_prompt = {}
    if os.path.exists(final_results_file):
        with open(final_results_file, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    iid = obj.get("instance_id")
                    prompt = obj.get("prompt")
                    if iid is not None and prompt is not None:
                        id_to_prompt[iid] = prompt
                except json.JSONDecodeError:
                    continue
    
    # 4. Write each instance to a json file in the correct folder
    counts = {"Ai_eval_0": 0, "Ai_eval_1": 0, "Ai_eval_null": 0}
    
    for inst in instances:
        chat_eval_raw = inst.get("chat_eval", None)
        instance_id = inst.get("instance_id", None)
        if instance_id is None:
            continue  # skip instances with no id
        
        if chat_eval_raw == "0":
            subdir = "Ai_eval_0"
        elif chat_eval_raw == "1":
            subdir = "Ai_eval_1"
        else:
            subdir = "Ai_eval_null"
        
        out_dir = os.path.join(output_root, subdir)
        out_filename = f"{instance_id}.json"
        out_path = os.path.join(out_dir, out_filename)
        
        # If null, insert the prompt property if not already present
        if subdir == "Ai_eval_null":
            prompt = id_to_prompt.get(instance_id)
            if prompt is not None and "prompt" not in inst:
                inst["prompt"] = prompt
        
        # Write instance as JSON pretty
        with open(out_path, "w") as out_f:
            json.dump(inst, out_f, indent=2, ensure_ascii=False)
        
        counts[subdir] += 1
    
    print(f"\nSeparation complete!")
    print(f"  Ai_eval_0 (Wrong):    {counts['Ai_eval_0']} instances")
    print(f"  Ai_eval_1 (Correct):  {counts['Ai_eval_1']} instances")
    print(f"  Ai_eval_null (Error): {counts['Ai_eval_null']} instances")
    print(f"  Output directory: {output_root}/")

if __name__ == "__main__":
    input_file = "instances_to_be_checked_by_chat.jsonl"
    output_file = "combined_instances_with_chat_eval.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print(f"Current directory: {os.getcwd()}")
        exit(1)
    
    process_instances(input_file, output_file)

