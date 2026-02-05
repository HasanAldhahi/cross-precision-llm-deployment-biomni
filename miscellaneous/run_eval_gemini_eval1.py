#!/usr/bin/env python3
"""
Gemini 2.5 Pro Evaluation on Eval1 Dataset
Features:
- Single-threaded (Strict Rate Limiting for 5 RPM)
- Automatic Key Rotation on 429 errors
- Full Trajectory Capture (Using A1.get_trajectory())
- No Contamination (Fresh Agent per Question)
- IMPORTANT: default_config is overridden before each agent call to ensure
  tools (database.py, etc.) use the correct Gemini API key
"""

import sys
import os
import time
import json
import re
import dotenv
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

print("=" * 100)
print("GEMINI 2.5 PRO EVALUATION (ROBUST ROTATION + FULL HISTORY)")
print("=" * 100)

# --- Load Environment ---
dotenv.load_dotenv(project_root / '.env')
dotenv.load_dotenv(project_root / '.env.local', override=True)

# --- Imports ---
from biomni.eval.biomni_eval1 import BiomniEval1
from biomni.agent.c1 import A1
from biomni.config import default_config

# --- Initial default_config setup ---
# Note: This will be overridden in process_single_item() with the rotating Gemini key
# to ensure tools (database.py etc.) use the correct API key
print("✓ default_config will be set dynamically per request with rotating Gemini API key")
print("=" * 100)


# --- SIMPLE KEY ROTATION MANAGER ---
class SimpleKeyRotator:
    """Manages multiple Gemini API keys with automatic rotation on errors"""
    
    def __init__(self):
        self.keys = []
        self.cooldown_until = {}  # key -> timestamp when cooldown ends
        
        # 1. Load from GEMINI_API_KEY_1 ... _50
        for i in range(1, 51):
            k = os.getenv(f"GEMINI_API_KEY_{i}")
            if k and k.strip():
                self.keys.append(k.strip())
        
        # 2. Load from single GEMINI_API_KEY (split by comma if needed)
        base = os.getenv("GEMINI_API_KEY")
        if base:
            for k in base.split(","):
                k = k.strip()
                if k and k not in self.keys:
                    self.keys.append(k)
        
        self.current_index = 0
        self.successful_requests = 0
        self.failed_requests = 0
        print(f"✓ Rotation Manager: Loaded {len(self.keys)} API keys.")

    def get_key(self) -> str:
        """Get the current active API key, skipping cooled-down keys"""
        if not self.keys:
            return "EMPTY"
        
        now = time.time()
        
        # Try to find a key that's not in cooldown
        for _ in range(len(self.keys)):
            key = self.keys[self.current_index]
            cooldown_end = self.cooldown_until.get(key, 0)
            
            if now >= cooldown_end:
                return key
            else:
                # Key is cooling down, rotate
                self.current_index = (self.current_index + 1) % len(self.keys)
        
        # All keys are in cooldown - return the one with shortest wait
        return self.keys[self.current_index]

    def rotate(self, cooldown_minutes: int = 60):
        """Rotate to next key and put current key in cooldown"""
        if not self.keys:
            return
        
        current_key = self.keys[self.current_index]
        self.cooldown_until[current_key] = time.time() + (cooldown_minutes * 60)
        
        self.current_index = (self.current_index + 1) % len(self.keys)
        print(f"🔄 Rotating to Key #{self.current_index + 1}/{len(self.keys)} (Previous key cooling down for {cooldown_minutes}m)")

    def record_success(self):
        self.successful_requests += 1

    def record_failure(self):
        self.failed_requests += 1

    def get_statistics(self) -> dict:
        return {
            'total_keys': len(self.keys),
            'current_key_index': self.current_index,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests
        }


class GeminiEval1Evaluator:
    def __init__(self):
        # Paths
        self.DATASET_PATH = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni_Env/biomni_data/benchmark/Eval1/biomni_eval1_dataset.parquet"
        self.DATA_LAKE_PATH = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni_Env"
        
        # Model to use
        self.GEMINI_MODEL = "gemini-2.5-pro"
        
        # Output
        self.output_dir = Path("eval_output")
        self.output_dir.mkdir(exist_ok=True)
        self.results_file = self.output_dir / "gemini_eval_results.jsonl"
        self.statistics_file = self.output_dir / "gemini_statistics.json"
        
        # Optional: Load specific instance IDs to evaluate
        self.target_ids_file = self.output_dir / "easy_instance_ids.json"
        self.target_instance_ids = self.load_target_instance_ids()
        
        # Load Dataset
        self.evaluator = BiomniEval1(dataset_path=self.DATASET_PATH)
        print(f"✓ Dataset loaded: {len(self.evaluator.df)} items")
        
        # Initialize Rotation Manager
        self.rotator = SimpleKeyRotator()
        
        # Rate Limiting Config for Gemini 2.5 Pro Free Tier
        # Limit: 2 RPM = 1 request every 30 seconds minimum
        # We use 35 seconds to be safe and avoid 429 errors
        self.DELAY_BETWEEN_CALLS = 35.0 
        self.last_call_time = 0.0
        
        # State Tracking
        self.processed_ids = self.load_processed_ids()
        self.stats = {
            'total': len(self.evaluator.df),
            'processed': len(self.processed_ids),
            'correct': 0,
            'errors': 0,
            'rotations': 0
        }
        
        # Count existing correct answers
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get('success') and data.get('reward', 0) > 0:
                            self.stats['correct'] += 1
                    except:
                        pass
        
        print(f"✓ Using model: {self.GEMINI_MODEL}")
        print(f"✓ Delay between calls: {self.DELAY_BETWEEN_CALLS}s (for 2 RPM limit on free tier)")
        print(f"⚠️  Note: Gemini 2.5 Pro free tier = 2 RPM, 50 RPD per key")
        print(f"✓ With {self.rotator.get_statistics()['total_keys']} keys: ~{50 * self.rotator.get_statistics()['total_keys']} requests/day possible")

    def load_target_instance_ids(self) -> Optional[set]:
        """Load specific instance IDs to evaluate from JSON file. Returns None if file doesn't exist."""
        if self.target_ids_file.exists():
            try:
                with open(self.target_ids_file, 'r', encoding='utf-8') as f:
                    ids = json.load(f)
                    if isinstance(ids, list):
                        print(f"✓ Loaded {len(ids)} target instance IDs from {self.target_ids_file}")
                        return set(ids)
            except Exception as e:
                print(f"⚠️ Failed to load target IDs: {e}")
        print(f"ℹ️ No target IDs file found, will evaluate entire dataset")
        return None

    def load_processed_ids(self) -> set:
        ids = set()
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        ids.add(data['instance_id'])
                    except:
                        pass
        return ids

    def enforce_rate_limit(self):
        """Strict sleep to ensure we don't exceed 5 RPM"""
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.DELAY_BETWEEN_CALLS:
            sleep_time = self.DELAY_BETWEEN_CALLS - elapsed
            print(f"⏳ Rate limit: Sleeping {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        self.last_call_time = time.time()

    def reconstruct_trajectory(self, messages) -> str:
        """
        Reconstructs the full trajectory into a clean, numbered format without separator lines.
        Format: [01] ROLE:\nContent
        """
        full_log = ""
        step_count = 1
        
        for msg in messages:
            role = "UNKNOWN"
            content = ""

            # 1. Handle LangChain message objects (Standard)
            if hasattr(msg, 'content'):
                msg_type = type(msg).__name__
                
                if 'Human' in msg_type:
                    role = 'USER'
                elif 'AI' in msg_type:
                    role = 'ASSISTANT'
                elif 'System' in msg_type:
                    role = 'SYSTEM'
                elif 'Tool' in msg_type:
                    role = 'ENVIRONMENT'
                else:
                    role = msg_type.upper().replace('MESSAGE', '')
                
                content = str(msg.content)
            
            # 2. Handle Dicts (Fallback)
            elif isinstance(msg, dict):
                raw_role = msg.get('role', 'unknown').lower()
                if raw_role == 'tool': role = 'ENVIRONMENT'
                elif raw_role == 'user': role = 'USER'
                elif raw_role == 'assistant': role = 'ASSISTANT'
                else: role = raw_role.upper()
                
                content = str(msg.get('content', ''))
            
            # 3. Handle Raw Strings (Legacy)
            elif isinstance(msg, str):
                if msg.strip():
                    full_log += f"{msg}\n"
                continue

            # Formatting: Clean and simple as requested
            if content.strip():
                full_log += f"[{step_count:02d}] {role}:\n{content}\n\n"
                step_count += 1
                
        return full_log

    def extract_answer(self, response: str, task_name: str) -> Optional[str]:
        """Heuristics to extract answer from text"""
        if not response:
            return None
        
        # 1. Check <solution> tag (Highest Priority)
        match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL)
        if match:
            solution_text = match.group(1).strip()
            # For letter answers, extract just the letter
            if task_name in ['crispr_delivery', 'hle'] or task_name.startswith('lab_bench'):
                letter_match = re.search(r'\b([A-F])\b', solution_text)
                if letter_match:
                    return letter_match.group(1).upper()
            return solution_text
        
        # Multiple choice tasks (letter answers)
        if task_name in ['crispr_delivery', 'hle'] or task_name.startswith('lab_bench'):
            match = re.search(r'\[ANSWER\]([A-F])\[/ANSWER\]', response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            match = re.search(r'(?:answer|choice):\s*([A-F])\b', response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            # Look for single letter at end
            lines = response.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if len(line) == 1 and line.upper() in 'ABCDEF':
                    return line.upper()
            
        # Gene symbol tasks
        elif task_name.startswith('gwas_causal_gene') or task_name == 'screen_gene_retrieval':
            match = re.search(r'\b([A-Z][A-Z0-9\-]{1,10})\b', response)
            if match:
                return match.group(1)
            
        # Variant ID tasks
        elif task_name == 'gwas_variant_prioritization':
            match = re.search(r'\b(rs\d+|chr\d+:\d+[ATCG>]+)\b', response)
            if match:
                return match.group(1)
            
        # JSON-based tasks
        elif task_name == 'rare_disease_diagnosis':
            try:
                match = re.search(r'\{[^}]*"OMIM_ID"[^}]*\}', response)
                if match:
                    return match.group(0)
            except:
                pass
            
        elif task_name == 'patient_gene_detection':
            try:
                match = re.search(r'\{[^}]*"causal_gene"[^}]*\}', response)
                if match:
                    return match.group(0)
            except:
                pass
            
        return None

    def process_single_item(self, row):
        """Process one question with full history capture"""
        instance_id = row['instance_id']
        task_name = row['task_name']
        
        print(f"\n▶ Processing Instance {instance_id} ({task_name})")
        
        # 1. Prepare Result Dict
        result = {
            'instance_id': instance_id,
            'task_name': task_name,
            'task_instance_id': row['task_instance_id'],
            'prompt': row['prompt'],
            'ground_truth': row['answer'],
            'full_response': None,
            'predicted_answer': None,
            'reward': 0.0,
            'success': False,
            'error': None,
            'execution_time': 0.0
        }

        # 2. Retry Loop (Key Rotation)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get Key & Enforce Rate Limit
                current_key = self.rotator.get_key()
                self.enforce_rate_limit()
                
                # --- CRITICAL: Override default_config for tools (database.py etc.) ---
                # Tools use default_config.llm, api_key, source, base_url directly
                default_config.llm = self.GEMINI_MODEL
                default_config.source = "Gemini"
                default_config.api_key = current_key
                default_config.base_url = None  # Gemini SDK handles this internally
                
                # --- CRITICAL: Create FRESH Agent ---
                # This prevents contamination
                agent = A1(
                    path=self.DATA_LAKE_PATH,
                    llm=self.GEMINI_MODEL,
                    api_key=current_key,
                    use_tool_retriever=True,
                    source="Gemini",
                    timeout_seconds=1000
                )
                
                # Run
                start_ts = time.time()
                _, response_content = agent.go(row['prompt'])
                duration = time.time() - start_ts
                self.rotator.record_success()

                # --- CRITICAL: Capture Full History using get_trajectory() ---
                full_trajectory = ""
                if hasattr(agent, 'get_trajectory'):
                    trajectory_msgs = agent.get_trajectory()
                    if trajectory_msgs:
                        full_trajectory = self.reconstruct_trajectory(trajectory_msgs)
                    else:
                        full_trajectory = f"[FINAL_ONLY]:\n{response_content}"
                else:
                    # Fallback to old method
                    if hasattr(agent, 'messages') and agent.messages:
                        for msg in agent.messages:
                            role = str(msg.get('role', 'unknown')).upper()
                            content = str(msg.get('content', ''))
                            full_trajectory += f"\n[{role}]:\n{content}\n{'-'*40}\n"
                    else:
                        full_trajectory = response_content

                # Extract & Evaluate
                pred = self.extract_answer(response_content, task_name)
                reward = 0.0
                if pred:
                    reward = self.evaluator.evaluate(task_name, row['task_instance_id'], pred)

                # Update Result
                result.update({
                    'full_response': full_trajectory,  # Full history saved!
                    'predicted_answer': pred,
                    'reward': reward,
                    'execution_time': duration,
                    'success': True
                })
                
                print(f"  ✓ Success (Reward: {reward}, Time: {duration:.2f}s)")
                return result  # Exit retry loop
                
            except Exception as e:
                error_str = str(e)
                print(f"  ⚠ Error (Attempt {attempt+1}): {error_str}")
                self.rotator.record_failure()
                
                # Handle Rate Limits via Rotation
                if '429' in error_str or 'rate' in error_str.lower() or 'quota' in error_str.lower():
                    # Check if it's daily quota (50 RPD) vs minute quota (2 RPM)
                    if 'day' in error_str.lower() or 'daily' in error_str.lower():
                        # Daily quota exhausted - longer cooldown
                        print(f"  ⚠️ Daily quota exhausted for current key, rotating...")
                        self.rotator.rotate(cooldown_minutes=1440)  # 24 hours
                    else:
                        # Minute quota - parse retry delay if available
                        retry_match = re.search(r'retry in (\d+\.?\d*)', error_str.lower())
                        if retry_match:
                            wait_time = float(retry_match.group(1)) + 5  # Add 5s buffer
                            print(f"  ⏳ Server says retry in {retry_match.group(1)}s, waiting {wait_time:.0f}s...")
                            time.sleep(wait_time)
                        else:
                            self.rotator.rotate(cooldown_minutes=60)
                    self.stats['rotations'] += 1
                elif 'timeout' in error_str.lower():
                    time.sleep(10)  # Wait longer for timeout issues
                else:
                    result['error'] = error_str
                    break  # Unknown error - don't retry
        
        if not result['error']:
            result['error'] = "Max retries exceeded"
        return result

    def save_statistics(self):
        """Save current statistics to JSON file"""
        try:
            stats_output = {
                **self.stats,
                'last_updated': datetime.now().isoformat(),
                'overall_accuracy': self.stats['correct'] / max(len(self.processed_ids), 1),
                'api_rotation_stats': self.rotator.get_statistics()
            }
            
            with open(self.statistics_file, 'w', encoding='utf-8') as f:
                json.dump(stats_output, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"ERROR saving statistics: {e}")

    def run(self):
        print("🚀 Starting Evaluation Loop...")
        
        # Convert dataframe to dicts for iteration
        all_rows = [row for _, row in self.evaluator.df.iterrows()]
        
        # Filter by target instance IDs if specified
        if self.target_instance_ids is not None:
            all_rows = [row for row in all_rows if row['instance_id'] in self.target_instance_ids]
            print(f"🎯 Filtering to {len(all_rows)} target instances from easy_instance_ids.json")
            # Update total in stats
            self.stats['total'] = len(all_rows)
        
        # Filter to only unprocessed items
        todo_rows = [row for row in all_rows if row['instance_id'] not in self.processed_ids]
        
        print(f"Total items: {len(all_rows)}")
        print(f"Already processed: {len(self.processed_ids)}")
        print(f"Remaining: {len(todo_rows)}")
        
        for row in todo_rows:
            # Process
            result = self.process_single_item(row)
            
            # Save result immediately
            with open(self.results_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # Stats
            if result['success'] and result['reward'] > 0:
                self.stats['correct'] += 1
            if result['error']:
                self.stats['errors'] += 1
            
            self.processed_ids.add(row['instance_id'])
            self.stats['processed'] = len(self.processed_ids)
            
            # Periodic Status
            if len(self.processed_ids) % 5 == 0:
                acc = (self.stats['correct'] / len(self.processed_ids)) * 100
                print(f"\n📊 Stats: {len(self.processed_ids)}/{self.stats['total']} | Acc: {acc:.2f}% | Rotations: {self.stats['rotations']}")
                self.save_statistics()
        
        # Final statistics
        self.save_statistics()
        
        print(f"\n{'#'*100}")
        print("EVALUATION COMPLETED")
        print(f"{'#'*100}")
        print(f"Total processed: {len(self.processed_ids)}")
        print(f"Correct: {self.stats['correct']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Accuracy: {self.stats['correct']/max(len(self.processed_ids),1)*100:.2f}%")
        print(f"API Rotations: {self.stats['rotations']}")
        print(f"Results: {self.results_file}")
        print(f"{'#'*100}\n")


if __name__ == "__main__":
    evaluator = GeminiEval1Evaluator()
    evaluator.run()
