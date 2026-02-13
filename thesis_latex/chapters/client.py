
"""""
Biomni-R0-32B Multi-Process Asynchronous Evaluation on Eval1 Dataset using B1 Agent.
This script is designed to run on the same node as the vLLM server.

Features:
- Uses B1 meta-agent instead of A1
- Can target specific error instances if instance_ids_to_be_evaluated.json exists
"""

import os
import sys
import time
import json
import re
import asyncio
import multiprocessing
import queue
from datetime import datetime
from typing import Dict, Any, Optional, Set, List
from pathlib import Path
import pandas as pd
import numpy as np
from filelock import FileLock
import dotenv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

print("=" * 100)
print("BIOMNI-R0-32B MULTI-PROCESS ASYNC EVALUATION (B1 AGENT)")
print("=" * 100)

# --- Environment Setup ---
dotenv.load_dotenv()
from biomni.eval.biomni_eval1 import BiomniEval1
from biomni.agent.c1 import A1  # Using B1 instead of A1
from biomni.config import default_config
print("✓ Imports successful")

# --- Server Configuration ---
# Since this script runs on the server node, the URL is always localhost.
R0_BASE_URL = "http://localhost:9200/v1"
print(f"✓ Target R0 Server URL is hardcoded to: {R0_BASE_URL}")
print("=" * 100)

print("\n--- [Step 1: Checking Environment Variables] ---")
https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
no_proxy = os.getenv("NO_PROXY") or os.getenv("no_proxy")
print(f"HTTPS_PROXY is set to: {https_proxy}")
print(f"NO_PROXY is set to: {no_proxy}")
if not https_proxy:
    print("⚠️ WARNING: HTTPS_PROXY is not set. External connections will likely fail.")
print("-" * 50)


NUM_PROCESSES = 18
ASYNC_CONCURRENCY_PER_PROCESS = 4


class BiomniR0EvaluatorB1:
    def __init__(self, process_id: int = 0):
        self.process_id = process_id
        self.CONCURRENCY_LIMIT = ASYNC_CONCURRENCY_PER_PROCESS
        self.DATASET_PATH = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni_Env/biomni_data/benchmark/Eval1/biomni_eval1_dataset.parquet"
        self.DATA_LAKE_PATH = "/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/Biomni_Env"
        self.R0_MODEL_NAME = "biomni-reasoning"
        self.output_dir = Path("Different_port")
        self.output_dir.mkdir(exist_ok=True)
        self.results_file = self.output_dir / "r0_eval_results_b1.jsonl"
        self.processed_instances_file = self.output_dir / "processed_instances_b1.jsonl"
        self.statistics_file = self.output_dir / "r0_statistics_b1.json"
        self.error_instance_ids_file = self.output_dir / "instance_ids_to_be_evaluated.json"
        self.results_lock = FileLock(str(self.results_file) + ".lock")
        self.processed_lock = FileLock(str(self.processed_instances_file) + ".lock")
        self.stats_lock = FileLock(str(self.statistics_file) + ".lock")
        self.evaluator = BiomniEval1(dataset_path=self.DATASET_PATH)
        default_config.llm = self.R0_MODEL_NAME
        default_config.source = "Custom"
        default_config.base_url = R0_BASE_URL
        default_config.api_key = "EMPTY"
        print(f"BiomniConfig detected proxies: {default_config.proxies}")
        
        # --- CRITICAL FIX: REMOVED AGENT CREATION FROM INIT ---
        # We do NOT create self.agent here anymore. 
        # Creating it here causes the "merged questions" bug.
        # Agent is now created fresh inside process_instance() for each question.
        # ------------------------------------------------------

    def load_error_instance_ids(self) -> Optional[List[int]]:
        if not self.error_instance_ids_file.exists():
            return None
        try:
            with open(self.error_instance_ids_file, 'r') as f:
                error_ids = json.load(f)
                if error_ids and isinstance(error_ids, list) and len(error_ids) > 0:
                    print(f"✓ Loaded {len(error_ids)} error instance IDs")
                    return error_ids
                else:
                    return None
        except:
            return None

    def load_already_processed_ids(self) -> Set[str]:
        processed = set()
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try: 
                        processed.add(json.loads(line)['instance_id'])
                    except (json.JSONDecodeError, KeyError): 
                        continue
        return processed
    
    def try_claim_instance(self, instance_id: str) -> bool:
        with self.processed_lock:
            if instance_id in self.load_already_processed_ids():
                return False
            
            claimed_ids = set()
            if self.processed_instances_file.exists():
                with open(self.processed_instances_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            claimed_ids.add(json.loads(line)['instance_id'])
                        except (json.JSONDecodeError, KeyError):
                            continue
            
            if instance_id in claimed_ids:
                return False
            
            with open(self.processed_instances_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'instance_id': instance_id, 
                    'process_id': self.process_id, 
                    'timestamp': datetime.now().isoformat()
                }) + '\n')
            return True

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

    def extract_answer_by_task(self, response: str, task_name: str) -> Optional[str]:
        if not response or not isinstance(response, str): return None
        if task_name in ['crispr_delivery', 'hle'] or task_name.startswith('lab_bench'):
            match = re.search(r'\[ANSWER\]([A-F])\[/ANSWER\]', response, re.IGNORECASE) or re.search(r'(?:answer|choice):\s*([A-F])\b', response, re.IGNORECASE)
            if match: return match.group(1).upper()
            lines = response.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if len(line) == 1 and line.upper() in 'ABCDEF': return line.upper()
        elif task_name.startswith('gwas_causal_gene') or task_name == 'screen_gene_retrieval':
            match = re.search(r'\b([A-Z][A-Z0-9\-]{1,10})\b', response)
            if match: return match.group(1)
        elif task_name == 'gwas_variant_prioritization':
            match = re.search(r'\b(rs\d+|chr\d+:\d+[ATCG>]+)\b', response)
            if match: return match.group(1)
        elif task_name in ['rare_disease_diagnosis', 'patient_gene_detection']:
            key = "OMIM_ID" if task_name == 'rare_disease_diagnosis' else "causal_gene"
            match = re.search(r'\{[^}]*"' + key + r'"[^}]*\}', response)
            if match: return match.group(0)
        return None

    async def process_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        instance_id = instance['instance_id']
        result = instance.copy()
        # Initialize result
        result.update({
            'predicted_answer': None, 
            'full_response': None, 
            'reward': 0.0, 
            'execution_time': 0.0, 
            'success': False, 
            'error': None, 
            'process_id': self.process_id, 
            'agent_type': 'B1'
        })
        
        start_time = time.time()
        try:
            print(f"  [Instance {instance_id}] Starting with A1 agent (Process {self.process_id})...")
            
            # --- 1. Create Fresh Agent ---
            current_agent = A1(
                path=self.DATA_LAKE_PATH,
                llm=self.R0_MODEL_NAME,
                use_tool_retriever=True,
                source="Custom",
                base_url=R0_BASE_URL,
                api_key="EMPTY",
            )
            
            # --- 2. Run Execution ---
            _, response_content = await asyncio.to_thread(current_agent.go, result['prompt'])
            
            # --- 3. CAPTURE FULL TRAJECTORY (Using new get_trajectory API) ---
            # Use the new get_trajectory() method added to A1 class
            full_trajectory_msgs = current_agent.get_trajectory()
            
            if full_trajectory_msgs:
                full_trajectory = self.reconstruct_trajectory(full_trajectory_msgs)
            else:
                # Fallback if something went wrong
                full_trajectory = f"[FINAL_ONLY_ERROR]:\n{response_content}"
            
            # --- 4. Extract Answer & Evaluate ---
            predicted_answer = self.extract_answer_by_task(response_content, result['task_name'])
            reward = 0.0
            if predicted_answer:
                reward = self.evaluator.evaluate(result['task_name'], result['task_instance_id'], predicted_answer)
            
            result.update({
                'predicted_answer': predicted_answer, 
                'full_response': full_trajectory,  # Now contains the full history
                'reward': reward, 
                'execution_time': time.time() - start_time, 
                'success': True
            })
            print(f"  [Instance {instance_id}] ✓ SUCCESS! Reward: {reward:.1f}, Time: {result['execution_time']:.2f}s")
            
        except Exception as e:
            result.update({'execution_time': time.time() - start_time, 'error': str(e), 'success': False})
            print(f"  [Instance {instance_id}] ✗ ERROR: {str(e)}")
            
        return result

    def save_result_and_update_stats(self, result: Dict[str, Any]):
        with self.results_lock:
            try:
                result_clean = {k: v.item() if isinstance(v, (np.integer, np.floating)) else v for k, v in result.items()}
                with open(self.results_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_clean, ensure_ascii=False) + '\n')
            except Exception as e: print(f"ERROR saving result: {e}")
        with self.stats_lock:
            stats = {}
            if self.statistics_file.exists():
                try: stats = json.load(open(self.statistics_file, 'r'))
                except json.JSONDecodeError: pass
            stats.setdefault('processed', 0)
            stats.setdefault('correct', 0)
            stats.setdefault('total_execution_time', 0.0)
            stats.setdefault('by_task', {})
            stats['processed'] += 1
            stats['total_execution_time'] += result.get('execution_time', 0.0)
            if result.get('success') and result.get('reward', 0.0) > 0: stats['correct'] += 1
            task_name = result['task_name']
            task_stats = stats['by_task'].setdefault(task_name, {'total': 0, 'correct': 0, 'total_time': 0.0})
            task_stats['total'] += 1
            task_stats['total_time'] += result.get('execution_time', 0.0)
            if result.get('success') and result.get('reward', 0.0) > 0: task_stats['correct'] += 1
            with open(self.statistics_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

async def run_evaluation_for_process(process_id: int, instance_queue: multiprocessing.Queue):
    evaluator = BiomniR0EvaluatorB1(process_id)
    tasks = []
    semaphore = asyncio.Semaphore(evaluator.CONCURRENCY_LIMIT)
    skipped_count = 0

    async def process_with_semaphore(instance):
        async with semaphore:
            result = await evaluator.process_instance(instance)
            evaluator.save_result_and_update_stats(result)

    while True:
        try:
            instance = instance_queue.get_nowait()
            instance_id = instance['instance_id']
            
            # Atomically check and claim instance
            if evaluator.try_claim_instance(instance_id):
                task = asyncio.create_task(process_with_semaphore(instance))
                tasks.append(task)
            else:
                skipped_count += 1
                print(f"  [Process {process_id}] Instance {instance_id} already claimed/processed, skipping")
        except queue.Empty:
            print(f"Process {process_id} found queue empty. Skipped {skipped_count} processed instances.")
            break
    
    if tasks:
        await asyncio.gather(*tasks)

def run_worker_process(process_id: int, instance_queue):
    print(f"✓ Started process {process_id} (PID: {os.getpid()})")
    asyncio.run(run_evaluation_for_process(process_id, instance_queue))
    print(f"✓ Process {process_id} completed its work.")

def main():
    print(f"\n{'#'*100}")
    print(f"STARTING MULTI-PROCESS EVALUATION WITH B1 AGENT")
    print(f"Processes: {NUM_PROCESSES}")
    print(f"Async concurrency per process: {ASYNC_CONCURRENCY_PER_PROCESS}")
    print(f"{'#'*100}\n")
    
    setup_evaluator = BiomniR0EvaluatorB1(process_id=-1)
    
    # Check for error instance IDs file
    error_instance_ids = setup_evaluator.load_error_instance_ids()
    
    # Clean up stale claims (instances in processed_instances but not in results)
    # This allows failed instances to be retried
    print("Cleaning up stale claims from previous runs...")
    if setup_evaluator.processed_instances_file.exists():
        with setup_evaluator.processed_lock:
            completed_ids = setup_evaluator.load_already_processed_ids()
            
            # Read all claims
            all_claims = []
            with open(setup_evaluator.processed_instances_file, 'r') as f:
                for line in f:
                    try:
                        claim = json.loads(line)
                        # Only keep claims that have results
                        if claim['instance_id'] in completed_ids:
                            all_claims.append(claim)
                    except: pass
            
            # Rewrite file with only valid claims
            with open(setup_evaluator.processed_instances_file, 'w') as f:
                for claim in all_claims:
                    f.write(json.dumps(claim) + '\n')
            
            print(f"✓ Cleaned up processed_instances_b1.jsonl: kept {len(all_claims)} valid claims")
    
    all_instances_df = setup_evaluator.evaluator.df
    processed_ids = setup_evaluator.load_already_processed_ids()
    
    if error_instance_ids is not None:
        print(f"\n{'='*100}")
        print(f"🎯 TARGETING SPECIFIC ERROR INSTANCES")
        print(f"Filtering to {len(error_instance_ids)} specific instance IDs from error file")
        all_instances_df = all_instances_df[all_instances_df['instance_id'].isin(error_instance_ids)]
        print(f"Found {len(all_instances_df)} matching instances in dataset")
    
    remaining_instances = [row.to_dict() for _, row in all_instances_df.iterrows() if row['instance_id'] not in processed_ids]
    
    print(f"\nTotal instances in dataset: {len(all_instances_df)}")
    print(f"Found {len(processed_ids)} already processed instances.")
    print(f"Remaining instances to process: {len(remaining_instances)}\n")

    if not remaining_instances:
        print("All instances have been processed. Exiting.")
        return

    manager = multiprocessing.Manager()
    instance_queue = manager.Queue()
    for instance in remaining_instances:
        instance_queue.put(instance)

    processes = []
    for i in range(NUM_PROCESSES):
        p = multiprocessing.Process(target=run_worker_process, args=(i + 1, instance_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"\n{'#'*100}\nALL PROCESSES COMPLETED\n{'#'*100}\n")

if __name__ == "__main__":
    main()

