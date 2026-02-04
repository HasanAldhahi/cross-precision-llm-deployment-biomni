#!/usr/bin/env python3
"""
Script to create r0_statistics_checked.json from chat evaluation results.
Format matches initial_statistics.json structure.
"""

import json
import os
from collections import defaultdict

def create_chat_statistics(
    chat_eval_file="combined_instances_with_chat_eval.jsonl",
    output_file="r0_statistics_checked.json"
):
    """
    Create statistics from chat evaluation results.
    """
    
    # Load chat evaluation results
    chat_instances = []
    if not os.path.exists(chat_eval_file):
        print(f"Error: Chat evaluation file '{chat_eval_file}' not found.")
        return
    
    with open(chat_eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                instance = json.loads(line)
                chat_instances.append(instance)
            except json.JSONDecodeError:
                continue
    
    if not chat_instances:
        print(f"Warning: No instances found in {chat_eval_file}")
        return
    
    # Calculate statistics by task
    task_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "total_time": 0.0
    })
    
    total_processed = 0
    total_correct = 0
    total_execution_time = 0.0
    
    for instance in chat_instances:
        task_name = instance.get("task_name", "unknown")
        chat_eval = instance.get("chat_eval")
        execution_time = instance.get("execution_time", 0.0)
        
        # Count this instance
        task_stats[task_name]["total"] += 1
        task_stats[task_name]["total_time"] += execution_time
        total_processed += 1
        total_execution_time += execution_time
        
        # Check if correct (chat_eval == "1")
        if chat_eval == "1":
            task_stats[task_name]["correct"] += 1
            total_correct += 1
    
    # Create output structure
    output = {
        "processed": total_processed,
        "correct": total_correct,
        "total_execution_time": total_execution_time,
        "by_task": {}
    }
    
    # Format by_task
    for task_name, stats in sorted(task_stats.items()):
        output["by_task"][task_name] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "total_time": stats["total_time"]
        }
    
    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Created {output_file}")
    print(f"  Processed: {total_processed} instances")
    print(f"  Correct: {total_correct} instances")
    print(f"  Accuracy: {total_correct/total_processed*100:.2f}%")
    print(f"  Total execution time: {total_execution_time:.2f}s")
    print(f"\nPer-task breakdown:")
    for task_name, stats in sorted(task_stats.items()):
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {task_name}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")

if __name__ == "__main__":
    create_chat_statistics()

