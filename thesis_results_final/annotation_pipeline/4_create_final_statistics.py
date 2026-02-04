#!/usr/bin/env python3
"""
Script to create final statistics from Final_r0_results_annotated.jsonl
which contains chat_eval property with values:
- "1" = correct
- "0" = wrong
- "2" = max_token_error
- "3" = proxy_error

Output format matches Final_statistics.json
"""

import json
import os
from collections import defaultdict

def create_final_statistics(
    annotated_file="Final_r0_results_annotated.jsonl",
    output_file="Final_statistics.json"
):
    """
    Create final statistics from annotated results file.
    """
    print("=" * 80)
    print("Creating Final Statistics from Annotated File")
    print("=" * 80)
    
    if not os.path.exists(annotated_file):
        print(f"Error: {annotated_file} not found!")
        return None
    
    # Initialize statistics tracking
    task_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "total_time": 0.0,
        "incorrect": {
            "max_token_error": 0,
            "proxy_error": 0,
            "wrong_answers": 0
        }
    })
    
    total_processed = 0
    total_correct = 0
    total_execution_time = 0.0
    
    # Read annotated file
    with open(annotated_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                instance = json.loads(line)
                instance_id = instance.get("instance_id")
                task_name = instance.get("task_name", "unknown")
                execution_time = instance.get("execution_time", 0.0)
                chat_eval = instance.get("chat_eval")
                
                # Update task statistics
                task_stats[task_name]["total"] += 1
                task_stats[task_name]["total_time"] += execution_time
                
                # Classify based on chat_eval
                if chat_eval == "1":  # Correct
                    task_stats[task_name]["correct"] += 1
                    total_correct += 1
                elif chat_eval == "0":  # Wrong answer
                    task_stats[task_name]["incorrect"]["wrong_answers"] += 1
                elif chat_eval == "2":  # Max token error
                    task_stats[task_name]["incorrect"]["max_token_error"] += 1
                elif chat_eval == "3":  # Proxy error
                    task_stats[task_name]["incorrect"]["proxy_error"] += 1
                
                total_processed += 1
                total_execution_time += execution_time
                
            except json.JSONDecodeError:
                continue
    
    # Create final statistics structure
    final_stats = {
        "processed": total_processed,
        "correct": total_correct,
        "total_execution_time": total_execution_time,
        "by_task": {}
    }
    
    # Add per-task statistics in sorted order
    for task_name in sorted(task_stats.keys()):
        stats = task_stats[task_name]
        final_stats["by_task"][task_name] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "total_time": stats["total_time"],
            "incorrect": stats["incorrect"]
        }
    
    # Save final statistics
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n✓ Created: {output_file}")
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Processed: {total_processed}")
    print(f"  Correct: {total_correct}")
    print(f"  Accuracy: {total_correct/total_processed*100:.2f}%")
    print(f"  Total execution time: {total_execution_time:.2f} seconds")
    
    # Calculate error totals
    total_max_token = sum(s["incorrect"]["max_token_error"] for s in task_stats.values())
    total_proxy = sum(s["incorrect"]["proxy_error"] for s in task_stats.values())
    total_wrong = sum(s["incorrect"]["wrong_answers"] for s in task_stats.values())
    
    print(f"\n❌ ERROR BREAKDOWN:")
    print(f"  Wrong answers: {total_wrong}")
    print(f"  Max token errors: {total_max_token}")
    print(f"  Proxy errors: {total_proxy}")
    print(f"  Total incorrect: {total_wrong + total_max_token + total_proxy}")
    
    print(f"\n📋 BY TASK:")
    for task_name, stats in sorted(final_stats["by_task"].items()):
        accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        inc = stats["incorrect"]
        total_inc = inc["wrong_answers"] + inc["max_token_error"] + inc["proxy_error"]
        print(f"  {task_name}:")
        print(f"    ✓ Correct: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
        print(f"    ✗ Wrong: {inc['wrong_answers']}, Max Token: {inc['max_token_error']}, Proxy: {inc['proxy_error']}")
    
    print("\n" + "=" * 80)
    
    return final_stats

if __name__ == "__main__":
    create_final_statistics()

