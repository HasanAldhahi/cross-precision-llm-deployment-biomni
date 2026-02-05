#!/usr/bin/env python3
"""
Generate comprehensive statistics from filtered unique results JSONL file.
Includes error tracking and visualizations.
"""

# Standard library imports
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def extract_error_code(error_str: str) -> str:
    """Extract error code from error string."""
    if not error_str:
        return None
    
    # Try to extract error code patterns
    # Pattern 1: "Error code: 400"
    match = re.search(r'Error code:\s*(\d+)', error_str)
    if match:
        return match.group(1)
    
    # Pattern 2: "400" at the start
    match = re.search(r'^(\d{3})', error_str)
    if match:
        return match.group(1)
    
    # Pattern 3: Look for common error types
    if 'timeout' in error_str.lower():
        return 'TIMEOUT'
    if 'badrequest' in error_str.lower() or 'bad request' in error_str.lower():
        return 'BAD_REQUEST'
    if 'connection' in error_str.lower():
        return 'CONNECTION_ERROR'
    if 'proxy' in error_str.lower():
        return 'PROXY_ERROR'
    
    return 'OTHER_ERROR'


def categorize_error(error_str: str) -> Dict[str, str]:
    """Categorize error into type and code."""
    if not error_str:
        return {'type': None, 'code': None, 'full_error': None}
    
    error_code = extract_error_code(error_str)
    
    # Determine error type
    error_lower = error_str.lower()
    if 'max_tokens' in error_lower or 'context length' in error_lower:
        error_type = 'CONTEXT_LENGTH'
    elif 'timeout' in error_lower:
        error_type = 'TIMEOUT'
    elif '400' in error_str or 'badrequest' in error_lower:
        error_type = 'BAD_REQUEST'
    elif 'connection' in error_lower:
        error_type = 'CONNECTION'
    elif 'proxy' in error_lower:
        error_type = 'PROXY'
    else:
        error_type = 'OTHER'
    
    return {
        'type': error_type,
        'code': error_code,
        'full_error': error_str[:200]  # Truncate long errors
    }


def generate_statistics(input_file: str) -> Dict[str, Any]:
    """Generate comprehensive statistics from JSONL file."""
    
    stats = {
        'processed': 0,
        'correct': 0,
        'incorrect': 0,
        'failed': 0,
        'total_execution_time': 0.0,
        'by_task': {},
        'errors': {
            'total_errors': 0,
            'by_type': defaultdict(int),
            'by_code': defaultdict(int),
            'by_task': defaultdict(int),
            'error_details': []
        }
    }
    
    print(f"Reading from: {input_file}")
    
    # Read and process all entries
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
                
                # Basic statistics
                stats['processed'] += 1
                
                execution_time = entry.get('execution_time', 0.0)
                stats['total_execution_time'] += execution_time
                
                success = entry.get('success', False)
                reward = entry.get('reward', 0.0)
                task_name = entry.get('task_name', 'unknown')
                
                # Initialize task statistics if needed
                if task_name not in stats['by_task']:
                    stats['by_task'][task_name] = {
                        'total': 0,
                        'correct': 0,
                        'incorrect': 0,
                        'failed': 0,
                        'total_time': 0.0,
                        'avg_time': 0.0,
                        'errors': []
                    }
                
                task_stats = stats['by_task'][task_name]
                task_stats['total'] += 1
                task_stats['total_time'] += execution_time
                
                # Check for errors
                error = entry.get('error')
                if error:
                    stats['errors']['total_errors'] += 1
                    task_stats['failed'] += 1
                    stats['failed'] += 1
                    
                    error_info = categorize_error(error)
                    error_type = error_info['type']
                    error_code = error_info['code']
                    
                    if error_type:
                        stats['errors']['by_type'][error_type] += 1
                    if error_code:
                        stats['errors']['by_code'][error_code] += 1
                    
                    stats['errors']['by_task'][task_name] += 1
                    task_stats['errors'].append({
                        'instance_id': entry.get('instance_id'),
                        'error_type': error_type,
                        'error_code': error_code,
                        'error_message': error_info['full_error']
                    })
                    
                elif success and reward > 0:
                    task_stats['correct'] += 1
                    stats['correct'] += 1
                elif success and reward == 0:
                    task_stats['incorrect'] += 1
                    stats['incorrect'] += 1
                else:
                    task_stats['failed'] += 1
                    stats['failed'] += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    # Calculate average times
    for task_name, task_stats in stats['by_task'].items():
        if task_stats['total'] > 0:
            task_stats['avg_time'] = task_stats['total_time'] / task_stats['total']
    
    # Convert defaultdicts to regular dicts for JSON serialization
    stats['errors']['by_type'] = dict(stats['errors']['by_type'])
    stats['errors']['by_code'] = dict(stats['errors']['by_code'])
    stats['errors']['by_task'] = dict(stats['errors']['by_task'])
    
    return stats


def create_visualizations(stats: Dict[str, Any], output_dir: Path):
    """Create visualizations from statistics."""
    
    output_dir.mkdir(exist_ok=True)
    
    # Prepare data for visualization
    tasks = list(stats['by_task'].keys())
    task_data = stats['by_task']
    
    # 1. Success Rate by Task
    fig, ax = plt.subplots(figsize=(14, 8))
    correct_counts = [task_data[task]['correct'] for task in tasks]
    total_counts = [task_data[task]['total'] for task in tasks]
    success_rates = [c / t * 100 if t > 0 else 0 for c, t in zip(correct_counts, total_counts)]
    
    colors = ['#2ecc71' if rate >= 50 else '#e74c3c' if rate < 20 else '#f39c12' 
              for rate in success_rates]
    
    bars = ax.bar(tasks, success_rates, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate by Task', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, rate, total) in enumerate(zip(bars, success_rates, total_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%\n({task_data[tasks[i]]["correct"]}/{total})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_by_task.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Errors by Task
    fig, ax = plt.subplots(figsize=(14, 8))
    error_counts = [stats['errors']['by_task'].get(task, 0) for task in tasks]
    
    bars = ax.bar(tasks, error_counts, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Errors', fontsize=12, fontweight='bold')
    ax.set_title('Errors by Task', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, count in zip(bars, error_counts):
        if count > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'errors_by_task.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Execution Time by Task
    fig, ax = plt.subplots(figsize=(14, 8))
    avg_times = [task_data[task]['avg_time'] for task in tasks]
    
    bars = ax.bar(tasks, avg_times, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Average Execution Time by Task', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, time_val in zip(bars, avg_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{time_val:.0f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'execution_time_by_task.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Error Types Distribution
    if stats['errors']['by_type']:
        fig, ax = plt.subplots(figsize=(10, 8))
        error_types = list(stats['errors']['by_type'].keys())
        error_counts = [stats['errors']['by_type'][et] for et in error_types]
        
        bars = ax.bar(error_types, error_counts, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Error Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Error Distribution by Type', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, error_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_types_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Error Codes Distribution
    if stats['errors']['by_code']:
        fig, ax = plt.subplots(figsize=(10, 8))
        error_codes = list(stats['errors']['by_code'].keys())
        error_counts = [stats['errors']['by_code'][ec] for ec in error_codes]
        
        bars = ax.bar(error_codes, error_counts, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Error Code', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Error Distribution by Code', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, error_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_codes_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Comprehensive Task Overview
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 6a. Success vs Failed by Task
    ax = axes[0, 0]
    task_indices = np.arange(len(tasks))
    width = 0.35
    correct_bars = [task_data[task]['correct'] for task in tasks]
    failed_bars = [task_data[task]['failed'] for task in tasks]
    
    ax.bar(task_indices - width/2, correct_bars, width, label='Success', color='#2ecc71', alpha=0.7)
    ax.bar(task_indices + width/2, failed_bars, width, label='Failed', color='#e74c3c', alpha=0.7)
    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Success vs Failed by Task', fontweight='bold')
    ax.set_xticks(task_indices)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 6b. Correct vs Incorrect by Task
    ax = axes[0, 1]
    correct_bars = [task_data[task]['correct'] for task in tasks]
    incorrect_bars = [task_data[task]['incorrect'] for task in tasks]
    
    ax.bar(task_indices - width/2, correct_bars, width, label='Correct', color='#2ecc71', alpha=0.7)
    ax.bar(task_indices + width/2, incorrect_bars, width, label='Incorrect', color='#f39c12', alpha=0.7)
    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Correct vs Incorrect by Task', fontweight='bold')
    ax.set_xticks(task_indices)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 6c. Total Execution Time by Task
    ax = axes[1, 0]
    total_times = [task_data[task]['total_time'] for task in tasks]
    ax.bar(tasks, total_times, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel('Total Time (seconds)', fontweight='bold')
    ax.set_title('Total Execution Time by Task', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 6d. Error Rate by Task
    ax = axes[1, 1]
    error_rates = [(stats['errors']['by_task'].get(task, 0) / task_data[task]['total'] * 100) 
                    if task_data[task]['total'] > 0 else 0 for task in tasks]
    ax.bar(tasks, error_rates, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel('Error Rate (%)', fontweight='bold')
    ax.set_title('Error Rate by Task', fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_task_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved visualizations to {output_dir}/")


def print_summary(stats: Dict[str, Any]):
    """Print a comprehensive summary of statistics."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICS SUMMARY")
    print("="*80)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total Processed:        {stats['processed']}")
    print(f"  Correct:                 {stats['correct']} ({stats['correct']/stats['processed']*100:.1f}%)")
    print(f"  Incorrect:               {stats['incorrect']} ({stats['incorrect']/stats['processed']*100:.1f}%)")
    print(f"  Failed:                  {stats['failed']} ({stats['failed']/stats['processed']*100:.1f}%)")
    print(f"  Total Execution Time:    {stats['total_execution_time']:.2f} seconds ({stats['total_execution_time']/3600:.2f} hours)")
    print(f"  Average Time per Task:   {stats['total_execution_time']/stats['processed']:.2f} seconds")
    
    print(f"\n❌ ERROR STATISTICS:")
    print(f"  Total Errors:            {stats['errors']['total_errors']}")
    print(f"  Error Rate:             {stats['errors']['total_errors']/stats['processed']*100:.1f}%")
    if stats['errors']['by_type']:
        print(f"  Error Types:")
        for err_type, count in sorted(stats['errors']['by_type'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {err_type}: {count}")
    if stats['errors']['by_code']:
        print(f"  Error Codes:")
        for err_code, count in sorted(stats['errors']['by_code'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {err_code}: {count}")
    
    print(f"\n📋 BY TASK:")
    print(f"{'Task':<40} {'Total':<8} {'Correct':<10} {'Incorrect':<12} {'Failed':<10} {'Avg Time':<12} {'Errors':<8}")
    print("-"*110)
    for task_name in sorted(stats['by_task'].keys()):
        task_stats = stats['by_task'][task_name]
        error_count = stats['errors']['by_task'].get(task_name, 0)
        success_rate = (task_stats['correct'] / task_stats['total'] * 100) if task_stats['total'] > 0 else 0
        print(f"{task_name:<40} {task_stats['total']:<8} {task_stats['correct']:<10} "
              f"{task_stats['incorrect']:<12} {task_stats['failed']:<10} "
              f"{task_stats['avg_time']:.1f}s{'':<8} {error_count:<8}")
    
    print("="*80)


def main():
    # File paths
    input_file = Path("eval_output/r0_eval_results_b1.jsonl")
    output_stats_file = Path("eval_output/final_r0_statistics_b1.json")
    output_viz_dir = Path("eval_output/visualizations")
    
    print("="*80)
    print("GENERATING COMPREHENSIVE STATISTICS")
    print("="*80)
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Generate statistics
    stats = generate_statistics(input_file)
    
    # Save statistics to JSON
    with open(output_stats_file, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Statistics saved to: {output_stats_file}")
    
    # Print summary
    print_summary(stats)
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(stats, output_viz_dir)
    
    print(f"\n✓ Complete! All results saved to eval_output/")
    print(f"  - Statistics: {output_stats_file}")
    print(f"  - Visualizations: {output_viz_dir}/")


if __name__ == "__main__":
    main()


