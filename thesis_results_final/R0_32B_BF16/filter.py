import json

def filter_results(input_file, output_file):
    """
    Filters the results from a JSONL file, keeping only those
    with no error (i.e., error is None or not present)
    and where success is True.
    Writes filtered results to a new JSONL file.
    """
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Check for 'error' key: must be absent or None/empty
            no_error = ("error" not in obj) or (obj["error"] is None) or (obj["error"] == "")
            # Check for 'success' key: must be True
            success_true = obj.get("success", False) is True

            if no_error and success_true:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def generate_statistics(filtered_file, stats_output_file):
    """
    Generates summary statistics with keys/structure analogous to r0_eval_results_b1.filtered.jsonl.
    Output includes: total, n_correct, accuracy, and per-task accuracy and counts.
    """
    from collections import defaultdict

    # Global counters
    total = 0
    n_correct = 0
    task_total = defaultdict(int)
    task_correct = defaultdict(int)
    split_total = defaultdict(int)
    split_correct = defaultdict(int)
    task_name_total = defaultdict(int)
    task_name_correct = defaultdict(int)

    with open(filtered_file, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            # Use the same correctness logic as the original stats:
            # correct if the run succeeded AND reward > 0
            correct = bool(obj.get("success")) and (obj.get("reward", 0.0) or 0.0) > 0
            if correct:
                n_correct += 1

            # Support both task and task_name
            task = obj.get("task", obj.get("task_name", "unknown_task"))
            task_total[task] += 1
            if correct:
                task_correct[task] += 1

            # Also gather stats by task_name if it exists
            task_name = obj.get("task_name", None)
            if task_name:
                task_name_total[task_name] += 1
                if correct:
                    task_name_correct[task_name] += 1

            # Also gather split (`val`, `test`, etc)
            split = obj.get("split", "unknown")
            split_total[split] += 1
            if correct:
                split_correct[split] += 1

    # Compose output with keys similar to the filtered results jsonl
    stats = {
        "total": total,
        "n_correct": n_correct,
        "accuracy": n_correct / total if total else None,
        "accuracy_by_task": {},
        "n_by_task": {},
        "n_correct_by_task": {},
        "accuracy_by_task_name": {},
        "n_by_task_name": {},
        "n_correct_by_task_name": {},
        "accuracy_by_split": {},
        "n_by_split": {},
        "n_correct_by_split": {},
    }
    for task in sorted(task_total.keys()):
        stats["n_by_task"][task] = task_total[task]
        stats["n_correct_by_task"][task] = task_correct[task]
        stats["accuracy_by_task"][task] = (task_correct[task] / task_total[task]) if task_total[task] else None

    for tname in sorted(task_name_total.keys()):
        stats["n_by_task_name"][tname] = task_name_total[tname]
        stats["n_correct_by_task_name"][tname] = task_name_correct[tname]
        stats["accuracy_by_task_name"][tname] = (task_name_correct[tname] / task_name_total[tname]) if task_name_total[tname] else None

    for split in sorted(split_total.keys()):
        stats["n_by_split"][split] = split_total[split]
        stats["n_correct_by_split"][split] = split_correct[split]
        stats["accuracy_by_split"][split] = (split_correct[split] / split_total[split]) if split_total[split] else None

    with open(stats_output_file, "w", encoding="utf-8") as fout:
        json.dump(stats, fout, indent=2, ensure_ascii=False)


# Example statistics generation usage:
filter_results("r0_eval_results_b1.jsonl", "r0_eval_results_b1.filtered.jsonl")
generate_statistics("r0_eval_results_b1.filtered.jsonl", "r0_statistics_b1.generated.json")

# Example usage:

