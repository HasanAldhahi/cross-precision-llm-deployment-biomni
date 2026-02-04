import json
import os

# File paths
orig_file = "Final_lora_256_results_annotated.jsonl"
proxy_corrected_file = "Final_lora_256_results_annotated_proxy_corrected.jsonl"

# Load corrected instances, make a dictionary: id -> instance
proxy_corrected_instances = {}
with open(proxy_corrected_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            inst = json.loads(line)
            if "instance_id" in inst:
                proxy_corrected_instances[inst["instance_id"]] = inst
        except Exception:
            continue

# The set of IDs to be replaced
replace_ids = set(proxy_corrected_instances.keys())

# Read original file and keep only those not in 'replace_ids'
remaining_instances = []
with open(orig_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            inst = json.loads(line)
            if "instance_id" in inst:
                instance_id = inst["instance_id"]
                if instance_id not in replace_ids:
                    remaining_instances.append(inst)
        except Exception:
            continue

# Add the corrected instances (replace on top)
combined_instances = list(proxy_corrected_instances.values()) + remaining_instances

# Backup original file
os.rename(orig_file, orig_file + ".bak")

# Write new result file
with open(orig_file, "w", encoding="utf-8") as f:
    for inst in combined_instances:
        f.write(json.dumps(inst, ensure_ascii=False) + "\n")

print(f"Done: replaced {len(replace_ids)} instances in {orig_file} using {proxy_corrected_file}. Backup at {orig_file}.bak")
