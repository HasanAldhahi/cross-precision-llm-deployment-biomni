import json
import sys

def load_instances(file_path):
    """Load jsonl file and return list of parsed objects."""
    print(f"DEBUG: Loading {file_path}...", file=sys.stderr)
    instances = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    obj = json.loads(line.strip())
                    instances.append(obj)
                except Exception as e:
                    print(f"DEBUG: Failed to parse line {line_num} in {file_path}: {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"DEBUG: Could not open {file_path}: {e}", file=sys.stderr)
    print(f"DEBUG: Loaded {len(instances)} instances from {file_path}", file=sys.stderr)
    return instances

def instance_dict_by_id(instances):
    """Return dict: instance_id -> instance (dict)."""
    d = {obj.get("instance_id"): obj for obj in instances if "instance_id" in obj}
    print(f"DEBUG: Built instance_id dict, unique IDs found: {len(d)}", file=sys.stderr)
    return d

def make_proxy_format(inst_obj):
    """Given base instance, construct output as described."""
    print(f"DEBUG: Making proxy format for instance_id={inst_obj.get('instance_id')}", file=sys.stderr)
    output = {}
    output['instance_id'] = inst_obj.get('instance_id')
    output['chat_eval'] = "3"
    orig_total_in_out = inst_obj.get('total_input_output_tokens', None)
    orig_total_out = inst_obj.get('total_output_tokens', None)

    try:
        orig_total_in_out = int(orig_total_in_out)
    except (TypeError, ValueError):
        print(f"DEBUG: instance_id={inst_obj.get('instance_id')} has invalid total_input_output_tokens ({orig_total_in_out}), setting to 0", file=sys.stderr)
        orig_total_in_out = 0
    try:
        orig_total_out = int(orig_total_out)
    except (TypeError, ValueError):
        print(f"DEBUG: instance_id={inst_obj.get('instance_id')} has invalid total_output_tokens ({orig_total_out}), setting to 0", file=sys.stderr)
        orig_total_out = 0

    output['total_output_tokens'] = 0
    output['total_input_output_tokens'] = orig_total_in_out - orig_total_out

    output['num_steps'] = 0
    output['task_instance_id'] = inst_obj.get('task_instance_id')
    output['prompt'] = inst_obj.get('prompt')
    output['task_name'] = inst_obj.get('task_name')
    output['split'] = inst_obj.get('split')
    output['answer'] = inst_obj.get('answer')
    output['predicted_answer'] = inst_obj.get('predicted_answer', None)
    output['full_response'] = None
    output['reward'] = inst_obj.get('reward', 0.0)
    output['execution_time'] = 0.0
    output['success'] = False
    output['error'] = "Connection error."
    return output

def main():
    base_file = "ORIGNIANL_RESULTS.jsonl"
    annotated_file = "Final_r0_results_annotated.jsonl"

    print("DEBUG: Starting main()", file=sys.stderr)

    annotated_instances = load_instances(annotated_file)
    annotated_dict = instance_dict_by_id(annotated_instances)
    base_instances = load_instances(base_file)
    base_dict = instance_dict_by_id(base_instances)

    print(f"DEBUG: base_file IDs: {len(base_dict.keys())}", file=sys.stderr)
    print(f"DEBUG: annotated_file IDs: {len(annotated_dict.keys())}", file=sys.stderr)

    missing_ids = sorted(set(base_dict.keys()) - set(annotated_dict.keys()))
    print(f"DEBUG: Found {len(missing_ids)} missing IDs in annotated_file", file=sys.stderr)
    if missing_ids:
        print(f"DEBUG: Missing IDs (first up to 10): {missing_ids[:10]}", file=sys.stderr)

    new_instances = []
    for count, iid in enumerate(missing_ids, 1):
        base_inst = base_dict[iid]
        out_obj = make_proxy_format(base_inst)
        new_instances.append(out_obj)
        print(f"DEBUG: {count}/{len(missing_ids)} Created proxy instance for {iid}", file=sys.stderr)
        print(json.dumps(out_obj, ensure_ascii=False))

    if new_instances:
        try:
            with open(annotated_file, "a", encoding="utf-8") as f:
                for idx, obj in enumerate(new_instances, 1):
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(f"DEBUG: Appended {len(new_instances)} new instances to {annotated_file}", file=sys.stderr)
        except Exception as e:
            print(f"DEBUG: ERROR writing to {annotated_file}: {e}", file=sys.stderr)
    else:
        print("DEBUG: No new missing instances to append.", file=sys.stderr)

if __name__ == "__main__":
    main()
