import json
import re
import os
from collections import defaultdict

input_file = "final.jsonl"
output_file_solution = "extracted_solutions.jsonl"
output_file_solution_with_tokens = "extracted_solutions_with_last_200_tokens.jsonl"
output_file_solution_without_tokens = "extracted_solutions_without_last_200_tokens.jsonl"
output_file_last_steps = "extracted_last_steps.jsonl"
output_file_proxy_error = "extracted_proxy_error.jsonl"
output_file_max_tokens_error = "extracted_max_tokens_error.jsonl"

seen_instance_ids = set()
solutions = []
last_steps = []
proxy_errors = []
max_tokens_errors = []
all_instance_ids = set()
error_stats = {"missing_instance_id": 0, "missing_full_response": 0, "duplicate_instance_id": 0, "malformed_json": 0}

def extract_solution_section(full_response):
    """
    Extract the solution section or tag, supporting:
    1. Markdown headings ("# Solution" etc.)
    2. Literal tags like <solution> ... </solution> or <Solution> ... </Solution>
    3. Square-bracket tags [Solution]
    Returns the string if found, else None.
    """

    # Try XML/HTML style tags first (with closing tag) - find the LAST one (usually the actual answer)
    xml_matches = list(re.finditer(
        r"<solution\b[^>]*>(.*?)</solution>", full_response, re.IGNORECASE | re.DOTALL))
    if xml_matches:
        # Return the last match (usually the actual answer, not the explanation)
        content = xml_matches[-1].group(1).strip()
        # If content contains another <solution> tag, extract only the part after the last one
        if "<solution" in content.lower():
            inner_solution = re.search(r"<solution[^>]*>(.*?)(?:</solution>|$)", content, re.IGNORECASE | re.DOTALL)
            if inner_solution:
                return inner_solution.group(1).strip()
        return content
    
    # Try XML/HTML style tags without closing tag - find the LAST one
    # But only if there are no closed tags (prioritize closed tags)
    xml_matches_unclosed = list(re.finditer(
        r"<solution\b[^>]*>(.*)", full_response, re.IGNORECASE | re.DOTALL))
    if xml_matches_unclosed:
        # If we have multiple unclosed tags, prefer the shortest one (usually the actual answer)
        # or the last one if they're similar length
        if len(xml_matches_unclosed) > 1:
            # Find the shortest content (usually the actual answer like "PRKCD")
            shortest_match = min(xml_matches_unclosed, key=lambda m: len(m.group(1).strip()))
            content = shortest_match.group(1).strip()
            # But if shortest is still very long (>100 chars), use the last one instead
            if len(content) > 100:
                content = xml_matches_unclosed[-1].group(1).strip()
            return content
        else:
            return xml_matches_unclosed[-1].group(1).strip()

    # Try Markdown headings (one or more # followed by "solution")
    mkdown_match = re.search(
        r"(?:^|[\n\r])\s*#+\s*solution\b[^\n\r]*([\s\S]+?)(?:^[#]+ |\n---|\n\*\*\*|\n###|\n##|\n#|\n\[\w+\]|$)", 
        full_response, re.IGNORECASE | re.MULTILINE)
    if mkdown_match:
        return mkdown_match.group(1).strip()

    # Try square brackets [Solution]
    bracket_match = re.search(
        r"(?:^|[\n\r])\[\s*solution\s*\][^\n\r]*([\s\S]+?)(?:^[#]+ |\n---|\n\*\*\*|\n###|\n##|\n#|\n\[\w+\]|$)", 
        full_response, re.IGNORECASE | re.MULTILINE)
    if bracket_match:
        return bracket_match.group(1).strip()

    # Fallback: old regex attempts (MarkDown, ...), just in case
    patterns = [
        r"(?:^|[\n\r])#?\s*Solution\b.*?$([\s\S]+)",
        r"(?:^|[\n\r])##?\s*Solution\b.*?$([\s\S]+)",
        r"(?:^|[\n\r])\[?Solution\]?.*?$([\s\S]+)",
    ]
    for pat in patterns:
        match = re.search(pat, full_response, re.IGNORECASE | re.MULTILINE)
        if match:
            text = match.group(1)
            cutoff = re.search(r"^#+\s|\n---|\n\*\*\*|\n###|\n##|\n#|\n\[\w+\]", text, flags=re.MULTILINE)
            if cutoff:
                text = text[:cutoff.start()]
            return text.strip()
    return None

def last_n_tokens(text, n_tokens=200):
    """Get the last n_tokens of text, tokenizing by whitespace (approximation)"""
    tokens = text.split()
    if len(tokens) <= n_tokens:
        return text.strip()
    else:
        return " ".join(tokens[-n_tokens:]).strip()

def count_answer_in_text(answer, text):
    """Count how many times the answer appears in the text (case-insensitive, whole word match)"""
    if not answer or len(str(answer).strip()) <= 1:
        return None  # Skip single letter answers
    
    answer_str = str(answer).strip()
    text_lower = text.lower()
    answer_lower = answer_str.lower()
    
    # Count whole word matches (word boundaries)
    pattern = r'\b' + re.escape(answer_lower) + r'\b'
    matches = re.findall(pattern, text_lower)
    return len(matches)

def first_n_tokens(text, n_tokens=200):
    """Get the first n_tokens of text, tokenizing by whitespace (approximation)"""
    tokens = text.split()
    if len(tokens) <= n_tokens:
        return text.strip()
    else:
        return " ".join(tokens[:n_tokens]).strip()

with open(input_file, "r", encoding="utf-8") as infile:
    for idx, line in enumerate(infile):
        raw_entry = line
        try:
            data = json.loads(line)
        except Exception:
            # Malformed JSON line - skip
            error_stats["malformed_json"] += 1
            continue

        instance_id = data.get("instance_id")
        all_instance_ids.add(instance_id)

        full_response = data.get("full_response")
        answer = data.get("answer")
        error = data.get("error")

        # Check for proxy error or max tokens error first
        error_str = str(error).lower() if error else ""
        if "proxy" in error_str:
            proxy_errors.append({
                "instance_id": instance_id,
                "answer": answer,
                "extracted_solution_or_last_step": None,
                "error": error
            })
            continue

        if "max token" in error_str or "max_token" in error_str or "maxtoken" in error_str:
            max_tokens_errors.append({
                "instance_id": instance_id,
                "answer": answer,
                "extracted_solution_or_last_step": None,
                "error": error
            })
            continue

        # If missing both fields
        if instance_id is None:
            error_stats["missing_instance_id"] += 1
            continue

        if instance_id in seen_instance_ids:
            error_stats["duplicate_instance_id"] += 1
            continue

        seen_instance_ids.add(instance_id)

        if full_response is None:
            error_stats["missing_full_response"] += 1
            continue

        snippet = extract_solution_section(full_response)
        if snippet:
            # If solution tag found (even if short), put it in solutions
            last_200 = last_n_tokens(full_response, n_tokens=200)
            # Count answer occurrences in last 200 tokens (only for non-single-letter answers)
            answer_count = count_answer_in_text(answer, last_200)
            
            solution_entry = {
                "instance_id": instance_id,
                "answer": answer,
                "extracted_solution_or_last_step": snippet,
                "error": None,
                "last_200_tokens": last_200
            }
            
            # Add answer count statistic if answer is not a single letter
            if answer_count is not None:
                solution_entry["answer_count_in_last_200_tokens"] = answer_count
            
            solutions.append(solution_entry)
        else:
            # No solution tag found at all - get last 100 tokens
            snippet = last_n_tokens(full_response, n_tokens=200)
            last_steps.append({
                "instance_id": instance_id,
                "answer": answer,
                "extracted_solution_or_last_step": snippet,
                "error": "no_solution_tag_found"
            })

# Write solutions to file (original with last_250_tokens)
with open(output_file_solution, "w", encoding="utf-8") as outfile:
    for entry in solutions:
        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Write solutions with last_250_tokens to separate file
with open(output_file_solution_with_tokens, "w", encoding="utf-8") as outfile:
    for entry in solutions:
        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Write last steps to file
with open(output_file_last_steps, "w", encoding="utf-8") as outfile:
    for entry in last_steps:
        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Write proxy errors to file
with open(output_file_proxy_error, "w", encoding="utf-8") as outfile:
    for entry in proxy_errors:
        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Write max tokens errors to file
with open(output_file_max_tokens_error, "w", encoding="utf-8") as outfile:
    for entry in max_tokens_errors:
        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Create instances folder - will be populated after adding features
instances_folder = "instances"
os.makedirs(instances_folder, exist_ok=True)

# Categorize instances based on answer matching
correct_instances = []
wrong_instances = []
instances_to_be_checked = []

# Helper function to check if extracted is a single character or single word (no tags/brackets)
def is_single_char_or_word(text):
    """Check if text is a single character or single word without tags/brackets"""
    text_clean = text.strip()
    # Single character
    if len(text_clean) == 1:
        return True
    # Check if it's a single word (no spaces) and no tags/brackets
    if ' ' not in text_clean:
        # Check for common tags/brackets
        if any(c in text_clean for c in ['[', ']', '<', '>', '{', '}', '(', ')']):
            return False
        # Single word without tags
        return True
    return False

def extract_potential_answer_with_has_is(extracted_text):
    """Extract potential answer from patterns like **answer** is or **answer** has
    Returns the answer that appears most frequently with is/has patterns.
    Includes parentheses if present (e.g., "KHDC3L (ENSG00000203908)")."""
    # Pattern: **answer** is or **answer** has (captures everything including parentheses)
    pattern = r'\*\*([^*]+)\*\*\s+(?:is|has)\b'
    matches = re.findall(pattern, extracted_text, re.IGNORECASE)
    if not matches:
        return None
    
    # Count occurrences of each answer with is/has pattern
    answer_counts = {}
    for answer in matches:
        answer_clean = answer.strip()
        # Count how many times this exact answer appears with is/has
        answer_pattern = r'\*\*' + re.escape(answer_clean) + r'\*\*\s+(?:is|has)\b'
        count = len(re.findall(answer_pattern, extracted_text, re.IGNORECASE))
        answer_counts[answer_clean] = count
    
    # Return the answer with the highest count (keeping parentheses if present)
    if answer_counts:
        max_count = max(answer_counts.values())
        # Get all answers with the maximum count
        candidates = [ans for ans, count in answer_counts.items() if count == max_count]
        
        # If there's a tie, prefer the one with parentheses (more complete)
        if len(candidates) > 1:
            # Sort by whether it contains parentheses (prefer those with parentheses), then by length
            candidates.sort(key=lambda x: (('(' not in x), len(x)))
            return candidates[0]
        
        # Return the candidate (with parentheses if present)
        return candidates[0]
    return None

def count_potential_answer_with_has_is(extracted_text, potential_answer):
    """Count how many times the potential answer appears in the extracted text"""
    if not potential_answer:
        return 0
    # For answers with parentheses, we need to count them as literal strings
    # Escape special regex characters and count occurrences
    escaped = re.escape(potential_answer)
    # Use word boundaries only if the answer doesn't contain parentheses
    if '(' in potential_answer or ')' in potential_answer:
        # For text with parentheses, count as literal substring (case-insensitive)
        pattern = re.escape(potential_answer)
        matches = len(re.findall(pattern, extracted_text, re.IGNORECASE))
        return matches
    else:
        # For text without parentheses, use word boundaries
        pattern = r'\b' + re.escape(potential_answer.lower()) + r'\b'
        matches = re.findall(pattern, extracted_text.lower())
        return len(matches)

def extract_potential_answer_emerges(extracted_text):
    """Extract potential answer from patterns like **answer** emerges or answer emerges
    Includes parentheses if present (e.g., "KHDC3L (ENSG00000203908)")."""
    # Pattern: **answer** emerges or answer emerges (without **)
    # For plain text, capture word(s) and optional parentheses before "emerges"
    patterns = [
        r'\*\*([^*]+)\*\*\s+emerges\b',  # **answer** emerges (captures everything including parentheses)
        r'\b([A-Z0-9]+(?:\s+\([^)]+\))?)\s+emerges\b'  # answer emerges (word/code with optional parentheses)
    ]
    for pattern in patterns:
        match = re.search(pattern, extracted_text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Keep parentheses if present (don't remove them)
            return answer
    return None

def count_potential_answer_emerges(extracted_text, potential_answer):
    """Count how many times the potential answer appears in the extracted text"""
    if not potential_answer:
        return 0
    # For answers with parentheses, we need to count them as literal strings
    # Escape special regex characters and count occurrences
    if '(' in potential_answer or ')' in potential_answer:
        # For text with parentheses, count as literal substring (case-insensitive)
        pattern = re.escape(potential_answer)
        matches = len(re.findall(pattern, extracted_text, re.IGNORECASE))
        return matches
    else:
        # For text without parentheses, use word boundaries
        pattern = r'\b' + re.escape(potential_answer.lower()) + r'\b'
        matches = re.findall(pattern, extracted_text.lower())
        return len(matches)

def extract_potential_answer_final_answer(extracted_text):
    """Extract potential answer from patterns like **Final Answer: answer**"""
    # Pattern: **Final Answer: answer** or Final Answer: answer
    patterns = [
        r'\*\*Final\s+Answer:\s*([^*]+)\*\*',
        r'Final\s+Answer:\s*([^\n]+)',
        r'\*\*Answer:\s*([^*]+)\*\*',
        r'Answer:\s*([^\n]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, extracted_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def count_potential_answer_final_answer(extracted_text, potential_answer):
    """Count how many times the potential answer appears in the extracted text"""
    if not potential_answer:
        return 0
    # Count all occurrences of the potential answer (case-insensitive, whole word match)
    pattern = r'\b' + re.escape(potential_answer.lower()) + r'\b'
    matches = re.findall(pattern, extracted_text.lower())
    return len(matches)

def extract_potential_answer_conclusion(extracted_text):
    """Extract potential answer from patterns like ## Conclusion\n\n**answer**"""
    # Pattern: ## Conclusion followed by **answer**
    pattern = r'##\s+Conclusion[^\n]*\n+\*\*([^*]+)\*\*'
    match = re.search(pattern, extracted_text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None

def count_potential_answer_conclusion(extracted_text, potential_answer):
    """Count how many times the potential answer appears in the extracted text"""
    if not potential_answer:
        return 0
    # Count all occurrences of the potential answer (case-insensitive, whole word match)
    pattern = r'\b' + re.escape(potential_answer.lower()) + r'\b'
    matches = re.findall(pattern, extracted_text.lower())
    return len(matches)

# Add potential answer features to all entries before writing to instances folder
def add_potential_answer_features(entry):
    """Add potential answer features to an entry"""
    extracted = str(entry.get("extracted_solution_or_last_step", "")).strip()
    
    # Extract potential answers and count occurrences
    potential_answer_has_is = extract_potential_answer_with_has_is(extracted)
    potential_answer_final = extract_potential_answer_final_answer(extracted)
    potential_answer_conclusion = extract_potential_answer_conclusion(extracted)
    potential_answer_emerges = extract_potential_answer_emerges(extracted)
    
    # Count how many times each potential answer appears in the extracted text
    count_has_is = count_potential_answer_with_has_is(extracted, potential_answer_has_is)
    count_final = count_potential_answer_final_answer(extracted, potential_answer_final)
    count_conclusion = count_potential_answer_conclusion(extracted, potential_answer_conclusion)
    count_emerges = count_potential_answer_emerges(extracted, potential_answer_emerges)
    
    # Create a copy and add features
    entry_copy = entry.copy()
    entry_copy["potential_answer_with_has_is"] = potential_answer_has_is if potential_answer_has_is else None
    entry_copy["potential_answer_final_answer"] = potential_answer_final if potential_answer_final else None
    entry_copy["potential_answer_conclusion"] = potential_answer_conclusion if potential_answer_conclusion else None
    entry_copy["potential_answer_emerges"] = potential_answer_emerges if potential_answer_emerges else None
    entry_copy["count_potential_answer_with_has_is"] = count_has_is
    entry_copy["count_potential_answer_final_answer"] = count_final
    entry_copy["count_potential_answer_conclusion"] = count_conclusion
    entry_copy["count_potential_answer_emerges"] = count_emerges
    
    return entry_copy

# Add potential answer features to all solutions
solutions_with_features = [add_potential_answer_features(entry) for entry in solutions]

# Add potential answer features to all last_steps
last_steps_with_features = [add_potential_answer_features(entry) for entry in last_steps]

# # Write each solution to a separate JSON file (with features)
# for idx, entry in enumerate(solutions_with_features):
#     instance_id = entry.get("instance_id", idx)
#     filename = f"solutions_{instance_id}.json"
#     filepath = os.path.join(instances_folder, filename)
#     with open(filepath, "w", encoding="utf-8") as outfile:
#         json.dump(entry, outfile, ensure_ascii=False, indent=2)

# # Write each last step to a separate JSON file (with features)
# for idx, entry in enumerate(last_steps_with_features):
#     instance_id = entry.get("instance_id", idx)
#     filename = f"last_steps_{instance_id}.json"
#     filepath = os.path.join(instances_folder, filename)
#     with open(filepath, "w", encoding="utf-8") as outfile:
#         json.dump(entry, outfile, ensure_ascii=False, indent=2)

# Check solutions (using entries with features)
for entry in solutions_with_features:
    answer = str(entry.get("answer", "")).strip()
    extracted = str(entry.get("extracted_solution_or_last_step", "")).strip()
    
    answer_lower = answer.lower()
    extracted_lower = extracted.lower()
    
    # Check if they match (case-insensitive)
    if answer_lower == extracted_lower:
        correct_instances.append(entry)
    else:
        # Check if extracted is one character or one word (no spaces, no tags)
        if is_single_char_or_word(extracted):
            # Single character or single word - mark as wrong
            wrong_instances.append(entry)
        else:
            # More than one word or contains tags - needs to be checked
            # Entry already has potential answer features from last_steps_with_features
            instances_to_be_checked.append(entry)

# Check last_steps (using entries with features)
for entry in last_steps_with_features:
    answer = str(entry.get("answer", "")).strip()
    extracted = str(entry.get("extracted_solution_or_last_step", "")).strip()
    
    answer_lower = answer.lower()
    extracted_lower = extracted.lower()
    
    # Check if they match (case-insensitive)
    if answer_lower == extracted_lower:
        correct_instances.append(entry)
    else:
        # Check if extracted is one character or one word (no spaces, no tags)
        if is_single_char_or_word(extracted):
            # Single character or single word - mark as wrong
            wrong_instances.append(entry)
        else:
            # More than one word or contains tags - needs to be checked
            # Entry already has potential answer features from last_steps_with_features
            instances_to_be_checked.append(entry)

# # Create correct_instances folder
# correct_instances_folder = "correct_instances"
# os.makedirs(correct_instances_folder, exist_ok=True)

# for idx, entry in enumerate(correct_instances):
#     instance_id = entry.get("instance_id", idx)
#     # Determine if it was from solutions or last_steps
#     if entry.get("error") is None:
#         filename = f"solutions_{instance_id}.json"
#     else:
#         filename = f"last_steps_{instance_id}.json"
#     filepath = os.path.join(correct_instances_folder, filename)
#     with open(filepath, "w", encoding="utf-8") as outfile:
#         json.dump(entry, outfile, ensure_ascii=False, indent=2)

# Load original data to get prompt, task_name, and execution_time
original_data = {}
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        try:
            data = json.loads(line)
            instance_id = data.get("instance_id")
            if instance_id is not None:
                original_data[instance_id] = {
                    "prompt": data.get("prompt", ""),
                    "task_name": data.get("task_name", ""),
                    "execution_time": data.get("execution_time", 0.0)
                }
        except Exception:
            continue

# Add prompt, task_name, and execution_time to all instances
for entry in correct_instances:
    instance_id = entry.get("instance_id")
    if instance_id in original_data:
        entry["prompt"] = original_data[instance_id]["prompt"]
        entry["task_name"] = original_data[instance_id]["task_name"]
        entry["execution_time"] = original_data[instance_id]["execution_time"]

for entry in wrong_instances:
    instance_id = entry.get("instance_id")
    if instance_id in original_data:
        entry["prompt"] = original_data[instance_id]["prompt"]
        entry["task_name"] = original_data[instance_id]["task_name"]
        entry["execution_time"] = original_data[instance_id]["execution_time"]

for entry in instances_to_be_checked:
    instance_id = entry.get("instance_id")
    if instance_id in original_data:
        entry["prompt"] = original_data[instance_id]["prompt"]
        entry["task_name"] = original_data[instance_id]["task_name"]
        entry["execution_time"] = original_data[instance_id]["execution_time"]

# Add task_name and execution_time to proxy_errors and max_tokens_errors
for entry in proxy_errors:
    instance_id = entry.get("instance_id")
    if instance_id in original_data:
        entry["task_name"] = original_data[instance_id]["task_name"]
        entry["execution_time"] = original_data[instance_id]["execution_time"]

for entry in max_tokens_errors:
    instance_id = entry.get("instance_id")
    if instance_id in original_data:
        entry["task_name"] = original_data[instance_id]["task_name"]
        entry["execution_time"] = original_data[instance_id]["execution_time"]

# Create instances_to_be_checked folder
instances_to_check_folder = "instances_to_be_checked"
os.makedirs(instances_to_check_folder, exist_ok=True)

for idx, entry in enumerate(instances_to_be_checked):
    instance_id = entry.get("instance_id", idx)
    # Determine if it was from solutions or last_steps
    if entry.get("error") is None:
        filename = f"solutions_{instance_id}.json"
    else:
        filename = f"last_steps_{instance_id}.json"
    filepath = os.path.join(instances_to_check_folder, filename)
    with open(filepath, "w", encoding="utf-8") as outfile:
        json.dump(entry, outfile, ensure_ascii=False, indent=2)

# Combine all instances_to_be_checked into one JSONL file
combined_output_file = "instances_to_be_checked_by_chat.jsonl"
with open(combined_output_file, "w", encoding="utf-8") as outfile:
    for entry in instances_to_be_checked:
        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Create correct_instances.jsonl with chat_eval = "1" (right after instance_id)
correct_instances_jsonl = "correct_instances.jsonl"
with open(correct_instances_jsonl, "w", encoding="utf-8") as outfile:
    for entry in correct_instances:
        # Create ordered dict with chat_eval right after instance_id
        from collections import OrderedDict
        ordered_entry = OrderedDict()
        ordered_entry["instance_id"] = entry.get("instance_id")
        ordered_entry["chat_eval"] = "1"
        # Add all other fields
        for key, value in entry.items():
            if key != "instance_id":
                ordered_entry[key] = value
        outfile.write(json.dumps(ordered_entry, ensure_ascii=False) + "\n")

# Create wrong_instances.jsonl with chat_eval = "0" (right after instance_id)
wrong_instances_jsonl = "wrong_instances.jsonl"
with open(wrong_instances_jsonl, "w", encoding="utf-8") as outfile:
    for entry in wrong_instances:
        # Create ordered dict with chat_eval right after instance_id
        from collections import OrderedDict
        ordered_entry = OrderedDict()
        ordered_entry["instance_id"] = entry.get("instance_id")
        ordered_entry["chat_eval"] = "0"
        # Add all other fields
        for key, value in entry.items():
            if key != "instance_id":
                ordered_entry[key] = value
        outfile.write(json.dumps(ordered_entry, ensure_ascii=False) + "\n")

# Create wrong_instances folder
# wrong_instances_folder = "wrong_instances"
# os.makedirs(wrong_instances_folder, exist_ok=True)

# for idx, entry in enumerate(wrong_instances):
#     instance_id = entry.get("instance_id", idx)
#     # Determine if it was from solutions or last_steps
#     if entry.get("error") is None:
#         filename = f"solutions_{instance_id}.json"
#     else:
#         filename = f"last_steps_{instance_id}.json"
#     filepath = os.path.join(wrong_instances_folder, filename)
#     with open(filepath, "w", encoding="utf-8") as outfile:
#         json.dump(entry, outfile, ensure_ascii=False, indent=2)

# Calculate statistics by task (including execution time and error breakdown)
task_stats = defaultdict(lambda: {
    "total": 0, 
    "correct": 0, 
    "wrong": 0, 
    "to_be_checked": 0,
    "total_time": 0.0,
    "incorrect": {
        "max_token_error": 0,
        "proxy_error": 0,
        "wrong_answers": 0
    }
})

# Count correct instances by task
for entry in correct_instances:
    task_name = entry.get("task_name", "unknown")
    execution_time = entry.get("execution_time", 0.0)
    task_stats[task_name]["total"] += 1
    task_stats[task_name]["correct"] += 1
    task_stats[task_name]["total_time"] += execution_time

# Count wrong instances by task
for entry in wrong_instances:
    task_name = entry.get("task_name", "unknown")
    execution_time = entry.get("execution_time", 0.0)
    task_stats[task_name]["total"] += 1
    task_stats[task_name]["wrong"] += 1
    task_stats[task_name]["total_time"] += execution_time
    task_stats[task_name]["incorrect"]["wrong_answers"] += 1

# Count instances_to_be_checked by task
for entry in instances_to_be_checked:
    task_name = entry.get("task_name", "unknown")
    execution_time = entry.get("execution_time", 0.0)
    task_stats[task_name]["total"] += 1
    task_stats[task_name]["to_be_checked"] += 1
    task_stats[task_name]["total_time"] += execution_time

# Count proxy errors by task
for entry in proxy_errors:
    task_name = entry.get("task_name", "unknown")
    execution_time = entry.get("execution_time", 0.0)
    task_stats[task_name]["total"] += 1
    task_stats[task_name]["total_time"] += execution_time
    task_stats[task_name]["incorrect"]["proxy_error"] += 1

# Count max_tokens errors by task
for entry in max_tokens_errors:
    task_name = entry.get("task_name", "unknown")
    execution_time = entry.get("execution_time", 0.0)
    task_stats[task_name]["total"] += 1
    task_stats[task_name]["total_time"] += execution_time
    task_stats[task_name]["incorrect"]["max_token_error"] += 1

# Convert to regular dict and format similar to r0_statistics_b1.json
by_task = {}
for task_name, stats in sorted(task_stats.items()):
    by_task[task_name] = {
        "total": stats["total"],
        "correct": stats["correct"],
        "wrong": stats["wrong"],
        "to_be_checked": stats["to_be_checked"],
        "total_time": stats["total_time"],
        "incorrect": stats["incorrect"]
    }

# Write detailed statistics file (for backward compatibility)
statistics_file = "statistics.json"
statistics = {
    "correct_instances": len(correct_instances),
    "wrong_instances": len(wrong_instances),
    "instances_to_be_checked": len(instances_to_be_checked),
    "proxy_errors": len(proxy_errors),
    "max_tokens_errors": len(max_tokens_errors),
    "wrong_instances_list": wrong_instances,
    "by_task": by_task
}

with open(statistics_file, "w", encoding="utf-8") as outfile:
    json.dump(statistics, outfile, ensure_ascii=False, indent=2)

# Create initial_statistics.json in r0_statistics_b1.json format
initial_statistics_file = "initial_statistics.json"
total_execution_time = sum(stats["total_time"] for stats in task_stats.values())
total_processed = len(correct_instances) + len(wrong_instances) + len(instances_to_be_checked) + len(proxy_errors) + len(max_tokens_errors)

initial_statistics = {
    "processed": total_processed,
    "correct": len(correct_instances),
    "total_execution_time": total_execution_time,
    "by_task": {}
}

# Format by_task for initial_statistics (similar to r0_statistics_b1.json)
for task_name, stats in sorted(task_stats.items()):
    initial_statistics["by_task"][task_name] = {
        "total": stats["total"],
        "correct": stats["correct"],
        "total_time": stats["total_time"],
        "incorrect": {
            "max_token_error": stats["incorrect"]["max_token_error"],
            "proxy_error": stats["incorrect"]["proxy_error"],
            "wrong_answers": stats["incorrect"]["wrong_answers"]
        }
    }

with open(initial_statistics_file, "w", encoding="utf-8") as outfile:
    json.dump(initial_statistics, outfile, ensure_ascii=False, indent=2)

print(f"\nCreated statistics files:")
print(f"  - {statistics_file} (detailed with lists)")
print(f"  - {initial_statistics_file} (summary format)")

print(f"\nCreated instance files:")
print(f"  - {correct_instances_jsonl} ({len(correct_instances)} instances with chat_eval='1')")
print(f"  - {wrong_instances_jsonl} ({len(wrong_instances)} instances with chat_eval='0')")
print(f"  - {combined_output_file} ({len(instances_to_be_checked)} instances to be checked)")

print(f"Processed entries:")
print(f"  - Solutions: {len(solutions)}")
print(f"  - Last steps: {len(last_steps)}")
print(f"  - Proxy errors: {len(proxy_errors)}")
print(f"  - Max tokens errors: {len(max_tokens_errors)}")
print(f"  - Unique instance ids seen: {len(seen_instance_ids)}")
print(f"Error statistics: {error_stats}")

print(f"\nAnswer matching statistics:")
print(f"  - Correct instances (answer matches extracted): {len(correct_instances)}")
print(f"  - Wrong instances (single char/word mismatch): {len(wrong_instances)}")
print(f"  - Instances to be checked (multi-word extracted): {len(instances_to_be_checked)}")

# Verify total count
total_count = len(solutions) + len(last_steps) + len(proxy_errors) + len(max_tokens_errors)
print(f"\nTotal count verification:")
print(f"  Solutions: {len(solutions)}")
print(f"  + Last steps: {len(last_steps)}")
print(f"  + Proxy errors: {len(proxy_errors)}")
print(f"  + Max tokens errors: {len(max_tokens_errors)}")
print(f"  = Total: {total_count}")
if total_count == 433:
    print(f"  ✓ Total count is correct: {total_count}")
else:
    print(f"  ✗ Total count mismatch! Expected 433, got {total_count}")
