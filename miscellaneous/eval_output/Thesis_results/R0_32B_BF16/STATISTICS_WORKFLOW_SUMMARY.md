# Statistics Workflow Summary

## Overview

This document summarizes the complete statistics workflow from initial extraction to final evaluation.

## Files Generated

### 1. **initial_statistics.json**
- **Source**: `extract_prompt.py`
- **Purpose**: Initial statistics from automated extraction
- **Format**: Similar to `r0_statistics_b1.json`
- **Content**:
  - Total processed: 432 instances
  - Initial correct: 106 (24.5%)
  - Error breakdown per task (max_token, proxy, wrong_answers)
  - Execution time tracking

### 2. **r0_statistics_checked.json**
- **Source**: `create_chat_statistics.py`
- **Purpose**: Statistics from Chat AI evaluation
- **Content**:
  - Evaluated: 200 instances (instances_to_be_checked)
  - Correct: 95 instances (47.5%)
  - Per-task breakdown

### 3. **final_statistics.json**
- **Source**: `combine_final_statistics.py`
- **Purpose**: Combined final statistics
- **Content**:
  - Total: 432 instances
  - Final correct: 201 (46.5%)
  - Combines initial + chat evaluation results
  - Preserves error breakdown from initial statistics

## Workflow Steps

```
Step 1: Initial Extraction
├─ Script: extract_prompt.py
├─ Input: Final_r0_results.jsonl
├─ Output: initial_statistics.json
└─ Result: 106 correct, 200 to_be_checked

Step 2: Chat AI Evaluation
├─ Script: ask_chat.py
├─ Input: instances_to_be_checked_by_gemini.jsonl
├─ Output: combined_instances_with_chat_eval_Chat_OBS.jsonl
└─ Result: 95 additional correct from 200 checked

Step 3: Chat Statistics
├─ Script: create_chat_statistics.py
├─ Input: combined_instances_with_chat_eval_Chat_OBS.jsonl
├─ Output: r0_statistics_checked.json
└─ Result: Formatted statistics from chat evaluation

Step 4: Combine Statistics
├─ Script: combine_final_statistics.py
├─ Input: initial_statistics.json + r0_statistics_checked.json
├─ Output: final_statistics.json
└─ Result: 201 total correct (46.5% accuracy)
```

## Results Summary

### Overall Performance

| Metric | Value |
|--------|-------|
| **Total Instances** | 432 |
| **Initial Correct** | 106 (24.5%) |
| **Chat Evaluation Checked** | 200 |
| **Additional Correct from Chat** | 95 (47.5% of checked) |
| **Final Total Correct** | 201 (46.5%) |
| **Improvement** | +95 instances (+22.0 percentage points) |

### Per-Task Results

| Task | Total | Initial | +Chat | Final | Final Accuracy |
|------|-------|---------|-------|-------|----------------|
| lab_bench_seqqa | 49 | 31 | +5 | 36 | 73.5% |
| gwas_causal_gene_opentargets | 50 | 13 | +17 | 30 | 60.0% |
| lab_bench_dbqa | 50 | 20 | +9 | 29 | 58.0% |
| gwas_causal_gene_pharmaprojects | 50 | 11 | +14 | 25 | 50.0% |
| gwas_variant_prioritization | 43 | 9 | +10 | 19 | 44.2% |
| rare_disease_diagnosis | 30 | 0 | +11 | 11 | 36.7% |
| gwas_causal_gene_gwas_catalog | 50 | 6 | +11 | 17 | 34.0% |
| screen_gene_retrieval | 50 | 14 | +3 | 17 | 34.0% |
| patient_gene_detection | 50 | 0 | +15 | 15 | 30.0% |
| crispr_delivery | 10 | 2 | +0 | 2 | 20.0% |

### Key Insights

1. **Massive Improvement**: Chat evaluation improved accuracy by 22.0 percentage points (from 24.5% to 46.5%)

2. **Best Performers**:
   - lab_bench_seqqa: 73.5% accuracy
   - gwas_causal_gene_opentargets: 60.0% accuracy
   - lab_bench_dbqa: 58.0% accuracy

3. **Most Improved by Chat AI**:
   - patient_gene_detection: +15 instances (0 → 15)
   - gwas_causal_gene_opentargets: +17 instances (13 → 30)
   - gwas_causal_gene_pharmaprojects: +14 instances (11 → 25)

4. **Chat AI Effectiveness**:
   - 200 instances evaluated
   - 95 marked as correct (47.5% success rate)
   - Particularly effective for complex answers requiring interpretation

## Error Breakdown (from initial_statistics.json)

Total Errors: 54 instances (12.5%)

- **Max Token Errors**: 43 instances (10.0%)
  - Model hit token generation limit
  - Most common in: gwas_causal_gene_gwas_catalog (10), lab_bench_dbqa (8)

- **Proxy Errors**: 11 instances (2.5%)
  - Network/connection issues during generation
  - Most common in: patient_gene_detection (3)

## Scripts Reference

### Running the Workflow

```bash
# Step 1: Initial extraction
python3 extract_prompt.py

# Step 2: Chat AI evaluation (if needed)
./run_step.sh chat
# OR
python3 ask_chat.py

# Step 3: Generate chat statistics
python3 create_chat_statistics.py

# Step 4: Combine into final statistics
python3 combine_final_statistics.py
```

### Script Locations

All scripts are in:
```
/projects/extern/kisski/kisski-narges-llm-interactive/dir.project/hasan/uni_work/biomni_integration/Biomni/compare/eval_output/Thesis_results/R0_32B_BF16/
```

## JSON Format

All statistics files follow this format:

```json
{
  "processed": <total instances>,
  "correct": <correct count>,
  "total_execution_time": <seconds>,
  "by_task": {
    "task_name": {
      "total": <count>,
      "correct": <count>,
      "total_time": <seconds>,
      "incorrect": {  // Only in initial_statistics and final_statistics
        "max_token_error": <count>,
        "proxy_error": <count>,
        "wrong_answers": <count>
      }
    }
  }
}
```

## Conclusion

The combination of automated extraction (extract_prompt.py) and Chat AI evaluation (ask_chat.py) provides a comprehensive assessment of model performance:

- **Automated extraction**: Fast, catches exact matches (106/432 = 24.5%)
- **Chat AI evaluation**: Interprets complex/ambiguous answers (95/200 = 47.5%)
- **Combined**: Achieves 46.5% overall accuracy with detailed error analysis

The workflow successfully doubled the accuracy from initial extraction through intelligent LLM-based evaluation of ambiguous cases.







