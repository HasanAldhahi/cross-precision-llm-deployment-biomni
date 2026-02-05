# Statistics Files Explanation

## Overview

The `extract_prompt.py` script now generates two statistics files:

1. **`statistics.json`** - Detailed statistics with instance lists
2. **`initial_statistics.json`** - Summary format similar to `r0_statistics_b1.json`

## File Format: initial_statistics.json

This file follows the format of `r0_statistics_b1.json` with additional error breakdown:

```json
{
  "processed": <total instances processed>,
  "correct": <number of correct instances>,
  "total_execution_time": <sum of execution times in seconds>,
  "by_task": {
    "task_name": {
      "total": <total instances for this task>,
      "correct": <correct instances>,
      "total_time": <execution time for this task>,
      "incorrect": {
        "max_token_error": <instances with max token errors>,
        "proxy_error": <instances with proxy errors>,
        "wrong_answers": <instances with wrong answers>
      }
    }
  }
}
```

## Understanding the Counts

For each task, the breakdown is:

```
total = correct + incorrect + to_be_checked
```

Where:
- **correct**: Instances where extracted solution exactly matches the answer
- **incorrect**: Sum of all error types:
  - `max_token_error`: Hit token limit during generation
  - `proxy_error`: Network/proxy errors
  - `wrong_answers`: Wrong single-word/single-char answers
- **to_be_checked**: Multi-word/complex answers that need LLM evaluation

### Example: lab_bench_dbqa

```
Total: 50 instances
├─ Correct: 20 (exact matches)
├─ Incorrect: 14
│  ├─ Max Token Error: 8
│  ├─ Proxy Error: 0
│  └─ Wrong Answers: 6
└─ To Be Checked: 16 (need evaluation)
```

Verification: 20 + 14 + 16 = 50 ✓

## Current Statistics (R0_32B_BF16)

### Overall
- **Total Processed**: 432 instances
- **Correct**: 106 (24.5%)
- **Wrong**: 72 (16.7%)
- **To Be Checked**: 200 (46.3%)
- **Errors**: 54 (12.5%)
  - Max Token Errors: 43
  - Proxy Errors: 11
- **Total Execution Time**: 2,739,971 seconds (~761 hours)

### Per-Task Breakdown

| Task | Total | Correct | Incorrect | To Check | Accuracy* |
|------|-------|---------|-----------|----------|-----------|
| lab_bench_seqqa | 49 | 31 | 11 | 7 | 63.3% |
| lab_bench_dbqa | 50 | 20 | 14 | 16 | 40.0% |
| screen_gene_retrieval | 50 | 14 | 26 | 10 | 28.0% |
| gwas_causal_gene_opentargets | 50 | 13 | 8 | 29 | 26.0% |
| gwas_causal_gene_pharmaprojects | 50 | 11 | 17 | 22 | 22.0% |
| gwas_variant_prioritization | 43 | 9 | 10 | 24 | 20.9% |
| gwas_causal_gene_gwas_catalog | 50 | 6 | 17 | 27 | 12.0% |
| crispr_delivery | 10 | 2 | 8 | 0 | 20.0% |
| patient_gene_detection | 50 | 0 | 9 | 41 | 0.0% |
| rare_disease_diagnosis | 30 | 0 | 6 | 24 | 0.0% |

*Accuracy = correct / total (before LLM evaluation)

## Next Steps

The 200 instances in "To Be Checked" need to be evaluated using:
1. ChatAI: `./run_step.sh chat`
2. OpenAI: `./run_step.sh openai`
3. Gemini: `./run_step.sh gemini`

After evaluation, run `create_final_statistics.py` to combine initial and evaluation results.

## Files Generated

1. **initial_statistics.json** - Summary format (this format)
2. **statistics.json** - Detailed with instance lists
3. **instances_to_be_checked_by_gemini.jsonl** - 200 instances for evaluation
4. **extracted_solutions.jsonl** - All extracted solutions
5. **extracted_proxy_error.jsonl** - 11 proxy errors
6. **extracted_max_tokens_error.jsonl** - 43 max token errors

## Error Analysis

### Max Token Errors (43 total)
Most common in:
- gwas_causal_gene_gwas_catalog: 10 instances
- lab_bench_dbqa: 8 instances
- rare_disease_diagnosis: 6 instances

These instances hit the model's token limit during generation.

### Proxy Errors (11 total)
Most common in:
- patient_gene_detection: 3 instances
- gwas_causal_gene_pharmaprojects: 2 instances

These instances had network/proxy connection issues during generation.

### Wrong Answers (72 total)
These are single-word or single-character answers that don't match the ground truth.
Most common in:
- screen_gene_retrieval: 22 instances
- gwas_causal_gene_pharmaprojects: 14 instances







