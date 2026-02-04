# Thesis Results

> **Experimental Results**: Complete evaluation results for all methods on the Eval1 biomedical benchmark.

This directory contains the annotated results, statistics, and evaluation outputs for all experimental configurations tested in the thesis.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Directory Structure](#-directory-structure)
- [Result Categories](#-result-categories)
- [Annotation Pipeline](#-annotation-pipeline)
- [Statistics Format](#-statistics-format)
- [Chat Evaluation Instances](#-chat-evaluation-instances)

---

## 🎯 Overview

Each subdirectory contains complete evaluation results for a specific model configuration:

| Directory | Method | Model | Description |
|-----------|--------|-------|-------------|
| `R0_32B_BF16/` | Baseline | Biomni-R0-32B | Full precision reference |
| `Qwen_32B_base_model/` | Control | Qwen3-32B | Non-adapted base model |
| `Qwen_FP8_LORA256/` | Method A | Qwen-FP8 + LoRA-256 | Naive transfer |
| `Qwen_FP8_with_extracted_dequantized_LORA_rank256/` | Method B | Qwen-FP8 + Corrective LoRA | Corrective extraction |
| `R0-32B-FP8/` | Method C | R0-32B-FP8 | Direct quantization |
| `Qwen_FP8_with_LORA_128/` | Variant | Qwen-FP8 + LoRA-128 | Reduced rank experiment |
| `R0_32B_INT8_quantized/` | Exploratory | R0-32B-INT8 | INT8 quantization test |

---

## 📁 Directory Structure

```
thesis_results_final/
├── README.md                         # This file
│
├── 📂 annotation_pipeline/           # Evaluation annotation tools
│   ├── 1_extract_answers.py          # Extract model outputs
│   ├── 2_ask_chat.py                 # AI-assisted evaluation
│   ├── 3_create_final_annotated_results.py  # Merge annotations
│   ├── 4_create_final_statistics.py  # Generate statistics
│   └── annotate_requirment.txt       # Python dependencies
│
├── 📂 R0_32B_BF16/                   # Baseline results
│   ├── R0_32B_BF16_results_annotated.jsonl
│   ├── R0_32B_BF16_results_not_annotated.jsonl
│   └── Final_statistics.json
│
├── 📂 Qwen_FP8_LORA256/              # Method A results
│   ├── Qwen_FP8_LORA256_results_annotated.jsonl
│   ├── Qwen_FP8_LORA256_results_not_annotated.jsonl
│   ├── Qwen_FP8_LORA256_statistics.json
│   └── Chat_eval_instances/
│       ├── Ai_eval_0/                # Incorrect predictions
│       ├── Ai_eval_1/                # Correct predictions
│       └── Ai_eval_null/             # Ambiguous cases
│
├── 📂 Qwen_FP8_with_extracted_dequantized_LORA_rank256/  # Method B
│   └── [same structure as above]
│
├── 📂 R0-32B-FP8/                    # Method C results
│   └── [same structure as above]
│
└── [Additional model directories...]
```

---

## 📊 Result Categories

### Annotation Labels (chat_eval)

| Label | Meaning | Description |
|-------|---------|-------------|
| `"1"` | Correct | Model answer matches ground truth |
| `"0"` | Wrong | Model answer incorrect |
| `"2"` | Max Token Error | Output truncated due to token limit |
| `"3"` | Proxy Error | Server/connection error |
| `null` | Not Evaluated | Requires manual review |

### Result Files

| File | Contents |
|------|----------|
| `*_annotated.jsonl` | Complete results with chat_eval labels |
| `*_not_annotated.jsonl` | Raw outputs before annotation |
| `*_statistics.json` | Aggregated performance metrics |

---

## 🔄 Annotation Pipeline

The annotation pipeline evaluates model outputs through a multi-stage process:

### Stage 1: Extract Answers (`1_extract_answers.py`)

Parses model outputs to extract final answers from various formats:
- XML tags: `<solution>answer</solution>`
- Markdown sections: `# Solution`
- Natural language: "The answer is..."

### Stage 2: AI Evaluation (`2_ask_chat.py`)

Uses ChatAI (GPT-4 class model) for automated evaluation:

```python
# Evaluation criteria:
# - Semantic equivalence (CCR4 == "CCR4" == ["CCR4"])
# - Multiple choice matching (B == "B. P335A")
# - JSON object matching (partial ID match acceptable)
```

### Stage 3: Merge Annotations (`3_create_final_annotated_results.py`)

Combines:
- Initial heuristic evaluation
- AI evaluation results
- Error detection (max_token, proxy)

### Stage 4: Generate Statistics (`4_create_final_statistics.py`)

Produces per-task and overall accuracy metrics.

---

## 📈 Statistics Format

### Example `Final_statistics.json`

```json
{
  "processed": 433,
  "correct": 312,
  "total_execution_time": 45678.9,
  "by_task": {
    "crispr_delivery": {
      "total": 45,
      "correct": 38,
      "total_time": 4521.3,
      "incorrect": {
        "max_token_error": 2,
        "proxy_error": 0,
        "wrong_answers": 5
      }
    },
    "genetic_variant": { ... },
    "gwas_causal_gene": { ... },
    ...
  }
}
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `processed` | Total instances evaluated |
| `correct` | Instances with chat_eval="1" |
| `total_execution_time` | Cumulative inference time (seconds) |
| `by_task.*.total` | Per-task instance count |
| `by_task.*.correct` | Per-task correct count |
| `by_task.*.incorrect.wrong_answers` | Incorrect predictions |
| `by_task.*.incorrect.max_token_error` | Truncation errors |
| `by_task.*.incorrect.proxy_error` | Connection errors |

---

## 📂 Chat Evaluation Instances

The `Chat_eval_instances/` subdirectory organizes individual evaluation cases:

```
Chat_eval_instances/
├── Ai_eval_0/     # chat_eval="0" (Wrong)
│   ├── 14.json
│   ├── 25.json
│   └── ...
├── Ai_eval_1/     # chat_eval="1" (Correct)
│   ├── 1.json
│   ├── 7.json
│   └── ...
└── Ai_eval_null/  # chat_eval=null (Ambiguous)
    ├── 145.json
    └── ...
```

### Instance JSON Format

```json
{
  "instance_id": 14,
  "chat_eval": "0",
  "answer": "CCR4",
  "extracted_solution_or_last_step": "CCR5",
  "prompt": "[01] USER: Which receptor...",
  "full_response": "[02] ASSISTANT: Let me analyze...",
  "task_name": "crispr_delivery",
  "execution_time": 45.2,
  "total_output_tokens": 1523,
  "num_steps": 12
}
```

---

## 💻 Usage

### Run Complete Pipeline

```bash
cd annotation_pipeline

# Step 1: Extract answers
python 1_extract_answers.py

# Step 2: AI evaluation (requires CUSTOM_MODEL_API_KEY)
python 2_ask_chat.py

# Step 3: Merge annotations
python 3_create_final_annotated_results.py

# Step 4: Generate statistics
python 4_create_final_statistics.py
```

### Quick Statistics Check

```bash
# View accuracy for a model
python -c "
import json
with open('R0-32B-FP8/R0-32B-FP8_results_annotated_statistics.json') as f:
    stats = json.load(f)
    print(f'Accuracy: {stats[\"correct\"]}/{stats[\"processed\"]} = {stats[\"correct\"]/stats[\"processed\"]*100:.1f}%')
"
```

---

## 📊 Results Summary

| Model | Accuracy | Correct | Total |
|-------|----------|---------|-------|
| R0-32B-BF16 (Baseline) | XX.X% | XXX | 433 |
| Method A (Naive Transfer) | XX.X% | XXX | 433 |
| Method B (Corrective) | XX.X% | XXX | 433 |
| Method C (Direct Quant) | XX.X% | XXX | 433 |

*See visualization/ for detailed performance comparisons.*
