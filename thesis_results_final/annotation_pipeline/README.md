# Annotation Pipeline

> **Evaluation Framework**: Automated and AI-assisted evaluation of model outputs for biomedical reasoning tasks.

This pipeline evaluates model outputs through a multi-stage process combining heuristic extraction, AI evaluation, and statistical aggregation.

---

## 📋 Pipeline Stages

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. Extract      │     │  2. AI           │     │  3. Merge        │     │  4. Generate     │
│     Answers      │────▶│     Evaluation   │────▶│     Annotations  │────▶│     Statistics   │
│                  │     │                  │     │                  │     │                  │
│ extract_answers  │     │  ask_chat.py     │     │  create_final    │     │  create_final    │
│     _1.py        │     │                  │     │  _annotated_3.py │     │  _statistics_4.py│
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## 📁 Module Structure

```
annotation_pipeline/
├── 1_extract_answers.py              # Stage 1: Answer extraction
├── 2_ask_chat.py                     # Stage 2: AI evaluation
├── 3_create_final_annotated_results.py  # Stage 3: Merge annotations
├── 4_create_final_statistics.py      # Stage 4: Statistics generation
├── annotate_requirment.txt           # Python dependencies
└── README.md                         # This file
```

---

## 🔧 Installation

```bash
pip install -r annotate_requirment.txt
```

### Dependencies

- `openai` - ChatAI API client
- `python-dotenv` - Environment variable management
- `tiktoken` - Token counting (optional)

### Environment Setup

Create `.env` file with:
```bash
CUSTOM_MODEL_API_KEY=your_chat_ai_api_key
```

---

## 💻 Usage

### Full Pipeline

```bash
# Run in order
python 1_extract_answers.py
python 2_ask_chat.py
python 3_create_final_annotated_results.py
python 4_create_final_statistics.py
```

### Input/Output Files

| Stage | Input | Output |
|-------|-------|--------|
| 1 | `r0_eval_results.jsonl` | `extracted_solutions.jsonl` |
| 2 | `instances_to_be_checked.jsonl` | `combined_with_chat_eval.jsonl` |
| 3 | Multiple sources | `Final_*_annotated.jsonl` |
| 4 | `Final_*_annotated.jsonl` | `Final_statistics.json` |

---

## 📊 Annotation Labels

| Label | Meaning |
|-------|---------|
| `"1"` | Correct - Answer matches ground truth |
| `"0"` | Wrong - Answer incorrect |
| `"2"` | Max Token Error - Output truncated |
| `"3"` | Proxy Error - Server/connection issue |
| `null` | Not Evaluated - Requires manual review |

---

## 🤖 AI Evaluation Prompt

The AI evaluator uses semantic matching rules:

1. **Ignore Formatting**: `CCR4` == `"CCR4"` == `['CCR4']`
2. **Multiple Choice**: `B` == `B. P335A` == `Answer: B`
3. **JSON Objects**: Partial ID/name matching acceptable
4. **Heuristics**: Trust pattern detection in ambiguous cases
