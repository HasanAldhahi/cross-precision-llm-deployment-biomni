# Quantization Module

> **Method C Implementation**: Direct FP8 quantization of Biomni-R0-32B with domain-specific calibration.

This module contains the implementation of **Method C (Direct Quantization)** from the thesis, which quantizes the domain-adapted Biomni-R0-32B model directly to FP8 using post-training quantization with biomedical calibration data.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Module Structure](#-module-structure)
- [Calibration Data](#-calibration-data)
- [FP8 Quantization](#-fp8-quantization)
- [Installation](#-installation)
- [Usage](#-usage)
- [Logs](#-logs)

---

## 🎯 Overview

Method C directly quantizes the Biomni-R0-32B model to FP8 format using the **llm-compressor** library with domain-specific calibration data from successful biomedical reasoning trajectories.

### Key Innovation: Deep Calibration

Unlike standard quantization recipes that use short context windows (512-1024 tokens), our calibration captures **full-trajectory activation patterns**:

| Metric | Value |
|--------|-------|
| Total Instances | 123 |
| Total Tokens | 3,163,274 |
| Average Context | 25,718 tokens |
| Maximum Context | 75,508 tokens |
| Sampling | Stratified across 10 Eval1 tasks |

---

## 📁 Module Structure

\`\`\`
quantization/
├── quantization_requirements.txt     # Python dependencies
├── README.md                         # This file
│
├── 📂 calibration_data/              # Biomedical calibration dataset
│   ├── Data_r0_annotated.jsonl       # Raw successful trajectories
│   ├── Data_r0_annotated_cleaned.jsonl  # Cleaned calibration data
│   ├── calibration_data.json         # Final calibration format
│   ├── calibration_preview.txt       # Sample preview
│   ├── prepare_calibration.py        # Data preparation script
│   └── clean_calibration_data.py     # Data cleaning script
│
└── 📂 scripts/
    ├── 📂 FP8_quantization/          # Method C implementation
    │   ├── quantize_FP8.py           # Main quantization script
    │   ├── quanitze_FP8.sh           # SLURM job script
    │   ├── quantize_FP8.log          # Execution log
    │   └── quantize_FP8.err          # Error log
    │
    └── 📂 INT4_quantization/         # AWQ INT4 (exploratory)
        ├── quantize_AWQ_INT4.py      # INT4 quantization script
        ├── quanitze_INT4.sh          # SLURM job script
        └── quantize_AWQ_INT4_*.log   # Execution logs
\`\`\`

---

## 📊 Calibration Data

Calibration data is derived from **successful biomedical reasoning trajectories** on the Eval1 benchmark with stratified sampling across all 10 tasks.

### Data Preparation

\`\`\`bash
cd calibration_data
python prepare_calibration.py --input ../results.jsonl --output calibration_data.json
\`\`\`

---

## ⚡ FP8 Quantization

### Configuration

- **Format**: FP8 E4M3 with Block-128 quantization
- **Excluded Layers**: lm_head (kept in higher precision)
- **Max Sequence Length**: 76,000 tokens

### Usage

\`\`\`bash
cd scripts/FP8_quantization
sbatch quanitze_FP8.sh  # SLURM
# OR
python quantize_FP8.py  # Direct
\`\`\`

---

## 📋 Logs

| Log File | Description |
|----------|-------------|
| \`scripts/FP8_quantization/quantize_FP8.log\` | FP8 quantization output |
| \`scripts/FP8_quantization/quantize_FP8.err\` | Error messages |
| \`scripts/INT4_quantization/quantize_AWQ_INT4_*.log\` | INT4 logs |

---

## 📚 References

- llm-compressor: https://github.com/neuralmagic/llm-compressor
- FP8 Training: https://arxiv.org/abs/2209.05433
