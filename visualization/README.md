# Visualization Module

> **Result Visualization**: Performance heatmaps and comparative charts for thesis figures.

This module contains scripts and outputs for visualizing experimental results across all methods and tasks.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Module Structure](#-module-structure)
- [Generated Figures](#-generated-figures)
- [Installation](#-installation)
- [Usage](#-usage)

---

## 🎯 Overview

The visualization module generates publication-quality figures for comparing model performance:

| Visualization | Script | Output |
|---------------|--------|--------|
| Task Heatmap | `heatmap.py` | `heatmap.png` |
| Accuracy Bars | `bar_chart.py` | `bar_chart.png` |
| Orthogonality Vectors | `other_charts/chart.py` | `orthogonality_vectors.png` |

---

## 📁 Module Structure

```
visualization/
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── 📂 Scripts
│   ├── heatmap.py                    # Task-model performance heatmap
│   ├── bar_chart.py                  # Per-method accuracy comparison
│   └── other_charts/
│       └── chart.py                  # Additional visualizations
│
├── 📂 Outputs
│   ├── heatmap.png                   # Performance heatmap (1000 DPI)
│   ├── bar_chart.png                 # Accuracy bar chart
│   └── other_charts/
│       └── orthogonality_vectors.png # Quantization noise visualization
│
└── 📂 Data (Statistics Files)
    ├── Final_R0_BF16_statistics.json                     # Baseline
    ├── Final_Qwen3_BF16_statistics.json                  # Base model
    ├── Final_(Method A)_*.json                           # Naive Transfer
    ├── Final_(Method B)_*.json                           # Corrective Extraction
    ├── Final_(Method C)_R0_32B_FP8_statistics.json       # Direct Quantization
    └── Final_Qwen_FP8_LoRA_128_statistics.json           # LoRA-128 variant
```

---

## 📊 Generated Figures

### 1. Performance Heatmap (`heatmap.png`)

Task-specific accuracy across all model configurations:

- **Rows**: 10 Eval1 biomedical tasks
- **Columns**: Model configurations (Methods A, B, C, Baseline)
- **Color Scale**: Plasma colormap (0.0 = dark purple → 1.0 = yellow)
- **Resolution**: 1000 DPI for publication

### 2. Accuracy Bar Chart (`bar_chart.png`)

Overall accuracy comparison:

- Per-method accuracy percentages
- Error bars for task variance
- Color-coded by method type

### 3. Orthogonality Vectors (`orthogonality_vectors.png`)

Visualization of quantization noise patterns:

- Weight difference distributions
- SVD component analysis
- Orthogonality verification

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- `matplotlib` - Core plotting library
- `numpy` - Numerical computing
- `seaborn` - Statistical visualization

---

## 💻 Usage

### Generate All Visualizations

```bash
cd visualization

# Generate heatmap
python heatmap.py

# Generate bar chart
python bar_chart.py

# Generate orthogonality plots
python other_charts/chart.py
```

### Custom Heatmap

```python
# heatmap.py loads all Final_*_statistics.json files automatically
# To add a new model, create a statistics file with the naming convention:
# Final_<model_name>_statistics.json
```

### Data File Format

Statistics files must follow this structure:

```json
{
  "processed": 433,
  "correct": 312,
  "by_task": {
    "task_name": {
      "total": 45,
      "correct": 38,
      "incorrect": {...}
    }
  }
}
```

---

## 🎨 Customization

### Heatmap Settings

```python
# In heatmap.py
fig, ax = plt.subplots(figsize=(2+len(model_names)*1.25, 1.5+len(all_task_names)*0.65))
im = ax.imshow(heatmap_data, cmap="plasma", vmin=0, vmax=1)  # Colormap
plt.savefig(output_path, dpi=1000)  # Resolution
```

### Model Name Labels

The scripts automatically convert internal names to thesis terminology:

```python
# Automatic label mapping
if "Method A" in name:
    model_names_labels[i] = "Method A"
elif "Method B" in name:
    model_names_labels[i] = "Method B"
elif "Method C" in name:
    model_names_labels[i] = "Method C"
```

---

## 📈 Output Specifications

| Figure | Dimensions | DPI | Format |
|--------|------------|-----|--------|
| heatmap.png | Auto-scaled | 1000 | PNG |
| bar_chart.png | Auto-scaled | 300 | PNG |
| orthogonality_vectors.png | 800x600 | 150 | PNG |
