import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

# Directory containing JSON files
current_dir = os.path.dirname(os.path.abspath(__file__))

# Dynamically grab all relevant files in this folder
json_files = []
for filename in os.listdir(current_dir):
    if (
        filename.startswith("Final_")
        and filename.endswith("_statistics.json")
    ):
        json_files.append(filename)

model_names = []
all_task_names = set()
task_accuracies = {}

# Gather accuracies
for f in json_files:
    fname = os.path.join(current_dir, f)
    with open(fname, "r") as pf:
        stats = json.load(pf)
        model_name = f.replace("Final_", "").replace("_statistics.json", "")
        model_names.append(model_name)
        task_accuracies[model_name] = {}
        for task, tdata in stats["by_task"].items():
            all_task_names.add(task)
            correct = tdata["correct"]
            total = tdata["total"]
            accuracy = correct / total if total > 0 else 0
            task_accuracies[model_name][task] = accuracy

all_task_names = sorted(all_task_names)

# Prepare heatmap data: rows=task, cols=model
heatmap_data = []
for task in all_task_names:
    row = []
    for model in model_names:
        acc = task_accuracies[model].get(task, np.nan)
        row.append(acc)
    heatmap_data.append(row)
heatmap_data = np.array(heatmap_data)

# Replace underscores with spaces and format into multiple lines
model_names_labels = []
for name in model_names:
    # Split by parentheses and plus signs, format into multiple lines
    parts = name.replace("_", " ").replace("(", "\n(").replace(") ", ")\n").replace(" + ", "\n+ ")
    model_names_labels.append(parts)


# Standardize model labels: If contains "Method A" "Method B" or "Method C", shorten to that
for i, name in enumerate(model_names_labels):
    if "Method A" in name:
        model_names_labels[i] = "Method A"
    elif "Method B" in name:
        model_names_labels[i] = "Method B"
    elif "Method C" in name:
        model_names_labels[i] = "Method C"
print(model_names_labels)
task_names_labels = [task.replace("_", " ") for task in all_task_names]

fig, ax = plt.subplots(figsize=(2+len(model_names)*1.25, 1.5+len(all_task_names)*0.65))

# Use Viridis colormap (dark purple to yellow)
im = ax.imshow(heatmap_data, cmap="plasma", vmin=0, vmax=1)

# Grid-like appearance
# Increase font size for x-axis and use multi-line labels
ax.set_xticks(np.arange(len(model_names)), labels=model_names_labels, rotation=45, ha='center', fontsize=8)
ax.set_yticks(np.arange(len(all_task_names)), labels=task_names_labels, fontsize=8)

# Set minor ticks for a grid
ax.set_xticks(np.arange(-.5, len(model_names), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(all_task_names), 1), minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

# Annotate cells
for i in range(len(all_task_names)):
    for j in range(len(model_names)):
        val = heatmap_data[i, j]
        if not np.isnan(val):
            color = "black" if 0.14 < val < 0.86 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight='bold')
        else:
            ax.text(j, i, "-", ha="center", va="center", color="grey", fontsize=8)

# Axis labels & title
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Task", fontsize=12)
ax.set_title("Task-Specific Performance Heatmap", fontsize=14)

# Custom colorbar with 0=red and 1=green, label as Reward (like the image)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Reward", rotation=270, labelpad=14)
cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))

plt.tight_layout()
# Add extra bottom margin for multi-line x-axis labels
plt.subplots_adjust(bottom=0.15)
output_path = os.path.join(current_dir, "heatmap.png")
plt.savefig(output_path, dpi=1000)
plt.close(fig)
