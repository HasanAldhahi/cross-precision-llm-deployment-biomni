import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Directory containing JSON stats files (current directory)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Gather all relevant statistics files
json_files = []
for filename in os.listdir(current_dir):
    if filename.startswith("Final_") and filename.endswith("_statistics.json"):
        json_files.append(filename)

# Extract average accuracies for each model
model_scores = {}
for f in json_files:
    fname = os.path.join(current_dir, f)
    with open(fname, "r") as pf:
        stats = json.load(pf)
        model_name = f.replace("Final_", "").replace("_statistics.json", "")
        task_accuracies = []
        for tdata in stats["by_task"].values():
            correct = tdata["correct"]
            total = tdata["total"]
            accuracy = correct / total if total > 0 else 0
            task_accuracies.append(accuracy)
        if len(task_accuracies) > 0:
            avg_score = sum(task_accuracies) / len(task_accuracies)
        else:
            avg_score = 0
        model_scores[model_name] = avg_score

# Sort models by value for consistent bar order (optionally adjust manually)
sorted_names = sorted(model_scores, key=model_scores.get)
scores = [model_scores[name] for name in sorted_names]

# Format pretty labels (replace _ with space)
labels = [name.replace("_", " ") for name in sorted_names]

fig, ax = plt.subplots(figsize=(max(9, 2 + len(labels)*1.7), 5))  # Wider for longer labels

bar_colors = [
    "#7BCC6A",    # light green
    "#E87FD3",    # pink
    "#F56D84",    # coral
    "#B39159",    # brown/gold
    "#5786C9",    # blue
    "#F9B66C",    # orange
]
# Extend or trim colors based on count
while len(bar_colors) < len(scores):
    bar_colors.append("#888888")
bar_colors = bar_colors[:len(scores)]

bars = ax.bar(range(len(labels)), scores, color=bar_colors, edgecolor="black", zorder=3)

# Annotate values inside or atop each bar
for idx, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.025,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )

# Set proper x ticks for long labels
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(
    labels,
    rotation=25,        # Angled for readability
    ha='right',
    fontsize=9,         # Small font for dense/long labels
    wrap=True           # Wrap long labels
)

# Reduce bottom margin for very long labels to fit
plt.subplots_adjust(bottom=0.23)

# Example improvement annotation as in the provided image, if two models are 'before' and 'after'
if len(scores) >= 2:
    for i, label in enumerate(labels):
        if "8B" in label and "After" not in label:
            for j, lab2 in enumerate(labels):
                if (lab2 != label) and ("8B" in lab2):
                    v0 = scores[i]
                    v1 = scores[j]
                    rel_change = ((v1-v0)/v0)*100 if v0 > 0 else 0
                    if rel_change > 0:
                        ax.annotate(
                            f"+{rel_change:.1f}%",
                            xy=((i+j)/2, max(v0,v1)+0.07),
                            xytext=((i+j)/2, max(v0,v1)+0.13),
                            textcoords="data",
                            ha='center',
                            va='bottom',
                            fontsize=12,
                            color='#D32424',
                            fontweight="bold",
                            arrowprops=dict(arrowstyle='-|>', lw=1, color="#D32424"),
                        )
                    break
            break

# Axis labels and formatting
ax.set_ylim(0, max(scores)+0.25)
ax.set_ylabel("Average Performance Across Tasks", fontsize=13)
ax.set_xlabel("Model", fontsize=13, labelpad=18)
ax.set_title("Average Task Performance (Agent Scaffold & E1 Environment)", fontsize=15, pad=18)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", linewidth=1, alpha=0.5)

# Save the figure as a PNG file in the current directory
output_path = os.path.join(current_dir, "bar_chart.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close(fig)
