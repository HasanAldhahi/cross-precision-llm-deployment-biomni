import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

# Set global style for academic poster
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})
colors = {'darkblue': '#254E71', 'lightblue': '#00ADEE', 'gray': '#808080', 'orange': '#FF8C00', 'red': '#E04F5F'}

def create_performance_chart():
    """Figure 1: Comparison of Methods"""
    methods = ['Baseline (BF16)', 'Method A\n(Naive Transfer)', 'Method C\n(Direct Quant)', 'Method B\n(Corrective)']
    scores = [44.5, 44.6, 40.9, 29.5]
    bar_colors = [colors['gray'], colors['orange'], colors['lightblue'], colors['red']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, scores, color=bar_colors, edgecolor='black', linewidth=1.5)
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}%', ha='center', va='bottom', fontsize=16, fontweight='bold')
        
    ax.set_ylim(0, 55)
    ax.set_ylabel('Accuracy (%) on Eval1 Benchmark', fontsize=14, fontweight='bold')
    ax.set_title('Method A Matches Full Precision Baseline', fontsize=18, fontweight='bold', pad=20)
    
    # Add dashed line for baseline
    ax.axhline(y=44.5, color='black', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(3.3, 45, 'Baseline Level', va='center', fontsize=12, fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig('performance_chart.png', dpi=300)
    plt.close()

def create_orthogonality_plot():
    """Figure 2: The Magnitude Paradox (3D Vector)"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Vectors: Noise is large, Signal is small but orthogonal
    origin = [0, 0, 0]
    
    # Noise Vector (Large Magnitude)
    noise = np.array([8, 2, 1]) 
    
    # Signal Vector (Small Magnitude, orthogonal to noise)
    # Finding a vector orthogonal to noise:
    signal = np.array([-2, 8, 0]) 
    signal = signal / np.linalg.norm(signal) * 1.5 # Scale down significantly
    
    ax.quiver(*origin, *noise, color=colors['red'], arrow_length_ratio=0.1, linewidth=3, label='Quantization Noise (||N||)')
    ax.quiver(*origin, *signal, color=colors['lightblue'], arrow_length_ratio=0.2, linewidth=3, label='LoRA Signal (||L||)')
    
    # Limits
    ax.set_xlim([-2, 10])
    ax.set_ylim([-2, 10])
    ax.set_zlim([-2, 5])
    
    ax.set_xlabel('Dimension X')
    ax.set_ylabel('Dimension Y')
    ax.set_zlabel('Dimension Z')
    
    # Add text
    ax.text(8, 2, 1, 'Noise (High Rank)\n||N|| ≈ 611x ||L||', color=colors['red'], fontsize=12)
    ax.text(-2, 8, 0, 'Signal (Low Rank)\nEncodes Knowledge', color=colors['darkblue'], fontsize=12)
    
    ax.set_title('The Magnitude Paradox:\nSignal is Orthogonal to Noise', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # View angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('orthogonality_vectors.png', dpi=300)
    plt.close()

def create_heatmap():
    """Figure 3: Task Breakdown"""
    # Data approximation from thesis
    tasks = ['CRISPR', 'GWAS Catalog', 'Rare Disease', 'Patient Gene', 'Lab Bench']
    data = np.array([
        [30, 20, 20, 30], # CRISPR
        [36, 18, 42, 36], # GWAS
        [37, 10, 10, 37], # Rare Disease
        [28, 12, 24, 32], # Patient Gene
        [54, 46, 52, 58]  # Lab Bench
    ])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Method A', 'Method B', 'Method C', 'Baseline'],
                yticklabels=tasks, cbar_kws={'label': 'Accuracy (%)'}, ax=ax, annot_kws={"size": 14})
    
    ax.set_title('Performance Consistency Across Domains', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('heatmap.png', dpi=300)
    plt.close()

def create_architecture_diagram():
    """Figure 4: The System Stack"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw FP8 Base
    rect_base = patches.Rectangle((0.1, 0.1), 0.8, 0.3, linewidth=2, edgecolor=colors['darkblue'], facecolor=colors['gray'], alpha=0.3)
    ax.add_patch(rect_base)
    ax.text(0.5, 0.25, 'Base Model (Qwen-32B)\nFormat: FP8 (Resident Memory)', 
            ha='center', va='center', fontsize=14, fontweight='bold', color=colors['darkblue'])
    ax.text(0.5, 0.15, 'Size: 32 GB', ha='center', va='center', fontsize=12)

    # Draw Adapters (Hot swappable)
    rect_bio = patches.Rectangle((0.1, 0.5), 0.2, 0.2, linewidth=2, edgecolor=colors['orange'], facecolor=colors['orange'], alpha=0.6)
    ax.add_patch(rect_bio)
    ax.text(0.2, 0.6, 'Biomedical\nAdapter\n(BF16)', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    rect_phys = patches.Rectangle((0.4, 0.5), 0.2, 0.2, linewidth=2, edgecolor=colors['lightblue'], facecolor=colors['lightblue'], alpha=0.6)
    ax.add_patch(rect_phys)
    ax.text(0.5, 0.6, 'Physics\nAdapter\n(BF16)', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    rect_code = patches.Rectangle((0.7, 0.5), 0.2, 0.2, linewidth=2, edgecolor='green', facecolor='green', alpha=0.6)
    ax.add_patch(rect_code)
    ax.text(0.8, 0.6, 'Coding\nAdapter\n(BF16)', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Arrows
    ax.annotate('', xy=(0.2, 0.42), xytext=(0.2, 0.48), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', xy=(0.5, 0.42), xytext=(0.5, 0.48), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('', xy=(0.8, 0.42), xytext=(0.8, 0.48), arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax.text(0.5, 0.85, 'Cross-Precision Transfer Architecture', ha='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.45, 'Dynamic Loading (The "Bridge")', ha='center', fontsize=12, fontstyle='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('architecture.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    create_performance_chart()
    create_orthogonality_plot()
    create_heatmap()
    create_architecture_diagram()
    print("All images generated successfully.")
    