"""Generate a visual for LinkedIn post about multi-head attention."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Multi-Head Attention: The Heart of GPT',
        fontsize=24, fontweight='bold', ha='center')

# Input
input_box = FancyBboxPatch((4, 8), 2, 0.6,
                           boxstyle="round,pad=0.1",
                           edgecolor='#2E86AB', facecolor='#A9D6E5',
                           linewidth=2)
ax.add_patch(input_box)
ax.text(5, 8.3, 'Input Embeddings\n[batch, seq, 768]',
        ha='center', va='center', fontsize=13, fontweight='bold')

# Q, K, V Projections
y_pos = 6.5
for i, (label, color) in enumerate([('Q', '#E63946'), ('K', '#F77F00'), ('V', '#06A77D')]):
    x_pos = 2 + i * 2.5
    box = FancyBboxPatch((x_pos - 0.5, y_pos), 1, 0.5,
                         boxstyle="round,pad=0.05",
                         edgecolor=color, facecolor=f'{color}40',
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x_pos, y_pos + 0.25, f'{label}\n[768]',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrow from input
    arrow = FancyArrowPatch((5, 8), (x_pos, y_pos + 0.5),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=1.5, color=color, alpha=0.7)
    ax.add_patch(arrow)

# Multi-Head Split
ax.text(5, 5.5, '↓ Split into 12 heads ↓',
        ha='center', fontsize=14, fontweight='bold', style='italic')

# Attention Heads (show 3 of 12)
heads_y = 4
for i in range(3):
    x_pos = 1.5 + i * 3

    # Head box
    box = FancyBboxPatch((x_pos - 0.6, heads_y - 0.4), 1.2, 0.8,
                         boxstyle="round,pad=0.05",
                         edgecolor='#9D4EDD', facecolor='#E0AAFF',
                         linewidth=2)
    ax.add_patch(box)

    if i == 1:
        ax.text(x_pos, heads_y, f'Head {i+1}\n...\nHead 12',
                ha='center', va='center', fontsize=11)
    else:
        ax.text(x_pos, heads_y, f'Head {i+1}\n[64 dim]',
                ha='center', va='center', fontsize=11)

    # Pattern description below
    patterns = ['Subject-Verb', '...', 'Long-Range']
    ax.text(x_pos, heads_y - 0.8, patterns[i],
            ha='center', fontsize=10, style='italic', color='#5A189A')

# Concatenate
ax.text(5, 2.5, '↓ Concatenate heads ↓',
        ha='center', fontsize=14, fontweight='bold', style='italic')

# Output
output_box = FancyBboxPatch((3.5, 1.2), 3, 0.6,
                           boxstyle="round,pad=0.1",
                           edgecolor='#2E86AB', facecolor='#A9D6E5',
                           linewidth=2)
ax.add_patch(output_box)
ax.text(5, 1.5, 'Context-Aware Output\n[batch, seq, 768]',
        ha='center', va='center', fontsize=13, fontweight='bold')

# Key insight box
insight_box = FancyBboxPatch((0.2, 0.1), 4, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor='#06A77D', facecolor='#D8F3DC',
                            linewidth=2, linestyle='--')
ax.add_patch(insight_box)
ax.text(2.2, 0.5, '💡 Each head learns different\npattern types in parallel!',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Stats box
stats_box = FancyBboxPatch((5.8, 0.1), 4, 0.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='#E63946', facecolor='#FFE5E5',
                          linewidth=2, linestyle='--')
ax.add_patch(stats_box)
ax.text(7.8, 0.5, '📊 GPT-2: 12 heads\n2.36M attention parameters',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Footer
ax.text(5, -0.3, 'Built from scratch following "Build a Large Language Model" by Sebastian Raschka',
        ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig('notes/concepts/multihead_attention_visual.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Visual saved to: notes/concepts/multihead_attention_visual.png")
print("📸 Upload this to your LinkedIn post!")
