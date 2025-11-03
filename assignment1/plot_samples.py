import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set publication-quality style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Load the qSamps.dat file
data = np.loadtxt('assignment1/qSamps.dat')

# Extract columns
n_samples = data[:, 0]  # Sample numbers
q1_p = data[:, 1]       # p̄ (max camber location) 
q2_t = data[:, 2]       # t̄ (thickness)

# Design space bounds
P_MIN, P_MAX = 1.5, 5.0
T_MIN, T_MAX = 12, 25

print(f"Loaded {len(data)} samples")
print(f"q₁ (p̄) range: [{q1_p.min():.3f}, {q1_p.max():.3f}]")
print(f"q₂ (t̄) range: [{q2_t.min():.3f}, {q2_t.max():.3f}]")
print(f"Design space coverage: p̄ = {100*(q1_p.max()-q1_p.min())/(P_MAX-P_MIN):.1f}%, t̄ = {100*(q2_t.max()-q2_t.min())/(T_MAX-T_MIN):.1f}%")

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Add subtle grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')

# Main scatter plot with professional styling
scatter = ax.scatter(q1_p, q2_t, c='crimson', s=100, alpha=0.85, 
                    edgecolors='darkred', linewidth=1.8, marker='o',
                    label='LHS Training Points', zorder=5)

# Add sample numbers as simple text annotations
for i, (p, t) in enumerate(zip(q1_p, q2_t)):
    ax.annotate(f'{int(n_samples[i])}', (p, t), xytext=(10, 10), 
                textcoords='offset points', fontsize=8, fontweight='bold',
                color='black', ha='center', va='center')

# Set exact design space boundaries with more padding
ax.set_xlim(P_MIN-0.2, P_MAX+0.2)
ax.set_ylim(T_MIN-1, T_MAX+1)

# Professional labels with mathematical notation
ax.set_xlabel(r'$\bar{p}$ - Location of Maximum Camber', fontweight='bold')
ax.set_ylabel(r'$\bar{t}$ - Airfoil Thickness [%chord]', fontweight='bold')

# Multi-line title with proper spacing
# fig.suptitle('Latin Hypercube Sampling Strategy for NACA Airfoil Design\nOptimization in Two-Dimensional Parameter Space', 
#              fontsize=15, fontweight='bold', y=0.96)

# Professional legend with frame
legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                  shadow=True, framealpha=0.95, edgecolor='black')
legend.get_frame().set_linewidth(1.2)

# Add axis ticks with proper spacing
ax.set_xticks(np.arange(1.5, 5.5, 0.5))
ax.set_yticks(np.arange(12, 26, 2))

# Ensure proper aspect ratio and layout
ax.set_aspect('auto')
plt.tight_layout()
plt.subplots_adjust(top=0.90)

# Save with high quality for publication
plt.savefig('LHS_Sampling_Strategy.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
plt.savefig('LHS_Sampling_Strategy.pdf', bbox_inches='tight', 
           facecolor='white', edgecolor='none')

plt.show()

# Print sample coordinates
print("\nSample coordinates:")
print("Sample#  q1 (p̄)   q2 (t̄)")
print("-" * 25)
for i, (n, p, t) in enumerate(zip(n_samples, q1_p, q2_t)):
    print(f"{int(n):6d}  {p:7.3f}  {t:7.3f}")