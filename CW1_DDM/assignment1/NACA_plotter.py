#!/usr/bin/env python3
"""
Professional NACA Airfoil Plotter
Visualizes optimal NACA airfoil geometries from GPR analysis
"""

import numpy as np
import matplotlib.pyplot as plt

# Set professional style for report
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8
})

# Load airfoil data
print("Loading NACA airfoil data...")
thin_data = np.loadtxt('naca_thin.dat', skiprows=1)
thick_data = np.loadtxt('naca_thick.dat', skiprows=1)

# Extract coordinates
x_thin, y_thin = thin_data[:, 0], thin_data[:, 1]
x_thick, y_thick = thick_data[:, 0], thick_data[:, 1]

print(f"NACA 4112 (Thin): {len(x_thin)} points")
print(f"NACA 4125 (Thick): {len(x_thick)} points")

# Create professional plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# ============================================================================
# Top plot: Both airfoils overlaid for comparison
# ============================================================================
ax = axes[0]

# Plot thin airfoil (optimal for Min CD and Min CD/CL)
ax.plot(x_thin, y_thin, 'b-', linewidth=2.5, label='NACA 4112 (t=12%, p=15%)\nOptimal: Min CD & Min CD/CL', alpha=0.9)
ax.fill(x_thin, y_thin, color='lightblue', alpha=0.3)

# Plot thick airfoil (optimal for Max CL)
ax.plot(x_thick, y_thick, 'r-', linewidth=2.5, label='NACA 4125 (t=25%, p=15%)\nOptimal: Max CL', alpha=0.9)
ax.fill(x_thick, y_thick, color='lightcoral', alpha=0.3)

# Add chord line
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.plot([0, 1], [0, 0], 'k-', linewidth=1.5, alpha=0.7, label='Chord Line')

# Formatting
ax.set_xlabel('x/c')
ax.set_ylabel('y/c')
ax.set_title('Optimal NACA Airfoil Comparison')
ax.legend(loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-0.05, 1.05)

# Add annotations
max_thin = np.max(y_thin)
max_thick = np.max(y_thick)
ax.annotate(f'Max thickness\n{max_thin:.3f}c', 
           xy=(0.15, max_thin), xytext=(0.25, max_thin + 0.05),
           arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
           fontsize=10, color='blue', ha='left')
ax.annotate(f'Max thickness\n{max_thick:.3f}c', 
           xy=(0.15, max_thick), xytext=(0.25, max_thick + 0.08),
           arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
           fontsize=10, color='red', ha='left')

# ============================================================================
# Bottom plots: Individual airfoils with details
# ============================================================================
ax = axes[1]

# Subplot for thin airfoil
ax1 = plt.subplot(2, 2, 3)
ax1.plot(x_thin, y_thin, 'b-', linewidth=2.5, alpha=0.9)
ax1.fill(x_thin, y_thin, color='lightblue', alpha=0.4)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.plot([0, 1], [0, 0], 'k-', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('x/c')
ax1.set_ylabel('y/c')
ax1.set_title('NACA 4112: Thin (Optimal Efficiency)')
ax1.grid(True, alpha=0.3, color='gray', linewidth=0.5)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim(-0.05, 1.05)

# Add performance text
perf_text_thin = (
    'Design Parameters:\n'
    '  p = 1.50 (15% chord)\n'
    '  t = 12.00 (12% chord)\n'
    '  m = 4.00 (4% camber)\n\n'
    'Performance:\n'
    '  CD = 0.1432 ± 0.0038\n'
    '  CL = 0.8995 ± 0.0040\n'
    '  CD/CL = 0.1593 ± 0.0038\n\n'
    'Optimal for:\n'
    '  • Minimum Drag\n'
    '  • Best Efficiency'
)
ax1.text(0.98, 0.05, perf_text_thin, transform=ax1.transAxes,
        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                 edgecolor='blue', alpha=0.8, linewidth=1.5))

# Subplot for thick airfoil
ax2 = plt.subplot(2, 2, 4)
ax2.plot(x_thick, y_thick, 'r-', linewidth=2.5, alpha=0.9)
ax2.fill(x_thick, y_thick, color='lightcoral', alpha=0.4)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.plot([0, 1], [0, 0], 'k-', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('x/c')
ax2.set_ylabel('y/c')
ax2.set_title('NACA 4125: Thick (Maximum Lift)')
ax2.grid(True, alpha=0.3, color='gray', linewidth=0.5)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(-0.05, 1.05)

# Add performance text
perf_text_thick = (
    'Design Parameters:\n'
    '  p = 1.50 (15% chord)\n'
    '  t = 25.00 (25% chord)\n'
    '  m = 4.00 (4% camber)\n\n'
    'Performance:\n'
    '  CD = 0.2072 ± 0.0039\n'
    '  CL = 0.9501 ± 0.0042\n'
    '  CD/CL = 0.2180 ± 0.0039\n\n'
    'Optimal for:\n'
    '  • Maximum Lift\n'
    '  • Takeoff/Landing'
)
ax2.text(0.98, 0.05, perf_text_thick, transform=ax2.transAxes,
        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', 
                 edgecolor='darkred', alpha=0.8, linewidth=1.5))

# Overall title
fig.suptitle('Optimal NACA Airfoil Geometries from GPR Analysis', 
            fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('NACA_airfoil_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('NACA_airfoil_comparison.pdf', bbox_inches='tight', facecolor='white')
print("\nSaved: NACA_airfoil_comparison.png")
print("Saved: NACA_airfoil_comparison.pdf")

# ============================================================================
# Create individual high-quality plots
# ============================================================================

# Thin airfoil only
fig2, ax = plt.subplots(figsize=(12, 5))
ax.plot(x_thin, y_thin, 'b-', linewidth=3, alpha=0.9)
ax.fill(x_thin, y_thin, color='lightblue', alpha=0.4)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
ax.plot([0, 1], [0, 0], 'k-', linewidth=2, alpha=0.7)
ax.set_xlabel('x/c', fontsize=13)
ax.set_ylabel('y/c', fontsize=13)
ax.set_title('NACA: m=4%, p=15%, t=13%', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('NACA_4112_thin.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: NACA_4112_thin.png")

# Thick airfoil only
fig3, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_thick, y_thick, 'r-', linewidth=3, alpha=0.9)
ax.fill(x_thick, y_thick, color='lightcoral', alpha=0.4)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
ax.plot([0, 1], [0, 0], 'k-', linewidth=2, alpha=0.7)
ax.set_xlabel('x/c', fontsize=13)
ax.set_ylabel('y/c', fontsize=13)
ax.set_title('NACA: m=4%, p=15%, t=24%', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('NACA_4125_thick.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: NACA_4125_thick.png")

plt.show()

print("\n" + "=" * 80)
print("NACA AIRFOIL VISUALIZATION COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. NACA_airfoil_comparison.png - Comprehensive comparison plot")
print("  2. NACA_airfoil_comparison.pdf - PDF version for reports")
print("  3. NACA_4112_thin.png - Individual thin airfoil plot")
print("  4. NACA_4125_thick.png - Individual thick airfoil plot")
print("\nKey Findings:")
print("  • NACA 4112 (thin): Optimal for efficiency (Min CD/CL)")
print("  • NACA 4125 (thick): Optimal for lift generation (Max CL)")
print("  • Both use forward camber position (p=15%)")
print("  • Thickness is the primary design variable for performance trade-offs")
print("=" * 80)
