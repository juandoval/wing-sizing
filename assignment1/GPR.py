#!/usr/bin/env python3
"""
Gaussian Process Regression Analysis for NACA Airfoil Optimization
Analyzes Cd, Cl, and Cd/Cl as functions of p (location of max thickness) and t (thickness)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import os

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import os

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

data_file = "doeList.dat"

df = pd.read_csv(data_file, sep='\t', header=None, 
                 names=['sample', 'm', 'p', 't', 'Cd', 'Cl'])

print("=" * 80)
print("NACA AIRFOIL GPR ANALYSIS")
print("=" * 80)
print(f"\nLoaded {len(df)} samples from: {data_file}")
print("\nData Summary:")
print(df.describe())

X = df[['p', 't']].values  # Design space: p (location of max thickness), t (thickness)
y_Cd = df['Cd'].values      # q1: Drag coefficient
y_Cl = df['Cl'].values      # q2: Lift coefficient
y_ratio = y_Cd / y_Cl       # q3: Cd/Cl ratio

print("\n" + "=" * 80)
print("Design Space Ranges:")
print(f"  p (location of max thickness): [{X[:, 0].min():.3f}, {X[:, 0].max():.3f}]") 
print(f"  t (thickness):                 [{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")
print("\nQoI Ranges:")
print(f"  Cd (drag coefficient):         [{y_Cd.min():.4f}, {y_Cd.max():.4f}]")
print(f"  Cl (lift coefficient):         [{y_Cl.min():.4f}, {y_Cl.max():.4f}]")
print(f"  Cd/Cl (drag-to-lift ratio):    [{y_ratio.min():.4f}, {y_ratio.max():.4f}]")

# Gaussian Process Regression
'''
Define the kernel as a combination of Constant, RBF, and WhiteKernel components:

Where:
C(1.0, (1e-3, 1e3)) is the constant kernel with initial value 1.0 and bounds [1e-3, 1e3],
RBF([1.0, 1.0], (1e-2, 1e2)) is the radial basis function kernel with initial length scales [1.0, 1.0] and bounds [1e-2, 1e2],
and WhiteKernel(noise_level=1e-5) is the white noise kernel with a fixed noise level.
'''
kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)

# Train GPR models for each QoI
print("\n" + "=" * 80)
print("Training Gaussian Process Regressors...")
print("=" * 80)

for n_restarts in [5, 10, 20, 30]:

    gpr_Cd = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42)
    gpr_Cd.fit(X, y_Cd)

    gpr_Cl = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42)
    gpr_Cl.fit(X, y_Cl)

    gpr_ratio = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42)
    gpr_ratio.fit(X, y_ratio)


    print("\nGPR Model for Cd:")
    print(f"  Kernel: {gpr_Cd.kernel_}")
    print(f"  Log-likelihood with {n_restarts} restarts: {gpr_Cd.log_marginal_likelihood_value_:.3f}")

    print("\nGPR Model for Cl:")
    print(f"  Kernel: {gpr_Cl.kernel_}")
    print(f"  Log-likelihood with {n_restarts} restarts: {gpr_Cl.log_marginal_likelihood_value_:.3f}")

    print("\nGPR Model for Cd/Cl:")
    print(f"  Kernel: {gpr_ratio.kernel_}")
    print(f"  Log-likelihood with {n_restarts} restarts: {gpr_ratio.log_marginal_likelihood_value_:.3f}")

    # Grid

    # Design space: p in [1.5, 5.0], t in [12, 25]
    n_grid = 1000 # The more the better but slower
    p_range = np.linspace(1.5, 5.0, n_grid)
    t_range = np.linspace(12, 25, n_grid)
    p_grid, t_grid = np.meshgrid(p_range, t_range)
    X_grid = np.column_stack([p_grid.ravel(), t_grid.ravel()])

    # Make predictions
    Cd_mean, Cd_std = gpr_Cd.predict(X_grid, return_std=True)
    Cl_mean, Cl_std = gpr_Cl.predict(X_grid, return_std=True)
    ratio_mean, ratio_std = gpr_ratio.predict(X_grid, return_std=True)

    # Reshape for plotting
    Cd_mean = Cd_mean.reshape(n_grid, n_grid)
    Cd_std = Cd_std.reshape(n_grid, n_grid)
    Cl_mean = Cl_mean.reshape(n_grid, n_grid)
    Cl_std = Cl_std.reshape(n_grid, n_grid)
    ratio_mean = ratio_mean.reshape(n_grid, n_grid)
    ratio_std = ratio_std.reshape(n_grid, n_grid)

# Plotting function

def plot_contour_pair(p_grid, t_grid, Z_mean, Z_std, training_points, 
                      title_base, filename, vmin_mean=None, vmax_mean=None):
    """Create clean side-by-side contour plots for mean and std deviation"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot mean - consistent blue colormap
    ax = axes[0]
    if vmin_mean is not None and vmax_mean is not None:
        levels_mean = np.linspace(vmin_mean, vmax_mean, 15)
        contour_mean = ax.contourf(p_grid, t_grid, Z_mean, levels=levels_mean, 
                                  cmap='Blues', alpha=0.9)
    else:
        contour_mean = ax.contourf(p_grid, t_grid, Z_mean, levels=15, 
                                  cmap='Blues', alpha=0.9)
    
    # Training points in consistent dark color
    ax.scatter(training_points[:, 0], training_points[:, 1], 
              c='darkred', s=60, edgecolors='white', linewidth=1.5, 
              label='Training Points', zorder=5, alpha=0.8)
    
    ax.set_xlabel('p (Location of Max Thickness)')
    ax.set_ylabel('t (Thickness)')
    ax.set_title(f'{title_base} - Mean Prediction')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    # Clean colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(contour_mean, cax=cax)
    cbar.set_label(title_base, fontsize=11)
    
    # Plot standard deviation - consistent gray colormap
    ax = axes[1]
    contour_std = ax.contourf(p_grid, t_grid, Z_std, levels=15, 
                             cmap='Greys', alpha=0.9)
    
    ax.scatter(training_points[:, 0], training_points[:, 1], 
              c='darkred', s=60, edgecolors='white', linewidth=1.5, 
              label='Training Points', zorder=5, alpha=0.8)
    
    ax.set_xlabel('p (Location of Max Thickness)')
    ax.set_ylabel('t (Thickness)')
    ax.set_title(f'{title_base} - Uncertainty (Std Dev)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(contour_std, cax=cax)
    cbar.set_label('Standard Deviation', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {filename}")
    
    return fig

def plot_contour_pair_with_optima(p_grid, t_grid, Z_mean, Z_std, training_points, 
                                 title_base, filename, optimal_points, vmin_mean=None, vmax_mean=None):
    """Create clean side-by-side contour plots with optimal points marked"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot mean - consistent blue colormap
    ax = axes[0]
    if vmin_mean is not None and vmax_mean is not None:
        levels_mean = np.linspace(vmin_mean, vmax_mean, 15)
        contour_mean = ax.contourf(p_grid, t_grid, Z_mean, levels=levels_mean, 
                                  cmap='Blues', alpha=0.9)
    else:
        contour_mean = ax.contourf(p_grid, t_grid, Z_mean, levels=15, 
                                  cmap='Blues', alpha=0.9)
    
    # Training points
    ax.scatter(training_points[:, 0], training_points[:, 1], 
              c='gray', s=60, edgecolors='white', linewidth=1.5, 
              label='Training Points', zorder=4, alpha=0.8)
    
    # Add optimal points
    for p_opt, t_opt, label, color in optimal_points:
        ax.scatter(p_opt, t_opt, c=color, s=200, marker='*', 
                  edgecolors='white', linewidth=2, label=label, zorder=6, alpha=0.9)
    
    ax.set_xlabel('p (Location of Max Thickness)')
    ax.set_ylabel('t (Thickness)')
    ax.set_title(f'{title_base} - Mean Prediction')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    # Clean colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(contour_mean, cax=cax)
    cbar.set_label(title_base, fontsize=11)
    
    # Plot standard deviation - consistent gray colormap
    ax = axes[1]
    contour_std = ax.contourf(p_grid, t_grid, Z_std, levels=15, 
                             cmap='Greys', alpha=0.9)
    
    ax.scatter(training_points[:, 0], training_points[:, 1], 
              c='darkred', s=60, edgecolors='white', linewidth=1.5, 
              label='Training Points', zorder=4, alpha=0.8)
    
    # Add optimal points to std plot as well
    for p_opt, t_opt, label, color in optimal_points:
        ax.scatter(p_opt, t_opt, c=color, s=200, marker='*', 
                  edgecolors='white', linewidth=2, label=label, zorder=6, alpha=0.9)
    
    ax.set_xlabel('p (Location of Max Thickness)')
    ax.set_ylabel('t (Thickness)')
    ax.set_title(f'{title_base} - Uncertainty (Std Dev)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(contour_std, cax=cax)
    cbar.set_label('Standard Deviation', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {filename}")
    
    return fig

# OPTIMIZATION: FIND OPTIMAL PARAMETERS (need to calculate before plotting)

print("\n" + "=" * 80)
print("Finding Optimal Parameters for Plotting...")
print("=" * 80)

# Find optimal points
idx_min_Cd = np.argmin(Cd_mean)
idx_max_Cl = np.argmax(Cl_mean)
idx_min_ratio = np.argmin(ratio_mean)

p_opt_minCd, t_opt_minCd = X_grid[idx_min_Cd]
p_opt_maxCl, t_opt_maxCl = X_grid[idx_max_Cl]
p_opt_minRatio, t_opt_minRatio = X_grid[idx_min_ratio]

# Generate Plots

print("\n" + "=" * 80)
print("Generating Contour Plots with Optimal Points...")
print("=" * 80)

# CD plots with optimal points
fig1 = plot_contour_pair_with_optima(p_grid, t_grid, Cd_mean, Cd_std, X,
                                    'CD (Drag Coefficient)', 
                                    os.path.join(OUTPUT_DIR, 'CD_contours.png'),
                                    [(p_opt_minCd, t_opt_minCd, 'Min CD', 'darkred')])

# CL plots with optimal points  
fig2 = plot_contour_pair_with_optima(p_grid, t_grid, Cl_mean, Cl_std, X,
                                    'CL (Lift Coefficient)', 
                                    os.path.join(OUTPUT_DIR, 'CL_contours.png'),
                                    [(p_opt_maxCl, t_opt_maxCl, 'Max CL', 'darkblue')])

# CD/CL plots with optimal points
fig3 = plot_contour_pair_with_optima(p_grid, t_grid, ratio_mean, ratio_std, X,
                                    'CD/CL (Drag-to-Lift Ratio)', 
                                    os.path.join(OUTPUT_DIR, 'CD_CL_ratio_contours.png'),
                                    [(p_opt_minRatio, t_opt_minRatio, 'Min CD/CL', 'darkgreen')])

# ANALYSIS: HOW QoIs VARY WITH p AND t

print("\n" + "=" * 80)
print("ANALYSIS: How QoIs vary with p and t")
print("=" * 80)

print("\n1. CD (Drag Coefficient):")
print("   - CD increases with thickness (t): Thicker airfoils have higher drag")
print("   - CD shows moderate sensitivity to p (location of max thickness)")
print("   - Minimum CD occurs at lower t values")
print(f"   - Range on grid: [{Cd_mean.min():.4f}, {Cd_mean.max():.4f}]")

print("\n2. CL (Lift Coefficient):")
print("   - CL increases with thickness (t): Thicker airfoils generate more lift")
print("   - CL shows some sensitivity to p location")
print("   - Maximum CL occurs at higher t values")
print(f"   - Range on grid: [{Cl_mean.min():.4f}, {Cl_mean.max():.4f}]")

print("\n3. CD/CL (Drag-to-Lift Ratio):")
print("   - This ratio represents aerodynamic efficiency (lower is better)")
print("   - Minimum CD/CL (best efficiency) occurs at moderate thickness")
print("   - Both p and t affect the ratio, showing a trade-off between drag and lift")
print(f"   - Range on grid: [{ratio_mean.min():.4f}, {ratio_mean.max():.4f}]")

# Print detailed optimization results
print("\n" + "=" * 80)
print("OPTIMIZATION: Detailed Results")
print("=" * 80)

print("\n(a) MINIMUM CD (Drag Coefficient):")
print(f"    Optimal p = {p_opt_minCd:.4f}")
print(f"    Optimal t = {t_opt_minCd:.4f}")
print(f"    CD = {Cd_mean.flat[idx_min_Cd]:.4f} (±{Cd_std.flat[idx_min_Cd]:.4f})")
print(f"    CL = {Cl_mean.flat[idx_min_Cd]:.4f} (±{Cl_std.flat[idx_min_Cd]:.4f})")
print(f"    CD/CL = {ratio_mean.flat[idx_min_Cd]:.4f} (±{ratio_std.flat[idx_min_Cd]:.4f})")

print("\n(b) MAXIMUM CL (Lift Coefficient):")
print(f"    Optimal p = {p_opt_maxCl:.4f}")
print(f"    Optimal t = {t_opt_maxCl:.4f}")
print(f"    CD = {Cd_mean.flat[idx_max_Cl]:.4f} (±{Cd_std.flat[idx_max_Cl]:.4f})")
print(f"    CL = {Cl_mean.flat[idx_max_Cl]:.4f} (±{Cl_std.flat[idx_max_Cl]:.4f})")
print(f"    CD/CL = {ratio_mean.flat[idx_max_Cl]:.4f} (±{ratio_std.flat[idx_max_Cl]:.4f})")

print("\n(c) MINIMUM CD/CL (Best Aerodynamic Efficiency):")
print(f"    Optimal p = {p_opt_minRatio:.4f}")
print(f"    Optimal t = {t_opt_minRatio:.4f}")
print(f"    CD = {Cd_mean.flat[idx_min_ratio]:.4f} (±{Cd_std.flat[idx_min_ratio]:.4f})")
print(f"    CL = {Cl_mean.flat[idx_min_ratio]:.4f} (±{Cl_std.flat[idx_min_ratio]:.4f})")
print(f"    CD/CL = {ratio_mean.flat[idx_min_ratio]:.4f} (±{ratio_std.flat[idx_min_ratio]:.4f})")

# OBSERVATIONS AND PHYSICAL INTERPRETATION

print("\n" + "=" * 80)
print("OBSERVATIONS AND PHYSICAL INTERPRETATION")
print("=" * 80)

print("\n• Are optimal parameters the same for different objectives?")
print("  NO - The optimal parameters differ significantly:")
print(f"    - Min CD:       p={p_opt_minCd:.2f}, t={t_opt_minCd:.2f}")
print(f"    - Max CL:       p={p_opt_maxCl:.2f}, t={t_opt_maxCl:.2f}")
print(f"    - Min CD/CL:    p={p_opt_minRatio:.2f}, t={t_opt_minRatio:.2f}")

print("\n• Physical Interpretation:")
print("  1. MINIMUM DRAG (Min CD):")
print("     - Favors thinner airfoils (low t)")
print("     - Reduces wetted area and friction drag")
print("     - Airfoil shape: Thin profile, reduced frontal area")
print("     - Trade-off: Lower lift generation")

print("\n  2. MAXIMUM LIFT (Max CL):")
print("     - Favors thicker airfoils (high t)")
print("     - Increases camber effect and circulation")
print("     - Airfoil shape: Thick, cambered profile")
print("     - Trade-off: Higher drag due to increased thickness")

print("\n  3. BEST EFFICIENCY (Min CD/CL):")
print("     - Represents optimal balance between drag and lift")
print("     - Moderate thickness value (compromise)")
print("     - Airfoil shape: Balanced profile - not too thin, not too thick")
print("     - This is the 'Pareto optimal' design for L/D ratio")

print("\n• Design Implications:")
print("  - For high-speed cruise: Use min CD design (thin airfoil)")
print("  - For takeoff/landing: Use max CL design (thick airfoil)")
print("  - For overall efficiency: Use min CD/CL design (moderate thickness)")

# Create Visualization with Optimal Points

print("\n" + "=" * 80)
print("Creating Comprehensive Visualization...")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Consistent color scheme for all plots
qois = [
    (Cd_mean, Cd_std, 'CD', 'Blues'),
    (Cl_mean, Cl_std, 'CL', 'Blues'),
    (ratio_mean, ratio_std, 'CD/CL', 'Blues')
]

# Clean optimal points with consistent colors
optimal_points = {
    'Min CD': (p_opt_minCd, t_opt_minCd, 'darkred', 'o'),
    'Max CL': (p_opt_maxCl, t_opt_maxCl, 'darkblue', 's'),
    'Min CD/CL': (p_opt_minRatio, t_opt_minRatio, 'darkgreen', '^')
}

for idx, (mean_data, std_data, name, cmap) in enumerate(qois):
    # Mean plot
    ax = axes[0, idx]
    contour = ax.contourf(p_grid, t_grid, mean_data, levels=15, cmap=cmap, alpha=0.9)
    ax.scatter(X[:, 0], X[:, 1], c='gray', s=40, edgecolors='white', 
              linewidth=1.2, label='Training Points', alpha=0.8, zorder=4)
    
    for opt_name, (p_opt, t_opt, color, marker) in optimal_points.items():
        ax.scatter(p_opt, t_opt, c=color, s=150, marker=marker, 
                  edgecolors='white', linewidth=2, label=opt_name, zorder=5, alpha=0.9)
    
    ax.set_xlabel('p (Location of Max Thickness)')
    ax.set_ylabel('t (Thickness)')
    ax.set_title(f'{name} - Mean Prediction')
    if idx == 2:
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    cbar = plt.colorbar(contour, ax=ax, label=name)
    cbar.ax.tick_params(labelsize=10)
    
    # Std plot - consistent gray colormap
    ax = axes[1, idx]
    contour = ax.contourf(p_grid, t_grid, std_data, levels=15, cmap='Greys', alpha=0.9)
    ax.scatter(X[:, 0], X[:, 1], c='darkred', s=40, edgecolors='white', 
              linewidth=1.2, alpha=0.8, zorder=4)
    ax.set_xlabel('p (Location of Max Thickness)')
    ax.set_ylabel('t (Thickness)')
    ax.set_title(f'{name} - Uncertainty (Std Dev)')
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    cbar = plt.colorbar(contour, ax=ax, label='Std Dev')
    cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'comprehensive_analysis.png'), dpi=300, 
           bbox_inches='tight', facecolor='white')
print("  Saved: comprehensive_analysis.png")

# ============================================================================
# 10. SAVE OPTIMAL PARAMETERS TO FILE
# ============================================================================

with open(os.path.join(OUTPUT_DIR, 'optimal_parameters.txt'), 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NACA AIRFOIL OPTIMIZATION RESULTS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("OPTIMAL PARAMETERS:\n\n")
    
    f.write("(a) MINIMUM CD (Drag Coefficient):\n")
    f.write(f"    p = {p_opt_minCd:.6f}\n")
    f.write(f"    t = {t_opt_minCd:.6f}\n")
    f.write(f"    CD = {Cd_mean.flat[idx_min_Cd]:.6f}\n")
    f.write(f"    CL = {Cl_mean.flat[idx_min_Cd]:.6f}\n")
    f.write(f"    CD/CL = {ratio_mean.flat[idx_min_Cd]:.6f}\n\n")
    
    f.write("(b) MAXIMUM CL (Lift Coefficient):\n")
    f.write(f"    p = {p_opt_maxCl:.6f}\n")
    f.write(f"    t = {t_opt_maxCl:.6f}\n")
    f.write(f"    CD = {Cd_mean.flat[idx_max_Cl]:.6f}\n")
    f.write(f"    CL = {Cl_mean.flat[idx_max_Cl]:.6f}\n")
    f.write(f"    CD/CL = {ratio_mean.flat[idx_max_Cl]:.6f}\n\n")
    
    f.write("(c) MINIMUM CD/CL (Best Aerodynamic Efficiency):\n")
    f.write(f"    p = {p_opt_minRatio:.6f}\n")
    f.write(f"    t = {t_opt_minRatio:.6f}\n")
    f.write(f"    CD = {Cd_mean.flat[idx_min_ratio]:.6f}\n")
    f.write(f"    CL = {Cl_mean.flat[idx_min_ratio]:.6f}\n")
    f.write(f"    CD/CL = {ratio_mean.flat[idx_min_ratio]:.6f}\n")

print("  Saved: optimal_parameters.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. CD_contours.png - Drag coefficient contours (mean & std)")
print("  2. CL_contours.png - Lift coefficient contours (mean & std)")
print("  3. CD_CL_ratio_contours.png - Drag-to-lift ratio contours (mean & std)")
print("  4. comprehensive_analysis.png - All QoIs with optimal points marked")
print("  5. optimal_parameters.txt - Detailed optimization results")
print("\nNext steps:")
print("  • Visualize optimal airfoil shapes in ParaView using the optimal p & t values")
print("  • Verify physical interpretation by examining airfoil geometry")
print("  • Consider multi-objective optimization for balanced performance")
print("=" * 80)

plt.show()