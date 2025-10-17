"""
UAV Aerodynamics Design Calculator
Preliminary sizing and performance analysis for cardboard UAV
Includes parametric sweep and visualization capabilities

STRUCTURE:
1. Imports and Constants
2. Helper Functions (input, calculations)
3. Batch Mode Function
4. Parametric Sweep Functions
5. Plotting Functions
6. Main Analysis Function
"""

import math
import csv
import os
import numpy as np

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Note: matplotlib not found. Plotting features disabled.")
    print("Install with: pip install matplotlib\n")

# ============================================================================
# SECTION 1: CONSTANTS
# ============================================================================
g = 9.81  # m/s^2 - Gravitational acceleration
rho = 1.225  # kg/m^3 - Air density at sea level
nu = 1.46e-5  # m^2/s - Kinematic viscosity of air

# ============================================================================
# SECTION 2: HELPER FUNCTIONS
# ============================================================================

def get_float_input(prompt, default=None):
    """
    Get float input from user with optional default value
    Handles invalid inputs gracefully
    """
    if default is not None:
        prompt = f"{prompt} (default: {default}): "
    else:
        prompt = f"{prompt}: "
    
    while True:
        try:
            value = input(prompt).strip()
            if value == "" and default is not None:
                return default
            return float(value)
        except ValueError:
            print("Invalid input. Please enter a number.")

def calculate_wing_geometry(AR, wingspan):
    """
    Calculate wing geometric parameters
    S = b²/AR (wing area)
    MAC = S/b (mean aerodynamic chord)
    """
    S = wingspan**2 / AR
    MAC = S / wingspan
    return S, MAC

def calculate_vstall(W, S, Cl_max):
    """
    Calculate stall speed from lift equation
    At stall: L = W and Cl = Cl_max
    V_stall = sqrt(2W / (ρ × S × Cl_max))
    """
    return math.sqrt((2 * W) / (rho * S * Cl_max))

def calculate_drag_polar(AR, e, Cd0):
    """
    Calculate induced drag factor k
    CD = CD_0 + k×Cl²
    k = 1/(π×e×AR)
    """
    k = 1 / (math.pi * e * AR)
    return k

def calculate_best_ld(Cd0, k):
    """
    Calculate optimal Cl and best L/D ratio
    At best L/D: Cl_opt = sqrt(CD_0/k)
    L/D_max = Cl_opt / (CD_0 + k×Cl_opt²)
    """
    Cl_opt = math.sqrt(Cd0 / k)
    LD_max = Cl_opt / (Cd0 + k * Cl_opt**2)
    return Cl_opt, LD_max

def calculate_cruise_speed(W, S, Cl_opt):
    """
    Calculate cruise speed at best L/D
    V = sqrt(2W / (ρ × S × Cl))
    """
    return math.sqrt((2 * W) / (rho * S * Cl_opt)) # Extract from airfoil

def calculate_reynolds(V, chord):
    """
    Calculate Reynolds number
    Re = V×c/ν
    """
    return (V * chord) / nu

def check_constraints(vstall, wingspan, wing_loading, vstall_max, span_max, wl_max):
    """
    Check if design meets all constraints
    Returns dictionary of constraint checks
    """
    checks = {
        f"Vstall <= {vstall_max} m/s": vstall <= vstall_max,
        f"Wingspan <= {span_max} m": wingspan <= span_max,
        f"Wing Loading <= {wl_max} N/m^2": wing_loading <= wl_max
    }
    return checks

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_results(results_dict):
    """Print results in formatted table"""
    for key, value in results_dict.items():
        if isinstance(value, float):
            print(f"  {key:.<40} {value:.3f}")
        else:
            print(f"  {key:.<40} {value}")

# ============================================================================
# SECTION 3: PARAMETRIC SWEEP FUNCTIONS
# ============================================================================

# def run_speed_sweep(W, S, Cl_max, Cd0, e, AR, vstall_max, V_wind=10.0, V_gust=3.0):
#     """
#     Run parametric sweep across multiple cruise speeds
    
#     For each speed, calculates:
#     - Required Cl for level flight
#     - Drag and power requirements
#     - Ground speed in headwind
#     - Efficiency metrics
#     - Mass impact estimates
    
#     Returns: (results_list, optimal_cruise_speed, max_L/D)
#     """
#     speeds = [15, 17.5, 20, 22.5, 25, 27.5, 30]  # m/s
    
#     # Calculate baseline optimal cruise
#     k = calculate_drag_polar(AR, e, Cd0)
#     Cl_opt, LD_max = calculate_best_ld(Cd0, k)
#     V_optimal = calculate_cruise_speed(W, S, Cl_opt)
    
#     results = []
    
#     for V in speeds:
#         # Calculate required Cl at this speed
#         Cl = (2 * W) / (rho * V**2 * S)
        
#         # Skip if not achievable
#         if Cl < -0.5 or Cl > Cl_max:
#             continue
        
#         # Drag polar
#         Cd = Cd0 + k * Cl**2
#         LD = Cl / Cd
#         drag = 0.5 * rho * V**2 * S * Cd
#         power = drag * V
#         motor_power = power / 0.9
        
#         # Wind penetration
#         V_headwind = V_wind + V_gust
#         ground_speed = V - V_headwind
        
#         # Efficiency metrics
#         LD_ratio = LD / LD_max
#         power_ratio = power / (calculate_drag_polar(AR, e, Cd0) * V_optimal)
        
#         # Mission energy (2 min high speed + 8 min cruise)
#         energy_high = (motor_power / 60) * 2.0
#         energy_low = ((drag * V_optimal / 0.75) / 60) * 8.0
#         total_energy = energy_high + energy_low
        
#         # Mass estimates (empirical scaling)
#         motor_mass = 0.15 * (motor_power / 100)
#         battery_mass = 0.12 * (total_energy / 16)
#         mass_increase = (motor_mass - 0.15) + (battery_mass - 0.12)
        
#         results.append({
#             'V': V,
#             'Cl': Cl,
#             'Cd': Cd,
#             'LD': LD,
#             'drag': drag,
#             'power': power,
#             'motor_power': motor_power,
#             'ground_speed': ground_speed,
#             'LD_ratio': LD_ratio,
#             'power_ratio': power_ratio,
#             'total_energy': total_energy,
#             'mass_increase': mass_increase
#         })
    
#     return results, V_optimal, LD_max

# ============================================================================
# SECTION 4: PLOTTING FUNCTIONS
# ============================================================================

# def plot_parametric_sweep(results, V_optimal, LD_max, W, S, vstall, vstall_max, 
#                           wingspan, MAC, save_path=None):
#     """
#     Create comprehensive 9-plot parametric analysis figure
    
#     Plots:
#     1. Power vs Speed
#     2. L/D vs Speed
#     3. Ground Speed (bar chart)
#     4. Cl vs Speed
#     5. Energy vs Speed
#     6. Mass Impact (bar chart)
#     7. Power Ratio
#     8. Efficiency Loss
#     9. Summary Table
#     """
#     if not PLOTTING_AVAILABLE:
#         print("  Matplotlib not available. Skipping plots.")
#         return
    
#     speeds = [r['V'] for r in results]
    
#     # Create figure with 3x3 grid
#     fig = plt.figure(figsize=(16, 10))
#     gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
#     # Color scheme
#     color_primary = '#667eea'
#     color_secondary = '#764ba2'
#     color_optimal = '#28a745'
#     color_warning = '#ffc107'
#     color_danger = '#dc3545'
    
#     # PLOT 1: Power vs Speed
#     ax1 = fig.add_subplot(gs[0, 0])
#     motor_powers = [r['motor_power'] for r in results]
#     ax1.plot(speeds, motor_powers, 'o-', color=color_primary, linewidth=2, markersize=8)
#     ax1.axvline(V_optimal, color=color_optimal, linestyle='--', label=f'Optimal ({V_optimal:.1f} m/s)')
#     ax1.set_xlabel('Cruise Speed [m/s]', fontsize=11, fontweight='bold')
#     ax1.set_ylabel('Motor Power [W]', fontsize=11, fontweight='bold')
#     ax1.set_title('Power Requirement vs Cruise Speed', fontsize=12, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
    
#     # PLOT 2: L/D vs Speed
#     ax2 = fig.add_subplot(gs[0, 1])
#     ld_values = [r['LD'] for r in results]
#     ax2.plot(speeds, ld_values, 'o-', color=color_secondary, linewidth=2, markersize=8)
#     ax2.axhline(LD_max, color=color_optimal, linestyle='--', label=f'Max L/D ({LD_max:.1f})')
#     ax2.axhline(LD_max * 0.7, color=color_warning, linestyle=':', alpha=0.5)
#     ax2.set_xlabel('Cruise Speed [m/s]', fontsize=11, fontweight='bold')
#     ax2.set_ylabel('L/D Ratio', fontsize=11, fontweight='bold')
#     ax2.set_title('Aerodynamic Efficiency vs Cruise Speed', fontsize=12, fontweight='bold')
#     ax2.grid(True, alpha=0.3)
#     ax2.legend()
    
#     # PLOT 3: Ground Speed (bar chart)
#     ax3 = fig.add_subplot(gs[0, 2])
#     ground_speeds = [r['ground_speed'] for r in results]
#     colors = [color_danger if gs < 0 else color_optimal if gs > 5 else color_warning 
#               for gs in ground_speeds]
#     ax3.bar(speeds, ground_speeds, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
#     ax3.axhline(0, color='black', linestyle='-', linewidth=2)
#     ax3.set_xlabel('Cruise Speed [m/s]', fontsize=11, fontweight='bold')
#     ax3.set_ylabel('Ground Speed in Headwind [m/s]', fontsize=11, fontweight='bold')
#     ax3.set_title('Wind Penetration Capability', fontsize=12, fontweight='bold')
#     ax3.grid(True, alpha=0.3, axis='y')
    
#     # PLOT 4: Cl vs Speed
#     ax4 = fig.add_subplot(gs[1, 0])
#     cl_values = [r['Cl'] for r in results]
#     ax4.plot(speeds, cl_values, 'o-', color=color_primary, linewidth=2, markersize=8)
#     ax4.axhline(1.0, color=color_optimal, linestyle='--', alpha=0.5, label='Cl = 1.0')
#     ax4.axhline(0.5, color=color_warning, linestyle=':', alpha=0.5, label='Cl = 0.5')
#     ax4.set_xlabel('Cruise Speed [m/s]', fontsize=11, fontweight='bold')
#     ax4.set_ylabel('Lift Coefficient (Cl)', fontsize=11, fontweight='bold')
#     ax4.set_title('Required Cl vs Cruise Speed', fontsize=12, fontweight='bold')
#     ax4.grid(True, alpha=0.3)
#     ax4.legend()
    
#     # PLOT 5: Energy vs Speed
#     ax5 = fig.add_subplot(gs[1, 1])
#     energies = [r['total_energy'] for r in results]
#     ax5.plot(speeds, energies, 'o-', color=color_secondary, linewidth=2, markersize=8)
#     ax5.set_xlabel('Cruise Speed [m/s]', fontsize=11, fontweight='bold')
#     ax5.set_ylabel('Mission Energy [Wh]', fontsize=11, fontweight='bold')
#     ax5.set_title('Energy Requirement (2min high + 8min cruise)', fontsize=12, fontweight='bold')
#     ax5.grid(True, alpha=0.3)
    
#     # PLOT 6: Mass Impact (bar chart)
#     ax6 = fig.add_subplot(gs[1, 2])
#     mass_increases = [r['mass_increase'] * 1000 for r in results]  # Convert to grams
#     colors_mass = [color_optimal if m < 200 else color_warning if m < 400 else color_danger 
#                    for m in mass_increases]
#     ax6.bar(speeds, mass_increases, color=colors_mass, alpha=0.7, edgecolor='black', linewidth=1.5)
#     ax6.set_xlabel('Cruise Speed [m/s]', fontsize=11, fontweight='bold')
#     ax6.set_ylabel('Added Mass [g]', fontsize=11, fontweight='bold')
#     ax6.set_title('Propulsion System Mass Penalty', fontsize=12, fontweight='bold')
#     ax6.grid(True, alpha=0.3, axis='y')
    
#     # PLOT 7: Power Ratio
#     ax7 = fig.add_subplot(gs[2, 0])
#     power_ratios = [r['power_ratio'] for r in results]
#     ax7.plot(speeds, power_ratios, 'o-', color=color_primary, linewidth=2, markersize=8)
#     ax7.axhline(1.0, color=color_optimal, linestyle='--', label='1× (optimal)')
#     ax7.axhline(2.0, color=color_warning, linestyle=':', label='2× threshold')
#     ax7.axhline(3.0, color=color_danger, linestyle=':', label='3× threshold')
#     ax7.set_xlabel('Cruise Speed [m/s]', fontsize=11, fontweight='bold')
#     ax7.set_ylabel('Power Ratio vs Optimal', fontsize=11, fontweight='bold')
#     ax7.set_title('Power Penalty Factor', fontsize=12, fontweight='bold')
#     ax7.grid(True, alpha=0.3)
#     ax7.legend()
    
#     # PLOT 8: Efficiency Loss
#     ax8 = fig.add_subplot(gs[2, 1])
#     ld_ratios = [r['LD_ratio'] * 100 for r in results]
#     efficiency_loss = [100 - r for r in ld_ratios]
#     ax8.plot(speeds, efficiency_loss, 'o-', color=color_danger, linewidth=2, markersize=8)
#     ax8.axhline(30, color=color_warning, linestyle='--', alpha=0.5)
#     ax8.set_xlabel('Cruise Speed [m/s]', fontsize=11, fontweight='bold')
#     ax8.set_ylabel('L/D Efficiency Loss [%]', fontsize=11, fontweight='bold')
#     ax8.set_title('Aerodynamic Efficiency Penalty', fontsize=12, fontweight='bold')
#     ax8.grid(True, alpha=0.3)
    
#     # PLOT 9: Summary Table
#     ax9 = fig.add_subplot(gs[2, 2])
#     ax9.axis('off')
    
#     summary_text = "DESIGN TRADE-OFF SUMMARY\n" + "="*35 + "\n\n"
#     summary_text += f"Wing Area: {S:.2f} m²\n"
#     summary_text += f"Wingspan: {wingspan:.2f} m\n"
#     summary_text += f"Chord (MAC): {MAC:.3f} m\n"
#     summary_text += f"Weight: {W/9.81:.2f} kg\n"
#     summary_text += f"Vstall: {vstall:.2f} m/s\n"
#     summary_text += f"Optimal Cruise: {V_optimal:.1f} m/s\n"
#     summary_text += f"Max L/D: {LD_max:.1f}\n\n"
#     summary_text += "SPEED OPTIONS:\n" + "-"*35 + "\n"
    
#     # Show 3 key speeds
#     for v_key in [18, 22, 25]:
#         matching = [r for r in results if abs(r['V'] - v_key) < 0.5]
#         if matching:
#             r = matching[0]
#             summary_text += f"\n{r['V']:.0f} m/s:\n"
#             summary_text += f"  Power: {r['motor_power']:.0f}W ({r['power_ratio']:.1f}×)\n"
#             summary_text += f"  Ground: {r['ground_speed']:.1f} m/s\n"
#             summary_text += f"  L/D loss: {(1-r['LD_ratio'])*100:.0f}%\n"
#             summary_text += f"  +Mass: {r['mass_increase']*1000:.0f}g\n"
    
#     ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
#              fontsize=10, verticalalignment='top', fontfamily='monospace',
#              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
#     fig.suptitle('UAV Parametric Cruise Speed Analysis', 
#                  fontsize=16, fontweight='bold', y=0.995)
    
#     # Save plot
#     try:
#         filepath = save_path if save_path else os.path.join(os.getcwd(), 'uav_parametric_sweep.png')
#         plt.savefig(filepath, dpi=300, bbox_inches='tight')
#         print(f"\n  ✓ Comprehensive plot saved to: {filepath}")
#         print(f"    Current directory: {os.getcwd()}")
        
#         # Show plot in window
#         plt.show()  # This will block until user closes window
        
#     except Exception as e:
#         print(f"\n  ✗ Error saving/showing plot: {e}")

# def plot_simple_comparison(results, V_optimal, wingspan, MAC, save_path=None):
#     """
#     Create simple 2-plot comparison for quick analysis
#     Shows: Power requirement and Wind penetration
#     """
#     if not PLOTTING_AVAILABLE:
#         return
    
#     speeds = [r['V'] for r in results]
#     motor_powers = [r['motor_power'] for r in results]
#     ground_speeds = [r['ground_speed'] for r in results]
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
#     # Plot 1: Power
#     ax1.plot(speeds, motor_powers, 'o-', color='#667eea', linewidth=2, markersize=8)
#     ax1.axvline(V_optimal, color='green', linestyle='--', label=f'Optimal ({V_optimal:.1f} m/s)')
#     ax1.set_xlabel('Cruise Speed [m/s]', fontsize=12)
#     ax1.set_ylabel('Motor Power [W]', fontsize=12)
#     ax1.set_title('Power Requirement', fontsize=14, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
    
#     # Plot 2: Ground Speed
#     colors = ['red' if gs < 0 else 'green' if gs > 5 else 'orange' for gs in ground_speeds]
#     ax2.bar(speeds, ground_speeds, color=colors, alpha=0.7, edgecolor='black')
#     ax2.axhline(0, color='black', linestyle='-', linewidth=2)
#     ax2.set_xlabel('Cruise Speed [m/s]', fontsize=12)
#     ax2.set_ylabel('Ground Speed in Headwind [m/s]', fontsize=12)
#     ax2.set_title('Wind Penetration', fontsize=14, fontweight='bold')
#     ax2.grid(True, alpha=0.3, axis='y')
    
#     info_text = f"b={wingspan:.2f}m, MAC={MAC:.3f}m"
#     fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')
    
#     plt.tight_layout()
    
#     try:
#         filepath = save_path if save_path else os.path.join(os.getcwd(), 'uav_simple_comparison.png')
#         plt.savefig(filepath, dpi=300, bbox_inches='tight')
#         print(f"\n  ✓ Simple plot saved to: {filepath}")
        
#         # Show plot in window
#         plt.show()  # This will block until user closes window
        
#     except Exception as e:
#         print(f"\n  ✗ Error saving/showing plot: {e}")

# ============================================================================
# SECTION 5: BATCH MODE FUNCTION
# ============================================================================
def plot_response_surfaces(results_batch, target_cruise, vstall_max, save_path=None):
    """
    Create response surface plots for design optimization
    Shows contour plots for:
    1. Vstall (minimize)
    2. Power at target cruise (minimize)
    3. Optimal cruise speed (maximize)
    4. Endurance at target cruise (maximize)
    """
    
    # Extract data
    AR_vals = sorted(list(set([r['AR'] for r in results_batch])))
    b_vals = sorted(list(set([r['b'] for r in results_batch])))
    
    # Create meshgrid
    AR_grid, b_grid = [], []
    vstall_grid, power_grid, vcruise_grid, endurance_grid = [], [], [], []
    pass_grid = []
    
    for AR in AR_vals:
        AR_row, b_row = [], []
        vstall_row, power_row, vcruise_row, endurance_row = [], [], [], []
        pass_row = []
        
        for b in b_vals:
            # Find matching result
            match = [r for r in results_batch if r['AR'] == AR and r['b'] == b]
            if match:
                r = match[0]
                AR_row.append(AR)
                b_row.append(b)
                vstall_row.append(r['vstall'])
                power_row.append(r['power_target'] if r['power_target'] > 0 else 1000)
                vcruise_row.append(r['vcruise_opt'])
                endurance_row.append(r['endurance_target'] if r['endurance_target'] > 0 else 0)
                pass_row.append(r['pass'])
        
        AR_grid.append(AR_row)
        b_grid.append(b_row)
        vstall_grid.append(vstall_row)
        power_grid.append(power_row)
        vcruise_grid.append(vcruise_row)
        endurance_grid.append(endurance_row)
        pass_grid.append(pass_row)
    
    AR_grid = np.array(AR_grid)
    b_grid = np.array(b_grid)
    vstall_grid = np.array(vstall_grid)
    power_grid = np.array(power_grid)
    vcruise_grid = np.array(vcruise_grid)
    endurance_grid = np.array(endurance_grid)
    pass_grid = np.array(pass_grid)
    
    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()  # Convert 2x2 array to 1D array for easier indexing
    
    # Define color maps
    cmap_vstall = 'RdYlGn_r'  # Red=high (bad), Green=low (good)
    cmap_power = 'RdYlGn_r'   # Red=high (bad), Green=low (good)
    cmap_vcruise = 'RdYlGn'   # Red=low (bad), Green=high (good)
    
    # PLOT 1: Vstall Response Surface
    ax1 = axes[0]
    levels_vstall = np.linspace(vstall_grid.min(), vstall_grid.max(), 15)
    contour1 = ax1.contourf(b_grid, AR_grid, vstall_grid, levels=levels_vstall, 
                            cmap=cmap_vstall, alpha=0.8)
    contour1_lines = ax1.contour(b_grid, AR_grid, vstall_grid, levels=levels_vstall, 
                                 colors='black', linewidths=0.5, alpha=0.4)
    ax1.clabel(contour1_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Mark constraint boundary
    ax1.contour(b_grid, AR_grid, vstall_grid, levels=[vstall_max], 
                colors='red', linewidths=3, linestyles='--')
    
    # Mark passing designs
    for i, AR in enumerate(AR_vals):
        for j, b in enumerate(b_vals):
            if pass_grid[i][j]:
                ax1.plot(b, AR, 'go', markersize=6, markeredgecolor='black', markeredgewidth=1)
    
    cbar1 = plt.colorbar(contour1, ax=ax1)
    cbar1.set_label('Vstall [m/s]', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Wingspan [m]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Aspect Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Response Surface: Stall Speed (Minimize)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Lower Vstall (Better)'),
        Patch(facecolor='red', alpha=0.7, label='Higher Vstall (Worse)'),
        plt.Line2D([0], [0], color='red', linewidth=3, linestyle='--', label=f'Vstall = {vstall_max} m/s limit'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
                   markeredgecolor='k', markersize=8, label='Passing Design')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # PLOT 2: Power at Target Cruise
    ax2 = axes[1]
    # Mask infeasible designs
    power_grid_masked = np.ma.masked_where(power_grid > 900, power_grid)
    levels_power = np.linspace(power_grid_masked.min(), power_grid_masked.max(), 15)
    contour2 = ax2.contourf(b_grid, AR_grid, power_grid_masked, levels=levels_power, 
                            cmap=cmap_power, alpha=0.8)
    contour2_lines = ax2.contour(b_grid, AR_grid, power_grid_masked, levels=levels_power, 
                                 colors='black', linewidths=0.5, alpha=0.4)
    ax2.clabel(contour2_lines, inline=True, fontsize=8, fmt='%.0f')
    
    # Mark passing designs
    for i, AR in enumerate(AR_vals):
        for j, b in enumerate(b_vals):
            if pass_grid[i][j]:
                ax2.plot(b, AR, 'go', markersize=6, markeredgecolor='black', markeredgewidth=1)
    
    cbar2 = plt.colorbar(contour2, ax=ax2)
    cbar2.set_label('Power [W]', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Wingspan [m]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Aspect Ratio', fontsize=12, fontweight='bold')
    ax2.set_title(f'Response Surface: Power @ {target_cruise:.0f} m/s (Minimize)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    legend_elements2 = [
        Patch(facecolor='green', alpha=0.7, label='Lower Power (Better)'),
        Patch(facecolor='red', alpha=0.7, label='Higher Power (Worse)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
                   markeredgecolor='k', markersize=8, label='Passing Design')
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=8)
    
    # PLOT 3: Optimal Cruise Speed
    ax3 = axes[2]
    levels_vcruise = np.linspace(vcruise_grid.min(), vcruise_grid.max(), 15)
    contour3 = ax3.contourf(b_grid, AR_grid, vcruise_grid, levels=levels_vcruise, 
                            cmap=cmap_vcruise, alpha=0.8)
    contour3_lines = ax3.contour(b_grid, AR_grid, vcruise_grid, levels=levels_vcruise, 
                                 colors='black', linewidths=0.5, alpha=0.4)
    ax3.clabel(contour3_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Mark passing designs
    for i, AR in enumerate(AR_vals):
        for j, b in enumerate(b_vals):
            if pass_grid[i][j]:
                ax3.plot(b, AR, 'go', markersize=6, markeredgecolor='black', markeredgewidth=1)
    
    cbar3 = plt.colorbar(contour3, ax=ax3)
    cbar3.set_label('Vcruise [m/s]', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Wingspan [m]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Aspect Ratio', fontsize=12, fontweight='bold')
    ax3.set_title('Response Surface: Optimal Cruise Speed (Maximize)', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    legend_elements3 = [
        Patch(facecolor='green', alpha=0.7, label='Higher Speed (Better)'),
        Patch(facecolor='red', alpha=0.7, label='Lower Speed (Worse)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
                   markeredgecolor='k', markersize=8, label='Passing Design')
    ]
    ax3.legend(handles=legend_elements3, loc='upper right', fontsize=8)
    
    # PLOT 4: Endurance at Target Cruise
    ax4 = axes[3]
    # Mask infeasible designs
    endurance_grid_masked = np.ma.masked_where(endurance_grid <= 0, endurance_grid)
    levels_endurance = np.linspace(endurance_grid_masked.min(), endurance_grid_masked.max(), 15)
    contour4 = ax4.contourf(b_grid, AR_grid, endurance_grid_masked, levels=levels_endurance, 
                            cmap=cmap_vcruise, alpha=0.8)
    contour4_lines = ax4.contour(b_grid, AR_grid, endurance_grid_masked, levels=levels_endurance, 
                                 colors='black', linewidths=0.5, alpha=0.4)
    ax4.clabel(contour4_lines, inline=True, fontsize=8, fmt='%.1f')
    
    for i, AR in enumerate(AR_vals):
        for j, b in enumerate(b_vals):
            if pass_grid[i][j]:
                ax4.plot(b, AR, 'go', markersize=6, markeredgecolor='black', markeredgewidth=1)
    
    cbar4 = plt.colorbar(contour4, ax=ax4)
    cbar4.set_label('Endurance [min]', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Wingspan [m]', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Aspect Ratio', fontsize=12, fontweight='bold')
    ax4.set_title(f'Response Surface: Endurance @ {target_cruise:.0f} m/s (Maximize)', 
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    legend_elements4 = [
        Patch(facecolor='green', alpha=0.7, label='Longer Endurance (Better)'),
        Patch(facecolor='red', alpha=0.7, label='Shorter Endurance (Worse)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
                   markeredgecolor='k', markersize=8, label='Passing Design')
    ]
    ax4.legend(handles=legend_elements4, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    try:
        filepath = save_path if save_path else os.path.join(os.getcwd(), 'uav_response_surfaces.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ Response surface plots saved to: {filepath}")
        plt.show()
    except Exception as e:
        print(f"\n  ✗ Error saving/showing response surfaces: {e}")

def batch_mode():
    """
    Run batch analysis for multiple AR × wingspan combinations
    Tests all combinations and identifies best configurations
    Shows only passing designs and generates response surfaces
    Includes battery endurance calculations
    """
    print("\n" + "#"*60)
    print("#  BATCH MODE - Multiple Configuration Analysis".center(60) + "#")
    print("#"*60)
    
    print("\nThis will analyze multiple AR and wingspan combinations.")
    

    AR_values = np.arange(4.0, 8.05, 0.05)
    b_values = np.arange(1.4, 2.05, 0.05)

    print(f"\nAR values to test: {AR_values}")
    print(f"Wingspan values to test: {b_values} m")
    print(f"Total configurations: {len(AR_values) * len(b_values)}")
    
    proceed = input("\nProceed with batch run? (y/n): ").strip().lower()
    if proceed != 'y':
        return
    
    print_header("COMMON PARAMETERS")
    MTOW = get_float_input("MTOW [kg]", 6.0)
    Cl_max = get_float_input("Cl,max", 1.52)
    Cd0 = get_float_input("CD,0", 0.016)
    e = get_float_input("Oswald e", 0.7)
    vstall_max = get_float_input("Max Vstall [m/s]", 8.7)
    span_max = get_float_input("Max wingspan [m]", 2.7)
    power_max = get_float_input("Max power [W]", 400.0)
    target_cruise = get_float_input("Target cruise speed [m/s]", 25.0)
    
    print_header("BATTERY CONFIGURATION")
    print("\nLiPo Battery Specifications:")
    print("  Common configurations:")
    print("    - 3S (11.1V nominal, 12.6V max)")
    print("    - 4S (14.8V nominal, 16.8V max)")
    print("    - 5S (18.5V nominal, 21.0V max)")
    print("    - 6S (22.2V nominal, 25.2V max)")
    
    num_cells = get_float_input("Number of cells (S)", 4.0)
    capacity_mah = get_float_input("Battery capacity [mAh]", 5000.0)
    discharge_limit = get_float_input("Usable capacity [%]", 80.0) / 100.0
    
    voltage_avg = num_cells * 3.8
    capacity_ah = capacity_mah / 1000.0
    battery_energy_wh = voltage_avg * capacity_ah
    usable_energy_wh = battery_energy_wh * discharge_limit
    
    print(f"\n  Battery Configuration Summary:")
    print(f"    Cells: {num_cells:.0f}S")
    print(f"    Nominal Voltage: {num_cells * 3.7:.1f}V")
    print(f"    Average Voltage: {voltage_avg:.1f}V")
    print(f"    Capacity: {capacity_mah:.0f}mAh ({capacity_ah:.1f}Ah)")
    print(f"    Total Energy: {battery_energy_wh:.1f}Wh")
    print(f"    Usable Energy ({discharge_limit*100:.0f}%): {usable_energy_wh:.1f}Wh")
    
    W = MTOW * g
    
    results_batch = []
    
    print("\n" + "="*80)
    print("ANALYZING ALL CONFIGURATIONS...")
    print("="*80)
    
    for AR in AR_values:
        for b in b_values:
            S = b**2 / AR
            MAC = S / b
            
            vstall = math.sqrt((2 * W) / (rho * S * Cl_max))
            
            k = 1 / (math.pi * e * AR)
            Cl_opt = math.sqrt(Cd0 / k)
            vcruise_opt = math.sqrt((2 * W) / (rho * S * Cl_opt))

            Cd_opt = Cd0 + k * Cl_opt**2
            drag_opt = 0.5 * rho * vcruise_opt**2 * S * Cd_opt
            power_opt = drag_opt * vcruise_opt / 0.9
            
            endurance_opt = (usable_energy_wh / power_opt) * 60.0 if power_opt > 0 else 0
            
            Cl_target = (2 * W) / (rho * target_cruise**2 * S)
            if -0.5 < Cl_target < Cl_max:
                Cd_target = Cd0 + k * Cl_target**2
                drag_target = 0.5 * rho * target_cruise**2 * S * Cd_target
                power_target = drag_target * target_cruise / 0.9
                endurance_target = (usable_energy_wh / power_target) * 60.0 if power_target > 0 else 0
            else:
                power_target = -1
                endurance_target = 0
        
            vstall_ok = vstall <= vstall_max
            span_ok = b <= span_max
            power_ok = power_target < power_max if power_target > 0 else False
            
            status = "✓ PASS" if (vstall_ok and span_ok and power_ok) else "✗ FAIL"
            
            results_batch.append({
                'AR': AR, 
                'b': b, 
                'S': S, 
                'MAC': MAC,
                'vstall': vstall, 
                'vcruise_opt': vcruise_opt,
                'power_opt': power_opt, 
                'power_target': power_target,
                'endurance_opt': endurance_opt, 
                'endurance_target': endurance_target,
                'pass': status == "✓ PASS"
            })
    
    feasible = [r for r in results_batch if r['pass']]
    
    print(f"\n  Total configurations analyzed: {len(results_batch)}")
    print(f"  Passing configurations: {len(feasible)}")
    print(f"  Failed configurations: {len(results_batch) - len(feasible)}")
    
    if not feasible:
        print("\n  ✗ No configurations meet all requirements!")
        print("  Consider relaxing constraints or expanding design space.")
        return
    
    print("\n" + "="*110)
    print("PASSING CONFIGURATIONS ONLY")
    print("="*110)
    print(f"{'AR':>5} | {'b[m]':>5} | {'S[m²]':>6} | {'Vstall':>7} | {'Vcr.opt':>8} | " +
          f"{'P@opt':>7} | {'P@{:.0f}'.format(target_cruise):>7} | " +
          f"{'t@opt':>8} | {'t@{:.0f}'.format(target_cruise):>8}")
    print(f"{'':>5} | {'':>5} | {'':>6} | {'[m/s]':>7} | {'[m/s]':>8} | " +
          f"{'[W]':>7} | {'[W]':>7} | {'[min]':>8} | {'[min]':>8}")
    print("-"*110)
    
    for r in feasible:
        print(f"{r['AR']:5.1f} | {r['b']:5.2f} | {r['S']:6.3f} | " +
              f"{r['vstall']:7.2f} | {r['vcruise_opt']:8.2f} | " +
              f"{r['power_opt']:7.0f} | {r['power_target']:7.0f} | " +
              f"{r['endurance_opt']:8.1f} | {r['endurance_target']:8.1f}")
    
    print("="*110)
    
    print("\n" + "="*60)
    print("  OPTIMAL CONFIGURATIONS")
    print("="*60)
    
    best_vstall = min(feasible, key=lambda x: x['vstall'])
    print(f"\n  MINIMUM STALL SPEED:")
    print(f"    AR={best_vstall['AR']:.1f}, b={best_vstall['b']:.2f}m, S={best_vstall['S']:.3f}m²")
    print(f"    Vstall={best_vstall['vstall']:.2f} m/s")
    print(f"    Power @ {target_cruise:.0f} m/s = {best_vstall['power_target']:.0f} W")
    print(f"    Endurance @ {target_cruise:.0f} m/s = {best_vstall['endurance_target']:.1f} min")
    
    best_power = min(feasible, key=lambda x: x['power_target'])
    print(f"\n  MINIMUM POWER @ {target_cruise:.0f} m/s:")
    print(f"    AR={best_power['AR']:.1f}, b={best_power['b']:.2f}m, S={best_power['S']:.3f}m²")
    print(f"    Power={best_power['power_target']:.0f} W")
    print(f"    Vstall={best_power['vstall']:.2f} m/s")
    print(f"    Endurance @ {target_cruise:.0f} m/s = {best_power['endurance_target']:.1f} min")
    
    best_vcruise = max(feasible, key=lambda x: x['vcruise_opt'])
    print(f"\n  MAXIMUM OPTIMAL CRUISE SPEED:")
    print(f"    AR={best_vcruise['AR']:.1f}, b={best_vcruise['b']:.2f}m, S={best_vcruise['S']:.3f}m²")
    print(f"    Vcruise={best_vcruise['vcruise_opt']:.2f} m/s")
    print(f"    Power @ {target_cruise:.0f} m/s = {best_vcruise['power_target']:.0f} W")
    print(f"    Endurance @ {target_cruise:.0f} m/s = {best_vcruise['endurance_target']:.1f} min")
    
    best_endurance_target = max(feasible, key=lambda x: x['endurance_target'])
    print(f"\n  MAXIMUM ENDURANCE @ {target_cruise:.0f} m/s:")
    print(f"    AR={best_endurance_target['AR']:.1f}, b={best_endurance_target['b']:.2f}m, S={best_endurance_target['S']:.3f}m²")
    print(f"    Endurance = {best_endurance_target['endurance_target']:.1f} min ({best_endurance_target['endurance_target']/60:.2f} hours)")
    print(f"    Power required = {best_endurance_target['power_target']:.0f} W")
    print(f"    Vstall = {best_endurance_target['vstall']:.2f} m/s")
    
    best_endurance_opt = max(feasible, key=lambda x: x['endurance_opt'])
    print(f"\n  MAXIMUM ENDURANCE @ OPTIMAL CRUISE:")
    print(f"    AR={best_endurance_opt['AR']:.1f}, b={best_endurance_opt['b']:.2f}m, S={best_endurance_opt['S']:.3f}m²")
    print(f"    Vcruise optimal = {best_endurance_opt['vcruise_opt']:.2f} m/s")
    print(f"    Endurance = {best_endurance_opt['endurance_opt']:.1f} min ({best_endurance_opt['endurance_opt']/60:.2f} hours)")
    print(f"    Power required = {best_endurance_opt['power_opt']:.0f} W")
    
    try:
        with open('uav_batch_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Batch Analysis Results - PASSING CONFIGURATIONS ONLY"])
            writer.writerow([])
            writer.writerow(["Constraints:"])
            writer.writerow([f"Max Vstall: {vstall_max} m/s"])
            writer.writerow([f"Max Wingspan: {span_max} m"])
            writer.writerow([f"Max Power: {power_max} W"])
            writer.writerow([f"Target Cruise: {target_cruise} m/s"])
            writer.writerow([])
            writer.writerow(["Battery Configuration:"])
            writer.writerow([f"Cells: {num_cells:.0f}S"])
            writer.writerow([f"Capacity: {capacity_mah:.0f} mAh"])
            writer.writerow([f"Usable Energy: {usable_energy_wh:.1f} Wh"])
            writer.writerow([])
            writer.writerow(["AR", "Wingspan [m]", "Area [m²]", "MAC [m]", "Vstall [m/s]", 
                           "Vcruise opt [m/s]", "Power opt [W]", 
                           f"Power @ {target_cruise:.0f} m/s [W]",
                           "Endurance @ opt [min]", f"Endurance @ {target_cruise:.0f} m/s [min]"])
            for r in feasible:
                writer.writerow([r['AR'], r['b'], f"{r['S']:.3f}", f"{r['MAC']:.3f}", 
                               f"{r['vstall']:.2f}", f"{r['vcruise_opt']:.2f}", 
                               f"{r['power_opt']:.0f}", f"{r['power_target']:.0f}",
                               f"{r['endurance_opt']:.1f}", f"{r['endurance_target']:.1f}"])
            
            writer.writerow([])
            writer.writerow(["OPTIMAL DESIGNS"])
            writer.writerow(["Objective", "AR", "Wingspan [m]", "Value", "Additional Info"])
            writer.writerow(["Min Vstall", best_vstall['AR'], best_vstall['b'], 
                           f"{best_vstall['vstall']:.2f} m/s",
                           f"Endurance @ {target_cruise:.0f} m/s: {best_vstall['endurance_target']:.1f} min"])
            writer.writerow([f"Min Power @ {target_cruise:.0f} m/s", best_power['AR'], 
                           best_power['b'], f"{best_power['power_target']:.0f} W",
                           f"Endurance: {best_power['endurance_target']:.1f} min"])
            writer.writerow(["Max Vcruise", best_vcruise['AR'], best_vcruise['b'], 
                           f"{best_vcruise['vcruise_opt']:.2f} m/s",
                           f"Endurance @ {target_cruise:.0f} m/s: {best_vcruise['endurance_target']:.1f} min"])
            writer.writerow([f"Max Endurance @ {target_cruise:.0f} m/s", 
                           best_endurance_target['AR'], best_endurance_target['b'],
                           f"{best_endurance_target['endurance_target']:.1f} min",
                           f"Power: {best_endurance_target['power_target']:.0f} W"])
            writer.writerow(["Max Endurance @ optimal", 
                           best_endurance_opt['AR'], best_endurance_opt['b'],
                           f"{best_endurance_opt['endurance_opt']:.1f} min",
                           f"Vcruise: {best_endurance_opt['vcruise_opt']:.2f} m/s"])
            
        print(f"\n  ✓ Batch results saved to: {os.path.join(os.getcwd(), 'uav_batch_results.csv')}")
    except Exception as e:
        print(f"\n  Error saving batch results: {e}")
    
    if PLOTTING_AVAILABLE:
        print("\n" + "="*60)
        print("  RESPONSE SURFACE GENERATION")
        print("="*60)
        
        generate_plots = input("\nGenerate response surface plots? (y/n): ").strip().lower()
        if generate_plots == 'y':
            print("\n  Creating response surface plots...")
            print("  This will show:")
            print("    1. Vstall optimization surface")
            print("    2. Power @ target cruise optimization surface")
            print("    3. Optimal cruise speed surface")
            print("    4. Endurance @ target cruise optimization surface")
            print("\n  Close the plot window to continue...")
            
            plot_response_surfaces(results_batch, target_cruise, vstall_max)
            print("\n  Response surface generation complete!")
    
    print("\n" + "="*60 + "\n")

# def plot_response_surfaces(results_batch, target_cruise, vstall_max, save_path=None):
#     """
#     Create response surface plots for design optimization
#     Shows contour plots for:
#     1. Vstall (minimize)
#     2. Power at target cruise (minimize)
#     3. Optimal cruise speed (maximize)
#     """
#     if not PLOTTING_AVAILABLE:
#         print("  Matplotlib not available. Skipping response surface plots.")
#         return
    
#     # Extract data
#     AR_vals = sorted(list(set([r['AR'] for r in results_batch])))
#     b_vals = sorted(list(set([r['b'] for r in results_batch])))
    
#     # Create meshgrid
#     AR_grid, b_grid = [], []
#     vstall_grid, power_grid, vcruise_grid = [], [], []
#     pass_grid = []
    
#     for AR in AR_vals:
#         AR_row, b_row = [], []
#         vstall_row, power_row, vcruise_row = [], [], []
#         pass_row = []
        
#         for b in b_vals:
#             # Find matching result
#             match = [r for r in results_batch if r['AR'] == AR and r['b'] == b]
#             if match:
#                 r = match[0]
#                 AR_row.append(AR)
#                 b_row.append(b)
#                 vstall_row.append(r['vstall'])
#                 power_row.append(r['power_target'] if r['power_target'] > 0 else 1000)
#                 vcruise_row.append(r['vcruise_opt'])
#                 pass_row.append(r['pass'])
        
#         AR_grid.append(AR_row)
#         b_grid.append(b_row)
#         vstall_grid.append(vstall_row)
#         power_grid.append(power_row)
#         vcruise_grid.append(vcruise_row)
#         pass_grid.append(pass_row)
    
#     import numpy as np
#     AR_grid = np.array(AR_grid)
#     b_grid = np.array(b_grid)
#     vstall_grid = np.array(vstall_grid)
#     power_grid = np.array(power_grid)
#     vcruise_grid = np.array(vcruise_grid)
#     pass_grid = np.array(pass_grid)
    
#     # Create figure with 3 subplots
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
#     # Define color maps
#     cmap_vstall = 'RdYlGn_r'  # Red=high (bad), Green=low (good)
#     cmap_power = 'RdYlGn_r'   # Red=high (bad), Green=low (good)
#     cmap_vcruise = 'RdYlGn'   # Red=low (bad), Green=high (good)
    
#     # PLOT 1: Vstall Response Surface
#     ax1 = axes[0]
#     levels_vstall = np.linspace(vstall_grid.min(), vstall_grid.max(), 15)
#     contour1 = ax1.contourf(b_grid, AR_grid, vstall_grid, levels=levels_vstall, 
#                             cmap=cmap_vstall, alpha=0.8)
#     contour1_lines = ax1.contour(b_grid, AR_grid, vstall_grid, levels=levels_vstall, 
#                                  colors='black', linewidths=0.5, alpha=0.4)
#     ax1.clabel(contour1_lines, inline=True, fontsize=8, fmt='%.2f')
    
#     # Mark constraint boundary
#     ax1.contour(b_grid, AR_grid, vstall_grid, levels=[vstall_max], 
#                 colors='red', linewidths=3, linestyles='--')
    
#     # Mark passing designs
#     for i, AR in enumerate(AR_vals):
#         for j, b in enumerate(b_vals):
#             if pass_grid[i][j]:
#                 ax1.plot(b, AR, 'go', markersize=6, markeredgecolor='black', markeredgewidth=1)
    
#     cbar1 = plt.colorbar(contour1, ax=ax1)
#     cbar1.set_label('Vstall [m/s]', fontsize=11, fontweight='bold')
#     ax1.set_xlabel('Wingspan [m]', fontsize=12, fontweight='bold')
#     ax1.set_ylabel('Aspect Ratio', fontsize=12, fontweight='bold')
#     ax1.set_title('Response Surface: Stall Speed (Minimize)', fontsize=13, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
    
#     # Add legend
#     from matplotlib.patches import Patch
#     legend_elements = [
#         Patch(facecolor='green', alpha=0.7, label='Lower Vstall (Better)'),
#         Patch(facecolor='red', alpha=0.7, label='Higher Vstall (Worse)'),
#         plt.Line2D([0], [0], color='red', linewidth=3, linestyle='--', label=f'Vstall = {vstall_max} m/s limit'),
#         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
#                    markeredgecolor='k', markersize=8, label='Passing Design')
#     ]
#     ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
#     # PLOT 2: Power at Target Cruise
#     ax2 = axes[1]
#     # Mask infeasible designs
#     power_grid_masked = np.ma.masked_where(power_grid > 900, power_grid)
#     levels_power = np.linspace(power_grid_masked.min(), power_grid_masked.max(), 15)
#     contour2 = ax2.contourf(b_grid, AR_grid, power_grid_masked, levels=levels_power, 
#                             cmap=cmap_power, alpha=0.8)
#     contour2_lines = ax2.contour(b_grid, AR_grid, power_grid_masked, levels=levels_power, 
#                                  colors='black', linewidths=0.5, alpha=0.4)
#     ax2.clabel(contour2_lines, inline=True, fontsize=8, fmt='%.0f')
    
#     # Mark passing designs
#     for i, AR in enumerate(AR_vals):
#         for j, b in enumerate(b_vals):
#             if pass_grid[i][j]:
#                 ax2.plot(b, AR, 'go', markersize=6, markeredgecolor='black', markeredgewidth=1)
    
#     cbar2 = plt.colorbar(contour2, ax=ax2)
#     cbar2.set_label('Power [W]', fontsize=11, fontweight='bold')
#     ax2.set_xlabel('Wingspan [m]', fontsize=12, fontweight='bold')
#     ax2.set_ylabel('Aspect Ratio', fontsize=12, fontweight='bold')
#     ax2.set_title(f'Response Surface: Power @ {target_cruise:.0f} m/s (Minimize)', 
#                   fontsize=13, fontweight='bold')
#     ax2.grid(True, alpha=0.3)
    
#     legend_elements2 = [
#         Patch(facecolor='green', alpha=0.7, label='Lower Power (Better)'),
#         Patch(facecolor='red', alpha=0.7, label='Higher Power (Worse)'),
#         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
#                    markeredgecolor='k', markersize=8, label='Passing Design')
#     ]
#     ax2.legend(handles=legend_elements2, loc='upper right', fontsize=8)
    
#     # PLOT 3: Optimal Cruise Speed
#     ax3 = axes[2]
#     levels_vcruise = np.linspace(vcruise_grid.min(), vcruise_grid.max(), 15)
#     contour3 = ax3.contourf(b_grid, AR_grid, vcruise_grid, levels=levels_vcruise, 
#                             cmap=cmap_vcruise, alpha=0.8)
#     contour3_lines = ax3.contour(b_grid, AR_grid, vcruise_grid, levels=levels_vcruise, 
#                                  colors='black', linewidths=0.5, alpha=0.4)
#     ax3.clabel(contour3_lines, inline=True, fontsize=8, fmt='%.1f')
    
#     # Mark passing designs
#     for i, AR in enumerate(AR_vals):
#         for j, b in enumerate(b_vals):
#             if pass_grid[i][j]:
#                 ax3.plot(b, AR, 'go', markersize=6, markeredgecolor='black', markeredgewidth=1)
    
#     cbar3 = plt.colorbar(contour3, ax=ax3)
#     cbar3.set_label('Vcruise [m/s]', fontsize=11, fontweight='bold')
#     ax3.set_xlabel('Wingspan [m]', fontsize=12, fontweight='bold')
#     ax3.set_ylabel('Aspect Ratio', fontsize=12, fontweight='bold')
#     ax3.set_title('Response Surface: Optimal Cruise Speed (Maximize)', 
#                   fontsize=13, fontweight='bold')
#     ax3.grid(True, alpha=0.3)
    
#     legend_elements3 = [
#         Patch(facecolor='green', alpha=0.7, label='Higher Speed (Better)'),
#         Patch(facecolor='red', alpha=0.7, label='Lower Speed (Worse)'),
#         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
#                    markeredgecolor='k', markersize=8, label='Passing Design')
#     ]
#     ax3.legend(handles=legend_elements3, loc='upper right', fontsize=8)
    
#     plt.tight_layout()
    
#     # Save plot
#     try:
#         filepath = save_path if save_path else os.path.join(os.getcwd(), 'uav_response_surfaces.png')
#         plt.savefig(filepath, dpi=300, bbox_inches='tight')
#         print(f"\n  ✓ Response surface plots saved to: {filepath}")
#         plt.show()
#     except Exception as e:
#         print(f"\n  ✗ Error saving/showing response surfaces: {e}")

# def batch_mode():
#     """
#     Run batch analysis for multiple AR × wingspan combinations
#     Tests all combinations and identifies best configurations
#     Shows only passing designs and generates response surfaces
#     """
#     print("\n" + "#"*60)
#     print("#  BATCH MODE - Multiple Configuration Analysis".center(60) + "#")
#     print("#"*60)
    
#     print("\nThis will analyze multiple AR and wingspan combinations.")
    
#     # Define parameter ranges
#     AR_values = np.arange(5.5, 7.55, 0.05) 
#     print(f"\nAR values to test: {AR_values}")
#     b_values = np.arange(1.6, 2.65, 0.05)
#     print(f"Wingspan values to test: {b_values} m")
#     print(f"Total configurations: {len(AR_values) * len(b_values)}")
    
#     proceed = input("\nProceed with batch run? (y/n): ").strip().lower()
#     if proceed != 'y':
#         return
    
#     # Get common parameters
#     print_header("COMMON PARAMETERS")
#     MTOW = get_float_input("MTOW [kg]", 6.0)
#     Cl_max = get_float_input("Cl,max", 1.52)
#     Cd0 = get_float_input("CD,0", 0.016)
#     e = get_float_input("Oswald e", 0.7)
#     vstall_max = get_float_input("Max Vstall [m/s]", 8.7)
#     span_max = get_float_input("Max wingspan [m]", 2.7)
#     power_max = get_float_input("Max power [W]", 400.0)
#     target_cruise = get_float_input("Target cruise speed [m/s]", 25.0)
    
#     W = MTOW * g
    
#     # Run batch analysis
#     results_batch = []
    
#     print("\n" + "="*80)
#     print("ANALYZING ALL CONFIGURATIONS...")
#     print("="*80)
    
#     for AR in AR_values:
#         for b in b_values:
#             S = b**2 / AR
#             MAC = S / b
            
#             # Calculate stall speed
#             vstall = math.sqrt((2 * W) / (rho * S * Cl_max))
            
#             # Calculate optimal cruise
#             k = 1 / (math.pi * e * AR)
#             Cl_opt = math.sqrt(Cd0 / k)
#             vcruise_opt = math.sqrt((2 * W) / (rho * S * Cl_opt))

#             # Power at optimal
#             Cd_opt = Cd0 + k * Cl_opt**2
#             drag_opt = 0.5 * rho * vcruise_opt**2 * S * Cd_opt
#             power_opt = drag_opt * vcruise_opt / 0.9
            
#             # Power at target
#             Cl_target = (2 * W) / (rho * target_cruise**2 * S)
#             if -0.5 < Cl_target < Cl_max:
#                 Cd_target = Cd0 + k * Cl_target**2
#                 drag_target = 0.5 * rho * target_cruise**2 * S * Cd_target
#                 power_target = drag_target * target_cruise / 0.9
#             else:
#                 power_target = -1
            
#             # Check constraints
#             vstall_ok = vstall <= vstall_max
#             span_ok = b <= span_max
#             power_ok = power_target < power_max if power_target > 0 else False
            
#             status = "✓ PASS" if (vstall_ok and span_ok and power_ok) else "✗ FAIL"
            
#             results_batch.append({
#                 'AR': AR, 'b': b, 'S': S, 'MAC': MAC,
#                 'vstall': vstall, 'vcruise_opt': vcruise_opt,
#                 'power_opt': power_opt, 'power_target': power_target,
#                 'pass': status == "✓ PASS"
#             })
    
#     # Filter for passing designs only
#     feasible = [r for r in results_batch if r['pass']]
    
#     print(f"\n  Total configurations analyzed: {len(results_batch)}")
#     print(f"  Passing configurations: {len(feasible)}")
#     print(f"  Failed configurations: {len(results_batch) - len(feasible)}")
    
#     if not feasible:
#         print("\n  ✗ No configurations meet all requirements!")
#         print("  Consider relaxing constraints or expanding design space.")
#         return
    
#     # Display only passing configurations
#     print("\n" + "="*80)
#     print("PASSING CONFIGURATIONS ONLY")
#     print("="*80)
#     print(f"{'AR':>5} | {'b[m]':>5} | {'S[m²]':>6} | {'MAC[m]':>7} | {'Vstall':>7} | {'Vcr.opt':>8} | " +
#           f"{'P@opt':>7} | {'P@{:.0f}'.format(target_cruise):>7}")
#     print("-"*80)
    
#     for r in feasible:
#         print(f"{r['AR']:5.1f} | {r['b']:5.2f} | {r['S']:6.3f} | {r['MAC']:7.3f} | " +
#               f"{r['vstall']:7.2f} | {r['vcruise_opt']:8.2f} | " +
#               f"{r['power_opt']:7.0f} | {r['power_target']:7.0f}")
    
#     print("="*80)
    
#     # Find optimal configurations
#     print("\n" + "="*60)
#     print("  OPTIMAL CONFIGURATIONS")
#     print("="*60)
    
#     best_vstall = min(feasible, key=lambda x: x['vstall'])
#     print(f"\n  MINIMUM STALL SPEED:")
#     print(f"    AR={best_vstall['AR']:.1f}, b={best_vstall['b']:.2f}m, S={best_vstall['S']:.3f}m²")
#     print(f"    Vstall={best_vstall['vstall']:.2f} m/s")
#     print(f"    Vcruise={best_vstall['vcruise_opt']:.2f} m/s")
#     print(f"    Power @ {target_cruise:.0f} m/s = {best_vstall['power_target']:.0f} W")
    
#     best_power = min(feasible, key=lambda x: x['power_target'])
#     print(f"\n  MINIMUM POWER @ {target_cruise:.0f} m/s:")
#     print(f"    AR={best_power['AR']:.1f}, b={best_power['b']:.2f}m, S={best_power['S']:.3f}m²")
#     print(f"    Power={best_power['power_target']:.0f} W")
#     print(f"    Vstall={best_power['vstall']:.2f} m/s")
#     print(f"    Vcruise={best_power['vcruise_opt']:.2f} m/s")

#     best_vcruise = max(feasible, key=lambda x: x['vcruise_opt'])
#     print(f"\n  MAXIMUM OPTIMAL CRUISE SPEED:")
#     print(f"    AR={best_vcruise['AR']:.1f}, b={best_vcruise['b']:.2f}m, S={best_vcruise['S']:.3f}m²")
#     print(f"    Vcruise={best_vcruise['vcruise_opt']:.2f} m/s")
#     print(f"    Power @ {target_cruise:.0f} m/s = {best_vcruise['power_target']:.0f} W")
#     print(f"    Vstall={best_vcruise['vstall']:.2f} m/s")
    
#     # Save results
#     try:
#         with open('uav_batch_results.csv', 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(["Batch Analysis Results - PASSING CONFIGURATIONS ONLY"])
#             writer.writerow([])
#             writer.writerow(["Constraints:"])
#             writer.writerow([f"Max Vstall: {vstall_max} m/s"])
#             writer.writerow([f"Max Wingspan: {span_max} m"])
#             writer.writerow([f"Max Power: {power_max} W"])
#             writer.writerow([f"Target Cruise: {target_cruise} m/s"])
#             writer.writerow([])
#             writer.writerow(["AR", "Wingspan [m]", "Area [m²]", "MAC [m]", "Vstall [m/s]", 
#                            "Vcruise opt [m/s]", "Power opt [W]", 
#                            f"Power @ {target_cruise:.0f} m/s [W]"])
#             for r in feasible:
#                 writer.writerow([r['AR'], r['b'], f"{r['S']:.3f}", f"{r['MAC']:.3f}", 
#                                f"{r['vstall']:.2f}", f"{r['vcruise_opt']:.2f}", 
#                                f"{r['power_opt']:.0f}", f"{r['power_target']:.0f}"])
            
#             writer.writerow([])
#             writer.writerow(["OPTIMAL DESIGNS"])
#             writer.writerow(["Objective", "AR", "Wingspan [m]", "Value"])
#             writer.writerow(["Min Vstall", best_vstall['AR'], best_vstall['b'], 
#                            f"{best_vstall['vstall']:.2f} m/s"])
#             writer.writerow([f"Min Power @ {target_cruise:.0f} m/s", best_power['AR'], 
#                            best_power['b'], f"{best_power['power_target']:.0f} W"])
#             writer.writerow(["Max Vcruise", best_vcruise['AR'], best_vcruise['b'], 
#                            f"{best_vcruise['vcruise_opt']:.2f} m/s"])
            
#         print(f"\n  ✓ Batch results saved to: {os.path.join(os.getcwd(), 'uav_batch_results.csv')}")
#     except Exception as e:
#         print(f"\n  Error saving batch results: {e}")
    
#     # Generate response surface plots
#     if PLOTTING_AVAILABLE:
#         print("\n" + "="*60)
#         print("  RESPONSE SURFACE GENERATION")
#         print("="*60)
        
#         generate_plots = input("\nGenerate response surface plots? (y/n): ").strip().lower()
#         if generate_plots == 'y':
#             print("\n  Creating response surface plots...")
#             print("  This will show:")
#             print("    1. Vstall optimization surface")
#             print("    2. Power @ target cruise optimization surface")
#             print("    3. Optimal cruise speed surface")
#             print("\n  Close the plot window to continue...")
            
#             plot_response_surfaces(results_batch, target_cruise, vstall_max)
#             print("\n  Response surface generation complete!")
    
#     print("\n" + "="*60 + "\n")

# ============================================================================
# SECTION 6: MAIN ANALYSIS FUNCTION
# ============================================================================

def main():
    """
    Main analysis function
    Runs interactive UAV design calculator with optional parametric sweeps
    """
    # print("\n" + "#"*60)
    # print("#" + " "*58 + "#")
    # print("#  UAV AERODYNAMICS DESIGN CALCULATOR".center(60) + "#")
    print("#  Sizing and Performance Analysis".center(60) + "#")
    # print("#" + " "*58 + "#")
    # print("#"*60)
    
    print(f"\n  Working Directory: {os.getcwd()}")
    print(f"  All output files will be saved here.")
    
    # Mode selection
    # print("\nSelect mode:")
    # print("  1. Single configuration analysis (interactive)")
    # print("  2. Batch mode (multiple AR × wingspan combinations)")
    # print("  3. Exit")
    
    batch_mode() #input("\nEnter choice [1/2/3]: ").strip()
    
    # if mode == '2':
    #     batch_mode()
    #     return
    # elif mode == '3':
    #     print("\nExiting calculator.")
    #     return
    # elif mode != '1':
    #     print("\nInvalid choice. Defaulting to single configuration mode.")
    
    # ==================== INPUTS ====================
    print_header("MISSION REQUIREMENTS")
    payload_mass = get_float_input("Payload mass [kg]", 2.0)
    vstall_max = get_float_input("Maximum stall speed [m/s]", 8.0)
    span_max = get_float_input("Maximum wingspan [m]", 2.7)
    tail_span_max = get_float_input("Maximum tail span [m]", 1.7)
    
    print_header("DESIGN PARAMETERS")
    AR = get_float_input("Aspect Ratio (AR)", 6.5)
    wingspan = get_float_input("Wingspan [m]", 2.5)
    MTOW = get_float_input("Estimated MTOW [kg]", 6.0)
    
    print_header("AERODYNAMIC PROPERTIES")
    Cl_max = get_float_input("Maximum lift coefficient Cl,max (3D)", 1.52)
    Cd0 = get_float_input("Zero-lift drag coefficient CD,0", 0.016)
    e = get_float_input("Oswald efficiency factor (e)", 0.7)
    
    # ==================== CALCULATIONS ====================
    print_header("CALCULATING...")
    
    # Initialize variables (prevents NameError)
    target_analysis_data = {}
    sweep_results = []
    
    # Weight
    W = MTOW * g
    
    # Wing geometry
    S, MAC = calculate_wing_geometry(AR, wingspan)
    wing_loading = W / S
    
    # Stall speed
    vstall = calculate_vstall(W, S, Cl_max)
    
    # Drag polar
    k = calculate_drag_polar(AR, e, Cd0)
    Cl_opt, LD_max = calculate_best_ld(Cd0, k)
    
    # Cruise performance
    vcruise = calculate_cruise_speed(W, S, Cl_opt)
    Re_cruise = calculate_reynolds(vcruise, MAC)
    
    # Drag and power at stall
    Cd_stall = Cd0 + k * Cl_max**2
    drag_stall = 0.5 * rho * vstall**2 * S * Cd_stall
    power_stall = drag_stall * vstall
    
    # Drag and power at cruise
    Cd_cruise = Cd0 + k * Cl_opt**2
    drag_cruise = 0.5 * rho * vcruise**2 * S * Cd_cruise
    power_cruise = drag_cruise * vcruise
    
    v_design = 1.2 * vstall
    
    # ==================== RESULTS ====================
    print_header("GEOMETRY RESULTS")
    geometry = {
        "Aspect Ratio": AR,
        "Wingspan [m]": wingspan,
        "Wing Area [m^2]": S,
        "Mean Aerodynamic Chord (MAC) [m]": MAC,
        "Root Chord (approx) [m]": MAC * 1.2,
        "Tip Chord (approx) [m]": MAC * 0.8,
        "Wing Loading [N/m^2]": wing_loading
    }
    print_results(geometry)
    
    print_header("PERFORMANCE RESULTS")
    performance = {
        "Stall Speed [m/s]": vstall,
        "Design Speed (1.2*Vstall) [m/s]": v_design,
        "Cruise Speed (best L/D) [m/s]": vcruise,
        "Maximum L/D": LD_max,
        "Optimal Cl for cruise": Cl_opt,
        "Reynolds Number @ cruise": Re_cruise
    }
    print_results(performance)
    
    print_header("DRAG ANALYSIS")
    drag_results = {
        "Induced drag factor (k)": k,
        "CD @ stall (Cl=Cl_max)": Cd_stall,
        "Drag @ stall [N]": drag_stall,
        "CD @ cruise (Cl=Cl_opt)": Cd_cruise,
        "Drag @ cruise [N]": drag_cruise
    }
    print_results(drag_results)
    
    print_header("POWER REQUIREMENTS")
    power_results = {
        "Power @ stall [W]": power_stall,
        "Power @ cruise [W]": power_cruise,
        "Motor power @ stall (η=0.75) [W]": power_stall / 0.75,
        "Motor power @ cruise (η=0.75) [W]": power_cruise / 0.75
    }
    print_results(power_results)
    
    # ==================== CONSTRAINT CHECKS ====================
    print_header("CONSTRAINT CHECKS")
    
    constraints = check_constraints(vstall, wingspan, wing_loading, 
                                   vstall_max, span_max, 80.0)
    
    all_pass = True
    for constraint, passed in constraints.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {constraint:.<45} {status}")
        if not passed:
            all_pass = False
    
    print("\n" + "-"*60)
    if all_pass:
        print("  Overall Status: ALL CONSTRAINTS SATISFIED ✓".center(60))
    else:
        print("  Overall Status: SOME CONSTRAINTS VIOLATED ✗".center(60))
    print("-"*60)
    
    # ==================== TAIL SIZING ====================
    print_header("TAIL SIZING (PRELIMINARY)")
    
    Vh = get_float_input("Horizontal tail volume coefficient", 0.55)
    lt = get_float_input("Tail moment arm [m]", 0.5)
    
    St = (Vh * S * MAC) / lt
    ARt = get_float_input("Tail aspect ratio", 4.0)
    bt = math.sqrt(ARt * St)
    
    tail_results = {
        "H-tail area [m^2]": St,
        "H-tail span [m]": bt,
        f"H-tail check (< {tail_span_max} m)": "PASS ✓" if bt <= tail_span_max else "FAIL ✗"
    }
    print_results(tail_results)
    
    # ==================== SPEED SWEEP ====================
    print_header("PERFORMANCE ACROSS SPEED RANGE")
    print("\n  V [m/s]  |   Cl   |    CD    | Drag [N] | Power [W] | Condition")
    print("  " + "-"*65)
    
    speeds = [
        (vstall, "Stall"),
        (vstall * 1.2, "Takeoff/Landing"),
        (vstall * 1.5, "Climb"),
        (vcruise, "Cruise (best L/D)"),
        (vcruise * 1.2, "Fast cruise"),
        (vcruise * 1.5, "High speed")
    ]
    
    speed_data = []
    for V, condition in speeds:
        Cl = (2 * W) / (rho * V**2 * S)
        Cd = Cd0 + k * Cl**2
        drag = 0.5 * rho * V**2 * S * Cd
        power = drag * V
        
        print(f"  {V:7.2f}  | {Cl:6.3f} | {Cd:8.5f} | {drag:8.2f} | {power:9.1f} | {condition}")
        speed_data.append([V, Cl, Cd, drag, power, condition])
    
    # # ==================== PARAMETRIC SWEEP ====================
    # print_header("PARAMETRIC CRUISE SPEED SWEEP")
    # print("\nWould you like to run a parametric sweep across multiple cruise speeds?")
    # print("This will analyze performance from 15-30 m/s and generate plots.")
    
    # run_sweep = input("Run parametric sweep? (y/n): ").strip().lower()
    
    # if run_sweep == 'y':
    #     print("\n  Running parametric sweep...")
        
    #     V_wind_sweep = get_float_input("\n  Mean wind speed [m/s]", 10.0)
    #     V_gust_sweep = get_float_input("  Gust speed [m/s]", 3.0)
        
    #     # Run sweep
    #     sweep_results, V_opt, LD_opt = run_speed_sweep(
    #         W, S, Cl_max, Cd0, e, AR, vstall_max, V_wind_sweep, V_gust_sweep
    #     )
        
    #     if not sweep_results:
    #         print("\n  ✗ No feasible speeds found!")
    #     else:
    #         # Print table
    #         print("\n  " + "="*110)
    #         print("  PARAMETRIC SWEEP RESULTS")
    #         print("  " + "="*110)
    #         print(f"  {'V':>5} | {'Cl':>6} | {'L/D':>6} | {'Power':>7} | {'Motor':>7} | " +
    #               f"{'Ground':>7} | {'E.Loss':>7} | {'P.Ratio':>7} | {'+Mass':>7}")
    #         print(f"  {'[m/s]':>5} | {' ':>6} | {' ':>6} | {'[W]':>7} | {'[W]':>7} | " +
    #               f"{'[m/s]':>7} | {'[%]':>7} | {'[×]':>7} | {'[g]':>7}")
    #         print("  " + "-"*110)
            
    #         for r in sweep_results:
    #             print(f"  {r['V']:5.0f} | {r['Cl']:6.3f} | {r['LD']:6.1f} | " +
    #                   f"{r['power']:7.0f} | {r['motor_power']:7.0f} | " +
    #                   f"{r['ground_speed']:7.1f} | {(1-r['LD_ratio'])*100:7.0f} | " +
    #                   f"{r['power_ratio']:7.1f} | {r['mass_increase']*1000:7.0f}")
            
    #         print("  " + "="*110)
            
    #         # Save CSV
    #         try:
    #             with open('uav_parametric_sweep.csv', 'w', newline='') as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(["Parametric Cruise Speed Sweep Results"])
    #                 writer.writerow([])
    #                 writer.writerow(["Speed [m/s]", "Cl", "CD", "L/D", "Drag [N]", 
    #                                "Power [W]", "Motor [W]", "Ground Speed [m/s]",
    #                                "Efficiency Loss [%]", "Power Ratio", "Energy [Wh]", "Added Mass [g]"])
                    
    #                 for r in sweep_results:
    #                     writer.writerow([
    #                         f"{r['V']:.1f}", f"{r['Cl']:.4f}", f"{r['Cd']:.5f}",
    #                         f"{r['LD']:.2f}", f"{r['drag']:.2f}", f"{r['power']:.1f}",
    #                         f"{r['motor_power']:.1f}", f"{r['ground_speed']:.2f}",
    #                         f"{(1-r['LD_ratio'])*100:.1f}", f"{r['power_ratio']:.2f}",
    #                         f"{r['total_energy']:.2f}", f"{r['mass_increase']*1000:.1f}"
    #                     ])
                
    #             print(f"\n  ✓ Sweep results saved to: {os.path.join(os.getcwd(), 'uav_parametric_sweep.csv')}")
    #         except Exception as e:
    #             print(f"\n  Error saving sweep results: {e}")
            
    #         # Generate plots
    #         if PLOTTING_AVAILABLE:
    #             print("\n  Generating plots...")
    #             print(f"  Working directory: {os.getcwd()}")
                
    #             plot_type = input("\n  Plot type: (1) Comprehensive 9-plot, (2) Simple 2-plot, (3) Both? [1/2/3]: ").strip()
                
    #             if plot_type == '1' or plot_type == '3':
    #                 print("\n  Creating comprehensive 9-plot figure...")
    #                 print("  Close the plot window to continue...")
    #                 plot_parametric_sweep(sweep_results, vcruise, LD_max, W, S, vstall, vstall_max, wingspan, MAC)
                
    #             if plot_type == '2' or plot_type == '3':
    #                 print("\n  Creating simple 2-plot figure...")
    #                 print("  Close the plot window to continue...")
    #                 plot_simple_comparison(sweep_results, vcruise, wingspan, MAC)
                
    #             print("\n  Plotting complete!")
    #         else:
    #             print("\n  Install matplotlib to generate plots: pip install matplotlib")
    
    # ==================== SAVE RESULTS ====================
    print_header("SAVE RESULTS")
    save = input("\nSave results to CSV file? (y/n): ").strip().lower()
    
    if save == 'y':
        filename = input("Enter filename (default: uav_results.csv): ").strip()
        if not filename:
            filename = "uav_results.csv"
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["UAV AERODYNAMICS ANALYSIS RESULTS"])
                writer.writerow([])
                writer.writerow(["INPUTS"])
                writer.writerow(["Parameter", "Value", "Unit"])
                writer.writerow(["Payload mass", payload_mass, "kg"])
                writer.writerow(["MTOW", MTOW, "kg"])
                writer.writerow(["Aspect Ratio", AR, "-"])
                writer.writerow(["Wingspan", wingspan, "m"])
                writer.writerow(["Cl,max", Cl_max, "-"])
                writer.writerow(["CD,0", Cd0, "-"])
                writer.writerow(["Oswald e", e, "-"])
                writer.writerow([])
                writer.writerow(["RESULTS"])
                writer.writerow(["Parameter", "Value", "Unit"])
                writer.writerow(["Wing Area", f"{S:.3f}", "m^2"])
                writer.writerow(["MAC", f"{MAC:.3f}", "m"])
                writer.writerow(["Wing Loading", f"{wing_loading:.1f}", "N/m^2"])
                writer.writerow(["Vstall", f"{vstall:.2f}", "m/s"])
                writer.writerow(["Vcruise", f"{vcruise:.2f}", "m/s"])
                writer.writerow(["L/D max", f"{LD_max:.1f}", "-"])
                writer.writerow(["Power @ stall", f"{power_stall:.1f}", "W"])
                writer.writerow(["Power @ cruise", f"{power_cruise:.1f}", "W"])
                writer.writerow([])
                writer.writerow(["SPEED SWEEP"])
                writer.writerow(["V [m/s]", "Cl", "CD", "Drag [N]", "Power [W]", "Condition"])
                for row in speed_data:
                    writer.writerow([f"{row[0]:.2f}", f"{row[1]:.3f}", f"{row[2]:.5f}", 
                                   f"{row[3]:.2f}", f"{row[4]:.1f}", row[5]])
            
            print(f"\n  ✓ Results saved to: {os.path.join(os.getcwd(), filename)}")
        except Exception as e:
            print(f"\n  Error saving file: {e}")
    
    # ==================== SUMMARY ====================
    print_header("DESIGN SUMMARY")
    summary_text = f"""
  Configuration: AR={AR}, b={wingspan}m, S={S:.2f}m², MAC={MAC:.3f}m
  
  Key Performance:
    • Stall Speed:    {vstall:.2f} m/s  ({'✓ PASS' if vstall <= vstall_max else '✗ FAIL'})
    • Cruise Speed:   {vcruise:.2f} m/s (best L/D = {LD_max:.1f})
    • Wing Loading:   {wing_loading:.1f} N/m²
    • Power @ Cruise: {power_cruise:.0f} W (motor: {power_cruise/0.75:.0f} W)
"""
    
    if sweep_results:
        summary_text += f"""
  Parametric Sweep Completed:
    • Speed range analyzed: 15-30 m/s
    • {len(sweep_results)} feasible configurations found
    • Results saved to: uav_parametric_sweep.csv
    • Plots generated: Check PNG files
"""
    
    summary_text += f"""
  Status: {'ALL REQUIREMENTS MET ✓' if all_pass else 'REQUIREMENTS NOT MET ✗'}
    """
    
    print(summary_text)
    print("="*60 + "\n")
    
    # List generated files
    print_header("GENERATED FILES")
    generated_files = []
    
    for fname in ['uav_results.csv', 'uav_parametric_sweep.csv', 
                  'uav_parametric_sweep.png', 'uav_simple_comparison.png']:
        fpath = os.path.join(os.getcwd(), fname)
        if os.path.exists(fpath):
            fsize = os.path.getsize(fpath)
            generated_files.append((fname, fsize))
    
    if generated_files:
        print("\n  Files created in this session:")
        for fname, fsize in generated_files:
            size_kb = fsize / 1024
            print(f"    ✓ {fname:<30} ({size_kb:.1f} KB)")
        print(f"\n  Location: {os.getcwd()}")
    else:
        print("\n  No files were generated in this session.")
    
    print("\n" + "="*60 + "\n")

# ============================================================================
# PROGRAM ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCalculation interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()