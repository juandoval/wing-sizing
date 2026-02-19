"""
Unity vs 2D NeuralFoil Comparison
Plots Unity experimental data alongside 2D airfoil analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
import aerosandbox as asb

# ============================================================================
# CONFIGURATION
# ============================================================================

# Unity data file
UNITY_DATA_FILE = 'data_good_2.txt'

# Cm vs CL specific data files (better quality)
UNITY_CM_FILE = 'mom_cg14.txt'  # 0° elevator deflection
UNITY_CM_FILE_UP = 'mom_cg14_-5.txt'  # -5° elevator deflection (up)
UNITY_CM_FILE_DOWN = 'mom_cg14_5.txt'  # +5° elevator deflection (down)

# CG position data files
UNITY_CG_FILES = {
    'CG 0.1': 'mom_cg1.txt',
    'CG 0.12': 'mom_cg12.txt',
    'CG 0.14': 'mom_cg14.txt',
    'CG 0.16': 'mom_cg16.txt',
    'CG 0.18': 'mom_cg18.txt'
}

# Airfoil data
AIRFOIL_FILE = 'supercub_normalized.dat'

# Analysis parameters
velocity = [16]  # m/s
Re = [400e3]

# Unity parameters (for coefficient calculation)
CHORD_UNITY = 0.367  # m
WINGSPAN_UNITY = 1.8  # m
S_UNITY = CHORD_UNITY * WINGSPAN_UNITY  # Reference area (m²)
RHO = 1.225  # kg/m³
WING_INCIDENCE = 2.5  # deg

# Column names in Unity data
AOA_COLUMN = 'Aircraft Simulation Angle Of Attack_Deg'
DRAG_COLUMN = 'Aircraft Simulation Net Global Force Z'
LIFT_COLUMN = 'Aircraft Simulation Net Global Force Y'
MOMENT_COLUMN = 'Aircraft Simulation Net Global Moment Y'

# 2D Analysis parameters
CHORD = 0.367  # m
SPAN = 1.8  # m
S = CHORD * SPAN  # m²
Weight = 4.9 * 9.81  # N

# ============================================================================
# FUNCTIONS
# ============================================================================

def compute_mach(velocity: float) -> float:
    SPEED_OF_SOUND = 343
    return velocity / SPEED_OF_SOUND

def load_unity_data(filename):
    """Load and process Unity data"""
    df = pd.read_csv(filename, sep='\t')
    df.columns = df.columns.str.strip()
    
    # Extract values
    aoa = df[AOA_COLUMN].values
    F_Y = df[LIFT_COLUMN].values
    F_Z = df[DRAG_COLUMN].values
    M_Y = df[MOMENT_COLUMN].values
    
    # Transform to aerodynamic frame
    alpha_rad = np.deg2rad(aoa)
    lift = F_Y * np.cos(alpha_rad) - F_Z * np.sin(alpha_rad)
    drag = F_Y * np.sin(alpha_rad) + F_Z * np.cos(alpha_rad)
    moment = M_Y
    
    # Get velocity
    if 'Aircraft Simulation Airspeed' in df.columns:
        V = df['Aircraft Simulation Airspeed'].values
    else:
        V = np.full_like(aoa, 16.0)
    
    # Calculate coefficients
    q = 0.5 * RHO * V**2
    Cl = lift / (q * S_UNITY)
    Cd_raw = drag / (q * S_UNITY)
    
    # Apply Cd0 correction to match 2D profile drag baseline
    target_Cd0 = 0.02
    current_min_Cd = Cd_raw.min()
    correction = target_Cd0 - current_min_Cd
    Cd = Cd_raw + correction
    
    Cm = -moment / (q * S_UNITY * CHORD_UNITY)  # Flip sign convention
    
    return aoa, Cl, Cd, Cm

def load_unity_cm_data(filename):
    """Load and process Unity Cm vs CL specific data (forces in N, moments in Nm)"""
    df = pd.read_csv(filename, sep='\t')
    df.columns = df.columns.str.strip()
    
    # Extract raw forces and moments (in N and Nm)
    # This file only has lift force and moment - no angle of attack or drag
    F_Y = df['Aircraft Simulation Net Global Force Y'].values  # N (lift force)
    M_Y = df['Aircraft Simulation Net Global Moment X'].values  # Nm (pitching moment)
    #Aircraft Simulation Net Global Moment X	Aircraft Simulation Net Glo
    
    # Get velocity (assume 16 m/s if not in file)
    if 'Aircraft Simulation Airspeed' in df.columns:
        V = df['Aircraft Simulation Airspeed'].values
    else:
        V = 16.0  # Fixed velocity for this dataset
    
    # Calculate coefficients
    q = 0.5 * RHO * V**2
    Cl = F_Y / (q * S_UNITY)  # Lift force already in aerodynamic frame
    Cm = -M_Y / (q * S_UNITY * CHORD_UNITY)  # Flip sign convention
    
    return Cl, Cm

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("  Loading Unity Data")
print("="*80)

aoa_unity, Cl_unity, Cd_unity, Cm_unity = load_unity_data(UNITY_DATA_FILE)
LD_unity = Cl_unity / Cd_unity

# Convert Unity angles to wing reference (add incidence)
aoa_wing_unity = aoa_unity + WING_INCIDENCE

print(f"\nUnity data loaded: {len(aoa_unity)} points")
print(f"  AoA range (fuselage): {aoa_unity.min():.1f}° to {aoa_unity.max():.1f}°")
print(f"  AoA range (wing): {aoa_wing_unity.min():.1f}° to {aoa_wing_unity.max():.1f}°")
print(f"  CL range: {Cl_unity.min():.3f} to {Cl_unity.max():.3f}")
print(f"  CD range: {Cd_unity.min():.4f} to {Cd_unity.max():.4f}")

max_LD_unity = LD_unity.max()
max_LD_idx_unity = LD_unity.argmax()
print(f"  Max L/D: {max_LD_unity:.2f} at a_fuselage={aoa_unity[max_LD_idx_unity]:.1f} deg")

# Load Cm vs CL specific data (elevator deflections)
print("\nLoading Cm vs CL data (elevator deflections)...")
Cl_unity_cm_0, Cm_unity_cm_0 = load_unity_cm_data(UNITY_CM_FILE)
print(f"  0° deflection: {len(Cl_unity_cm_0)} points")

Cl_unity_cm_up, Cm_unity_cm_up = load_unity_cm_data(UNITY_CM_FILE_UP)
print(f"  -5° deflection (up): {len(Cl_unity_cm_up)} points")

Cl_unity_cm_down, Cm_unity_cm_down = load_unity_cm_data(UNITY_CM_FILE_DOWN)
print(f"  +5° deflection (down): {len(Cl_unity_cm_down)} points")

# Load CG position data
print("\nLoading CG position data...")
cg_data = {}
for cg_label, cg_file in UNITY_CG_FILES.items():
    Cl_cg, Cm_cg = load_unity_cm_data(cg_file)
    cg_data[cg_label] = (Cl_cg, Cm_cg)
    print(f"  {cg_label}: {len(Cl_cg)} points")

# Load CG position data
print("\nLoading CG position data...")
cg_data = {}
for cg_label, cg_file in UNITY_CG_FILES.items():
    Cl_cg, Cm_cg = load_unity_cm_data(cg_file)
    cg_data[cg_label] = (Cl_cg, Cm_cg)
    print(f"  {cg_label}: {len(Cl_cg)} points")

# ============================================================================
# 2D NEURALFOIL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("  Running 2D NeuralFoil Analysis")
print("="*80)

# Load airfoil
supercub_airfoil = asb.Airfoil(name='supercub', coordinates=AIRFOIL_FILE)
print(f"\nAirfoil loaded: {supercub_airfoil.name}")
print(f"  Coordinate points: {len(supercub_airfoil.coordinates)}")

# Create flat plate for comparison
flat_plate = asb.Airfoil(
    name="Flat Plate",
    coordinates=np.array([
        [1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0]
    ])
)

# Analysis parameters
alphas = np.linspace(-20, 20, 100)
V = velocity[0]
mach = compute_mach(V)

print(f"\nAnalysis conditions:")
print(f"  Velocity: {V} m/s")
print(f"  Reynolds number: {Re[0]:,.0f}")
print(f"  Mach number: {mach:.4f}")
print(f"  Alpha range: {alphas[0]:.1f}° to {alphas[-1]:.1f}°")

# Run analysis
print("\nRunning NeuralFoil...")
aero_supercub = supercub_airfoil.get_aero_from_neuralfoil(
    alpha=alphas,
    Re=Re[0],
    mach=mach,
)

aero_flatplate = flat_plate.get_aero_from_neuralfoil(
    alpha=alphas,
    Re=Re[0],
    mach=mach,
)

# Calculate L/D
LD_supercub = aero_supercub['CL'] / aero_supercub['CD']
LD_flatplate = aero_flatplate['CL'] / aero_flatplate['CD']

idx_best_LD = np.argmax(LD_supercub)
print(f"\n✓ Analysis complete")
print(f"  Max L/D (SuperCub): {LD_supercub[idx_best_LD]:.2f} at a={alphas[idx_best_LD]:.1f} deg")

# ============================================================================
# CREATE PLOTS
# ============================================================================

print("\n" + "="*80)
print("  Creating Comparison Plots")
print("="*80)

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax = ax.flatten()

# Plot 1: Lift Curve (CL vs Alpha)
ax[0].plot(alphas, aero_supercub['CL'], '-', color='red', linewidth=2.5, 
           label='Custom 35b')
ax[0].plot(alphas, aero_flatplate['CL'], '--', color='gray', linewidth=2, 
           label='Flat Plate')
ax[0].plot(aoa_unity, Cl_unity, '-', color='blue', linewidth=2,
           label='Unity')
ax[0].grid(True, alpha=0.3)
ax[0].set_xlabel('Angle of Attack (degrees)', fontsize=11)
ax[0].set_ylabel('Lift Coefficient (CL)', fontsize=11)
ax[0].set_title('Lift Curve', fontsize=12, fontweight='bold')
ax[0].legend(fontsize=10)
ax[0].set_xlim(-20, 20)

# Plot 2: Drag Polar (CL vs CD)
ax[1].plot(aero_supercub['CD'], aero_supercub['CL'], '-', color='red', 
           linewidth=2.5, label='Custom 35b')
ax[1].plot(aero_flatplate['CD'], aero_flatplate['CL'], '--', color='gray', 
           linewidth=2, label='Flat Plate')
ax[1].plot(Cd_unity, Cl_unity, '-', color='blue', linewidth=2,
           label='Unity')
ax[1].grid(True, alpha=0.3)
ax[1].set_xlabel('Drag Coefficient (CD)', fontsize=11)
ax[1].set_ylabel('Lift Coefficient (CL)', fontsize=11)
ax[1].set_title('Drag Polar', fontsize=12, fontweight='bold')
ax[1].legend(fontsize=10)

# Plot 3: Elevator Pitching Moment (Cm vs CL)
cm_key = 'CM' if 'CM' in aero_supercub else 'Cm'
if cm_key in aero_supercub:
    # ax[2].plot(aero_supercub['CL'], aero_supercub[cm_key], '-', color='red',
    #            linewidth=2.5, label='Custom 35b')
    # ax[2].plot(aero_flatplate['CL'], aero_flatplate[cm_key], '--', color='gray',
    #            linewidth=2, label='Flat Plate')
    # Unity data with different elevator deflections
    # Filter to remove initial sweep (from -2 to -20), keep only from max Cm onwards
    idx_0 = np.argmax(Cm_unity_cm_0)
    idx_up = np.argmax(Cm_unity_cm_up)
    idx_down = np.argmax(Cm_unity_cm_down)
    
    ax[2].scatter(Cl_unity_cm_0[idx_0:], Cm_unity_cm_0[idx_0:], s=15, color='blue', alpha=0.7,
                  label='0°')
    ax[2].scatter(Cl_unity_cm_up[idx_up:], Cm_unity_cm_up[idx_up:], s=15, color='orange', alpha=0.7,
                  label='-5° (up)')
    ax[2].scatter(Cl_unity_cm_down[idx_down:], Cm_unity_cm_down[idx_down:], s=15, color='green', alpha=0.7,
                  label='+5° (down)')
    ax[2].axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax[2].grid(True, alpha=0.3)
    ax[2].set_xlabel('Lift Coefficient (CL)', fontsize=11)
    ax[2].set_ylabel('Moment Coefficient (Cm)', fontsize=11)
    ax[2].set_title('Elevator Pitching Moment', fontsize=12, fontweight='bold')
    ax[2].legend(fontsize=9, loc='best')

# Plot 4: Pitch Stability (Cm vs CL for different CG positions)
if cm_key in aero_supercub:
    # ax[3].plot(aero_supercub['CL'], aero_supercub[cm_key], '-', color='red',
    #            linewidth=2.5, label='Custom 35b')
    # ax[3].plot(aero_flatplate['CL'], aero_flatplate[cm_key], '--', color='gray',
    #            linewidth=2, label='Flat Plate')
    
    # Unity data with different CG positions
    # Filter to remove initial sweep (from -2 to -20), keep only from max Cm onwards
    colors = ['green', 'cyan', 'blue', 'orange', 'purple']
    for (cg_label, (Cl_cg, Cm_cg)), color in zip(cg_data.items(), colors):
        idx_max = np.argmax(Cm_cg)
        ax[3].scatter(Cl_cg[idx_max:], Cm_cg[idx_max:], s=15, color=color, alpha=0.7,
                      label=f'{cg_label}')
    
    ax[3].axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax[3].grid(True, alpha=0.3)
    ax[3].set_xlabel('Lift Coefficient (CL)', fontsize=11)
    ax[3].set_ylabel('Moment Coefficient (Cm)', fontsize=11)
    ax[3].set_title('Pitch Stability (CG Positions)', fontsize=12, fontweight='bold')
    ax[3].legend(fontsize=9, loc='best')

plt.tight_layout()
plt.savefig('unity_vs_2d_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved: unity_vs_2d_comparison.png")
plt.show()

# ============================================================================
# PERFORMANCE COMPARISON (Console Output)
# ============================================================================

print("\n" + "="*80)
print("  PERFORMANCE COMPARISON")
print("="*80)

print(f"\nCustom 35b:")
print(f"  Max L/D: {LD_supercub[idx_best_LD]:.2f} at a = {alphas[idx_best_LD]:.1f} deg")
print(f"  CL = {aero_supercub['CL'][idx_best_LD]:.4f}")
print(f"  CD = {aero_supercub['CD'][idx_best_LD]:.5f} (profile drag only)")

print(f"\nUnity 3D (Full Aircraft):")
print(f"  Max L/D: {max_LD_unity:.2f} at a_fuselage = {aoa_unity[max_LD_idx_unity]:.1f} deg")
print(f"  CL = {Cl_unity[max_LD_idx_unity]:.4f}")
print(f"  CD = {Cd_unity[max_LD_idx_unity]:.5f} (profile + induced + fuselage)")

print(f"\nDrag Difference:")
print(f"  ΔCD = {Cd_unity[max_LD_idx_unity] - aero_supercub['CD'][idx_best_LD]:.5f}")
print(f"  Unity has {100*(Cd_unity[max_LD_idx_unity]/aero_supercub['CD'][idx_best_LD] - 1):.0f}% more drag")

print("\n" + "="*80)
print("  Analysis Complete!")
print("="*80)
