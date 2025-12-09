import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============== CHANGE THESE ==============
# Add your files here - each entry is (filename, label)
# Files for all plots (Cl, Cd, Cm)
FILES_ALL = [
    # ('unity\\data_12.txt', '12 m/s'),
    # ('unity\\data_14.txt', '14 m/s'),
    ('unity\\data_good_2.txt', '16 m/s'),
]

# Files ONLY for Cm vs Cl plot (elevator deflection data)
FILES_CM_ONLY = [
    ('unity\\Cm_Cl_10deg_v2.txt', '10 deg'),
    ('unity\\Cm_Cl_min10deg.txt', '-10 deg')
]

OUTPUT_FILE = 'aero_plots.png'
TITLE = 'Aerodynamic Data - Velocity Comparison'

# Column names
AOA_COLUMN = 'Aircraft Simulation Angle Of Attack_Deg'
DRAG_COLUMN = 'Aircraft Simulation Net Global Force Z'  # Force Z = Drag
LIFT_COLUMN = 'Aircraft Simulation Net Global Force Y'  # Force Y = Lift
MOMENT_COLUMN = 'Aircraft Simulation Net Global Moment Y'  # Moment Y = Pitching moment

# Aircraft parameters (for coefficient calculation)
CHORD = 0.367  # m
WINGSPAN = 1.8  # m
S = CHORD * WINGSPAN  # Reference area (m²)
RHO = 1.225  # kg/m³

# Filtering parameters
AOA_THRESHOLD = 10
DRAG_THRESHOLD = 10
# ==========================================

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# Create 4 subplots: Cl vs AoA, Cd vs AoA, Cl vs Cd, Cm vs Cl
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Process files for all plots
for i, (data_file, label) in enumerate(FILES_ALL):
    try:
        # Read data file
        df = pd.read_csv(data_file, sep='\t')
        df.columns = df.columns.str.strip()
        
        # Extract values
        aoa = df[AOA_COLUMN].values
        F_Y = df[LIFT_COLUMN].values  # Global Force Y (vertical)
        F_Z = df[DRAG_COLUMN].values  # Global Force Z (forward/backward)
        M_Y = df[MOMENT_COLUMN].values  # Pitching moment about Y axis
        
        # Transform global forces to aerodynamic lift and drag
        # Unity: Y=up, Z=forward, aircraft pitches about Y axis
        # At angle of attack α:
        # Lift (perpendicular to velocity) = F_Y * cos(α) - F_Z * sin(α)
        # Drag (parallel to velocity) = F_Y * sin(α) + F_Z * cos(α)
        alpha_rad = np.deg2rad(aoa)
        lift = F_Y * np.cos(alpha_rad) - F_Z * np.sin(alpha_rad)
        drag = F_Y * np.sin(alpha_rad) + F_Z * np.cos(alpha_rad)
        moment = M_Y  # Pitching moment (positive = nose up)
        
        # Get actual airspeed from data if available
        if 'Aircraft Simulation Airspeed' in df.columns:
            V = df['Aircraft Simulation Airspeed'].values
        else:
            # Extract velocity from label as fallback
            V_val = float(label.split()[0])
            V = np.full_like(aoa, V_val)
        
        # Calculate dynamic pressure (element-wise if V is array)
        q = 0.5 * RHO * V**2
        
        # Calculate coefficients (no filtering)
        Cl = lift / (q * S)
        Cd = (drag / (q * S)) + 0.1  # +0.1 empirical correction for parasitic drag not captured by Unity
        Cm = moment / (q * S * CHORD)  # Moment coefficient normalized by chord
        
        # Filter out noisy Cm values for Cm vs Cl plot
        mask_cm = (Cm >= -0.8) & (Cm <= 0.8)
        Cl_filtered = Cl[mask_cm]
        Cm_filtered = Cm[mask_cm]
        
        color = COLORS[i % len(COLORS)]
        
        # Plot 1: Cl vs AoA (top-left)
        axes[0,0].scatter(aoa, Cl, alpha=0.5, s=15, c=color, label=label)
        
        # Plot 2: Cd vs AoA (top-right)
        axes[0,1].scatter(aoa, Cd, alpha=0.5, s=15, c=color, label=label)
        
        # Plot 3: Cl vs Cd (bottom-left)
        axes[1,0].scatter(Cd, Cl, alpha=0.7, s=30, c=color, label=label)
        
        # Don't plot FILES_ALL in Cm vs Cl (bottom-right) - only FILES_CM_ONLY will be plotted there
        
        print(f"{label}: {len(aoa)} data points, Cm range: [{Cm.min():.4f}, {Cm.max():.4f}]")
        
    except FileNotFoundError as e:
        print(f"Warning: {e}, skipping...")
    except Exception as e:
        print(f"Error processing {label}: {e}")

# Process files for Cm plot only (elevator deflection data)
for i, (data_file, label) in enumerate(FILES_CM_ONLY):
    try:
        df = pd.read_csv(data_file, sep='\t')
        df.columns = df.columns.str.strip()
        
        aoa = df[AOA_COLUMN].values
        F_Y = df[LIFT_COLUMN].values
        F_Z = df[DRAG_COLUMN].values
        M_Y = df[MOMENT_COLUMN].values
        
        alpha_rad = np.deg2rad(aoa)
        lift = F_Y * np.cos(alpha_rad) - F_Z * np.sin(alpha_rad)
        moment = M_Y
        
        if 'Aircraft Simulation Airspeed' in df.columns:
            V = df['Aircraft Simulation Airspeed'].values
        else:
            V_val = float(label.split()[0])
            V = np.full_like(aoa, V_val)
        
        q = 0.5 * RHO * V**2
        Cl = lift / (q * S)
        Cm = moment / (q * S * CHORD)
        
        # Debug: print raw values
        print(f"\n{label} DEBUG:")
        print(f"  Raw moment range: [{moment.min():.4f}, {moment.max():.4f}] N·m")
        print(f"  q*S*CHORD = {(q[0] * S * CHORD):.2f}")
        print(f"  Cm range: [{Cm.min():.6f}, {Cm.max():.6f}]")
        
        # Don't filter for elevator data - we want to see the actual range
        color = COLORS[(i + len(FILES_ALL)) % len(COLORS)]
        
        # Only plot in Cm vs Cl (bottom-right)
        axes[1,1].scatter(Cl, Cm, alpha=0.7, s=30, c=color, label=label)
        
        print(f"{label} (Cm only): {len(aoa)} data points")
        
    except FileNotFoundError as e:
        print(f"Warning: {e}, skipping...")
    except Exception as e:
        print(f"Error processing {label}: {e}")

# Format plots
# Top-left: Cl vs AoA
axes[0,0].set_xlabel('Angle of Attack (°)', fontsize=12)
axes[0,0].set_ylabel('$C_L$', fontsize=12)
axes[0,0].set_title('$C_L$ vs AoA', fontsize=12)
axes[0,0].grid(True, alpha=0.3)
axes[0,0].axhline(y=0, color='k', linewidth=0.5)
axes[0,0].axvline(x=0, color='k', linewidth=0.5)
axes[0,0].legend()

# Top-right: Cd vs AoA
axes[0,1].set_xlabel('Angle of Attack (°)', fontsize=12)
axes[0,1].set_ylabel('$C_D$', fontsize=12)
axes[0,1].set_title('$C_D$ vs AoA', fontsize=12)
axes[0,1].grid(True, alpha=0.3)
axes[0,1].axhline(y=0, color='k', linewidth=0.5)
axes[0,1].axvline(x=0, color='k', linewidth=0.5)
axes[0,1].legend()

# Bottom-left: Drag Polar
axes[1,0].set_xlabel('$C_D$', fontsize=12)
axes[1,0].set_ylabel('$C_L$', fontsize=12)
axes[1,0].set_title('Drag Polar ($C_L$ vs $C_D$)', fontsize=12)
axes[1,0].grid(True, alpha=0.3)
axes[1,0].axhline(y=0, color='k', linewidth=0.5)
axes[1,0].axvline(x=0, color='k', linewidth=0.5)
axes[1,0].legend()

# Bottom-right: Cm vs Cl
axes[1,1].set_xlabel('$C_L$', fontsize=12)
axes[1,1].set_ylabel('$C_m$', fontsize=12)
axes[1,1].set_title('$C_m$ vs $C_L$', fontsize=12)
axes[1,1].grid(True, alpha=0.3)
axes[1,1].axhline(y=0, color='k', linewidth=0.5)
axes[1,1].axvline(x=0, color='k', linewidth=0.5)
axes[1,1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
plt.show()

print(f"Saved to {OUTPUT_FILE}")