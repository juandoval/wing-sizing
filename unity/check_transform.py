import pandas as pd
import numpy as np

# Check the transformation for one file
df = pd.read_csv('unity/data_16.txt', sep='\t')
df.columns = df.columns.str.strip()

aoa = df['Aircraft Simulation Angle Of Attack_Deg'].values
F_Y = df['Aircraft Simulation Net Global Force Y'].values
F_Z = df['Aircraft Simulation Net Global Force Z'].values

# Current transformation
alpha_rad = np.deg2rad(aoa)
lift_v1 = F_Y * np.cos(alpha_rad) - F_Z * np.sin(alpha_rad)
drag_v1 = F_Y * np.sin(alpha_rad) + F_Z * np.cos(alpha_rad)

# Alternative transformation (reversed signs)
lift_v2 = F_Y * np.cos(alpha_rad) + F_Z * np.sin(alpha_rad)
drag_v2 = -F_Y * np.sin(alpha_rad) + F_Z * np.cos(alpha_rad)

# Alternative 3 (Z is drag direction, Y is lift direction with rotation)
lift_v3 = F_Y
drag_v3 = -F_Z

V = 16
q = 0.5 * 1.225 * V**2
S = 0.367 * 1.8

# Check at AoA near 0
idx = np.abs(aoa) < 1
print("At AoA ≈ 0°:")
print(f"  F_Y: {F_Y[idx].mean():.3f} N, F_Z: {F_Z[idx].mean():.3f} N")
print(f"\nVersion 1 (current):")
print(f"  Lift: {lift_v1[idx].mean():.3f} N, Drag: {drag_v1[idx].mean():.3f} N")
print(f"  Cl: {(lift_v1[idx]/q/S).mean():.3f}, Cd: {(drag_v1[idx]/q/S).mean():.3f}")
print(f"\nVersion 2 (+ instead of -):")
print(f"  Lift: {lift_v2[idx].mean():.3f} N, Drag: {drag_v2[idx].mean():.3f} N")
print(f"  Cl: {(lift_v2[idx]/q/S).mean():.3f}, Cd: {(drag_v2[idx]/q/S).mean():.3f}")
print(f"\nVersion 3 (no transform):")
print(f"  Lift: {lift_v3[idx].mean():.3f} N, Drag: {drag_v3[idx].mean():.3f} N")
print(f"  Cl: {(lift_v3[idx]/q/S).mean():.3f}, Cd: {(drag_v3[idx]/q/S).mean():.3f}")

# Check at AoA = 5
idx5 = (aoa > 4) & (aoa < 6)
print("\n" + "="*60)
print("At AoA ≈ 5°:")
print(f"  F_Y: {F_Y[idx5].mean():.3f} N, F_Z: {F_Z[idx5].mean():.3f} N")
print(f"\nVersion 1 (current):")
print(f"  Lift: {lift_v1[idx5].mean():.3f} N, Drag: {drag_v1[idx5].mean():.3f} N")
print(f"  Cl: {(lift_v1[idx5]/q/S).mean():.3f}, Cd: {(drag_v1[idx5]/q/S).mean():.3f}")
print(f"\nVersion 2 (+ instead of -):")
print(f"  Lift: {lift_v2[idx5].mean():.3f} N, Drag: {drag_v2[idx5].mean():.3f} N")
print(f"  Cl: {(lift_v2[idx5]/q/S).mean():.3f}, Cd: {(drag_v2[idx5]/q/S).mean():.3f}")

print("\n" + "="*60)
print("PHYSICS CHECK:")
print("- Lift should increase with AoA")
print("- Drag should be positive and increase with AoA")
print("- At AoA=0, Cl should be small positive (cambered airfoil)")
print("- Cd should always be positive (drag always opposes motion)")
