import pandas as pd
import numpy as np

df = pd.read_csv('unity/data_good_2.txt', sep='\t')
df.columns = df.columns.str.strip()

aoa = df['Aircraft Simulation Angle Of Attack_Deg'].values
F_Y = df['Aircraft Simulation Net Global Force Y'].values
F_Z = df['Aircraft Simulation Net Global Force Z'].values

# Check at AoA near 0
idx0 = np.abs(aoa) < 1
print("At AoA ≈ 0°:")
print(f"  F_Y: {F_Y[idx0].mean():.2f} N")
print(f"  F_Z: {F_Z[idx0].mean():.2f} N")

# Check at AoA = 5
idx5 = (aoa > 4) & (aoa < 6)
print("\nAt AoA ≈ 5°:")
print(f"  F_Y: {F_Y[idx5].mean():.2f} N")
print(f"  F_Z: {F_Z[idx5].mean():.2f} N")

print("\n" + "="*60)
print("ANALYSIS:")
print("If F_Z is NEGATIVE at cruise, it means drag opposes forward (+Z) motion")
print("So we probably DON'T need transformation at all!")
print("\nTry:")
print("  lift = F_Y  (already in vertical direction)")
print("  drag = -F_Z (negate to make positive)")
