import pandas as pd
import numpy as np

df = pd.read_csv('unity/liftCurve_16.txt', sep='\t')
df.columns = df.columns.str.strip()

print("Raw Force Ranges:")
print(f"Force Y: {df['Aircraft Simulation Net Global Force Y'].min():.2f} to {df['Aircraft Simulation Net Global Force Y'].max():.2f} N")
print(f"Force Z: {df['Aircraft Simulation Net Global Force Z'].min():.2f} to {df['Aircraft Simulation Net Global Force Z'].max():.2f} N")

# Check at different AoA
for target_aoa in [0, 5, 10]:
    idx = np.abs(df['Aircraft Simulation Angle Of Attack_Deg'].values - target_aoa) < 0.5
    if idx.sum() > 0:
        F_Y = df['Aircraft Simulation Net Global Force Y'].values[idx].mean()
        F_Z = df['Aircraft Simulation Net Global Force Z'].values[idx].mean()
        alpha = np.deg2rad(target_aoa)
        
        # Current transformation
        L_current = -F_Y * np.cos(alpha) + F_Z * np.sin(alpha)
        D_current = -F_Y * np.sin(alpha) - F_Z * np.cos(alpha)
        
        # Maybe just F_Y is lift?
        L_simple = -F_Y
        
        V = 16
        q = 0.5 * 1.225 * V**2
        S = 0.367 * 1.8
        
        print(f"\nAt AoA ≈ {target_aoa}°:")
        print(f"  F_Y = {F_Y:.2f} N, F_Z = {F_Z:.2f} N")
        print(f"  Current transform -> Cl = {L_current/(q*S):.3f}, Cd = {D_current/(q*S):.3f}")
        print(f"  Simple F_Y -> Cl = {L_simple/(q*S):.3f}")
        print(f"  Simple -F_Z -> Cd = {-F_Z/(q*S):.3f}")
