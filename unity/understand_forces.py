import pandas as pd
import numpy as np

df = pd.read_csv('unity/liftCurve_16.txt', sep='\t')
df.columns = df.columns.str.strip()

# Look at all the force columns available
force_cols = [col for col in df.columns if 'Force' in col]
print("Available Force Columns:")
for col in force_cols:
    print(f"  - {col}")

print("\n" + "="*80)
print("Let's compare Global vs Local forces:")
print("="*80)

# Check at a specific AoA
for target_aoa in [0, 5, 10]:
    idx = np.abs(df['Aircraft Simulation Angle Of Attack_Deg'].values - target_aoa) < 0.5
    if idx.sum() > 0:
        print(f"\nAt AoA ≈ {target_aoa}°:")
        print(f"  Global Force X: {df['Aircraft Simulation Net Global Force X'].values[idx].mean():7.2f} N")
        print(f"  Global Force Y: {df['Aircraft Simulation Net Global Force Y'].values[idx].mean():7.2f} N (vertical/up)")
        print(f"  Global Force Z: {df['Aircraft Simulation Net Global Force Z'].values[idx].mean():7.2f} N (forward)")
        print(f"  ---")
        print(f"  Local Force X:  {df['Aircraft Simulation Net Local Force X'].values[idx].mean():7.2f} N (body/side)")
        print(f"  Local Force Y:  {df['Aircraft Simulation Net Local Force Y'].values[idx].mean():7.2f} N (body/up)")
        print(f"  Local Force Z:  {df['Aircraft Simulation Net Local Force Z'].values[idx].mean():7.2f} N (body/forward)")
        
        # In body/local frame at AoA:
        # Local Y should be perpendicular to body (lift-ish)
        # Local Z should be along body axis (drag-ish)
        V = df['Aircraft Simulation Airspeed'].values[idx].mean()
        q = 0.5 * 1.225 * V**2
        S = 0.367 * 1.8
        
        Cl_local = -df['Aircraft Simulation Net Local Force Y'].values[idx].mean() / (q * S)
        Cd_local = -df['Aircraft Simulation Net Local Force Z'].values[idx].mean() / (q * S)
        
        print(f"  Airspeed: {V:.2f} m/s")
        print(f"  If using Local Forces directly:")
        print(f"    Cl = {Cl_local:.3f}")
        print(f"    Cd = {Cd_local:.3f}")
