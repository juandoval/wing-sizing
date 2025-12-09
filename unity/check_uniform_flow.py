import pandas as pd
import numpy as np

df = pd.read_csv('unity/liftCurve_16.txt', sep='\t')
df.columns = df.columns.str.strip()

aoa = df['Aircraft Simulation Angle Of Attack_Deg'].values
V = 16
q = 0.5 * 1.225 * V**2
S = 0.367 * 1.8

print("UNIFORM FLOW FORCES (aerodynamic only, no propeller/weight):")
print("="*80)

for target in [0, 5, 10]:
    idx = np.abs(aoa - target) < 0.5
    if idx.sum() > 0:
        UF_Y = df['Uniform Flow Net Local Force Y'].values[idx].mean()
        UF_Z = df['Uniform Flow Net Local Force Z'].values[idx].mean()
        
        Cl = -UF_Y / (q * S)
        Cd = -UF_Z / (q * S)
        
        print(f'\nAoA ~ {target}°:')
        print(f'  Uniform Flow Y (lift): {UF_Y:7.2f} N')
        print(f'  Uniform Flow Z (drag): {UF_Z:7.2f} N')
        print(f'  -> Cl = {Cl:.3f}')
        print(f'  -> Cd = {Cd:.3f}')

print("\n" + "="*80)
print("SHOULD USE: Uniform Flow Net Local Force Y and Z!")
print("These are the pure aerodynamic forces in the body frame.")
