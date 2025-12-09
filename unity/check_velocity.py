import pandas as pd
import numpy as np

# Read both files
df_drag = pd.read_csv('unity/dragPolar_16.txt', sep='\t')
df_lift = pd.read_csv('unity/liftCurve_16.txt', sep='\t')

# Clean column names
df_drag.columns = df_drag.columns.str.strip()
df_lift.columns = df_lift.columns.str.strip()

print("=" * 80)
print("DRAG POLAR FILE (dragPolar_16.txt)")
print("=" * 80)
print(f"Columns: {list(df_drag.columns)}")
print(f"\nNumber of data points: {len(df_drag)}")
print(f"\nAoA range: {df_drag['Aircraft Simulation Angle Of Attack_Deg'].min():.2f} to {df_drag['Aircraft Simulation Angle Of Attack_Deg'].max():.2f} deg")
print(f"Drag force range: {df_drag['Aircraft Simulation Net Global Force Z'].min():.2f} to {df_drag['Aircraft Simulation Net Global Force Z'].max():.2f} N")

print("\n" + "=" * 80)
print("LIFT CURVE FILE (liftCurve_16.txt)")
print("=" * 80)
print(f"Number of columns: {len(df_lift.columns)}")
print(f"Number of data points: {len(df_lift)}")

# Check if airspeed column exists
if 'Aircraft Simulation Airspeed' in df_lift.columns:
    airspeed = df_lift['Aircraft Simulation Airspeed']
    print(f"\n✓ Airspeed column found!")
    print(f"  Min airspeed: {airspeed.min():.2f} m/s")
    print(f"  Max airspeed: {airspeed.max():.2f} m/s")
    print(f"  Mean airspeed: {airspeed.mean():.2f} m/s")
    print(f"  Std dev: {airspeed.std():.3f} m/s")
else:
    print("✗ No airspeed column found")

print(f"\nAoA range: {df_lift['Aircraft Simulation Angle Of Attack_Deg'].min():.2f} to {df_lift['Aircraft Simulation Angle Of Attack_Deg'].max():.2f} deg")
print(f"Lift force range: {df_lift['Aircraft Simulation Net Global Force Y'].min():.2f} to {df_lift['Aircraft Simulation Net Global Force Y'].max():.2f} N")
print(f"Drag force range: {df_lift['Aircraft Simulation Net Global Force Z'].min():.2f} to {df_lift['Aircraft Simulation Net Global Force Z'].max():.2f} N")

print("\n" + "=" * 80)
print("KEY FINDING")
print("=" * 80)
print("\nThe dragPolar_16.txt file has NO airspeed column.")
print("The liftCurve_16.txt file HAS actual airspeed measurements.")
print("\nWhen plot.py uses a fixed 16 m/s for both files:")
print(f"  - dragPolar: Assumes 16.00 m/s (unknown actual velocity)")
print(f"  - liftCurve: Assumes 16.00 m/s (but actual is ~{df_lift['Aircraft Simulation Airspeed'].mean():.2f} m/s)")
print(f"\nVelocity error: {((16.0 / df_lift['Aircraft Simulation Airspeed'].mean()) - 1) * 100:.1f}%")
print(f"Coefficient error: {((16.0**2 / df_lift['Aircraft Simulation Airspeed'].mean()**2) - 1) * 100:.1f}% (velocity squared in dynamic pressure)")

# Check if dragPolar data might be from the same simulation
print("\n" + "=" * 80)
print("COMPARING DATA SOURCES")
print("=" * 80)

# Check if drag forces overlap
drag_from_lift = df_lift['Aircraft Simulation Net Global Force Z']
drag_from_drag = df_drag['Aircraft Simulation Net Global Force Z']

print(f"\nDrag force comparison:")
print(f"  liftCurve.txt drag: {drag_from_lift.min():.2f} to {drag_from_lift.max():.2f} N")
print(f"  dragPolar.txt drag: {drag_from_drag.min():.2f} to {drag_from_drag.max():.2f} N")

# Check time ranges
if 'Time' in df_drag.columns and 'Time' in df_lift.columns:
    print(f"\nTime ranges:")
    print(f"  liftCurve.txt: {df_lift['Time'].min():.1f} to {df_lift['Time'].max():.1f} s")
    print(f"  dragPolar.txt: {df_drag['Time'].min():.1f} to {df_drag['Time'].max():.1f} s")
    
    if df_drag['Time'].max() < df_lift['Time'].max():
        print("\n  → dragPolar.txt appears to be from a LONGER simulation run")
    else:
        print("\n  → Files may be from different simulation runs")
