import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── aero parameters ───────────────────────────────────────────────────────────
rho   = 1.225          # air density [kg/m³]
span  = 1.8            # wing span [m]
chord = 0.367          # chord [m]
S     = span * chord   # wing reference area [m²]
V_min = 3.0            # min airspeed for CD calc (avoids /0 noise) [m/s]

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(
    r'C:\Users\juand\OneDrive\Documents\GitHub\wing-sizing\unity\glide8_60.txt',
    sep='\t', names=['time', 'altitude', 'pos_z', 'fx', 'fy', 'fz', 'airspeed'], header=0
)

# ── clip to flight phase (launch → first ground contact after peak) ───────────
peak_idx   = df['altitude'].idxmax()
after_peak = df.loc[peak_idx:]
landing    = after_peak[after_peak['altitude'] <= 0]
end_idx    = landing.index[0] if not landing.empty else df.index[-1]
flight     = df.loc[:end_idx].copy()
peak       = df.loc[peak_idx]

# ── CD = |Fz| / (½ρV²S)  ─────────────────────────────────────────────────────
q = 0.5 * rho * flight['airspeed']**2 * S          # dynamic pressure × area
flight['cd'] = np.abs(flight['fz']) / q.replace(0, np.nan)  # NaN where V≈0
cd_valid = flight[flight['airspeed'] >= V_min].copy()
cd_valid['cd_smooth'] = cd_valid['cd'].rolling(window=30, center=True,
                                                min_periods=1).mean()

# ── plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=False)

# ── 1 · trajectory ────────────────────────────────────────────────────────────
ax1.plot(flight['pos_z'], flight['altitude'], color='steelblue', linewidth=1.8)
ax1.fill_between(flight['pos_z'], flight['altitude'].min() - 2, 0,
                 color='saddlebrown', alpha=0.25, label='Ground')
ax1.axhline(0, color='saddlebrown', linewidth=1.5)
ax1.plot(peak['pos_z'], peak['altitude'], 'r^', markersize=9, zorder=5,
         label=f"Peak  {peak['altitude']:.1f} m  @  {peak['pos_z']:.0f} m")
ax1.plot(flight['pos_z'].iloc[0],  flight['altitude'].iloc[0],  'go',
         markersize=8, label='Launch')
ax1.plot(flight['pos_z'].iloc[-1], flight['altitude'].iloc[-1], 'ks',
         markersize=8, label='Landing')
ax1.set_xlabel('Horizontal position Z  [m]', fontsize=11)
ax1.set_ylabel('Altitude  [m]', fontsize=11)
ax1.set_title('Glide Trajectory  —  60 m launch', fontsize=12)
ax1.set_ylim(bottom=-3)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(fontsize=9)

# ── 2 · altitude vs time ──────────────────────────────────────────────────────
ax2.plot(flight['time'], flight['altitude'], color='steelblue', linewidth=1.8)
ax2.fill_between(flight['time'], flight['altitude'].min() - 2, 0,
                 color='saddlebrown', alpha=0.25)
ax2.axhline(0, color='saddlebrown', linewidth=1.5)
ax2.plot(peak['time'], peak['altitude'], 'r^', markersize=9, zorder=5)
ax2.set_xlabel('Time  [s]', fontsize=11)
ax2.set_ylabel('Altitude  [m]', fontsize=11)
ax2.set_title('Altitude vs Time', fontsize=12)
ax2.set_ylim(bottom=-3)
ax2.grid(True, linestyle=':', alpha=0.5)

# ── 3 · CD vs time ────────────────────────────────────────────────────────────
ax3.plot(cd_valid['time'], cd_valid['cd'], color='lightgray', linewidth=0.8,
         label='raw')
ax3.plot(cd_valid['time'], cd_valid['cd_smooth'], color='darkorange',
         linewidth=2, label='smoothed (30-pt)')
ax3.set_xlabel('Time  [s]', fontsize=11)
ax3.set_ylabel('$C_D$  [–]', fontsize=11)
ax3.set_title(f'Drag Coefficient vs Time   '
              f'(S = {S:.4f} m², ρ = {rho} kg/m³)', fontsize=12)
ax3.set_ylim(bottom=0)
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.legend(fontsize=9)

# ── print stats ───────────────────────────────────────────────────────────────
glide_range     = flight['pos_z'].iloc[-1] - flight['pos_z'].iloc[0]
glide_from_peak = flight['pos_z'].iloc[-1] - peak['pos_z']
glide_ratio     = glide_from_peak / peak['altitude'] if peak['altitude'] > 0 else np.nan
print(f"Peak altitude  : {peak['altitude']:.1f} m  at t = {peak['time']:.1f} s")
print(f"Total range    : {glide_range:.0f} m")
print(f"Range from peak: {glide_from_peak:.0f} m")
print(f"Glide ratio    : {glide_ratio:.1f}  (from peak)")
print(f"Flight duration: {flight['time'].iloc[-1]:.1f} s")
print(f"Wing area S    : {S:.4f} m²")
if not cd_valid.empty:
    cd_med = cd_valid['cd_smooth'].median()
    print(f"Median CD      : {cd_med:.4f}")

plt.tight_layout()
plt.savefig('unity/glide_60_plot.png', dpi=150)
plt.show()
