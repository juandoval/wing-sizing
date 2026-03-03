import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── inputs ────────────────────────────────────────────────────────────────────
V_stall_level = 9.5          # level-flight stall speed [m/s]
bank_angles   = [20, 60]     # bank angles to plot [deg]
V_min         = 5.0          # min airspeed for x-axis [m/s]
V_max         = 35.0         # max airspeed for x-axis [m/s]
g             = 9.81         # m/s²

# ── validation files ──────────────────────────────────────────────────────────
val_files = {
    20: r'bank_radius\bankStall_20.txt',
    60: r'bank_radius\bankStall_60.txt',
}
col_order = {
    20: ['time', 'airspeed', 'pos_z', 'altitude', 'fy'],
    60: ['time', 'fy',       'pos_z', 'altitude', 'airspeed'],
}

def load_stall(phi_deg):
    df = pd.read_csv(val_files[phi_deg], sep='\t',
                     names=col_order[phi_deg], header=0)
    peak_idx    = df['altitude'].idxmax()
    V_stall_meas = df.loc[peak_idx, 'airspeed']
    return df, peak_idx, V_stall_meas

colors    = plt.cm.plasma(np.linspace(0.15, 0.85, len(bank_angles)))
color_map = dict(zip(bank_angles, colors))
V         = np.linspace(V_min, V_max, 500)

# ══ Figure 1: bank radius (standalone) ═══════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(9, 6))

for phi_deg in bank_angles:
    color = color_map[phi_deg]
    phi   = np.radians(phi_deg)

    R        = V**2 / (g * np.tan(phi))
    V_s_th   = V_stall_level / np.sqrt(np.cos(phi))   # theory
    _, _, V_s_sim = load_stall(phi_deg)                # simulation

    ax1.plot(V, R, color=color, linewidth=2, label=f'φ = {phi_deg}°')

    # theory stall: solid vertical
    ax1.axvline(V_s_th,  color=color, linewidth=1.4, linestyle='-',
                label=f'Vs {phi_deg}° theory  {V_s_th:.2f} m/s')
    # sim stall: dashed vertical
    ax1.axvline(V_s_sim, color=color, linewidth=1.4, linestyle='--',
                label=f'Vs {phi_deg}° sim  {V_s_sim:.2f} m/s')

ax1.set_xlabel('Airspeed  V  [m/s]', fontsize=12)
ax1.set_ylabel('Turn Radius  R  [m]', fontsize=12)
ax1.set_title('Banked Turn Radius vs Airspeed\n'
              r'$R = V^2/(g\tan\varphi)$   —   '
              r'$V_{s,\varphi} = V_{s,0}/\sqrt{\cos\varphi}$',
              fontsize=11)
ax1.set_xlim(V_min, V_max)
ax1.set_ylim(0, 500)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.legend(fontsize=9, loc='upper right')
fig1.tight_layout()
fig1.savefig('bank_radius/bank_radius.png', dpi=150)

# ══ Figure 2: altitude vs time validation (2 panels) ═════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_b, phi_deg in zip(axes, bank_angles):
    color = color_map[phi_deg]
    df, peak_idx, V_s_sim = load_stall(phi_deg)

    phi    = np.radians(phi_deg)
    V_s_th = V_stall_level / np.sqrt(np.cos(phi))
    t_stall = df.loc[peak_idx, 'time']

    t_start = max(0, t_stall - 20)
    t_end   = t_stall + 30
    window  = df[(df['time'] >= t_start) & (df['time'] <= t_end)]

    ax_b.plot(window['time'], window['altitude'], color=color, linewidth=1.8)

    # single vertical line at the measured stall event
    ax_b.axvline(t_stall, color='k', linewidth=1.4, linestyle='--',
                 label=f'Stall event\n'
                       f'Theory  Vs = {V_s_th:.2f} m/s\n'
                       f'Sim       Vs = {V_s_sim:.2f} m/s')

    ax_b.set_xlabel('Time  [s]', fontsize=11)
    ax_b.set_ylabel('Altitude  [m]', fontsize=11)
    ax_b.set_title(f'φ = {phi_deg}°  —  Altitude drop at stall', fontsize=11)
    ax_b.grid(True, linestyle=':', alpha=0.5)
    ax_b.legend(fontsize=9)

fig2.tight_layout()
fig2.savefig('bank_radius/bank_radius_validation.png', dpi=150)

plt.show()
