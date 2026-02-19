import numpy as np
import matplotlib.pyplot as plt

# ── inputs ────────────────────────────────────────────────────────────────────
V_stall_level = 9.5          # level-flight stall speed [m/s]
bank_angles   = [20, 30, 40, 50, 60]   # bank angles to plot [deg]
V_min         = 5.0          # min airspeed for x-axis [m/s]
V_max         = 35.0          # max airspeed for x-axis [m/s]
g             = 9.81          # m/s²

# ── setup ─────────────────────────────────────────────────────────────────────
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(bank_angles)))
V      = np.linspace(V_min, V_max, 500)

fig, ax = plt.subplots(figsize=(9, 6))

for i, (phi_deg, color) in enumerate(zip(bank_angles, colors)):
    phi   = np.radians(phi_deg)
    label = f"φ = {phi_deg}°"

    # turn radius (only meaningful above stall)
    R = V**2 / (g * np.tan(phi))

    # stall speed at this bank angle
    V_s = V_stall_level / np.sqrt(np.cos(phi))   # = V_s_level * sqrt(1/cos φ)

    ax.plot(V, R, color=color, linewidth=2, label=label)

    # vertical stall-speed line
    ax.axvline(V_s, color=color, linewidth=1.2, linestyle="--", alpha=0.75)
    y_pos = 0.97 - i * 0.12
    ax.text(V_s + 0.3, y_pos, f"Vs {phi_deg}°\n{V_s:.1f} m/s",
            color=color, fontsize=7.5, va="top",
            transform=ax.get_xaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

ax.set_xlabel("Airspeed  V  [m/s]", fontsize=12)
ax.set_ylabel("Turn Radius  R  [m]", fontsize=12)
ax.set_title("Banked Turn Radius vs Airspeed\n"
             r"$R = V^2 \,/\, (g\,\tan\varphi)$   —   "
             r"$V_{s,\varphi} = V_{s,0}\,/\,\sqrt{\cos\varphi}$",
             fontsize=11)
ax.set_xlim(V_min, V_max)
ax.set_ylim(0, 500)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=10, loc="upper left")
plt.tight_layout()
plt.savefig("bank_radius/bank_radius.png", dpi=150)
plt.show()
