import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(
    r'C:\Users\juand\OneDrive\Documents\GitHub\wing-sizing\unity\wind_yawrate_bomboo.txt',
    sep='\t', names=['time', 'yaw_rate'], header=0
)

t   = df['time'].values
yaw = df['yaw_rate'].values

# ── gust segments ─────────────────────────────────────────────────────────────
gusts = [
    (100, 108,  4, 'tab:green'),
    (138, 146,  6, 'tab:orange'),
    (190, 198,  8, 'tab:red'),
    (220, 230, 10, 'tab:purple'),
    (254, 262, 12, 'tab:brown'),
]

# ── damping ratio via logarithmic decrement ───────────────────────────────────
def calc_damping(t_all, y_all, t0, t1):
    """
    Half-cycle logarithmic decrement: first positive peak → next negative trough.
    δ_half = ln(|A_high| / |A_low|)
    ζ = δ_half / sqrt(π² + δ_half²)
    T_d estimated from time between that peak and trough * 2.
    """
    mask  = (t_all >= t0) & (t_all <= t1)
    seg_t = t_all[mask]
    seg_y = y_all[mask]

    p_idx, _ = find_peaks( seg_y, prominence=0.01)
    n_idx, _ = find_peaks(-seg_y, prominence=0.01)

    if len(p_idx) == 0 or len(n_idx) == 0:
        return np.nan, np.nan, np.nan, []

    # first positive peak
    t_high = seg_t[p_idx[0]];  a_high = seg_y[p_idx[0]]
    # first negative trough that comes AFTER the high peak
    after = n_idx[seg_t[n_idx] > t_high]
    if len(after) == 0:
        return np.nan, np.nan, np.nan, []
    t_low  = seg_t[after[0]];  a_low  = np.abs(seg_y[after[0]])

    delta_half = np.log(np.abs(a_high) / a_low)
    zeta = delta_half / np.sqrt(np.pi**2 + delta_half**2)
    T_d  = 2 * (t_low - t_high)          # half-period × 2
    w_d  = 2 * np.pi / T_d
    w_n  = w_d / np.sqrt(max(1 - zeta**2, 1e-9))
    return zeta, w_d, w_n, [(t_high, a_high), (t_low, -a_low)]

# ── MIL-SPEC Dutch Roll handling qualities thresholds ────────────────────────
# Criterion        Level 1    Level 2    Level 3
# zeta_dr          >= 0.08    >= 0.02    >= 0
# wn_dr  [rad/s]   >= 0.4     >= 0.4     >= 0.4
# zeta*wn [rad/s]  >= 0.15    >= 0.05    —

def hq_level(zeta, wn):
    zw = zeta * wn
    if zeta >= 0.08 and wn >= 0.4 and zw >= 0.15:
        return "Level 1"
    elif zeta >= 0.02 and wn >= 0.4 and zw >= 0.05:
        return "Level 2"
    elif zeta >= 0.0  and wn >= 0.4:
        return "Level 3"
    else:
        return "< Level 3"

print(f"\n{'Gust':>8}  {'zeta':>7}  {'wn[r/s]':>9}  {'zeta*wn':>9}  {'Level':>9}")
print("-" * 52)
gust_results = []
for t0, t1, speed, col in gusts:
    zeta, w_d, w_n, peaks = calc_damping(t, yaw, t0, t1)
    level = hq_level(zeta, w_n) if not np.isnan(zeta) else "N/A"
    gust_results.append((t0, t1, speed, col, zeta, w_d, w_n, peaks, level))
    zw = zeta * w_n if not np.isnan(zeta) else np.nan
    print(f"{speed:>5} m/s  {zeta:>7.4f}  {w_n:>9.4f}  {zw:>9.4f}  {level:>9}")

# ── full-signal peaks (for markers) ──────────────────────────────────────────
pos_idx, _ = find_peaks( yaw, prominence=0.02)
neg_idx, _ = find_peaks(-yaw, prominence=0.02)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

# shaded gust windows
for t0, t1, speed, col, zeta, w_d, w_n, peaks, level in gust_results:
    ax.axvspan(t0, t1, color=col, alpha=0.15, zorder=0)
    ax.text((t0 + t1) / 2, 0.99, f'{speed} m/s',
            ha='center', va='top', fontsize=8, color=col, fontweight='bold',
            transform=ax.get_xaxis_transform())
    if not np.isnan(zeta):
        ax.text((t0 + t1) / 2, 0.89, f'z={zeta:.3f}',
                ha='center', va='top', fontsize=7.5, color=col,
                transform=ax.get_xaxis_transform())
        lv_color = 'green' if level == 'Level 1' else 'darkorange'
        ax.text((t0 + t1) / 2, 0.79, level,
                ha='center', va='top', fontsize=7.5, color=lv_color,
                fontweight='bold', transform=ax.get_xaxis_transform())

# signal and peaks
ax.plot(t, yaw, color='steelblue', linewidth=1.0, label='Yaw rate')
ax.plot(t[pos_idx], yaw[pos_idx], 'r^', markersize=5, label='Positive peaks')
ax.plot(t[neg_idx], yaw[neg_idx], 'bv', markersize=5, label='Negative peaks')
ax.axhline(0, color='k', linewidth=0.7, linestyle='--', alpha=0.5)

ax.set_xlabel('Time  [s]', fontsize=12)
ax.set_ylabel('Yaw Rate  [rad/s]', fontsize=12)
ax.set_title('Yaw Rate vs Time  —  Damping ratio per gust', fontsize=12)
ax.set_xticks(np.arange(0, t.max() + 1, 10))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.grid(True, which='major', linestyle=':', alpha=0.5)
ax.grid(True, which='minor', linestyle=':', alpha=0.2)
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig('unity/yawrate_plot.png', dpi=150)
plt.show()
