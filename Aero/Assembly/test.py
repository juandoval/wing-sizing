import aerosandbox as asb
from custom_airfoil_utils import create_airfoil_auto
import matplotlib.pyplot as plt

# ── FUSELAGE ──────────────────────────────────────────────────────────────
nose_length = 0.204
body_length = 0.4729
tail_length = 0.63 #0.65 #0.7321

nose_tip_wh = 0.04
body_w = 0.15
body_h = 0.15
tail_w = 0.15
tail_h = 0.04

x0 = 0.0
x1 = nose_length
x2 = nose_length + body_length
x3 = nose_length + body_length + tail_length

fuselage = asb.Fuselage(
    xsecs=[
        asb.FuselageXSec(xyz_c=[x0, 0, -body_h / 2 + nose_tip_wh / 2], width=nose_tip_wh, height=nose_tip_wh),
        asb.FuselageXSec(xyz_c=[x1, 0, 0],           width=body_w,      height=body_h),
        asb.FuselageXSec(xyz_c=[x2, 0, 0],           width=body_w,      height=body_h),
        asb.FuselageXSec(xyz_c=[x3, 0, body_h/2 - tail_h/2],           width=tail_w,      height=tail_h),
    ]
)

# ── MAIN WING ─────────────────────────────────────────────────────────────
wing_chord = 0.367      # m  (from SUPERCUB.ipynb)
wing_span  = 1.8        # m
x_wing_le  = 0.31       # LE at ~mid body section
z_wing     = body_h / 2 # high-wing: sits on top of fuselage

airfoil_file = "Aero/Assembly/supercub_normalized.dat"
supercub_airfoil = create_airfoil_auto(airfoil_file, name="SuperCub")

kulfan_airfoil = supercub_airfoil.to_kulfan_airfoil()
# kulfan_airfoil.plot()


fig, ax = plt.subplots(figsize=(6, 2))
coords = kulfan_airfoil.to_airfoil().coordinates
ax.plot(coords[:, 0], coords[:, 1])
ax.set_aspect("equal")
ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.grid(True)
# plt.show()

wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[x_wing_le, 0, z_wing + 0.015],
            chord=wing_chord,
            airfoil=kulfan_airfoil, #Airfoil("naca6409"),
            twist=2.6
        ),
        asb.WingXSec(
            xyz_le=[x_wing_le, wing_span/2, z_wing + 0.015],
            chord=wing_chord,
            airfoil=kulfan_airfoil, #Airfoil("naca6409"),
            twist=2.6
        ),
    ]
)

# ── HORIZONTAL TAIL ───────────────────────────────────────────────────────

# Sized via Vh=0.40: Sh = Vh*S*MAC/lt ≈ 0.114 m²  →  span≈0.67m, chord≈0.17m
htail_chord = 0.274
htail_span  = 0.648
x_htail_le  = 1.135 
z_htail     = body_h / 2

htail = asb.Wing(
    name="Horizontal Flat Tail",
    symmetric=True,
    xsecs=[
        asb.WingXSec(
            xyz_le=[x_htail_le, 0,            z_htail],
            chord=htail_chord,
            airfoil=asb.Airfoil("naca0006"),
            twist=3.0
        ),
        asb.WingXSec(
            xyz_le=[x_htail_le, htail_span/2, z_htail],
            chord=htail_chord,
            airfoil=asb.Airfoil("naca0006"),
            twist=3.0
        ),
    ]
)

# ── VERTICAL TAIL ─────────────────────────────────────────────────────────
# Sized via Vv=0.025: Sv = Vv*S*b/lt ≈ 0.035 m²  →  height≈0.23m, chord≈0.15m
vtail_chord  = .168
vtail_height = 0.335
x_vtail_le   = 1.155 
z_vtail_base = body_h / 2

vtail = asb.Wing(
    name="Vertical Tail",
    symmetric=False,
    xsecs=[
        asb.WingXSec(
            xyz_le=[x_vtail_le, 0, z_vtail_base],
            chord=vtail_chord,
            airfoil=asb.Airfoil("naca0006"),
        ),
        asb.WingXSec(
            xyz_le=[x_vtail_le + 0.023, 0, z_vtail_base + vtail_height],
            chord=vtail_chord,
            airfoil=asb.Airfoil("naca0006"),
        ),
    ]
)

# ── AIRPLANE ──────────────────────────────────────────────────────────────
airplane = asb.Airplane(
    name="BUSR",
    xyz_ref=[x_wing_le + 0.25 * wing_chord, 0, z_wing],  # ~25% MAC
    wings=[wing, htail, vtail],
    fuselages=[fuselage],
)

airplane.draw()

# # ── AERO BUILDUP ──────────────────────────────────────────────────────────
def test_aero_buildup():
    analysis = asb.AeroBuildup(
        airplane=airplane,
        op_point=asb.OperatingPoint(),
    )
    aero = analysis.run()
    assert aero is not None
    return aero

if __name__ == "__main__":
    import numpy as np

    alphas = np.linspace(-15, 25, 100)  # degrees

    op_points = [
        asb.OperatingPoint(
            velocity=15,   # m/s — adjust to your cruise speed
            alpha=a,
        )
        for a in alphas
    ]

    analyses = [
        asb.AeroBuildup(airplane=airplane, op_point=op).run()
        for op in op_points
    ]

    CLs = [float(a["CL"]) for a in analyses]
    CDs = [float(a["CD"]) for a in analyses]
    CMs = [float(a["Cm"]) for a in analyses]

    # Airplane without fuselage → isolate fuselage drag contribution
    airplane_no_fus = asb.Airplane(
        name="BUSR_no_fus",
        xyz_ref=airplane.xyz_ref,
        wings=airplane.wings,
    )
    analyses_no_fus = [
        asb.AeroBuildup(airplane=airplane_no_fus, op_point=op).run()
        for op in op_points
    ]
    CDs_no_fus = [float(a["CD"]) for a in analyses_no_fus]
    CDs_fus    = [cd - cd_nf for cd, cd_nf in zip(CDs, CDs_no_fus)]

    # Print summary
    print(f"{'Alpha':>8}  {'CL':>8}  {'CD':>8}  {'L/D':>8}  {'Cm':>8}")
    for a, cl, cd, cm in zip(alphas, CLs, CDs, CMs):
        print(f"{a:8.1f}  {cl:8.4f}  {cd:8.4f}  {cl/cd:8.1f}  {cm:8.4f}")

    # ── Airfoil polars (2D) ───────────────────────────────────────────────
    Re = 15 * wing_chord / 1.46e-5  # Re at cruise speed, ~sea level

    wing_foil_polar  = kulfan_airfoil.get_aero_from_neuralfoil(
        alpha=alphas, Re=Re, model_size="large"
    )
    htail_foil_polar = asb.Airfoil("naca0006").get_aero_from_neuralfoil(
        alpha=alphas, Re=15 * htail_chord / 1.46e-5, model_size="large"
    )

    wing_CL  = wing_foil_polar["CL"]
    wing_CD  = wing_foil_polar["CD"]
    wing_CM  = wing_foil_polar["CM"]
    htail_CL = htail_foil_polar["CL"]
    htail_CD = htail_foil_polar["CD"]
    htail_CM = htail_foil_polar["CM"]

    # ── Window 1: CL vs AoA, Efficiency ──────────────────────────────────
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))

    ax1[0].plot(alphas, CLs,      label="Airplane (3D)")
    ax1[0].plot(alphas, wing_CL,  label="Wing airfoil (2D)", linestyle="--")
    ax1[0].plot(alphas, htail_CL, label="HTail airfoil (2D)", linestyle=":")
    ax1[0].set_xlabel("Alpha (deg)"); ax1[0].set_ylabel("CL")
    ax1[0].set_title("Lift Curve"); ax1[0].grid(True); ax1[0].legend()

    ax1[1].plot(alphas, [cl / cd for cl, cd in zip(CLs, CDs)], label="Airplane (3D)")
    ax1[1].plot(alphas, wing_CL  / wing_CD,  label="Wing airfoil (2D)", linestyle="--")
    ax1[1].plot(alphas, htail_CL / htail_CD, label="HTail airfoil (2D)", linestyle=":")
    ax1[1].set_xlabel("Alpha (deg)"); ax1[1].set_ylabel("L/D")
    ax1[1].set_title("Efficiency"); ax1[1].grid(True); ax1[1].legend()

    fig1.tight_layout()

    # ── Window 2: CD vs AoA, Drag Polar ──────────────────────────────────
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))

    ax2[0].plot(alphas, CDs,      label="Airplane (3D, total)")
    ax2[0].plot(alphas, CDs_fus,  label="Fuselage drag", linestyle="-.")
    ax2[0].plot(alphas, wing_CD,  label="Wing airfoil (2D)", linestyle="--")
    ax2[0].plot(alphas, htail_CD, label="HTail airfoil (2D)", linestyle=":")
    ax2[0].set_xlabel("Alpha (deg)"); ax2[0].set_ylabel("CD")
    ax2[0].set_title("Drag vs AoA"); ax2[0].grid(True); ax2[0].legend()

    ax2[1].plot(CDs,      CLs,      label="Airplane (3D)")
    ax2[1].plot(wing_CD,  wing_CL,  label="Wing airfoil (2D)", linestyle="--")
    ax2[1].plot(htail_CD, htail_CL, label="HTail airfoil (2D)", linestyle=":")
    ax2[1].set_xlabel("CD"); ax2[1].set_ylabel("CL")
    ax2[1].set_title("Drag Polar"); ax2[1].grid(True); ax2[1].legend()

    fig2.tight_layout()
    plt.show()
