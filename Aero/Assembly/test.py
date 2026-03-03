import aerosandbox as asb

# Dimension in meters

nose_length   = 0.204   # nose cone length
body_length   = 0.4     # rectangular section length
tail_length   = 0.750   # trapezoid tail length

nose_tip_wh = 0.04        # nose tip width

body_w = 0.15        # body width
body_h = 0.15        # body height

tail_w = 0.15         # tail tip width
tail_h = 0.04         # tail tip height

# x positions
x0 = 0.0                              # nose tip
x1 = nose_length                      # nose -> body
x2 = nose_length + body_length        # body -> tail
x3 = nose_length + body_length + tail_length  # tail tip

fuselage = asb.Fuselage(
    xsecs=[
        # Nose cone: tip at lower wall level → body centerline
        asb.FuselageXSec(xyz_c=[x0, 0, -body_h / 2], width=nose_tip_wh, height=nose_tip_wh),
        asb.FuselageXSec(xyz_c=[x1, 0, 0], width=body_w, height=body_h),

        # Rectangular body: constant cross-section
        asb.FuselageXSec(xyz_c=[x2, 0, 0], width=body_w, height=body_h),

        # Trapezoid tail: tapers to smaller tip
        asb.FuselageXSec(xyz_c=[x3, 0, 0], width=tail_w, height=tail_h),
    ]
)

def test_aero_buildup():
    analysis = asb.AeroBuildup(
        airplane=airplane,
        op_point=asb.OperatingPoint(),
    )
    aero = analysis.run()
    assert aero is not None  ### Verify analysis produces results

if __name__ == "__main__":
    test_aero_buildup()
    # or inline:
    analysis = asb.AeroBuildup(airplane=airplane, op_point=asb.OperatingPoint())
    aero = analysis.run()
    print(aero)