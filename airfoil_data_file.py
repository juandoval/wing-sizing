MANUAL_CUSTOM_COORDS = """
 0.0000000 0.0000000
 0.0125000 0.0239000
 0.0250000 0.0335000
 0.0500000 0.0476000
 0.0750000 0.0589000
 0.1000000 0.0669000
 0.1500000 0.0780000
 0.2000000 0.0852000
 0.3000000 0.0900000
 0.4000000 0.0866000
 0.5000000 0.0757000
 0.6000000 0.0605000
 0.7000000 0.0432000
 0.8000000 0.0226000
 0.9000000 -.0004000
 0.9500000 -.0126000
 1.0000000 -.0275000
 0.0000000 0.0000000
 0.0125000 -.0173000
 0.0250000 -.0213000
 0.0500000 -.0248000
 0.0750000 -.0262000
 0.1000000 -.0269000
 0.1500000 -.0276000
 0.2000000 -.0270000
 0.3000000 -.0261000
 0.4000000 -.0250000
 0.5000000 -.0237000
 0.6000000 -.0231000
 0.7000000 -.0234000
 0.8000000 -.0241000
 0.9000000 -.0256000
 0.9500000 -.0264000
 1.0000000 -.0276000"""

# Parse coordinates
lines = MANUAL_CUSTOM_COORDS.strip().split('\n')
coords = []
for line in lines:
    parts = line.split()
    coords.append([float(parts[0]), float(parts[1])])

# Find the split point
split_idx = 17  # First 17 points are upper surface

upper_surface = coords[:split_idx]
lower_surface = coords[split_idx:]

# ADJUST THIS VALUE: maximum thickness to add in millimeters
CHORD = 0.4  # meters
MAX_THICKNESS_TO_ADD_MM = 40.0  # millimeters - ADJUST THIS VALUE

# ADJUST THIS VALUE: x-location where maximum thickness is added (0 to 1)
# Typically around 0.3-0.5 for most airfoils
MAX_THICKNESS_LOCATION = 0.4  # normalized x-coordinate

max_thickness_normalized = (MAX_THICKNESS_TO_ADD_MM / 1000) / CHORD

print(f"Adding up to {MAX_THICKNESS_TO_ADD_MM} mm to upper surface")
print(f"Maximum addition at x/c = {MAX_THICKNESS_LOCATION}")
print(f"Chord length: {CHORD} m")
print()

# Thicken upper surface with tapering to zero at leading and trailing edges
thickened_upper = []
for x, y in upper_surface:
    # Calculate thickness multiplier based on x-position
    # Taper from 0 at leading edge (x=0) to max at MAX_THICKNESS_LOCATION
    # Then taper back to 0 at trailing edge (x=1)
    
    if x <= MAX_THICKNESS_LOCATION:
        # Leading edge to max thickness location
        if MAX_THICKNESS_LOCATION > 0:
            multiplier = x / MAX_THICKNESS_LOCATION
        else:
            multiplier = 0
    else:
        # Max thickness location to trailing edge
        if (1.0 - MAX_THICKNESS_LOCATION) > 0:
            multiplier = (1.0 - x) / (1.0 - MAX_THICKNESS_LOCATION)
        else:
            multiplier = 0
    
    thickness_to_add = max_thickness_normalized * multiplier
    new_y = y + thickness_to_add
    thickened_upper.append([x, new_y])

# Combine surfaces
new_coords = thickened_upper + lower_surface

# Print in original format
print("Modified coordinates:")
print()
for x, y in new_coords:
    print(f" {x:.7f} {y:.7f}")

# Verify trailing edge closure
print("\n" + "="*50)
print("Trailing edge check:")
print(f"Upper surface last point: ({thickened_upper[-1][0]:.7f}, {thickened_upper[-1][1]:.7f})")
print(f"Lower surface last point: ({lower_surface[-1][0]:.7f}, {lower_surface[-1][1]:.7f})")
print(f"Trailing edge gap: {abs(thickened_upper[-1][1] - lower_surface[-1][1]) * CHORD * 1000:.4f} mm")

# Statistics
print("\n" + "="*50)
print("Statistics:")
print(f"Original max upper surface y: {max(y for x, y in upper_surface):.7f}")
print(f"New max upper surface y: {max(y for x, y in thickened_upper):.7f}")
print(f"Original max thickness: {(max(y for x, y in upper_surface) - min(y for x, y in lower_surface)) * CHORD * 1000:.2f} mm")
print(f"New max thickness: {(max(y for x, y in thickened_upper) - min(y for x, y in lower_surface)) * CHORD * 1000:.2f} mm")
print(f"Thickness added: {(max(y for x, y in thickened_upper) - max(y for x, y in upper_surface)) * CHORD * 1000:.2f} mm")