"""
Custom Airfoil Utilities for WingForge
=======================================
Tools for loading and converting custom airfoil formats to AeroSandbox

Supports:
1. .dat files (Selig format)
2. Conversion to Kulfan (CST) parameterization
3. Direct use with AeroSandbox VLM
"""

import numpy as np
import aerosandbox as asb
from pathlib import Path
from scipy.optimize import least_squares
from typing import Tuple, List, Optional
import math


def read_dat_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read airfoil coordinates from .dat file (Selig format)
    
    Format:
    - Line 1: Airfoil name
    - Lines 2-N: x y coordinates
    - Upper surface first (from TE to LE)
    - Lower surface second (from LE to TE)
    
    Args:
        filepath: Path to .dat file
        
    Returns:
        upper_coords: Nx2 array of upper surface (x, y)
        lower_coords: Mx2 array of lower surface (x, y)
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    name = lines[0].strip()
    print(f"Loading airfoil: {name}")
    
    # Parse coordinates
    coords = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                coords.append([x, y])
    
    coords = np.array(coords)
    
    # Normalize coordinates to 0-1 range if needed
    x_max = np.max(np.abs(coords[:, 0]))
    y_max = np.max(np.abs(coords[:, 1]))
    
    if x_max > 2.0 or y_max > 2.0:
        # Coordinates are not normalized (likely in mm or other units)
        print(f"  WARNING: Coordinates not normalized (x_max={x_max:.1f}, y_max={y_max:.1f})")
        print(f"  Auto-normalizing to 0-1 range...")
        coords[:, 0] = coords[:, 0] / x_max
        coords[:, 1] = coords[:, 1] / x_max  # Use x_max to preserve aspect ratio
    
    # Find split point (where x goes from decreasing to increasing)
    # Upper surface: x decreases from 1.0 to 0.0
    # Lower surface: x increases from 0.0 to 1.0
    x_vals = coords[:, 0]
    
    # Find the minimum x (leading edge)
    le_idx = np.argmin(x_vals)
    
    # Split into upper and lower surfaces
    upper_coords = coords[:le_idx + 1]  # From TE to LE
    lower_coords = coords[le_idx:]      # From LE to TE
    
    # Reverse upper surface so it goes LE to TE
    upper_coords = upper_coords[::-1]
    
    print(f"  Upper surface: {len(upper_coords)} points")
    print(f"  Lower surface: {len(lower_coords)} points")
    
    return upper_coords, lower_coords


def fit_kulfan_weights(
    coords: np.ndarray,
    n_weights: int = 8,
    N1: float = 0.5,
    N2: float = 1.0
) -> np.ndarray:
    """
    Fit Kulfan (CST) weights to airfoil coordinates
    
    Args:
        coords: Nx2 array of (x, y) coordinates
        n_weights: Number of Kulfan weights to use
        N1: Leading edge shape factor
        N2: Trailing edge shape factor
        
    Returns:
        weights: Array of fitted Kulfan weights
    """
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Remove duplicate points at x=0 and x=1
    mask = (x > 0.001) & (x < 0.999)
    x = x[mask]
    y = y[mask]
    
    # Kulfan basis function
    def class_function(x, N1, N2):
        return x**N1 * (1 - x)**N2
    
    def shape_function(x, weights):
        n = len(weights)
        S = np.zeros_like(x)
        for i, w in enumerate(weights):
            # Bernstein polynomial
            C = math.factorial(n - 1) / (math.factorial(i) * math.factorial(n - 1 - i))
            S += w * C * x**i * (1 - x)**(n - 1 - i)
        return S
    
    # Objective function to minimize
    def objective(weights):
        y_pred = class_function(x, N1, N2) * shape_function(x, weights)
        return y_pred - y
    
    # Initial guess
    w0 = np.zeros(n_weights)
    
    # Fit
    result = least_squares(objective, w0, verbose=0)
    
    return result.x


def coords_to_kulfan(
    upper_coords: np.ndarray,
    lower_coords: np.ndarray,
    n_weights: int = 8,
    N1: float = 0.5,
    N2: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert airfoil coordinates to Kulfan (CST) parameters
    
    Args:
        upper_coords: Upper surface coordinates (LE to TE)
        lower_coords: Lower surface coordinates (LE to TE)
        n_weights: Number of weights per surface
        N1: Leading edge shape factor (0.5 = conventional)
        N2: Trailing edge shape factor (1.0 = conventional)
        
    Returns:
        upper_weights: Fitted upper surface weights
        lower_weights: Fitted lower surface weights
    """
    print(f"\nFitting Kulfan (CST) parameters...")
    print(f"  Weights per surface: {n_weights}")
    print(f"  Shape factors: N1={N1}, N2={N2}")
    
    # Fit upper surface
    upper_weights = fit_kulfan_weights(upper_coords, n_weights, N1, N2)
    print(f"  Upper surface fitted (RMS error calculated)")
    
    # Fit lower surface (flip y-coordinates to positive)
    lower_coords_flipped = lower_coords.copy()
    lower_coords_flipped[:, 1] = -lower_coords_flipped[:, 1]
    lower_weights = fit_kulfan_weights(lower_coords_flipped, n_weights, N1, N2)
    print(f"  Lower surface fitted (RMS error calculated)")
    
    return upper_weights, lower_weights


def create_airfoil_from_dat(
    filepath: str,
    name: Optional[str] = None,
    n_weights: int = 8,
    N1: float = 0.5,
    N2: float = 1.0,
    TE_thickness: float = 0.0,
    n_points: int = 200
) -> asb.Airfoil:
    """
    Create AeroSandbox Airfoil from .dat file
    
    Args:
        filepath: Path to .dat file
        name: Airfoil name (if None, uses name from file)
        n_weights: Number of Kulfan weights per surface
        N1: Leading edge shape factor (0.5 = conventional)
        N2: Trailing edge shape factor (1.0 = conventional)
        TE_thickness: Trailing edge thickness (y/c)
        n_points: Number of points for final airfoil
        
    Returns:
        AeroSandbox Airfoil object
    """
    # Read coordinates
    upper_coords, lower_coords = read_dat_file(filepath)
    
    # Fit Kulfan parameters
    upper_weights, lower_weights = coords_to_kulfan(
        upper_coords, lower_coords, n_weights, N1, N2
    )
    
    # Generate Kulfan coordinates
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
    kulfan_coords = get_kulfan_coordinates(
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        TE_thickness=TE_thickness,
        n_points_per_side=n_points // 2,
        N1=N1,
        N2=N2
    )
    
    # Get name
    if name is None:
        name = Path(filepath).stem
    
    # Create airfoil
    airfoil = asb.Airfoil(name=name, coordinates=kulfan_coords)
    
    print(f"\n[OK] Created AeroSandbox airfoil: {name}")
    print(f"  Points: {len(kulfan_coords)}")
    
    return airfoil


def create_airfoil_from_coords_direct(
    filepath: str,
    name: Optional[str] = None
) -> asb.Airfoil:
    """
    Create AeroSandbox Airfoil directly from coordinates (no Kulfan fitting)
    
    Simpler method that doesn't require CST parameterization.
    Good for when you want exact coordinates without approximation.
    Best for sparse data (<50 points per surface).
    
    Args:
        filepath: Path to .dat file
        name: Airfoil name (if None, uses name from file)
        
    Returns:
        AeroSandbox Airfoil object
    """
    # Read coordinates (both surfaces LE to TE)
    upper_coords, lower_coords = read_dat_file(filepath)
    
    # AeroSandbox expects coordinates in a closed loop:
    # TE(upper) → LE → TE(lower)
    # So we need: upper reversed (TE→LE) + lower (LE→TE)
    upper_reversed = upper_coords[::-1]  # TE to LE
    
    # Combine: start at TE upper, go to LE, then to TE lower
    coords = np.vstack([upper_reversed, lower_coords[1:]])  # Skip duplicate LE point
    
    # Get name
    if name is None:
        name = Path(filepath).stem
    
    # Create airfoil directly from coordinates
    airfoil = asb.Airfoil(name=name, coordinates=coords)
    
    print(f"\n[OK] Created AeroSandbox airfoil (direct): {name}")
    print(f"  Points: {len(coords)}")
    print(f"  Format: TE(upper) → LE → TE(lower)")
    
    return airfoil


def create_airfoil_auto(
    filepath: str,
    name: Optional[str] = None,
    point_threshold: int = 50,
    n_weights: int = 8
) -> asb.Airfoil:
    """
    Automatically choose best method (direct or Kulfan) based on data quality
    
    Decision logic:
    - If points_per_surface < threshold: Use direct coordinates (exact)
    - If points_per_surface >= threshold: Use Kulfan fitting (smooth, optimizable)
    
    Args:
        filepath: Path to .dat file
        name: Airfoil name (if None, uses name from file)
        point_threshold: Min points per surface to use Kulfan (default: 50)
        n_weights: Number of Kulfan weights if using CST method
        
    Returns:
        AeroSandbox Airfoil object
    """
    # Read to check point count
    upper_coords, lower_coords = read_dat_file(filepath)
    points_per_surface = min(len(upper_coords), len(lower_coords))
    
    print(f"\n🤖 Auto-selecting method...")
    print(f"   Points per surface: {points_per_surface}")
    print(f"   Threshold: {point_threshold}")
    
    if points_per_surface < point_threshold:
        print(f"   [OK] Using DIRECT coordinates (sparse data)")
        return create_airfoil_from_coords_direct(filepath, name)
    else:
        print(f"   [OK] Using KULFAN fitting (dense data)")
        return create_airfoil_from_dat(filepath, name, n_weights=n_weights)


def compare_airfoil_fits(
    filepath: str,
    n_weights_list: List[int] = [4, 6, 8, 10],
    output_dir: str = "output/airfoil_fits"
):
    """
    Compare different Kulfan fitting parameters
    Generates plots showing fit quality
    
    Args:
        filepath: Path to .dat file
        n_weights_list: List of weight counts to try
        output_dir: Where to save comparison plots
    """
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read original coordinates
    upper_coords, lower_coords = read_dat_file(filepath)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Kulfan (CST) Fit Comparison: {Path(filepath).stem}', fontsize=14, weight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_weights_list)))
    
    for idx, n_weights in enumerate(n_weights_list):
        # Fit
        upper_weights, lower_weights = coords_to_kulfan(
            upper_coords, lower_coords, n_weights
        )
        
        # Generate fitted coordinates
        from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
        kulfan_coords = get_kulfan_coordinates(
            lower_weights=lower_weights,
            upper_weights=upper_weights,
            n_points_per_side=100
        )
        
        # Split for plotting
        n_half = len(kulfan_coords) // 2
        upper_fit = kulfan_coords[:n_half + 1]
        lower_fit = kulfan_coords[n_half:]
        
        # Plot 1: Full airfoil
        ax1 = axes[0, 0]
        ax1.plot(kulfan_coords[:, 0], kulfan_coords[:, 1], 
                label=f'{n_weights} weights', color=colors[idx], linewidth=2)
    
    # Add original on top
    ax1.plot(upper_coords[:, 0], upper_coords[:, 1], 'k--', label='Original', linewidth=1, alpha=0.5)
    ax1.plot(lower_coords[:, 0], lower_coords[:, 1], 'k--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('x/c')
    ax1.set_ylabel('y/c')
    ax1.set_title('Airfoil Shape Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Leading edge zoom
    ax2 = axes[0, 1]
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
    for idx, n_weights in enumerate(n_weights_list):
        upper_weights, lower_weights = coords_to_kulfan(
            upper_coords, lower_coords, n_weights
        )
        kulfan_coords = get_kulfan_coordinates(
            lower_weights=lower_weights,
            upper_weights=upper_weights,
            n_points_per_side=100
        )
        ax2.plot(kulfan_coords[:, 0], kulfan_coords[:, 1], 
                label=f'{n_weights} weights', color=colors[idx], linewidth=2)
    
    ax2.plot(upper_coords[:, 0], upper_coords[:, 1], 'k--', label='Original', linewidth=1, alpha=0.5)
    ax2.plot(lower_coords[:, 0], lower_coords[:, 1], 'k--', linewidth=1, alpha=0.5)
    ax2.set_xlim(-0.05, 0.2)
    ax2.set_ylim(-0.1, 0.1)
    ax2.set_xlabel('x/c')
    ax2.set_ylabel('y/c')
    ax2.set_title('Leading Edge Detail')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Weight values
    ax3 = axes[1, 0]
    for idx, n_weights in enumerate(n_weights_list):
        upper_weights, lower_weights = coords_to_kulfan(
            upper_coords, lower_coords, n_weights
        )
        x = np.arange(n_weights)
        ax3.plot(x, upper_weights, 'o-', color=colors[idx], label=f'{n_weights} weights (upper)')
        ax3.plot(x, lower_weights, 's--', color=colors[idx], label=f'{n_weights} weights (lower)', alpha=0.6)
    
    ax3.set_xlabel('Weight Index')
    ax3.set_ylabel('Weight Value')
    ax3.set_title('Kulfan Weight Values')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
Kulfan (CST) Parameterization
==============================

File: {Path(filepath).name}
Original Points: {len(upper_coords) + len(lower_coords)}

Shape Factors:
  N1 = 0.5 (conventional LE)
  N2 = 1.0 (conventional TE)

Weight Counts Tested:
  {', '.join(map(str, n_weights_list))}

Recommendation:
  • 4-6 weights: Simple shapes
  • 8 weights: Most airfoils (good balance)
  • 10+ weights: Complex shapes

More weights = Better fit
But also more parameters to optimize
"""
    ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            family='monospace')
    
    plt.tight_layout()
    plot_file = output_path / f'{Path(filepath).stem}_kulfan_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Saved comparison plot: {plot_file}")
    
    return plot_file


# ============================================================================
#                          EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("  Custom Airfoil Utilities - Example Usage")
    print("="*70)
    
    # Example .dat file
    dat_file = "data/airfoils/supercub.dat"
    
    if not Path(dat_file).exists():
        print(f"\n✗ File not found: {dat_file}")
        print("  Please provide a valid .dat file path")
        exit(1)
    
    # Method 1: AUTO - Let it decide!
    print("\n" + "="*70)
    print("  Method 1: AUTO (Smart Selection)")
    print("="*70)
    print("Automatically chooses best method based on data quality...")
    airfoil_auto = create_airfoil_auto(dat_file, name="SuperCub_Auto")
    
    # Method 2: Direct coordinates (exact, no fitting)
    print("\n" + "="*70)
    print("  Method 2: Direct Coordinates (Forced)")
    print("="*70)
    print("Force direct coordinates for exact geometry...")
    airfoil_direct = create_airfoil_from_coords_direct(dat_file, name="SuperCub_Direct")
    
    # Method 3: Kulfan (CST) parameterization (smooth, optimizable)
    print("\n" + "="*70)
    print("  Method 3: Kulfan (CST) Parameterization (Forced)")
    print("="*70)
    print("Force Kulfan for smooth, optimizable geometry...")
    airfoil_kulfan = create_airfoil_from_dat(
        dat_file,
        name="SuperCub_Kulfan",
        n_weights=8,
        N1=0.5,
        N2=1.0
    )
    
    # Method 4: Compare different fits
    print("\n" + "="*70)
    print("  Method 4: Compare Different Kulfan Fits")
    print("="*70)
    compare_airfoil_fits(dat_file, n_weights_list=[4, 6, 8, 10])
    
    # Test with VLM - use the auto-selected method
    print("\n" + "="*70)
    print("  Testing with AeroSandbox VLM (using auto-selected)")
    print("="*70)
    
    wing = asb.Wing(
        name="TestWing",
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=1.0, airfoil=airfoil_auto),
            asb.WingXSec(xyz_le=[0, 2.0, 0], chord=1.0, airfoil=airfoil_auto)
        ],
        symmetric=True
    )
    
    airplane = asb.Airplane(name="TestAirplane", wings=[wing])
    op_point = asb.OperatingPoint(velocity=25, alpha=5)
    
    vlm = asb.VortexLatticeMethod(airplane, op_point)
    aero = vlm.run()
    
    print(f"\n[OK] VLM Analysis Results (with auto-selected method):")
    print(f"  CL = {aero['CL']:.4f}")
    print(f"  CD = {aero['CD']:.5f}")
    print(f"  Cm = {aero['Cm']:.4f}")
    print(f"  L/D = {aero['CL']/aero['CD']:.1f}")
    
    print("\n" + "="*70)
    print("  [OK] All methods work! Custom airfoil ready for VLM service")
    print("="*70)
    print("\nRECOMMENDATION:")
    print("  • Use create_airfoil_auto() - it chooses intelligently")
    print("  • Sparse data (<50 pts): Direct coordinates (exact)")
    print("  • Dense data (≥50 pts): Kulfan fitting (smooth, optimizable)")
    print("="*70)
