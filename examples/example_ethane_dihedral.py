#!/usr/bin/env python3
"""
Metadynamics of ethane dihedral angle: PySAGES + GPUMD

Author: Jaafar Mehrez
(Shanghai Jiao Tong University, Shanghai, China;
 HPQC Labs, Waterloo, Canada;
 jaafarmehrez@sjtu.edu.cn, jaafar@hpqc.org)

This script uses metadynamics to compute the free energy surface (FES)
along the H-C-C-H dihedral angle of ethane. The FES shows the classic
three-fold rotational barrier (~12 kJ/mol) with minima at the staggered
conformations (60°, 180°, 300°) and maxima at the eclipsed conformations
(0°, 120°, 240°).

For background on ethane conformations, see:
https://chem.libretexts.org/Courses/Athabasca_University/...

Before running:
1. Build gpumd.so:   cd GPUMD/src && make pygpumd
2. Ensure gpumd.so is on PYTHONPATH
3. Have a GPUMD simulation directory with run.in and model.xyz
   (the provided ethane model with 8 atoms works well)

Usage:
    python example_gpumd_ethane_dihedral.py

SPDX-License-Identifier: MIT
"""

import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# 0. Ensure the compiled GPUMD module is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GPUMD_SRC = os.path.join(_SCRIPT_DIR, "GPUMD", "src")
if os.path.isdir(_GPUMD_SRC) and _GPUMD_SRC not in sys.path:
    sys.path.insert(0, _GPUMD_SRC)

import gpumd

# JAX defaults to float32. GPUMD uses double (float64) for positions and
# forces, so we need 64-bit precision for the CV and bias calculations.
import jax

jax.config.update("jax_enable_x64", True)

import pysages
from pysages.approxfun import compute_mesh
from pysages.backends.core import SamplingContext
from pysages.colvars import DihedralAngle
from pysages.methods import MetaDLogger, Metadynamics
from pysages.methods.core import Result

# ---------------------------------------------------------------------------
# 1. Simulation setup
# ---------------------------------------------------------------------------
SIMULATION_DIR = "/path/to/your/gpumd/simulation"
RUN_IN_PATH = os.path.join(SIMULATION_DIR, "run.in")

if not os.path.isfile(RUN_IN_PATH):
    raise FileNotFoundError(
        f"Cannot find {RUN_IN_PATH}. Please create a GPUMD simulation directory first."
    )

# IMPORTANT: Use a simulation box large enough that the molecule does not
# wrap across periodic boundaries during the entire metadynamics run.
# GPUMD returns wrapped positions, and the PySAGES backend now re-images
# atoms relative to atom 0 automatically.  However, for absolute safety
# with very long runs, use a box ≥ 50 Å (copy ``model_large_box.xyz`` to
# ``model.xyz`` in your simulation directory) instead of the default 15 Å box.


def generate_simulation(**kwargs):
    """Return a GPUMD simulation object (backend context)."""
    os.chdir(SIMULATION_DIR)
    return gpumd.Simulation(RUN_IN_PATH)


# ---------------------------------------------------------------------------
# 2. Collective variable: H-C-C-H dihedral angle
# ---------------------------------------------------------------------------
# From model.xyz (ethane, 8 atoms):
#   0  C   (carbon 1)
#   1  H   (hydrogen on C1)
#   2  H   (hydrogen on C1)
#   3  H   (hydrogen on C1)
#   4  C   (carbon 2)
#   5  H   (hydrogen on C2)
#   6  H   (hydrogen on C2)
#   7  H   (hydrogen on C2)
#
# Dihedral angle: H(1) -- C(0) -- C(4) -- H(5)
# This spans the full 360° rotation about the C-C bond.

pi = np.pi

cvs = [DihedralAngle([1, 0, 4, 5])]

# ---------------------------------------------------------------------------
# 3. Metadynamics parameters
# ---------------------------------------------------------------------------
# For ethane, the rotational barrier is ~12 kJ/mol (~0.124 eV).
# We choose hill parameters that resolve this barrier:
#   height  = 0.02 eV  (~2 kJ/mol, smaller than barrier to allow escape)
#   sigma   = 0.3 rad  (~17°, broad enough to smooth the FES)
#   stride  = 100      (deposit every 100 steps)
#   timesteps = 50_000  (500 hills total, enough to fill the FES)

height = 0.02  # Gaussian height in eV (GPUMD energy unit)
sigma = [0.3]  # Gaussian width in radians
stride = 100  # Deposit a hill every 100 steps
timesteps = 50_000  # Total simulation steps (yields ~500 hills)
ngauss = timesteps // stride + 1

# Grid: dihedral angles are periodic, so we use a periodic grid
# from -π to +π.  Ethane has 3-fold symmetry, but we sample the
# full circle to see all three minima and maxima.
grid = pysages.Grid(
    lower=(-pi,),
    upper=(pi,),
    shape=(200,),
    periodic=True,
)

method = Metadynamics(cvs, height, sigma, stride, ngauss, grid=grid)

# ---------------------------------------------------------------------------
# 4. Logging
# ---------------------------------------------------------------------------
hills_file = "hills_ethane.dat"
callback = MetaDLogger(hills_file, stride)

# ---------------------------------------------------------------------------
# 5. Run the simulation
# ---------------------------------------------------------------------------
# We use the lower-level SamplingContext API so we can access the GPUMD
# backend sampler and print its per-step timing breakdown after the run.
print("Starting ethane dihedral metadynamics...")
print(f"  CV: dihedral angle H(1)-C(0)-C(4)-H(5)")
print(f"  Grid: [{-pi:.3f}, {pi:.3f}] rad  (periodic)")
print(f"  Hills: height={height} eV, sigma={sigma[0]} rad, stride={stride}")
print(f"  Total steps: {timesteps} -> ~{ngauss} hills")

sim = generate_simulation()
tic = time.perf_counter()

sampling_context = SamplingContext(method, lambda: sim)
with sampling_context:
    sampling_context.run(timesteps)
    # The sampler lives on the backend; grab it for timing printout
    sampler = sampling_context.sampler

toc = time.perf_counter()
print(f"Completed in {toc - tic:0.1f} seconds.")

# Print backend timing breakdown
if hasattr(sampler, "print_timings"):
    sampler.print_timings()

# Build a Result object for downstream analysis (same shape as pysages.run)
run_result = Result(
    method,
    [sampler.state],
    None if sampler.callback is None else [sampler.callback],
    [sampler.take_snapshot()],
)

# ---------------------------------------------------------------------------
# 6. Free energy analysis
# ---------------------------------------------------------------------------
# For standard metadynamics (not well-tempered), the free energy is
# approximately the negative of the accumulated bias potential.

# Fine grid for plotting
plot_grid = pysages.Grid(
    lower=(-pi,),
    upper=(pi,),
    shape=(400,),
    periodic=True,
)
xi = compute_mesh(plot_grid)

result = pysages.analyze(run_result)
metapotential = result["metapotential"]

# Free energy = -metapotential (standard metadynamics, alpha = 1)
A = -metapotential(xi)
A = A - A.min()  # Set global minimum to zero

# Save to file
output = np.column_stack((xi.flatten() * 180 / pi, A.flatten()))  # convert to degrees
np.savetxt(
    "fes_ethane.dat",
    output,
    header="dihedral_angle_deg  free_energy_eV",
    comments="",
    fmt="%.6f",
)
print("Free energy surface saved to fes_ethane.dat")

# ---------------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xi.flatten() * 180 / pi, A.flatten(), lw=2, color="steelblue")
    ax.axhline(y=0, color="gray", ls="--", lw=0.5)

    # Annotate key conformations
    ax.annotate(
        "staggered (min)",
        xy=(60, 0),
        xytext=(60, 0.05),
        ha="center",
        fontsize=9,
        color="green",
    )
    ax.annotate(
        "eclipsed (max)",
        xy=(0, A.max()),
        xytext=(0, A.max() + 0.02),
        ha="center",
        fontsize=9,
        color="red",
    )

    ax.set_xlabel(r"Dihedral angle $\phi$ (degrees)")
    ax.set_ylabel(r"Free energy $\Delta G$ (eV)")
    ax.set_title("Ethane rotational free energy (metadynamics)")
    ax.set_xlim(-180, 180)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])

    fig.tight_layout()
    fig.savefig("fes_ethane.png", dpi=150)
    print("Plot saved to fes_ethane.png")
except ImportError:
    print("matplotlib not available; skipping plot.")

# ---------------------------------------------------------------------------
# 8. Sanity check: barrier height
# ---------------------------------------------------------------------------
barrier = A.max() - A.min()
print(f"\nFES barrier height: {barrier:.4f} eV")
print(f"  (Literature value for ethane: ~0.12 eV = ~12 kJ/mol)")

if barrier < 0.05:
    print("WARNING: barrier seems very low. Try increasing timesteps or")
    print("         decreasing hill height/sigma for better resolution.")
elif barrier > 0.5:
    print("WARNING: barrier seems very high. Check units or hill parameters.")
else:
    print("Barrier height is in a physically reasonable range.")

print("\nDone.")
