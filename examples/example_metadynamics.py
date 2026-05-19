#!/usr/bin/env python3
"""
Metadynamics example: PySAGES + GPUMD

This demonstrates metadynamics with GPUMD as the backend.
A Gaussian hill is deposited on the collective variable every `stride` steps.
After the run, the free energy surface (FES) is reconstructed from the
bias potential.

Before running:
1. Build gpumd.so:   cd GPUMD/src && make pygpumd
2. Ensure gpumd.so is on PYTHONPATH
3. Have a GPUMD simulation directory with run.in and model.xyz

Usage:
    python example_gpumd_metadynamics.py

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

# JAX defaults to float32, but GPUMD uses double (float64) for forces.
# Enable x64 so that CV gradients and bias forces are computed in float64.
import jax

jax.config.update("jax_enable_x64", True)

import pysages
from pysages.approxfun import compute_mesh
from pysages.colvars import Component
from pysages.methods import MetaDLogger, Metadynamics

# ---------------------------------------------------------------------------
# 1. Simulation setup
# ---------------------------------------------------------------------------
SIMULATION_DIR = "/path/to/your/gpumd/simulation"
RUN_IN_PATH = os.path.join(SIMULATION_DIR, "run.in")

if not os.path.isfile(RUN_IN_PATH):
    raise FileNotFoundError(
        f"Cannot find {RUN_IN_PATH}. Please create a GPUMD simulation directory first."
    )


def generate_simulation(**kwargs):
    """Return a GPUMD simulation object (backend context)."""
    os.chdir(SIMULATION_DIR)
    return gpumd.Simulation(RUN_IN_PATH)


# ---------------------------------------------------------------------------
# 2. Define collective variables and metadynamics parameters
# ---------------------------------------------------------------------------
# For demonstration, bias the z-coordinate of atom 0.
# In a real simulation, use meaningful CVs (distances, angles, etc.)
cvs = [Component([0], 2)]  # z-coordinate of atom 0

# Metadynamics parameters
height = 0.05  # Gaussian height in GPUMD energy units (eV)
sigma = [0.5]  # Gaussian width in GPUMD length units (Angstrom)
stride = 50  # Deposit a hill every 50 steps
timesteps = 1000  # Total simulation steps
ngauss = timesteps // stride + 1

# Optional: use a grid for faster bias potential evaluation.
# For a single CV, define a 1-D grid.
grid = pysages.Grid(lower=(-5.0,), upper=(5.0,), shape=(100,), periodic=False)

method = Metadynamics(cvs, height, sigma, stride, ngauss, grid=grid)

# ---------------------------------------------------------------------------
# 3. Logging: write hills to a file for analysis
# ---------------------------------------------------------------------------
hills_file = "hills.dat"
callback = MetaDLogger(hills_file, stride)

# ---------------------------------------------------------------------------
# 4. Run the simulation
# ---------------------------------------------------------------------------
print("Starting metadynamics simulation...")
tic = time.perf_counter()

run_result = pysages.run(method, generate_simulation, timesteps, callback)

toc = time.perf_counter()
print(f"Completed in {toc - tic:0.4f} seconds.")

# ---------------------------------------------------------------------------
# 5. Free energy analysis
# ---------------------------------------------------------------------------
# Evaluate the bias potential on a fine grid and convert to free energy.
plot_grid = pysages.Grid(lower=(-5.0,), upper=(5.0,), shape=(200,), periodic=False)
xi = compute_mesh(plot_grid)

result = pysages.analyze(run_result)
metapotential = result["metapotential"]

# For standard metadynamics (not well-tempered), alpha = 1.
# The free energy is approximately -metapotential(xi).
A = -metapotential(xi)
A = A - A.min()  # Set minimum to zero

# Save to CSV
output = np.column_stack((xi.flatten(), A.flatten()))
np.savetxt("fes.dat", output, header="cv  free_energy", comments="")
print("Free energy surface saved to fes.dat")

# Optional: quick plot if matplotlib is available
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(xi.flatten(), A.flatten(), lw=2)
    ax.set_xlabel("CV (z-coordinate)")
    ax.set_ylabel(r"$\Delta$G (eV)")
    ax.set_title("Metadynamics Free Energy Surface")
    fig.savefig("fes.png", dpi=150)
    print("Plot saved to fes.png")
except ImportError:
    print("matplotlib not available; skipping plot.")

print("\nDone.")
