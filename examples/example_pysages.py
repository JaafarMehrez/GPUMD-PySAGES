#!/usr/bin/env python3
"""
Minimal example: PySAGES + GPUMD

This script demonstrates how to use the new GPUMD backend with PySAGES
for enhanced sampling.  Before running, ensure that:

1. GPUMD has been compiled with the pybind11 wrapper:

       cd GPUMD/src
       make pygpumd

   This produces ``gpumd.so`` in the current directory.

2. The ``gpumd.so`` module is on your ``PYTHONPATH``:

       export PYTHONPATH=$PYTHONPATH:/path/to/GPUMD/src

3. PySAGES has been installed with the GPUMD backend patches applied:

       cd PySAGES
       pip install -e .

Usage:
    python example_gpumd_pysages.py

SPDX-License-Identifier: MIT
"""

import os
import sys

# ---------------------------------------------------------------------------
# 0. Ensure the compiled GPUMD module is importable
# ---------------------------------------------------------------------------
# If gpumd.so lives next to this script (in GPUMD/src), add it automatically.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GPUMD_SRC = os.path.join(_SCRIPT_DIR, "..", "GPUMD", "src")
if os.path.isdir(_GPUMD_SRC) and _GPUMD_SRC not in sys.path:
    sys.path.insert(0, _GPUMD_SRC)

# ---------------------------------------------------------------------------
# 1. Create a minimal GPUMD simulation directory
# ---------------------------------------------------------------------------
# For this example we assume a small Lennard-Jones Ar system.
# You should replace the paths below with your own GPUMD input files.
# A typical GPUMD simulation needs:
#   - model.xyz    (atomic coordinates and types)
#   - run.in       (simulation commands)
#   - potential    (force-field definition, e.g. lj.txt or a NEP model)
#
# Example run.in for testing:
#   potential   lj.txt
#   velocity    300
#   time_step   1
#   dump_thermo 10
#   run         1000
#
# The Python wrapper parses run.in but defers execution of ``run`` commands
# until ``simulation.run(steps)`` is called from PySAGES.
# ---------------------------------------------------------------------------

import gpumd

# JAX defaults to float32, but GPUMD uses double (float64) for forces.
# Enable x64 so that CV gradients and bias forces are computed in float64.
import jax

jax.config.update("jax_enable_x64", True)

SIMULATION_DIR = "/path/to/your/gpumd/simulation"
RUN_IN_PATH = os.path.join(SIMULATION_DIR, "run.in")

# Verify input files exist
if not os.path.isfile(RUN_IN_PATH):
    raise FileNotFoundError(
        f"Cannot find {RUN_IN_PATH}. Please create a GPUMD simulation directory first."
    )

# ---------------------------------------------------------------------------
# 2. Set up a PySAGES sampling method
# ---------------------------------------------------------------------------
import pysages
from pysages.methods import HarmonicBias, HistogramLogger
from pysages.colvars import Component

# For demonstration, we bias the z-coordinate of the first atom.
# In a real simulation you would define a meaningful collective variable.
cvs = [Component([0], 2)]  # list of CVs: atom index 0, component 2 = z

center_cv = [5.0]  # target center for each CV

# Example 1: Simple harmonic bias (uncomment to use)
# k = 10.0
# method = HarmonicBias(cvs, k, center_cv)

# Example 2: Metadynamics (uncomment to use)
# from pysages.methods import Metadynamics
# grid = pysages.Grid(lower=(-5.0,), upper=(5.0,), shape=(50,), periodic=False)
# method = Metadynamics(cvs, height=0.1, sigma=[0.5], stride=50, ngauss=100, grid=grid)

# For this skeleton we use HarmonicBias as the simplest demonstration.
k = 10.0
method = HarmonicBias(cvs, k, center_cv)

callback = HistogramLogger(50)  # log histogram every 50 steps

# ---------------------------------------------------------------------------
# 3. Context generator and simulation run
# ---------------------------------------------------------------------------
# PySAGES.run creates a SamplingContext internally, auto-detects the
# backend (gpumd), and drives the simulation for the requested number
# of steps.


def generate_simulation(**kwargs):
    """Return a GPUMD simulation object (backend context)."""
    os.chdir(SIMULATION_DIR)
    sim = gpumd.Simulation(RUN_IN_PATH)
    print(f"GPUMD simulation loaded:")
    print(f"  Number of atoms : {sim.get_number_of_atoms()}")
    print(f"  Time step       : {sim.get_timestep()} (natural units)")
    box_h, box_origin = sim.get_box()
    print(f"  Box H           : {list(box_h)}")
    return sim


# Run 1000 MD steps with the PySAGES bias applied every step.
# Use keyword arguments for optional parameters to avoid ambiguity.
result = pysages.run(method, generate_simulation, 1000, callback=callback)

# ---------------------------------------------------------------------------
# 4. Post-processing / analysis
# ---------------------------------------------------------------------------
# The sampling method object (e.g. Metadynamics) may have accumulated
# a free-energy estimate.  Access it via the method's state or result.
# For HarmonicBias there is no free-energy estimate; the bias simply
# restrains the CV around the target center.
print("\nSimulation complete.")
