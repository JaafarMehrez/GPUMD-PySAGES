#!/usr/bin/env python3
"""
Quick timing diagnostic for the GPUMD + PySAGES interface.

Author: Jaafar Mehrez
(Shanghai Jiao Tong University, Shanghai, China;
 HPQC Labs, Waterloo, Canada;
 jaafarmehrez@sjtu.edu.cn, jaafar@hpqc.org)

Runs a short unbiased simulation (or with a minimal HarmonicBias) and prints
the per-step timing breakdown from the backend.

Usage:
    python diagnose_timing.py /path/to/gpumd/simulation

The directory must contain run.in, model.xyz and NEP potential.
"""

import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GPUMD_SRC = os.path.join(_SCRIPT_DIR, "GPUMD", "src")
if os.path.isdir(_GPUMD_SRC) and _GPUMD_SRC not in sys.path:
    sys.path.insert(0, _GPUMD_SRC)

import gpumd

import jax

jax.config.update("jax_enable_x64", True)

import pysages
from pysages.backends.core import SamplingContext
from pysages.colvars import Component
from pysages.methods import HarmonicBias

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <gpumd_simulation_directory>")
    sys.exit(1)

SIMULATION_DIR = sys.argv[1]
RUN_IN_PATH = os.path.join(SIMULATION_DIR, "run.in")

if not os.path.isfile(RUN_IN_PATH):
    raise FileNotFoundError(f"Cannot find {RUN_IN_PATH}")

print(f"Simulation directory: {SIMULATION_DIR}")
print(f"Run input: {RUN_IN_PATH}")

os.chdir(SIMULATION_DIR)
sim = gpumd.Simulation(RUN_IN_PATH)
print(f"  Atoms : {sim.get_number_of_atoms()}")
print(f"  Box H : {list(sim.get_box()[0])}")

# Minimal CV: z-coordinate of atom 0
cvs = [Component([0], 2)]
method = HarmonicBias(cvs, kspring=1.0, center=[0.0])

STEPS = 1000  # enough for stable averages
print(f"\nRunning {STEPS} steps with HarmonicBias for timing...")

tic = time.perf_counter()

sampling_context = SamplingContext(method, lambda: sim)
with sampling_context:
    sampling_context.run(STEPS)
    sampler = sampling_context.sampler

toc = time.perf_counter()
print(f"Wall time: {toc - tic:.2f} s  ({(toc - tic) / STEPS * 1e3:.3f} ms/step)")

if hasattr(sampler, "print_timings"):
    sampler.print_timings()
else:
    print("[WARN] Backend does not expose print_timings().")

print("\nSanity checks:")
print(f"  Sampler state type: {type(sampler.state).__name__}")
print(f"  Callback is None: {sampler.callback is None}")

print("\nDone.")
