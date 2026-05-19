# GPUMD-PySAGES

A GPU-native interface between [GPUMD](https://github.com/brucefan1983/GPUMD) and [PySAGES](https://github.com/SSAGESLabs/PySAGES) for enhanced-sampling molecular dynamics on machine-learning potentials.

## Overview

This package bridges GPUMD (a highly efficient GPU MD engine) with PySAGES (a JAX-based enhanced-sampling framework). It enables:

- **Metadynamics**, **ABF**, **Umbrella Sampling**, and all PySAGES methods on GPUMD simulations.
- **Zero-copy GPU↔GPU** data exchange via DLPack — no host↔device transfers.
- **Native GPU execution** of collective-variable and bias computations via JAX.
- **NEP potentials** — run enhanced sampling on GPUMD's neural-network potentials with full GPU acceleration.

## Dependencies

- [GPUMD](https://github.com/brucefan1983/GPUMD) (tested with v3.x / v5.x)
- [PySAGES](https://github.com/SSAGESLabs/PySAGES)
- Python ≥ 3.8
- pybind11
- JAX with CUDA support (`jax[cuda]`)
- CuPy
- CUDA toolkit ≥ 11.0

## Installation

### Step 1 — Install GPUMD

Clone and build GPUMD from source (do **not** run `make pygpumd` yet):

```bash
git clone https://github.com/brucefan1983/GPUMD.git
cd GPUMD/src
make gpumd
```

### Step 2 — Install PySAGES

```bash
git clone https://github.com/SSAGESLabs/PySAGES.git
cd PySAGES
pip install -e .
```

### Step 3 — Install this interface

```bash
git clone https://github.com/JaafarMehrez/GPUMD-PySAGES.git
cd GPUMD-PySAGES

# Option A: Apply patches to an existing GPUMD clone and build the wrapper
./build.sh /path/to/GPUMD

# Option B: Manual patching
cd /path/to/GPUMD/src
patch -p2 < /path/to/GPUMD-PySAGES/patches/01-run-cuh.patch
patch -p2 < /path/to/GPUMD-PySAGES/patches/02-run-cu.patch
patch -p2 < /path/to/GPUMD-PySAGES/patches/03-gpu_vector-cuh.patch
patch -p2 < /path/to/GPUMD-PySAGES/patches/04-makefile.patch

# Copy wrapper files into GPUMD source
cp /path/to/GPUMD-PySAGES/wrapper/* /path/to/GPUMD/src/main_gpumd/

# Build the Python extension module
make pygpumd

# Install the PySAGES backend
pip install -e .
```

## Quick start

```python
import gpumd
import pysages
from pysages.colvars import Component
from pysages.methods import HarmonicBias

# Load GPUMD simulation (parses run.in)
sim = gpumd.Simulation("run.in")

# Define CV and sampling method
cv = Component([0], 2)  # z-coordinate of atom 0
method = HarmonicBias(cv, kspring=10.0, center=5.0)

# Run with PySAGES bias
with pysages.SamplingContext(method, lambda: sim) as ctx:
    ctx.run(1000)
```

See `examples/` for complete tutorials:
- `example_ethane_dihedral.py` — Metadynamics of ethane H-C-C-H dihedral
- `example_ethane_dihedral_wt.py` — Well-tempered metadynamics
- `diagnose_timing.py` — Profile the interface overhead

## Performance

Benchmarked on ethane (8 atoms, NEP4, 1M steps, RTX 3090):

| Stage | Time / step |
|-------|-------------|
| Snapshot (DLPack rebuild) | ~0.51 ms |
| JAX CV + bias compute | ~0.16 ms |
| Bias apply (custom CUDA kernel) | ~0.10 ms |
| **Total interface overhead** | **~0.77 ms** |

For 5M steps, the total runtime is **~1.15×** that of GPUMD+PLUMED — essentially the same speed with the flexibility of JAX autodiff and Python-defined collective variables.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  PySAGES (JAX)                                              │
│  ├─ Collective Variables (Python / JAX)                    │
│  ├─ Bias computation (GPU, float64)                         │
│  └─ run() orchestration                                     │
├─────────────────────────────────────────────────────────────┤
│  GPUMD-PySAGES Backend (pysages_backend/gpumd.py)          │
│  ├─ Snapshot from DLPack (zero-copy)                        │
│  ├─ Cached constants (ids, masses, box for NVT)             │
│  └─ _update() registered as GPUMD step callback           │
├─────────────────────────────────────────────────────────────┤
│  pybind11 Wrapper (wrapper/gpumd_python.cpp)                │
│  ├─ DLPack capsules for positions/forces/velocities         │
│  ├─ set_step_callback() / run()                             │
│  └─ add_aos_bias_to_forces() (custom CUDA kernel)           │
├─────────────────────────────────────────────────────────────┤
│  GPUMD Engine (C++ / CUDA)                                  │
│  ├─ force.compute()                                         │
│  ├─ step_callback()  ← PySAGES hook                       │
│  └─ integrate.compute2()                                    │
└─────────────────────────────────────────────────────────────┘
```

## How it works

1. **GPUMD core** (`run.cuh` / `run.cu`) — a minimal `#ifdef USE_PYSAGES` hook is added to the MD loop. It calls a Python callback after force computation and before integration.
2. **pybind11 wrapper** (`gpumd_python.cpp`) — exposes GPUMD's GPU arrays as DLPack capsules that JAX can read directly, and accepts bias forces from JAX to add back to GPUMD's force buffer.
3. **Custom CUDA kernel** (`gpumd_python_kernels.cu`) — adds the JAX-computed AOS-format bias directly to GPUMD's SOA `force_per_atom` in a single kernel launch.
4. **PySAGES backend** (`gpumd.py`) — reconstructs the `Snapshot` from DLPack every step, caches constant data, and applies the computed bias via the custom kernel.

## Citation

If you use this interface in your research, please cite both GPUMD and PySAGES:

**GPUMD:**
```bibtex
@article{fan2017gpumd,
  title={{GPUMD}: A {GPU}-based molecular dynamics package},
  author={Fan, Zheyong and Zhan, Haikuan},
  journal={{Mater. Sci. Eng.}},
  volume={7},
  pages={1--8},
  year={2017},
  publisher={Wiley},
  doi={10.1002/mgea.70028}
}
```

**PySAGES:**
```bibtex
@software{pysages2022,
  title={PySAGES: {P}ython {S}uite for {A}dvanced {G}eneralized-{E}nsemble {S}imulations},
  author={{PySAGES Development Team}},
  year={2022},
  url={https://github.com/SSAGESLabs/PySAGES}
}
```

## License

- The GPUMD core patches and wrapper are released under the **MIT License** (compatible with GPUMD's GPL-3.0+ for linking).
- The PySAGES backend is released under the **MIT License** (same as PySAGES).
- See `LICENSE` for details.

## Author

**Jaafar Mehrez**  
Shanghai Jiao Tong University, Shanghai, China  
HPQC Labs, Waterloo, Canada  
jaafarmehrez@sjtu.edu.cn | jaafar@hpqc.org

## Acknowledgments

This interface was developed in collaboration with the GPUMD and PySAGES development teams. Special thanks to Zheyong Fan (GPUMD) for the suggestion to release this as a standalone package.
