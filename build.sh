#!/bin/bash
# Build script for GPUMD-PySAGES
#
# Usage: ./build.sh /path/to/GPUMD
#
# This script:
# 1. Verifies the GPUMD source directory
# 2. Copies the modified GPUMD core files (with USE_PYSAGES hooks)
# 3. Copies the pybind11 wrapper files
# 4. Builds the Python extension module (gpumd.so)

set -e

GPUMD_DIR="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$GPUMD_DIR" ]; then
    echo "Usage: ./build.sh /path/to/GPUMD"
    echo ""
    echo "Please provide the path to your GPUMD source directory."
    echo "GPUMD can be cloned from: https://github.com/brucefan1983/GPUMD"
    exit 1
fi

if [ ! -d "$GPUMD_DIR/src" ]; then
    echo "Error: $GPUMD_DIR does not look like a GPUMD source directory."
    echo "Expected to find: $GPUMD_DIR/src/"
    exit 1
fi

GPUMD_DIR="$(cd "$GPUMD_DIR" && pwd)"
echo "=========================================="
echo "GPUMD directory: $GPUMD_DIR"
echo "=========================================="

# --- Step 1: Backup original files ---
echo ""
echo "Step 1/5: Backing up original GPUMD files..."
cp "$GPUMD_DIR/src/main_gpumd/run.cuh"       "$GPUMD_DIR/src/main_gpumd/run.cuh.bak"
cp "$GPUMD_DIR/src/main_gpumd/run.cu"        "$GPUMD_DIR/src/main_gpumd/run.cu.bak"
cp "$GPUMD_DIR/src/utilities/gpu_vector.cuh" "$GPUMD_DIR/src/utilities/gpu_vector.cuh.bak"
cp "$GPUMD_DIR/src/makefile"                 "$GPUMD_DIR/src/makefile.bak"
echo "  ✓ Backups created (*.bak)"

# --- Step 2: Copy modified GPUMD core files ---
echo ""
echo "Step 2/5: Copying modified GPUMD core files..."
cp "$SCRIPT_DIR/gpumd_patches/run.cuh"       "$GPUMD_DIR/src/main_gpumd/"
cp "$SCRIPT_DIR/gpumd_patches/run.cu"        "$GPUMD_DIR/src/main_gpumd/"
cp "$SCRIPT_DIR/gpumd_patches/gpu_vector.cuh" "$GPUMD_DIR/src/utilities/"
cp "$SCRIPT_DIR/gpumd_patches/makefile"       "$GPUMD_DIR/src/"
echo "  ✓ Core files updated"

# --- Step 3: Copy wrapper files ---
echo ""
echo "Step 3/5: Copying pybind11 wrapper files..."
cp "$SCRIPT_DIR/wrapper/gpumd_python.cpp"       "$GPUMD_DIR/src/main_gpumd/"
cp "$SCRIPT_DIR/wrapper/gpumd_python_kernels.cu"  "$GPUMD_DIR/src/main_gpumd/"
cp "$SCRIPT_DIR/wrapper/gpumd_python_kernels.cuh" "$GPUMD_DIR/src/main_gpumd/"
cp "$SCRIPT_DIR/wrapper/dlpack.h"                 "$GPUMD_DIR/src/main_gpumd/"
echo "  ✓ Wrapper files copied"

# --- Step 4: Build the Python extension ---
echo ""
echo "Step 4/5: Building Python extension module (gpumd.so)..."
cd "$GPUMD_DIR/src"
make clean
make pygpumd

# --- Step 5: Install PySAGES backend ---
echo ""
echo "Step 5/5: Installing PySAGES backend..."
cd "$SCRIPT_DIR"
pip install -e . > /dev/null 2>&1 || pip install -e .
python -m gpumd_pysages.install

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Ensure $GPUMD_DIR/src is in your PYTHONPATH:"
echo "     export PYTHONPATH=\$PYTHONPATH:$GPUMD_DIR/src"
echo "  2. Verify installation:"
echo "     python3 -c 'import gpumd; print(\"gpumd imported OK\")'"
echo "  3. Run a quick test:"
echo "     python3 $SCRIPT_DIR/examples/diagnose_timing.py /path/to/your/simulation"
echo ""
