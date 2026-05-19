#!/bin/bash
# Build script for GPUMD-PySAGES
#
# Usage: ./build.sh /path/to/GPUMD
#
# This script:
# 1. Verifies the GPUMD source directory
# 2. Applies the necessary patches to GPUMD core files
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
echo "Patches source:  $SCRIPT_DIR/patches"
echo "Wrapper source:  $SCRIPT_DIR/wrapper"
echo "=========================================="

# --- Step 1: Apply patches ---
echo ""
echo "Step 1/4: Applying patches to GPUMD core..."
cd "$GPUMD_DIR/src"

for patch in "$SCRIPT_DIR/patches"/*.patch; do
    pname=$(basename "$patch")
    echo "  Applying $pname ..."
    if patch -p2 --forward -i "$patch" > /dev/null 2>&1; then
        echo "    ✓ Success"
    else
        echo "    ⚠ Already applied or failed (skipping)"
    fi
done

# --- Step 2: Copy wrapper files ---
echo ""
echo "Step 2/4: Copying pybind11 wrapper files..."
cp "$SCRIPT_DIR/wrapper/gpumd_python.cpp"       "$GPUMD_DIR/src/main_gpumd/"
cp "$SCRIPT_DIR/wrapper/gpumd_python_kernels.cu"  "$GPUMD_DIR/src/main_gpumd/"
cp "$SCRIPT_DIR/wrapper/gpumd_python_kernels.cuh" "$GPUMD_DIR/src/main_gpumd/"
cp "$SCRIPT_DIR/wrapper/dlpack.h"                 "$GPUMD_DIR/src/main_gpumd/"
echo "  ✓ Copied wrapper files"

# --- Step 3: Build the Python extension ---
echo ""
echo "Step 3/4: Building Python extension module (gpumd.so)..."
cd "$GPUMD_DIR/src"
make clean
make pygpumd

# --- Step 4: Install PySAGES backend ---
echo ""
echo "Step 4/4: Installing PySAGES backend..."
cd "$SCRIPT_DIR"
pip install -e . > /dev/null 2>&1 || pip install -e .

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
