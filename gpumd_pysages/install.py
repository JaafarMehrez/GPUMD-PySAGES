"""
Helper to install the GPUMD backend into the PySAGES package.
"""

import os
import shutil
import sys


def main():
    """Copy gpumd.py into pysages.backends."""
    try:
        import pysages
    except ImportError:
        print("Error: PySAGES is not installed. Please install it first:")
        print("  pip install pysages")
        sys.exit(1)

    pysages_dir = os.path.dirname(pysages.__file__)
    backend_src = os.path.join(
        os.path.dirname(__file__), "..", "pysages_backend", "gpumd.py"
    )
    backend_dst = os.path.join(pysages_dir, "backends", "gpumd.py")

    if not os.path.exists(backend_src):
        # When installed as editable, look relative to package root
        backend_src = os.path.join(
            os.path.dirname(__file__), "..", "..", "pysages_backend", "gpumd.py"
        )

    if not os.path.exists(backend_src):
        print(f"Error: Could not find gpumd.py backend source.")
        sys.exit(1)

    if os.path.exists(backend_dst):
        print(f"GPUMD backend already installed at {backend_dst}")
        print("Overwrite? [y/N] ", end="")
        response = input().strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    shutil.copy2(backend_src, backend_dst)
    print(f"Successfully installed GPUMD backend to {backend_dst}")
    print("")
    print("Next: update pysages/backends/core.py to detect gpumd.Simulation contexts.")
    print("See README.md for the one-line change needed.")


if __name__ == "__main__":
    main()
