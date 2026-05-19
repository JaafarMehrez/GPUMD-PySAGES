#!/usr/bin/env python3
"""
Setup script for GPUMD-PySAGES interface.

Usage:
    pip install -e .

After installation, install the PySAGES backend:
    python -m gpumd_pysages.install
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gpumd-pysages",
    version="0.1.0",
    author="Jaafar Mehrez",
    author_email="jaafarmehrez@sjtu.edu.cn",
    description="GPU-native interface between GPUMD and PySAGES",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JaafarMehrez/GPUMD-PySAGES",
    packages=find_packages(),
    package_data={
        "gpumd_pysages": ["*.py"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "jax[cuda]>=0.4.0",
        "cupy-cuda11x>=11.0.0",
        "pysages>=0.3.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "gpumd-pysages-install=gpumd_pysages.install:main",
        ],
    },
)
