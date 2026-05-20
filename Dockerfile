FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="Jaafar Mehrez <jaafarmehrez@sjtu.edu.cn>"
LABEL org.opencontainers.image.source="https://github.com/JaafarMehrez/GPUMD-PySAGES"
LABEL org.opencontainers.image.description="GPUMD-PySAGES: GPU-native enhanced sampling interface"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install jax[cuda11_cudnn82] cupy-cuda11x pysages matplotlib numpy

# Clone GPUMD (users must build it themselves due to GPU architecture differences)
RUN git clone https://github.com/brucefan1983/GPUMD.git /opt/GPUMD

# Install the PySAGES backend
COPY . /opt/GPUMD-PySAGES
RUN cd /opt/GPUMD-PySAGES && pip3 install -e .

WORKDIR /opt/GPUMD-PySAGES

CMD ["/bin/bash"]
