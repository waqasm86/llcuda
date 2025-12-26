# Installation Guide - llcuda Python Package

Complete installation guide for the llcuda Python package on different platforms.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Kaggle Installation](#kaggle-installation)
3. [Google Colab Installation](#colab-installation)
4. [Local Installation](#local-installation)
5. [Docker Installation](#docker-installation)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support
  - Kaggle: T4 (Compute 7.5)
  - Colab: T4, P100, or V100
  - Local: GTX 1060+ or Tesla series
- **CUDA**: 11.7+ or 12.0+
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 10GB free space (for models)

### Build Dependencies

- CMake 3.24+
- C++ compiler (GCC 9+, Clang 10+)
- CUDA Toolkit
- pybind11

---

## Kaggle Installation

### Step 1: Enable GPU Runtime

1. Go to Settings → Accelerator
2. Select "GPU T4 x2" or "GPU P100"
3. Click "Save"

### Step 2: Install Package

```python
# Install build dependencies
!pip install -q pybind11 numpy cmake ninja

# Clone repository
!git clone https://github.com/waqasm86/local-llama-cuda.git
%cd local-llama-cuda

# Set CUDA architecture for T4 GPU
import os
os.environ['CUDA_ARCHITECTURES'] = '75'

# Install package
!pip install -e .
```

### Step 3: Verify Installation

```python
import llcuda

# Check CUDA
print("CUDA Available:", llcuda.check_cuda_available())
print("GPU Info:", llcuda.get_cuda_device_info())

# Check package version
print("llcuda version:", llcuda.__version__)
```

### Step 4: Install llama-server

```python
# Clone llama.cpp
!git clone https://github.com/ggerganov/llama.cpp.git /kaggle/working/llama.cpp
%cd /kaggle/working/llama.cpp

# Build with CUDA
!mkdir -p build && cd build && \
  cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 && \
  cmake --build . --config Release -j4

# Verify build
!./build/bin/llama-server --version
```

---

## Google Colab Installation

### Step 1: Enable GPU Runtime

1. Runtime → Change runtime type
2. Hardware accelerator: GPU
3. GPU type: T4 (or P100/V100)
4. Save

### Step 2: Install Package

```python
# Check GPU
!nvidia-smi

# Install dependencies
!pip install -q pybind11 numpy cmake

# Clone and install
!git clone https://github.com/waqasm86/local-llama-cuda.git
%cd local-llama-cuda

import os
os.environ['CUDA_ARCHITECTURES'] = '75'  # T4 GPU

!pip install -e .
```

### Step 3: Install llama.cpp

```python
# Clone llama.cpp
!git clone https://github.com/ggerganov/llama.cpp.git /content/llama.cpp
%cd /content/llama.cpp

# Build with CUDA
!mkdir -p build && cd build && \
  cmake .. -DGGML_CUDA=ON && \
  cmake --build . --config Release -j$(nproc)

# Test
!./build/bin/llama-server --help
```

### Step 4: Download Model

```python
# Download from HuggingFace
!pip install -q huggingface_hub

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    local_dir="/content/models"
)

print(f"Model: {model_path}")
```

---

## Local Installation

### Ubuntu/Debian

#### Install CUDA Toolkit

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA
sudo apt-get install -y cuda-toolkit-12-3

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Install Build Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python3-dev \
    python3-pip \
    git
```

#### Install Python Package

```bash
# Clone repository
git clone https://github.com/waqasm86/local-llama-cuda.git
cd local-llama-cuda

# Install dependencies
pip3 install pybind11 numpy

# Set CUDA architecture (check your GPU)
# RTX 3060: 86, RTX 4090: 89, T4: 75
export CUDA_ARCHITECTURES=86

# Install
pip3 install -e .
```

#### Install llama.cpp

```bash
# Clone
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j$(nproc)

# Install
sudo cp bin/llama-server /usr/local/bin/
```

### Verify Installation

```bash
# Test llcuda
python3 -c "import llcuda; print(llcuda.check_cuda_available())"

# Test llama-server
llama-server --version
```

---

## Docker Installation

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install pybind11 numpy

# Clone and install llcuda
WORKDIR /opt
RUN git clone https://github.com/waqasm86/local-llama-cuda.git
WORKDIR /opt/local-llama-cuda

ENV CUDA_ARCHITECTURES=75
RUN pip3 install -e .

# Install llama.cpp
WORKDIR /opt
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /opt/llama.cpp
RUN mkdir build && cd build && \
    cmake .. -DGGML_CUDA=ON && \
    cmake --build . --config Release -j4 && \
    cp bin/llama-server /usr/local/bin/

WORKDIR /workspace
```

### Build and Run

```bash
# Build image
docker build -t llcuda:latest .

# Run container with GPU
docker run --gpus all -it llcuda:latest

# Inside container
python3 -c "import llcuda; print(llcuda.check_cuda_available())"
```

---

## Troubleshooting

### Issue: CUDA not found during build

**Error:**
```
Could not find CUDA toolkit
```

**Solution:**
```bash
# Set CUDA path
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reinstall
pip install --force-reinstall -e .
```

### Issue: Wrong CUDA architecture

**Error:**
```
nvcc fatal: Unsupported gpu architecture 'compute_75'
```

**Solution:**
```bash
# Find your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Set correct architecture (e.g., 86 for RTX 3060)
export CUDA_ARCHITECTURES=86
pip install --force-reinstall -e .
```

### Issue: ImportError after installation

**Error:**
```python
ImportError: cannot import name '_llcuda'
```

**Solution:**
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Reinstall
pip install --force-reinstall --no-cache-dir -e .
```

### Issue: Out of memory during build

**Solution:**
```bash
# Reduce parallel jobs
export CMAKE_BUILD_PARALLEL_LEVEL=2
pip install -e .
```

### Issue: llama-server not found

**Solution:**
```bash
# Add to PATH
export PATH=/path/to/llama.cpp/build/bin:$PATH

# Or use absolute path in Python
engine = llcuda.InferenceEngine(server_url="http://127.0.0.1:8090")
```

---

## GPU Compute Capabilities Reference

| GPU Model | Compute Capability | CUDA_ARCHITECTURES |
|-----------|-------------------|-------------------|
| Tesla T4 | 7.5 | 75 |
| Tesla P100 | 6.0 | 60 |
| Tesla V100 | 7.0 | 70 |
| RTX 2080 Ti | 7.5 | 75 |
| RTX 3060 | 8.6 | 86 |
| RTX 3090 | 8.6 | 86 |
| RTX 4090 | 8.9 | 89 |
| A100 | 8.0 | 80 |
| H100 | 9.0 | 90 |

---

## Platform-Specific Notes

### Kaggle

- GPU: NVIDIA T4 (2 GPUs available)
- VRAM: 16GB per GPU
- Time limit: 9 hours/week (GPU)
- Storage: 20GB persistent

### Google Colab

- **Free Tier:**
  - GPU: T4
  - VRAM: 15GB
  - Time limit: 12 hours continuous
  - Storage: 108GB (temporary)

- **Colab Pro:**
  - GPU: P100, V100, or A100
  - VRAM: Up to 40GB
  - Time limit: 24 hours
  - Priority access

---

## Next Steps

After successful installation:

1. **Download a model**: See [Model Guide](MODELS.md)
2. **Run examples**: Check [examples/](examples/)
3. **Start llama-server**: See [README.md](README.md#setup-with-llama-server)
4. **Run inference**: Try the Quick Start guide

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/waqasm86/local-llama-cuda/issues)
- **Documentation**: [Docs](https://waqasm86.github.io/projects/local-llama-cuda/)
- **Examples**: [Notebooks](examples/)

---

**Last Updated**: December 2024
