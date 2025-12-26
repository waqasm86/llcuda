# Kaggle Installation Fix for llcuda

## The Problem

When installing `llcuda` on Kaggle with `!pip install llcuda`, you get this error:

```
ERROR: Failed to build installable wheels for some pyproject.toml based projects (llcuda)
```

## The Solution

The package needs to be built from source with CUDA support. Use this instead:

### Quick Fix (Copy-Paste into Kaggle)

```python
# Cell 1: Install dependencies
!apt-get update -qq
!apt-get install -y cmake build-essential
!pip install -q pybind11 numpy

# Cell 2: Install llcuda
import os
os.environ['CUDA_ARCHITECTURES'] = '75'  # T4 GPU

!pip install llcuda --no-binary llcuda

# Cell 3: Verify
import llcuda
print(f"✅ llcuda {llcuda.__version__} installed!")
print(f"CUDA: {llcuda.check_cuda_available()}")
```

## Why This Happens

- `llcuda` contains C++/CUDA extensions
- These need to be compiled from source
- Requires: CMake, C++ compiler, CUDA toolkit
- The `--no-binary llcuda` flag forces building from source

## Installation Time

- Dependencies: ~30 seconds
- Building llcuda: ~2-3 minutes
- **Total: ~3-4 minutes**

## Alternative: GitHub Installation

If PyPI installation fails:

```python
!apt-get update -qq && apt-get install -y cmake build-essential
!pip install -q pybind11 numpy

import os
os.environ['CUDA_ARCHITECTURES'] = '75'

!git clone https://github.com/waqasm86/llcuda.git
%cd llcuda
!pip install -e .
```

## After Installation

Test that it works:

```python
import llcuda

# Create engine
engine = llcuda.InferenceEngine()

# Check CUDA
print(llcuda.check_cuda_available())
device_info = llcuda.get_cuda_device_info()
print(device_info)
```

## More Details

- Full guide: See [KAGGLE_COLAB.md](https://github.com/waqasm86/llcuda/blob/main/KAGGLE_COLAB.md)
- PyPI page: https://pypi.org/project/llcuda/
- GitHub: https://github.com/waqasm86/llcuda

## Need Help?

Open an issue: https://github.com/waqasm86/llcuda/issues
