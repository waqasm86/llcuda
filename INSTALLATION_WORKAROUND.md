# llcuda Installation Workaround for Kaggle

## ⚠️ Current Issue

The `llcuda` package on PyPI (v0.1.0) cannot be installed directly because it's missing required C++ source files for compilation.

**Error you'll see:**
```
ERROR: Failed building wheel for llcuda
```

## ✅ Working Solution: Install from GitHub

Until the PyPI package is fixed, install directly from GitHub:

### For Kaggle/Colab:

```python
# Cell 1: Install from GitHub (WORKING METHOD)
!pip install git+https://github.com/waqasm86/llcuda.git

# Verify installation
import llcuda
print(f"✅ llcuda {llcuda.__version__} installed!")
print(f"CUDA available: {llcuda.check_cuda_available()}")
```

This works because GitHub has the complete source code, while the PyPI package is incomplete.

## 📋 Complete Kaggle Setup

```python
# ==================================================================
# WORKING KAGGLE SETUP - Use this instead
# ==================================================================

# Install llcuda from GitHub
print("📦 Installing llcuda from GitHub...")
!pip install -q git+https://github.com/waqasm86/llcuda.git

# Verify
import llcuda
print(f"✅ llcuda {llcuda.__version__} installed!")

# Check CUDA
if llcuda.check_cuda_available():
    device_info = llcuda.get_cuda_device_info()
    print(f"🎮 GPU: {device_info['name']}")
else:
    print("⚠️ Enable GPU: Settings → Accelerator → GPU T4 x2")

# Create engine
engine = llcuda.InferenceEngine()
print("✅ Ready to use!")
```

## 🔧 What's Wrong with PyPI Package?

The PyPI package (llcuda-0.1.0.tar.gz) contains:
- ✅ `llcuda/__init__.py` (Python code)
- ✅ `llcuda/llcuda_py.cpp` (C++ bindings)
- ✅ `llcuda/CMakeLists.txt` (build config)

But CMakeLists.txt expects these files (which are MISSING):
- ❌ `src/core/*.cpp` (core library)
- ❌ `src/cuda/*.cu` (CUDA kernels)
- ❌ `src/storage/*.cpp` (storage system)
- ❌ `include/llcuda/*.hpp` (headers)

**Result**: CMake can't find the source files → build fails

## 🚀 Permanent Fix (For Package Maintainer)

You need to do ONE of the following:

### Option 1: Include Full Source in PyPI Package

Update MANIFEST.in to include all C++ source:

```
# Add to MANIFEST.in
recursive-include src *.cpp *.cu *.hpp
recursive-include include *.hpp *.h
```

Then rebuild and re-upload to PyPI with version 0.1.1.

### Option 2: Make Pure Python Package (Simpler)

Remove C++ extension entirely and make it pure Python:

1. Remove `ext_modules` from setup.py
2. Remove `CMakeBuild` class
3. Update `llcuda/__init__.py` to not import `_llcuda`
4. Implement everything in pure Python (HTTP client to llama-server)

This is simpler and will work everywhere without compilation.

## 📝 Temporary Note for Users

**Until version 0.1.1 is released**, use the GitHub installation method shown above.

The PyPI package will be fixed in the next release.

## 🆘 Alternative: Manual Installation

If GitHub install also fails:

```python
# Clone and install manually
!git clone https://github.com/waqasm86/llcuda.git
%cd llcuda

# Install dependencies
!pip install -q numpy pybind11

# Install in development mode
!pip install -e .

# Verify
import llcuda
print(llcuda.__version__)
```

## ✅ Summary

| Method | Status | Use When |
|--------|--------|----------|
| `pip install llcuda` | ❌ BROKEN | Don't use (v0.1.0) |
| `pip install git+https://github.com/waqasm86/llcuda.git` | ✅ WORKS | **Use this** |
| Manual git clone + pip install -e . | ✅ WORKS | Backup method |

---

**For package maintainer**: See Option 1 or Option 2 above to fix the PyPI package.

**For users**: Use the GitHub installation method until fixed.
