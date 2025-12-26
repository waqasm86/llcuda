# 🚀 Quick Start Guide - llcuda Python Package

## What You Have

A complete Python package for CUDA-accelerated LLM inference, ready for Kaggle/Colab with NVIDIA T4 GPUs.

---

## 📦 Package Contents

```
llcuda-python-package/
├── 📄 Core Package Files
│   ├── setup.py                     # Pip installation script
│   ├── pyproject.toml               # Modern Python packaging
│   ├── CMakeLists.txt               # CMake build configuration
│   ├── llcuda_py.cpp                # C++ Python bindings
│   ├── requirements.txt             # Python dependencies
│   └── MANIFEST.in                  # Package manifest
│
├── 🐍 Python Module
│   └── llcuda/
│       └── __init__.py              # Main Python API
│
├── 📚 Documentation
│   ├── README.md                    # Main documentation
│   ├── INSTALL.md                   # Installation guide
│   ├── KAGGLE_COLAB.md              # Kaggle/Colab guide
│   └── PACKAGE_SUMMARY.md           # Complete overview
│
├── 📓 Examples
│   └── examples/
│       └── kaggle_colab_example.ipynb   # Complete working notebook
│
├── 🧪 Tests
│   └── tests/
│       └── test_llcuda.py           # Unit tests
│
└── 🔨 Build Tools
    └── build_wheel.sh               # Wheel build script
```

---

## 🎯 Option 1: Use on Kaggle (Recommended)

### Step 1: Upload to Kaggle

1. Compress the package: `zip -r llcuda-python-package.zip llcuda-python-package/`
2. Go to Kaggle → Datasets → New Dataset
3. Upload `llcuda-python-package.zip`
4. Create dataset (name it "llcuda-package")

### Step 2: Use in Kaggle Notebook

Create a new Kaggle notebook with GPU enabled, then:

```python
# ============================================
# KAGGLE SETUP - COPY THIS ENTIRE CELL
# ============================================

# 1. Check GPU
!nvidia-smi

# 2. Extract package from dataset
!unzip -q /kaggle/input/llcuda-package/llcuda-python-package.zip
%cd llcuda-python-package

# 3. Install llcuda
!pip install -q pybind11 numpy cmake
import os
os.environ['CUDA_ARCHITECTURES'] = '75'  # T4 GPU
!pip install -q -e .

# 4. Install llama.cpp backend
!git clone https://github.com/ggerganov/llama.cpp.git /kaggle/working/llama_cpp
%cd /kaggle/working/llama_cpp
!mkdir build && cd build && cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 && cmake --build . --config Release -j4

# 5. Download a model (Gemma 2B - fast!)
!pip install -q huggingface_hub
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="google/gemma-2b-it-GGUF",
    filename="gemma-2b-it-q4_k_m.gguf",
    local_dir="/kaggle/working/models"
)
print(f"✅ Model: {model_path}")

# 6. Start llama-server (background)
import subprocess, time, requests

server = subprocess.Popen([
    '/kaggle/working/llama_cpp/build/bin/llama-server',
    '-m', model_path,
    '--port', '8090',
    '-ngl', '99',
    '-c', '4096'
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("⏳ Starting server...")
for i in range(30):
    try:
        r = requests.get('http://127.0.0.1:8090/health', timeout=1)
        if r.status_code == 200:
            print("✅ Server ready!")
            break
    except:
        time.sleep(1)

# 7. Test llcuda
import llcuda

engine = llcuda.InferenceEngine()
open('/tmp/model.gguf', 'a').close()
engine.load_model('/tmp/model.gguf')

result = engine.infer("What is artificial intelligence?", max_tokens=50)
print(f"\n✅ llcuda working! Speed: {result.tokens_per_sec:.1f} tok/s")
print(f"📝 Response: {result.text[:100]}...")

print("\n" + "="*60)
print("✅ SETUP COMPLETE! Ready to use llcuda.")
print("="*60)
```

### Step 3: Start Using!

```python
# Basic inference
result = engine.infer("Explain quantum computing", max_tokens=100)
print(result.text)

# Streaming
def callback(chunk):
    print(chunk, end='', flush=True)

result = engine.infer_stream("Write a story", callback=callback, max_tokens=200)

# Batch processing
prompts = ["What is ML?", "What is AI?", "What is DL?"]
results = engine.batch_infer(prompts, max_tokens=50)
```

---

## 🎯 Option 2: Use on Google Colab

### Step 1: Upload to Google Drive

1. Upload entire `llcuda-python-package` folder to Google Drive
2. Or use the notebook directly from Drive

### Step 2: Use in Colab Notebook

Enable GPU (Runtime → Change runtime type → T4 GPU), then:

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Go to package
%cd /content/drive/MyDrive/llcuda-python-package

# 3. Install
!pip install -q pybind11 numpy cmake
import os
os.environ['CUDA_ARCHITECTURES'] = '75'
!pip install -q -e .

# 4. Install llama.cpp
!git clone https://github.com/ggerganov/llama.cpp.git /content/llama_cpp
%cd /content/llama_cpp
!mkdir build && cd build && cmake .. -DGGML_CUDA=ON && cmake --build . --config Release -j$(nproc)

# 5. Download model
!pip install -q huggingface_hub
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="google/gemma-2b-it-GGUF",
    filename="gemma-2b-it-q4_k_m.gguf",
    local_dir="/content/models"
)

# 6. Start server
import subprocess, time, requests
server = subprocess.Popen([
    '/content/llama_cpp/build/bin/llama-server',
    '-m', model_path, '--port', '8090', '-ngl', '99'
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

for i in range(30):
    try:
        if requests.get('http://127.0.0.1:8090/health', timeout=1).status_code == 200:
            print("✅ Ready!")
            break
    except:
        time.sleep(1)

# 7. Use llcuda
import llcuda
engine = llcuda.InferenceEngine()
open('/tmp/model.gguf', 'a').close()
engine.load_model('/tmp/model.gguf')

print(engine.infer("Hello!", max_tokens=20).text)
```

---

## 🎯 Option 3: Local Installation

### Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.7+ or 12.0+
- Python 3.8 - 3.11
- Ubuntu 20.04+ or similar Linux

### Installation

```bash
# 1. Install CUDA (if not already installed)
# See INSTALL.md for detailed instructions

# 2. Install build dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3-dev python3-pip git

# 3. Go to package directory
cd llcuda-python-package

# 4. Install Python dependencies
pip3 install pybind11 numpy

# 5. Set CUDA architecture for your GPU
# T4: 75, RTX 3060: 86, RTX 4090: 89
export CUDA_ARCHITECTURES=75

# 6. Install llcuda
pip3 install -e .

# 7. Verify
python3 -c "import llcuda; print(llcuda.check_cuda_available())"
```

### Install llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j$(nproc)
sudo cp bin/llama-server /usr/local/bin/
```

---

## 📖 API Reference

### InferenceEngine

```python
import llcuda

# Create engine
engine = llcuda.InferenceEngine()

# Load model
engine.load_model(
    model_path="model.gguf",
    gpu_layers=99,        # 0-99 (99 = all layers on GPU)
    ctx_size=4096,        # Context window size
    batch_size=512,       # Batch size
    threads=4             # CPU threads
)

# Single inference
result = engine.infer(
    prompt="Your prompt here",
    max_tokens=128,       # Max tokens to generate
    temperature=0.7,      # Sampling temperature (0.0-2.0)
    top_p=0.9,           # Nucleus sampling (0.0-1.0)
    top_k=40,            # Top-k sampling
    seed=0               # Random seed (0 = random)
)

# Streaming inference
def callback(chunk):
    print(chunk, end='', flush=True)

result = engine.infer_stream(
    prompt="Your prompt",
    callback=callback,
    max_tokens=200
)

# Batch inference
results = engine.batch_infer(
    prompts=["Q1", "Q2", "Q3"],
    max_tokens=50
)

# Get metrics
metrics = engine.get_metrics()
print(metrics['latency']['mean_ms'])
print(metrics['throughput']['tokens_per_sec'])

# Reset metrics
engine.reset_metrics()

# Unload model
engine.unload_model()
```

### InferResult

```python
result = engine.infer("Test", max_tokens=50)

# Properties
result.success          # bool: Success status
result.text            # str: Generated text
result.tokens_generated # int: Number of tokens
result.latency_ms      # float: Latency in ms
result.tokens_per_sec  # float: Throughput
result.error_message   # str: Error if failed
```

### Utilities

```python
# Check CUDA availability
llcuda.check_cuda_available()  # Returns bool

# Get GPU info
llcuda.get_cuda_device_info()  # Returns dict or None

# Quick inference (convenience function)
text = llcuda.quick_infer(
    model_path="model.gguf",
    prompt="What is AI?",
    max_tokens=100,
    gpu_layers=99
)
```

---

## 🎨 Complete Example

```python
import llcuda

# Check CUDA
print("CUDA Available:", llcuda.check_cuda_available())
print("GPU:", llcuda.get_cuda_device_info())

# Create engine
engine = llcuda.InferenceEngine()

# Load model (placeholder - llama-server handles actual model)
open('/tmp/model.gguf', 'a').close()
engine.load_model('/tmp/model.gguf', gpu_layers=99)

# Basic inference
result = engine.infer(
    prompt="Explain machine learning in simple terms.",
    max_tokens=100,
    temperature=0.7
)

print("\n" + "="*60)
print("RESULT:")
print("="*60)
print(result.text)
print("\n" + "="*60)
print(f"📊 Tokens: {result.tokens_generated}")
print(f"⏱️  Latency: {result.latency_ms:.0f}ms")
print(f"🚀 Speed: {result.tokens_per_sec:.1f} tok/s")
print("="*60)

# Streaming example
print("\n📝 Streaming output:")
print("-" * 60)

def print_chunk(chunk):
    print(chunk, end='', flush=True)

result = engine.infer_stream(
    prompt="Write a haiku about programming.",
    callback=print_chunk,
    max_tokens=50
)
print("\n" + "-" * 60)

# Batch example
prompts = [
    "What is Python?",
    "What is CUDA?",
    "What is deep learning?"
]

print("\n📚 Batch processing:")
results = engine.batch_infer(prompts, max_tokens=30)

for i, (prompt, result) in enumerate(zip(prompts, results)):
    print(f"\n{i+1}. {prompt}")
    print(f"   → {result.text[:80]}...")

# Metrics
metrics = engine.get_metrics()
print("\n📈 Performance Metrics:")
print(f"   Mean latency: {metrics['latency']['mean_ms']:.0f}ms")
print(f"   p95 latency: {metrics['latency']['p95_ms']:.0f}ms")
print(f"   Throughput: {metrics['throughput']['tokens_per_sec']:.1f} tok/s")
```

---

## 🔍 Troubleshooting

### Import Error

```python
# Error: ImportError: cannot import name '_llcuda'

# Solution: Reinstall with correct CUDA architecture
!pip uninstall llcuda -y
!rm -rf build/ dist/ *.egg-info
import os
os.environ['CUDA_ARCHITECTURES'] = '75'  # Your GPU
!pip install -e .
```

### CUDA Not Found

```bash
# Error: Could not find CUDA toolkit

# Solution: Set CUDA path
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Out of Memory

```python
# Solution: Reduce GPU layers
engine.load_model(model_path, gpu_layers=10)  # Instead of 99

# Or use smaller model
# Use TinyLlama (1.1B) instead of Mistral (7B)
```

---

## 📚 Documentation Files

- **README.md** - Complete package documentation
- **INSTALL.md** - Detailed installation instructions
- **KAGGLE_COLAB.md** - Kaggle/Colab specific guide
- **PACKAGE_SUMMARY.md** - Technical overview
- **examples/kaggle_colab_example.ipynb** - Working notebook

---

## 🎯 Recommended Models for T4

### Small (< 3GB)
- **Gemma 2B Q4_K_M** - 1.6GB, ~45 tok/s
- **TinyLlama 1.1B Q4_K_M** - 0.6GB, ~80 tok/s

### Medium (3-5GB)
- **Mistral 7B Q4_K_M** - 4.4GB, ~28 tok/s (best quality)
- **Llama 2 7B Q4_K_M** - 4.1GB, ~26 tok/s

---

## 🚀 Next Steps

1. **Choose your platform**: Kaggle, Colab, or Local
2. **Follow the setup**: Use the copy-paste scripts above
3. **Download a model**: From HuggingFace
4. **Start inferencing**: Use the API examples
5. **Check the notebook**: See `examples/kaggle_colab_example.ipynb`

---

## 💡 Tips

- **First time?** Start with Kaggle - easiest setup
- **Want to experiment?** Use Colab - longer sessions
- **Building locally?** Check INSTALL.md for detailed steps
- **Need help?** Check the documentation files

---

## ✅ Verification Checklist

After setup, verify everything works:

```python
import llcuda

# 1. Check CUDA
assert llcuda.check_cuda_available(), "CUDA not available!"

# 2. Check GPU info
gpu_info = llcuda.get_cuda_device_info()
assert gpu_info is not None, "GPU not detected!"
print(f"✅ GPU: {gpu_info['name']}")

# 3. Create engine
engine = llcuda.InferenceEngine()
assert engine is not None, "Engine creation failed!"

# 4. Load model (dummy)
open('/tmp/model.gguf', 'a').close()
assert engine.load_model('/tmp/model.gguf'), "Model load failed!"

# 5. Run inference
result = engine.infer("Test", max_tokens=10)
assert result.success, f"Inference failed: {result.error_message}"

print("✅ All checks passed! Package working correctly.")
```

---

**You're ready to run CUDA-accelerated LLM inference!** 🎉

For more details, see the documentation files included in the package.
