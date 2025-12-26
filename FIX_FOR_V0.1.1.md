# Fix for llcuda v0.1.1 - Pure Python Version

## Problem Summary

llcuda v0.1.0 on PyPI fails to install because:
1. The package tries to compile C++ extensions
2. The required C++ source files are not included in the distribution
3. CMakeLists.txt expects files in `src/` and `include/` that don't exist in the package

## Solution: Make it Pure Python

Since your `llcuda/__init__.py` already provides a pure Python implementation that works via HTTP to llama-server, we should remove the C++ extension requirement entirely.

## Changes Needed for v0.1.1

### 1. Update `setup.py`

Remove the C++ extension build:

```python
# OLD setup.py (REMOVE THESE):
class CMakeExtension(Extension):
    ...

class CMakeBuild(build_ext):
    ...

setup(
    ...
    ext_modules=[CMakeExtension('llcuda._llcuda')],  # REMOVE THIS
    cmdclass={'build_ext': CMakeBuild},              # REMOVE THIS
    ...
)
```

```python
# NEW setup.py (SIMPLIFIED):
from setuptools import setup, find_packages
from pathlib import Path

def read_long_description():
    readme_path = Path(__file__).parent / 'README.md'
    if readme_path.exists():
        return readme_path.read_text(encoding='utf-8')
    return ''

setup(
    name='llcuda',
    version='0.1.1',  # Increment version!
    author='Waqas Muhammad',
    author_email='waqasm86@gmail.com',
    description='CUDA-accelerated LLM inference for Python',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/waqasm86/llcuda',
    project_urls={
        'Bug Tracker': 'https://github.com/waqasm86/llcuda/issues',
        'Documentation': 'https://github.com/waqasm86/llcuda#readme',
        'Source Code': 'https://github.com/waqasm86/llcuda',
    },
    packages=find_packages(include=['llcuda', 'llcuda.*']),
    python_requires='>=3.11',
    install_requires=[
        'numpy>=1.20.0',
        'requests>=2.20.0',  # For HTTP client
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'mypy>=0.950',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='llm cuda gpu inference deep-learning llama',
    zip_safe=True,  # Now it's pure Python!
)
```

### 2. Update `llcuda/__init__.py`

Remove the C++ import:

```python
# OLD (line 11-14):
try:
    from . import _llcuda
except ImportError:
    import _llcuda

# NEW (REMOVE THE IMPORT):
# No _llcuda import needed - pure Python implementation
```

Then update the classes to NOT use `_llcuda`:

```python
class InferenceEngine:
    def __init__(self, server_url: str = "http://127.0.0.1:8090"):
        """Initialize the inference engine."""
        # OLD:
        # self._engine = _llcuda.InferenceEngine()

        # NEW:
        self.server_url = server_url
        self._model_loaded = False
        self._metrics = {
            'latency': [],
            'throughput': {'total_tokens': 0, 'total_requests': 0}
        }

    def load_model(self, model_path, gpu_layers=0, **kwargs):
        """Load a model (connects to llama-server)."""
        import requests
        try:
            # Ping the server to check if it's running
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                self._model_loaded = True
                return True
        except requests.exceptions.RequestException:
            return False
        return False

    def infer(self, prompt, max_tokens=128, temperature=0.7, **kwargs):
        """Run inference via HTTP."""
        import requests
        import time

        start_time = time.time()

        try:
            response = requests.post(
                f"{self.server_url}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": kwargs.get('top_p', 0.9),
                    "top_k": kwargs.get('top_k', 40),
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                latency_ms = (time.time() - start_time) * 1000

                result = InferResult()
                result.success = True
                result.text = data.get('content', '')
                result.tokens_generated = data.get('tokens_predicted', 0)
                result.latency_ms = latency_ms
                result.tokens_per_sec = result.tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0

                return result
        except Exception as e:
            result = InferResult()
            result.success = False
            result.error_message = str(e)
            return result

    def get_metrics(self):
        """Get performance metrics."""
        return self._metrics

    def reset_metrics(self):
        """Reset metrics."""
        self._metrics = {
            'latency': [],
            'throughput': {'total_tokens': 0, 'total_requests': 0}
        }

    @property
    def is_loaded(self):
        """Check if model is loaded."""
        return self._model_loaded


class InferResult:
    """Inference result."""
    def __init__(self):
        self.success = False
        self.text = ""
        self.tokens_generated = 0
        self.latency_ms = 0.0
        self.tokens_per_sec = 0.0
        self.error_message = ""
```

### 3. Update `pyproject.toml`

```toml
[project]
name = "llcuda"
version = "0.1.1"  # Increment version!
...

dependencies = [
    "numpy>=1.20.0",
    "requests>=2.20.0",  # Add this
]

# REMOVE build-system dependencies on cmake and pybind11
[build-system]
requires = [
    "setuptools>=45",
    "wheel",
]
build-backend = "setuptools.build_meta"
```

### 4. Update `MANIFEST.in`

```
# Include documentation
include README.md
include LICENSE
include CHANGELOG.md
...

# REMOVE C++ files (not needed anymore)
# recursive-include llcuda *.cpp
# recursive-include llcuda *.h
# recursive-include llcuda CMakeLists.txt
```

### 5. Update `CHANGELOG.md`

```markdown
## [0.1.1] - 2024-12-27

### Changed
- Converted to pure Python package (removed C++ extensions)
- Now installs without compilation on all platforms
- Uses HTTP client to communicate with llama-server backend

### Fixed
- Fixed PyPI installation failure due to missing C++ source files
- Package now installs successfully with `pip install llcuda`

### Removed
- C++ extension compilation requirement
- CMake build system
- pybind11 dependency

## [0.1.0] - 2024-12-26

### Added
- Initial release (had installation issues - see v0.1.1)
```

### 6. Update `README.md`

Update installation section:

```markdown
## Installation

### From PyPI

```bash
pip install llcuda
```

**Note**: llcuda v0.1.1+ is pure Python and installs without compilation.
Version 0.1.0 had installation issues - please use 0.1.1 or later.

### Requirements

- Python 3.11+
- `requests` library (installed automatically)
- Running `llama-server` instance (backend)

**System requirements** (for llama-server):
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.7+ or 12.0+
- llama.cpp compiled with CUDA support
```

## Steps to Release v0.1.1

1. **Make all changes above** to your local repository

2. **Test locally**:
   ```bash
   pip install -e .
   python -c "import llcuda; print(llcuda.__version__)"
   ```

3. **Update version in 3 files**:
   - `setup.py`: `version='0.1.1'`
   - `pyproject.toml`: `version = "0.1.1"`
   - `llcuda/__init__.py`: `__version__ = "0.1.1"`

4. **Commit changes**:
   ```bash
   git add -A
   git commit -m "Release v0.1.1: Pure Python package (fix installation issues)"
   git tag -a v0.1.1 -m "Version 0.1.1"
   git push origin main --tags
   ```

5. **Build new distribution**:
   ```bash
   rm -rf dist/ build/ *.egg-info
   python -m build --sdist --wheel
   ```

6. **Upload to PyPI**:
   ```bash
   python -m twine check dist/*
   python -m twine upload dist/*
   ```

7. **Test installation**:
   ```bash
   pip install llcuda==0.1.1
   ```

## Benefits of Pure Python Version

✅ **Installs everywhere** - No compilation needed
✅ **Faster installation** - No 2-3 minute build time
✅ **Works on Windows** - No MSVC compiler needed
✅ **Simpler to maintain** - No C++ code to debug
✅ **Same functionality** - Still uses llama-server backend

## Backward Compatibility

The API remains exactly the same - users don't need to change their code!

```python
# This still works the same way:
import llcuda

engine = llcuda.InferenceEngine()
result = engine.infer("Hello!", max_tokens=50)
print(result.text)
```

---

**This fix will make llcuda installable on Kaggle/Colab with just `pip install llcuda`!**
