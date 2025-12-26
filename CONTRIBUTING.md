# Contributing to llcuda

Thank you for your interest in contributing to llcuda! This document provides guidelines for contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Documentation](#documentation)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and collaborative environment.

### Our Standards
- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Prioritize community benefit

## Getting Started

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub
   git clone https://github.com/YOUR_USERNAME/llcuda.git
   cd llcuda
   ```

2. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/waqasm86/llcuda.git
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites
- Python 3.11+
- CUDA 11.7+ or 12.0+
- CMake 3.24+
- NVIDIA GPU
- Git

### Install Development Dependencies

```bash
# Install package in development mode
pip install -e ".[dev]"

# Or install manually
pip install pytest pytest-cov black mypy sphinx
```

### Build the Extension

```bash
# Set CUDA architecture for your GPU
export CUDA_ARCHITECTURES=75  # For T4 GPU

# Build
python setup.py build_ext --inplace
```

## Making Changes

### 1. Keep Changes Focused
- One feature/fix per pull request
- Related changes should be grouped

### 2. Write Clear Commit Messages
```
feat: Add support for streaming inference
fix: Resolve memory leak in GPU allocation
docs: Update installation instructions
test: Add unit tests for batch inference
```

Prefix options:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `chore:` - Maintenance tasks

### 3. Update Tests
- Add tests for new features
- Ensure existing tests pass
- Aim for >80% code coverage

### 4. Update Documentation
- Update README.md if needed
- Add docstrings to new functions/classes
- Update relevant .md files

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=llcuda tests/

# Run specific test
pytest tests/test_llcuda.py::test_function_name
```

### Test Guidelines
- Write descriptive test names
- Test edge cases
- Mock external dependencies
- Keep tests fast and isolated

## Submitting Changes

### 1. Update Your Branch

```bash
# Fetch latest changes
git fetch upstream
git rebase upstream/main
```

### 2. Format Code

```bash
# Format with black
black llcuda/ tests/

# Check types
mypy llcuda/
```

### 3. Run Tests

```bash
pytest tests/
```

### 4. Push Changes

```bash
git push origin feature/your-feature-name
```

### 5. Create Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request"
3. Fill in the template:
   - **Title**: Clear, descriptive title
   - **Description**: What changed and why
   - **Related Issues**: Link to related issues
   - **Testing**: How you tested the changes

### Pull Request Checklist

- [ ] Code follows project style
- [ ] Tests added/updated
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Commits are clear and atomic
- [ ] Branch is up-to-date with main

## Coding Standards

### Python Code Style

- Follow [PEP 8](https://pep8.org/)
- Use `black` for formatting (line length: 88)
- Use type hints
- Write docstrings (Google style)

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative
    """
    pass
```

### C++ Code Style

- Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use descriptive variable names
- Add comments for complex logic

### File Organization

```
llcuda/
├── __init__.py          # Public API
├── llcuda_py.cpp        # C++ bindings
├── CMakeLists.txt       # Build config
tests/
├── __init__.py
├── test_llcuda.py       # Unit tests
examples/
├── kaggle_colab_example.ipynb
```

## Documentation

### Docstrings

- Use Google-style docstrings
- Document all public functions/classes
- Include examples where helpful

### Markdown Files

- Use clear headings
- Include code examples
- Keep formatting consistent

### README Updates

Update README.md for:
- New features
- Changed requirements
- Updated installation steps
- New examples

## Areas for Contribution

### High Priority
- Pre-built wheels for common platforms
- Additional GPU architecture support
- Performance optimizations
- More comprehensive tests
- Improved error messages

### Documentation
- Tutorial notebooks
- API reference documentation
- Video tutorials
- Translation to other languages

### Examples
- Real-world use cases
- Integration examples
- Benchmark scripts

## Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact waqasm86@gmail.com

## Recognition

Contributors will be:
- Listed in CHANGELOG.md
- Acknowledged in releases
- Added to contributors list

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to llcuda! 🚀
