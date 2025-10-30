# Contributing to GMCS

Thank you for your interest in contributing to the GMCS Universal Platform! This document provides guidelines and instructions for contributing.

## Ways to Contribute

- **Bug Reports**: Found a bug? Open an issue with detailed reproduction steps
- **Feature Requests**: Have an idea? Propose it in the discussions or issues
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Improve docs, add examples, or write tutorials
- **Testing**: Write tests, report edge cases, or improve test coverage
- **Domain Presets**: Create new application presets for different domains
- **GMCS Algorithms**: Implement new signal processing algorithms

## Getting Started

### Development Setup

```bash
# Clone the repository
git clone https://github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate.git
cd gmcs

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .  # Editable install

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run tests to verify setup
pytest tests/ -v
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Follow the existing code style (see Style Guide below)
- Write clear, descriptive commit messages
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run stability tests
pytest tests/test_stability.py -v --slow

# Run benchmarks
python -m src.tools.perf
```

### 4. Submit a Pull Request

- Push your branch to GitHub
- Create a pull request with a clear description
- Link related issues
- Wait for review and address feedback

## Code Style Guide

### Python

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Type hints**: Always use type hints for function signatures
- **Docstrings**: Google-style docstrings for all public functions
- **Formatting**: Use `black` for auto-formatting

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
    pass
```

### TypeScript/React

- **Style**: Follow Airbnb style guide
- **Components**: Use functional components with hooks
- **Types**: Prefer explicit interfaces over inline types
- **Formatting**: Use Prettier

```typescript
interface ComponentProps {
  value: number;
  onChange: (value: number) => void;
}

export const MyComponent: React.FC<ComponentProps> = ({ value, onChange }) => {
  return <div>{value}</div>;
};
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add optical Kerr algorithm to GMCS pipeline
fix: resolve NaN in wave PDE boundary conditions
docs: update API reference for /session/create
test: add stability tests for 1024 nodes
refactor: simplify EBM weight update logic
perf: optimize field downsampling for WebSocket
```

## Testing Guidelines

### Writing Tests

- **Unit tests**: Test individual functions/modules in isolation
- **Integration tests**: Test component interactions
- **Stability tests**: Test long-running scenarios and edge cases
- **Performance tests**: Benchmark critical paths

```python
def test_my_feature():
    """Test that my feature works correctly."""
    # Arrange
    state = initialize_system_state(jax.random.PRNGKey(0))
    
    # Act
    result = my_function(state)
    
    # Assert
    assert result is not None
    assert validate_state(result)
```

### Test Coverage

- Aim for >80% coverage on critical modules
- All new features must include tests
- Bug fixes should include regression tests

## Documentation

### Where to Document

- **Code**: Docstrings for public APIs
- **README.md**: High-level overview and quick start
- **docs/**: Detailed documentation and guides
- **Examples**: Standalone scripts in `examples/`

### Documentation Style

- Use clear, concise language
- Include code examples
- Explain *why*, not just *what*
- Keep it up-to-date with code changes

## Reporting Bugs

### Good Bug Reports Include

1. **Clear title**: Summarize the issue
2. **Environment**: OS, Python version, GPU/CPU
3. **Steps to reproduce**: Minimal code example
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Screenshots/logs**: If applicable

### Bug Report Template

```markdown
**Environment:**
- OS: Ubuntu 22.04
- Python: 3.10.12
- JAX: 0.4.23
- GPU: NVIDIA RTX 3090

**Describe the bug:**
Wave field becomes NaN after ~500 steps with high wave speed.

**To Reproduce:**
\`\`\`python
state = initialize_system_state(key, dt=0.01)
state = state._replace(c_val=jnp.array([5.0]))
for i in range(1000):
    state = simulation_step(state)
    print(f"Step {i}: max field = {jnp.max(jnp.abs(state.field_p))}")
\`\`\`

**Expected:** Field remains bounded
**Actual:** Field becomes NaN after ~500 steps

**Additional context:**
This only happens with c_val > 3.0
```

## Feature Requests

### Good Feature Requests Include

1. **Use case**: What problem does it solve?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Examples**: Similar features in other projects

## Architecture Guidelines

### Adding New GMCS Algorithms

1. Add algorithm constant to `src/core/gmcs_pipeline.py`
2. Implement pure function: `jnp.ndarray → jnp.ndarray`
3. Add to `apply_single_algo` switch statement
4. Write unit tests
5. Update documentation

```python
ALGO_NEW_FEATURE = 10

def new_feature_function(x: jnp.ndarray, param: float) -> jnp.ndarray:
    """Apply new feature transformation."""
    return jnp.tanh(x * param)
```

### Adding New Domain Presets

1. Create preset in `src/config/presets.py`
2. Define `ApplicationConfig` with appropriate adapters
3. Add tests in `tests/test_integration.py`
4. Document in `docs/domain_presets.md`

### Modifying Core Simulation

- ⚠️ Core modules (`state.py`, `simulation.py`) are sensitive
- Always maintain JAX immutability
- Preserve JIT compatibility
- Add extensive tests for any changes
- Benchmark performance impact

## Community

- **GitHub Discussions**: Ask questions, share ideas
- **Issue Tracker**: Report bugs, request features
- **Pull Requests**: Contribute code

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment.

### Expected Behavior

- Be respectful and constructive
- Welcome newcomers
- Give and receive feedback gracefully
- Focus on what's best for the community

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling or inflammatory remarks
- Spam or off-topic content

## Priority Areas

### High Priority

- Performance optimizations (GPU utilization, memory efficiency)
- Additional GMCS algorithms (optical Kerr, resonators, etc.)
- Domain-specific presets (RL, scientific computing, etc.)
- Comprehensive documentation and tutorials

### Medium Priority

- Web-based configuration UI
- Real-time parameter visualization
- Pre-built Docker images
- Integration examples (PyTorch, TensorFlow, etc.)

### Nice to Have

- Mobile visualization app
- Cloud deployment templates
- Community preset gallery
- Interactive tutorials

## Questions?

- Open a [GitHub Discussion](https://github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate/discussions)
- Check existing [issues](https://github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate/issues) and docs
- Website: [gavriel.tech](https://gavriel.tech)
- Twitter/X: [@GavrielTech](https://x.com/GavrielTech)

---

Thank you for contributing to GMCS! 🌊✨


