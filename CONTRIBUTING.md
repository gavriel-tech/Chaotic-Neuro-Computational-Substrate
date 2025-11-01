# Contributing to GMCS

Thank you for your interest in contributing to the GMCS (Generalized Modular Control System) project. This document provides guidelines for contributing bug fixes, improvements, and troubleshooting to the codebase. It reflects the November 2025 THRML/node integration refresh.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Architecture Principles](#architecture-principles)
- [Bug Fixes and Troubleshooting](#bug-fixes-and-troubleshooting)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Be respectful and professional. We value technical discussions and constructive feedback. Personal attacks or harassment will not be tolerated.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment (see below)
4. Create a feature branch from `main`
5. Make your changes
6. Test thoroughly
7. Submit a pull request

## Development Environment

### Backend Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Chaotic-Neuro-Computational-Substrate
cd Chaotic-Neuro-Computational-Substrate

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest tests/ -v
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev  # Next.js dev server (consumes REST + websocket APIs)
```

### GPU Setup

The system uses JAX for GPU acceleration. JAX automatically detects CUDA, ROCm, or Metal:

```bash
# Check GPU availability
python -c "import jax; print(jax.devices())"
```

If you encounter GPU issues, see the JAX documentation for platform-specific setup.

## Architecture Principles

The GMCS architecture follows several core design patterns. When contributing, respect these principles:

### 1. Immutable State (JAX PyTree)

The system uses JAX's functional programming paradigm with immutable PyTree structures. Never mutate `SystemState` directly:

```python
# Correct
new_state = state._replace(t=state.t + dt)

# Incorrect
state.t += dt  # This won't work - state is immutable
```

### 2. Pre-allocated Arrays

All arrays are pre-allocated to maximum size (`N_MAX`, `GRID_W`, `GRID_H`). Active nodes are controlled via `node_active_mask`. This enables JIT compilation without dynamic resizing:

```python
# Node activation/deactivation uses masking
new_mask = state.node_active_mask.at[node_id].set(1.0)  # Activate
new_mask = state.node_active_mask.at[node_id].set(0.0)  # Deactivate
```

### 3. Separation of Concerns

The system is divided into distinct layers:

- **Core Layer** - Oscillators, wave PDE, THRML integration (JIT-compiled where possible)
- **API Layer** - FastAPI endpoints, WebSocket handling
- **Node System** - High-level node graph execution
- **Frontend** - React/Next.js UI, completely decoupled from backend

Don't mix concerns between layers. For example, don't add JAX code to the API layer or business logic to the core simulation functions.

### 4. Plugin Architecture

Custom functionality should be added via the plugin system rather than modifying core code. Plugins live in `src/plugins/custom/` and are discovered dynamically.

## Bug Fixes and Troubleshooting

### Before Starting

1. **Search existing issues** - Your bug may already be reported or fixed
2. **Reproduce the bug** - Document exact steps to reproduce
3. **Check recent changes** - Review recent commits that might have introduced the issue
4. **Isolate the problem** - Narrow down which component is failing

### Common Bug Categories

#### Simulation Instability

If oscillators or the wave field are exploding:

- Check `dt` value (should be 0.01 or smaller)
- Verify forcing amplitudes are reasonable
- Check for NaN propagation in state arrays
- Ensure node positions are within grid bounds

#### THRML Integration Issues

If THRML sampling fails or produces errors:

- Verify THRML is installed: `pip show thrml`
- Check that `n_active_nodes > 0` before creating/rebuilding THRML models
- Ensure weight matrix is symmetric for Ising models
- Validate temperature parameter is positive and CD-k ≥ 1
- When exposing node/THRML data through new endpoints, reuse `_serialize_node` helpers so the frontend stays in sync

#### WebSocket Connection Problems

If the frontend can't connect:

- Verify backend is running on the expected port
- Check CORS configuration in `src/api/server.py`
- Look for WebSocket upgrade errors in browser console
- Ensure no firewall is blocking the connection
- Confirm `/status` and `/nodes` endpoints return healthy JSON; the node graph depends on both REST and websocket channels

#### Node Execution Failures

If specific nodes crash:

- Check node configuration parameters are within valid ranges
- Verify input data types match expected types
- Look for missing imports or dependencies
- Check node implementation for edge cases

### Making the Fix

1. **Create a test first** - Write a test that fails with the bug
2. **Fix the issue** - Make minimal changes to fix the bug
3. **Verify the fix** - Ensure your test now passes
4. **Test side effects** - Run full test suite to catch regressions
5. **Document the fix** - Update comments if behavior changed

### Example Bug Fix Process

```bash
# Create feature branch
git checkout -b fix/oscillator-nan-handling

# Write failing test
# Edit tests/test_stability.py

# Run test to confirm it fails
pytest tests/test_stability.py::test_oscillator_nan_handling -v

# Make the fix
# Edit src/core/integrators.py

# Verify fix
pytest tests/test_stability.py::test_oscillator_nan_handling -v

# Run full suite
pytest

# Commit
git add tests/test_stability.py src/core/integrators.py
git commit -m "Fix NaN handling in Chua integrator"
```

## Code Style

### Python

We use Black, Ruff, and MyPy for code quality:

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Style Guidelines

- Maximum line length: 100 characters (Black default)
- Use type hints for all function signatures
- Docstrings required for public functions (Google style)
- Prefer explicit over implicit (e.g., specify types, don't use `from module import *`)

### Docstring Format

```python
def simulation_step(
    state: SystemState,
    enable_ebm_feedback: bool = True
) -> Tuple[SystemState, Optional[THRMLWrapper]]:
    """
    Execute one simulation step with THRML integration.
    
    Args:
        state: Current system state
        enable_ebm_feedback: Whether to apply THRML feedback
        
    Returns:
        Tuple of (new_state, thrml_wrapper)
        
    Raises:
        ValueError: If state is invalid
    """
    pass
```

### JavaScript/TypeScript

```bash
# Frontend formatting
cd frontend
npm run lint
npm run type-check
```

Follow the existing patterns in the Next.js codebase.

## Testing

### Running Tests

```bash
# Backend: full suite (includes THRML + API checks)
python -m pytest

# Backend: focused THRML/node integration
python -m pytest tests/test_thrml_integration.py -v

# Backend: coverage report
python -m pytest --cov=src --cov-report=html

# Frontend lint + tests
cd frontend
npm run lint
npm run test
```

### Writing Tests

Tests should be:

- **Isolated** - Don't depend on other tests
- **Deterministic** - Use fixed random seeds
- **Fast** - Mock expensive operations when possible
- **Clear** - Test one thing at a time

Example test structure:

```python
def test_node_activation():
    """Test that node activation updates the mask correctly."""
    # Setup
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Execute
    state, node_id = add_node_to_state(state, position=[128, 128])
    
    # Verify
    assert state.node_active_mask[node_id] == 1.0
    assert jnp.sum(state.node_active_mask) == 1.0
```

### Test Categories

- `tests/test_*.py` - Unit tests for individual components
- `tests/test_thrml_integration.py` - Backend THRML/node sanity checks
- `tests/test_integration.py` - Integration tests across components
- `tests/test_api_*.py` - API endpoint tests
- `tests/test_stability.py` - Numerical stability tests

## Submitting Changes

### Pull Request Process

1. **Update from main** - Rebase your branch on latest main
2. **Run all tests** - Ensure nothing is broken
3. **Update documentation** - Add docstrings, update README if needed
4. **Write clear commit messages** - Explain what and why
5. **Create PR** - Use the PR template

### Commit Message Format

```
<type>: <short summary>

<detailed description>

Fixes #<issue_number>
```

Types: `fix`, `feat`, `docs`, `style`, `refactor`, `test`, `chore`

Example:

```
fix: Handle NaN propagation in Chua integrator

Added checks for NaN values in oscillator state and reset
to small random values to prevent complete simulation failure.
Also added unit test to verify NaN handling.

Fixes #123
```

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines (Black, Ruff, MyPy)
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] No merge conflicts with main
- [ ] Commit messages are clear
- [ ] PR description explains the change

### Review Process

1. Automated tests run on your PR
2. Maintainers review code
3. Address feedback if requested
4. Once approved, maintainers merge

## Reporting Issues

### Bug Reports

Include:

1. **Description** - What happened vs. what you expected
2. **Steps to reproduce** - Exact commands/actions to trigger the bug
3. **Environment** - OS, Python version, GPU type, JAX version
4. **Error messages** - Full stack traces
5. **Code samples** - Minimal code that reproduces the issue

### Issue Template

```markdown
## Bug Description
Brief description of the bug.

## Steps to Reproduce
1. Run command X
2. Click button Y
3. Observe error Z

## Expected Behavior
What should happen.

## Actual Behavior
What actually happens.

## Environment
- OS: Ubuntu 22.04
- Python: 3.10.8
- JAX: 0.4.20
- GPU: NVIDIA RTX 3080
- CUDA: 11.8

## Error Messages
```
<paste full error here>
```

## Additional Context
Any other relevant information.
```

## Questions?

For questions about the architecture or contribution process, open a GitHub discussion or issue.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

