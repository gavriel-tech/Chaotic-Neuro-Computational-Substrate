<h1 align='center'>THRML</h1>

THRML is a JAX library for building and sampling probabilistic graphical models, with a focus on efficient block Gibbs sampling and energy-based models. Extropic is developing hardware to make sampling from certain classes of discrete PGMs massively more energy‑efficient; THRML provides GPU‑accelerated tools for block sampling on sparse, heterogeneous graphs, making it a natural place to prototype today and experiment with future Extropic hardware.

Features include:

- Blocked Gibbs sampling for PGMs
- Arbitrary PyTree node states
- Support for heterogeneous graphical models
- Discrete EBM utilities
- Enables early experimentation with future Extropic hardware

From a technical point of view, the internal structure compiles factor-based interactions to a compact "global" state representation, minimising Python loops and maximising array-level parallelism in JAX.

## Installation

Requires Python 3.10+.

```bash
pip install thrml
```

## Documentation

Available at [docs.thrml.ai](https://docs.thrml.ai/en/latest/).


## Quick example

Sampling a small Ising chain with two-color block Gibbs:

```python
import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[i], nodes[i+1]) for i in range(4)]
biases = jnp.zeros((5,))
weights = jnp.ones((4,)) * 0.5
beta = jnp.array(1.0)
model = IsingEBM(nodes, edges, biases, weights, beta)

free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

key = jax.random.key(0)
k_init, k_samp = jax.random.split(key, 2)
init_state = hinton_init(k_init, model, free_blocks, ())
schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
```

## Developing

To get started, you'll need to create a virtual environment and install the requirements:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the package with development dependencies
pip install -e ".[development,testing,examples]"

# Install pre-commit hooks
pre-commit install
```

The pre-commit hooks will automatically run code formatting and linting tools (ruff, black, isort, pyright) on every commit to ensure consistent style.

If you want to skip pre-commit (for a WIP commit), you can use the `--no-verify` flag:

```bash
git commit --no-verify -m "Your commit message"
```
