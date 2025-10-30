# GMCS (Generalized Modular Control System) - Universal Chaotic-Neuro Computational Substrate Platform v0.1

A GPU-accelerated platform for chaotic-neuro computation, real-time audio-reactive simulation, and energy-based machine learning.

![GMCS Interface](docs/images/Screenshot%202025-10-30%20105601.png)

License: MIT | Python 3.10+ | JAX GPU Accelerated | Next.js 14 | THRML Integrated

**Status:** Under active development. We are still troubleshooting major systems and APIs, so expect rapid changes and occasional instability.

**THRML Notice:** The THRML simulator integration remains under active troubleshooting; live sampling, energy metrics, and p-bit visuals may be intermittent.

---

## Overview

The GMCS - Universal Chaotic-Neuro Computational Substrate Platform is an experimental programmable chaotic computer where thousands of coupled Chua oscillators interact through a 3D wave field while an Energy-Based Model (EBM) powered by THRML learns and modulates their collective dynamics. The system features 21 signal processing algorithms, universal modulation matrix, ML framework integration, and an extensible plugin architecture.

### Applications

Audio Synthesis - Real-time chaotic sound generation and music production  
Generative Art - Dynamic visual patterns and interactive installations  
Research - Chaos theory, energy-based models, and computational neuroscience  
Photonic Computing - Optical neural network simulation and design  
ML Integration - Feature extraction, data augmentation, and hybrid models  
Interactive Systems - Real-time responsive environments and installations  
Signal Processing - Advanced audio and signal transformation  
Control Systems - Nonlinear dynamics and adaptive control  
Cryptography - Chaotic encryption and secure key generation  
Anomaly Detection - Pattern recognition and outlier identification  
Optimization - Nonlinear optimization using chaotic dynamics  
Simulation - Complex system modeling and emergent behavior studies

---

## Key Features

### Node-Based Visual Interface
Modular Node Graph - Visual programming interface for building computational pipelines  
20+ Node Types - Oscillators, algorithms, THRML models, P-bit dynamics, and visualizers  
Embedded Visualizers - Real-time oscilloscopes, spectrograms, phase space plots, energy graphs, P-bit mapper  
Drag-and-Drop - Intuitive node placement and connection routing  
Live Configuration - Real-time parameter editing with instant feedback  
Professional Theme - Clean Linux-inspired dark interface

### Core System
1024 Chaotic Oscillators (Chua circuits) with real-time dynamics  
3D Wave Field (256×256 FDTD PDE) with complex-valued support for photonics  
21 GMCS Algorithms (7 basic, 7 audio/signal, 7 photonic)  
THRML Integration - Full energy-based model with block Gibbs sampling  
Universal Modulation Matrix - Bidirectional routing between all components  
Multi-GPU Support - JAX pmap parallelization  
Real-time Audio - Pitch and RMS detection with audio reactivity

### Advanced Features
Higher-Order Interactions - 3-way and 4-way THRML coupling  
Heterogeneous Nodes - Spin, continuous, and discrete node types  
Conditional Sampling - Clamped nodes for targeted generation  
Custom Energy Factors - Photonic, audio, ML regularization  
Advanced Sampling - Adaptive, annealed, parallel tempering schedules  
ML Integration - PyTorch, TensorFlow, HuggingFace support  
Plugin System - Extensible architecture for custom algorithms  \
Session Management - Save/load complete system state

### Production Features
80+ REST API Endpoints - Complete control via HTTP  
WebSocket Streaming - Binary msgpack for real-time data  
Next.js Frontend - Modern React UI with cyberpunk styling  
Docker Support - Containerized deployment  
Comprehensive Testing - 100+ test cases  
Full Documentation - API, algorithms, deployment guides

---

## 21 GMCS Algorithms

### Basic Algorithms (0-6)
| ID | Name | Description | Parameters |
|----|------|-------------|------------|
| 0 | No-Op | Pass-through | None |
| 1 | Limiter | Soft clipping (tanh) | A_max |
| 2 | Compressor | Dynamic range compression | R_comp, T_comp |
| 3 | Expander | Dynamic range expansion | R_exp, T_exp |
| 4 | Threshold | Hard threshold gate | T_comp |
| 5 | Phase Mod | Time-varying modulation | Phi, omega |
| 6 | Fold | Wave folding nonlinearity | gamma, beta |

### Audio/Signal Processing (7-13)
| ID | Name | Description | Parameters |
|----|------|-------------|------------|
| 7 | Resonator | Bandpass filter | f0, Q |
| 8 | Hilbert | 90° phase shift | None |
| 9 | Rectifier | Full-wave rectification | None |
| 10 | Quantizer | Bit-depth reduction | levels |
| 11 | Slew Limiter | Rate-of-change limiter | rate_limit |
| 12 | Cross Mod | Ring modulation | Phi, omega |
| 13 | Bipolar Fold | Symmetric wave folding | T_comp |

### Photonic Algorithms (14-20)
| ID | Name | Description | Parameters |
|----|------|-------------|------------|
| 14 | Optical Kerr | χ³ nonlinearity (SPM) | n2, beta |
| 15 | Electro-Optic | Pockels effect | V, V_pi |
| 16 | Optical Switch | Intensity-dependent switching | T_comp, gamma |
| 17 | Four-Wave Mixing | Parametric interaction | gamma, n2 |
| 18 | Raman Amplifier | Stimulated Raman scattering | gamma, n2, beta |
| 19 | Saturation | Soft saturation | A_max |
| 20 | Optical Gain | Linear amplification | gamma |

---

## P-Bit Dynamics Nodes

GMCS includes specialized nodes for probabilistic bit (p-bit) manipulation and THRML network control:

**P-Bit Compressor** - Probability distribution compression for stabilizing hot p-bits  
**P-Bit Limiter** - Hard limits on flip probability to prevent numerical instability  
**P-Bit Expander** - Amplify thermal fluctuations to escape local minima  
**P-Bit Gate** - Threshold-based activation for conditional p-bit routing  
**P-Bit Mapper Visualizer** - Real-time visualization of p-bit states with history tracking

These nodes enable fine-grained control over THRML energy landscapes, thermal noise levels, and exploration/exploitation dynamics in energy-based learning.

---

## Quick Start

### Prerequisites
Python 3.10+  
CUDA 11.8+ (for NVIDIA GPUs) or ROCm (for AMD)  
Node.js 18+ (for frontend)  
8GB+ RAM (16GB recommended)

### 1. Backend Setup
```bash
# Clone repository
git clone https://github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate
cd Chaotic-Neuro-Computational-Substrate

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install ML frameworks
pip install torch>=2.0.0  # PyTorch
pip install tensorflow>=2.13.0  # TensorFlow
pip install transformers>=4.30.0  # HuggingFace

# Run server
python src/main.py --host 0.0.0.0 --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev  # Serves UI on http://localhost:3000
```

### 3. Docker Deployment
```bash
# Build and run
docker-compose up --build

# With GPU support (NVIDIA)
docker-compose up --build --gpus all
```

---

## API Overview

### Core Endpoints
```bash
# System status
GET /health
GET /status

# Node management
POST /node/add
POST /node/update/{node_id}
DELETE /node/{node_id}

# Algorithm management
GET /algorithms/list
GET /algorithms/{algo_id}
GET /algorithms/categories
```

### THRML Endpoints
```bash
# Performance modes
POST /thrml/performance-mode  # speed, accuracy, research
POST /thrml/temperature
GET /thrml/energy

# Advanced features
POST /thrml/heterogeneous/configure
POST /thrml/clamp-nodes
POST /thrml/interactions/add
POST /thrml/factors/add
```

### Modulation Matrix
```bash
# Modulation routes
POST /modulation/routes
GET /modulation/routes
DELETE /modulation/routes/{route_id}
POST /modulation/presets/{preset_name}
GET /modulation/sources
GET /modulation/targets
```

### ML Integration
```bash
# Model management
GET /ml/models/list
POST /ml/models/register
POST /ml/models/{model_id}/forward
POST /ml/models/{model_id}/extract-features
GET /ml/frameworks/available
```

### Plugin System
```bash
# Plugin management
GET /plugins/list
POST /plugins/{plugin_id}/execute
POST /plugins/{plugin_id}/enable
POST /plugins/discover
GET /plugins/documentation
```

### Session Management
```bash
# Save/load sessions
POST /session/save?name=MySession
GET /session/list
POST /session/{session_id}/load
DELETE /session/{session_id}
```

Full API documentation: [docs/API_REFERENCE.md](docs/API_REFERENCE.md)

---

## Frontend Features

### Components
Algorithm Selector - Browse and select from 21 algorithms  
Modulation Matrix Editor - Visual routing between components  
Plugin Browser - Discover and execute custom plugins  
Observer Dashboard - Real-time THRML monitoring with charts  
Session Manager - Save/load complete configurations  
THRML Controls - Performance modes, temperature, sampling parameters



---

## ML Integration

### Supported Frameworks
PyTorch - Full wrapper with training support  
TensorFlow/Keras - Complete integration  
HuggingFace Transformers - Embeddings and pattern recognition

### Integration Patterns

#### 1. Feedback Control
```python
from src.ml.pytorch_integration import PyTorchModelWrapper

# Load model
model = create_gmcs_feedback_model(input_size=30, hidden_sizes=[64, 32], output_size=10)
wrapper = PyTorchModelWrapper(model)

# Forward pass
gmcs_state = jnp.array([...])  # Current state
output = wrapper.forward(gmcs_state)  # Get feedback
```

#### 2. Feature Extraction
```python
from src.ml.huggingface_integration import HuggingFaceModelWrapper

# Create wrapper
wrapper = HuggingFaceModelWrapper("bert-base-uncased")

# Extract features
oscillator_states = jnp.array([...])
features = wrapper.encode_oscillator_state(oscillator_states)
```

#### 3. Pattern Recognition
```python
from src.ml.huggingface_integration import GMCSPatternRecognizer

# Create recognizer
recognizer = GMCSPatternRecognizer()

# Add patterns
recognizer.add_pattern("stable", stable_state)
recognizer.add_pattern("chaotic", chaotic_state)

# Recognize current state
similarities = recognizer.recognize_pattern(current_state)
```

---

## Plugin System

### Creating Custom Plugins

```python
from src.plugins.plugin_base import AlgorithmPlugin, PluginMetadata
import jax.numpy as jnp

class MyCustomPlugin(AlgorithmPlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="MyPlugin",
            version="1.0.0",
            author="Your Name",
            description="Custom algorithm",
            category="algorithm",
            tags=["custom"],
            parameters=[
                {"name": "gain", "type": "float", "range": [0.0, 2.0], "default": 1.0}
            ]
        )
    
    def initialize(self, config):
        self.state = {"initialized": True}
    
    def compute(self, input_signal, params, **kwargs):
        gain = params[0]
        return input_signal * gain
```

Save to `src/plugins/custom/my_plugin.py` and call `/plugins/discover`.

Full guide: [docs/PLUGIN_DEVELOPMENT.md](docs/PLUGIN_DEVELOPMENT.md)

---

## Audio Reactivity

GMCS features real-time audio analysis with pitch detection (YIN algorithm) and RMS energy:

```python
# Audio automatically modulates:
# - Wave field speed (pitch → c)
# - Oscillator forcing (RMS → amplitude)
# - GMCS parameters (via modulation matrix)

# Disable audio
python src/main.py --no-audio

# List audio devices
python src/main.py --list-devices

# Select specific device
python src/main.py --audio-device 1
```

---

## Photonic Computing

GMCS includes 7 photonic algorithms for optical computing simulation:

### Features
Complex-valued fields - Full optical wave simulation  
Nonlinear optics - Kerr effect, four-wave mixing, Raman  
Electro-optic modulation - Pockels effect, optical switching  
Wavelength-dependent - Dispersion and coupling

### Example
```python
# Configure photonic node
requests.post("http://localhost:8000/node/add", json={
    "position": [128, 128],
    "chain": [14, 17, 20],  # Kerr → FWM → Gain
    "config": {
        "n2": 0.3,      # Nonlinear refractive index
        "gamma": 2.0,   # Nonlinear coefficient
        "beta": 1.5     # Propagation constant
    }
})
```

---

## Performance

### Benchmarks (NVIDIA RTX 3080)
| Configuration | FPS | GPU Memory |
|--------------|-----|------------|
| 64 nodes, 64×64 grid | 120 Hz | ~400 MB |
| 256 nodes, 128×128 grid | 60 Hz | ~1.5 GB |
| 1024 nodes, 256×256 grid | 30 Hz | ~6 GB |

### Optimization
```bash
# Run benchmarks
python -m src.tools.perf

# Profile
python -m cProfile -o profile.stats src/main.py
```

---

## Documentation

### Core Documentation
[Architecture](docs/architecture.md) - System design and data flow  
[Algorithm Reference](docs/ALGORITHM_REFERENCE.md) - All 21 algorithms detailed  
[Deployment Guide](docs/DEPLOYMENT.md) - Production deployment

### THRML Integration
[THRML Integration](docs/thrml_integration.md) - EBM integration guide  
[THRML Complete Integration](docs/THRML_COMPLETE_INTEGRATION.md) - Full verification

### Development
[Plugin Development](docs/PLUGIN_DEVELOPMENT.md) - Create custom plugins  
[Contributing](CONTRIBUTING.md) - Contribution guidelines  
[Security](SECURITY.md) - Security policy

### Project Information
[Repository Structure](REPOSITORY_STRUCTURE.md) - File organization  
[Feature Status](FEATURE_STATUS.md) - Current implementation status

---

## Testing

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_thrml_integration.py
pytest tests/test_ml_integration.py
pytest tests/test_plugin_system.py
pytest tests/test_api_integration.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run stability tests
pytest tests/test_stability.py -v
```

---

## Docker

### Basic Deployment
```bash
docker-compose up --build
```

### Production Deployment
```bash
# Build
docker build -t gmcs:latest .

# Run with GPU
docker run --gpus all -p 8000:8000 -p 3000:3000 gmcs:latest

# Scale horizontally
docker-compose up --scale gmcs-backend=4
```

---

## System Requirements

### Minimum
CPU: 4 cores  
RAM: 8 GB  
GPU: NVIDIA GTX 1060 / AMD RX 580 / Apple M1  
Storage: 10 GB  
OS: Linux, macOS, Windows 10+

### Recommended
CPU: 8+ cores  
RAM: 16+ GB  
GPU: NVIDIA RTX 3080 / AMD RX 6800 / Apple M2 Pro  
Storage: 50+ GB SSD  
OS: Linux (Ubuntu 20.04+)

### GPU Support
NVIDIA: CUDA 11.8+ (automatic via JAX)  
AMD: ROCm 5.0+ (JAX backend)  
Apple: Metal (automatic on M1/M2/M3)

---

## Use Cases

### Research
Chaotic dynamics studies  
Energy-based modeling  
Photonic computing  
Neural architecture search  
Emergent behavior

### Creative
Audio synthesis  
Visual generation  
Interactive art  
Music production  
Live performance

### Industrial
Signal processing  
Control systems  
Optimization  
Simulation  
Data augmentation

---

## Citation

```bibtex
@software{gmcs2025,
  title={GMCS: Universal Chaotic-Neuro Computational Platform},
  author={Luis Jake Gabriel III},
  organization={Gavriel Technologies},
  year={2025},
  version={0.1},
  url={https://github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate}
}
```

---

## Contributing

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pre-commit black ruff mypy pytest-cov

# Install pre-commit hooks
pre-commit install

# Run linters
black src/
ruff check src/
mypy src/
```

---

## License

MIT License - Copyright (c) 2025 Gavriel Technologies

See [LICENSE](LICENSE) for details.

---

## Acknowledgments

THRML library by Extropic AI  
JAX framework by Google  
FastAPI by Sebastián Ramírez  
Next.js by Vercel  
All contributors and users

---

## Contact

Organization: Gavriel Technologies  
Website: Gavriel.Tech  
Repository: github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate

---

## Status

**Version**: 0.1  
**Status**: In Development  
**Last Updated**: October 30, 2025
