# GMCS (Generalized Modular Control System) - Universal Chaotic-Neuro Computational Substrate Platform v0.1

A GPU-accelerated platform for chaotic-neuro computation, real-time audio-reactive simulation, and energy-based machine learning.

![GMCS Interface](docs/images/Screenshot%202025-11-01%20133848.png)

![GMCS Node Graph](docs/images/Screenshot%202025-11-01%20134332.png)

License: MIT | Python 3.10+ | JAX GPU Accelerated | Next.js 14 | THRML Integrated

**Status:** Active development. Core architecture is implemented with 75+ nodes, 16+ presets, ML framework integration, and comprehensive APIs. Currently working on troublshooting simulation functionality, node validation, preset validation, model training, and production infrastructure.

**Codebase:** 78,577+ lines of code (51,624+ Python backend + 26,953+ TypeScript/JavaScript frontend)

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

## ML + Gradient Integration

Chaos-gradient hybrid platform with end-to-end ML integration.

### Comprehensive ML Capabilities
- **Differentiable Chaos** - Gradients through oscillator dynamics using JAX
- **12 Pre-Implemented Model Architectures** - Model definitions for various domains (training in progress)
- **Hybrid Training** - Combined chaos exploration with gradient optimization
- **Model Zoo** - Pre-registered models with HuggingFace Hub integration
- **Model Registry** - Centralized model management with metadata tracking
- **Gradient Flow Graph** - Mix differentiable and non-differentiable nodes
- **Unified Trainer** - Supports all model types with callbacks and validation
- **15+ Loss Functions** - Chaos-specific (Lyapunov, attractor distance, energy) plus standard ML losses

### Implemented ML Models
**Music & Audio:**
- GenreClassifier - CNN for 10-genre music classification
- MusicTransformer - Transformer encoder for music embeddings

**Reinforcement Learning:**
- PPOAgent - Actor-critic policy learning with PPO
- ValueFunction - State value estimation for RL

**Generative Models:**
- PixelArtGAN - GAN for 32x32 pixel art generation
- CodeGenerator - Transformer for source code generation

**Scientific Computing:**
- LogicGateDetector - CNN for logic gate classification (AND, OR, XOR, NAND)
- PerformancePredictor - MLP for neural architecture performance prediction
- EfficiencyPredictor - MLP for solar cell efficiency prediction
- CognitiveStateDecoder - CNN for brain state classification (5 classes)
- BindingPredictor - MLP for molecular binding affinity prediction
- MLPerformanceSelector - Multi-output MLP for algorithm performance metrics

### Quick Start (Backend)
```python
# Differentiable chaos with gradients
from src.core.differentiable_chua import DifferentiableChuaOptimizer
optimizer = DifferentiableChuaOptimizer(params, lr=0.01)
optimizer.train(initial_state, target_trajectory, forcing, dt, loss_fn, 100)

# Use model architectures (training in progress)
from src.ml.concrete_models import GenreClassifier
classifier = GenreClassifier("genre_model", config)
predictions = classifier.forward(spectrogram_features)

# Hybrid chaos-gradient training
from src.ml.hybrid_training import HybridTrainer
trainer = HybridTrainer(oscillator, ml_model, config)
history = trainer.train(data_generator, n_steps=1000)
```

For examples, see `examples/ml/` directory.

---

## Key Features

### Node-Based Visual Interface
Modular Node Graph - Visual programming interface for building computational pipelines  
77 Node Types - Oscillators, algorithms, THRML models, P-bit dynamics, ML models, analysis tools, and visualizers  
26 Presets - Pre-configured systems for music, cryptography, ML, neuroscience, visualization, and more  
Embedded Visualizers - Real-time oscilloscopes, spectrograms, phase space plots, energy graphs, P-bit mapper  
Drag-and-Drop - Intuitive node placement and connection routing  
Live Configuration - Real-time parameter editing with instant feedback  
Professional Theme - Clean Linux-inspired dark interface
### Core System
1024 Chaotic Oscillators (Chua circuits) with real-time dynamics  
3D Wave Field (256×256 FDTD PDE) with complex-valued support for photonics  
21 GMCS Algorithms (7 basic, 7 audio/signal, 7 photonic)  
THRML Integration - Block Gibbs sampling + CD learning with async rebuilds  
Universal Modulation Matrix - Bidirectional routing between oscillators, THRML, audio, ML  
Multi-GPU Support - JAX pmap parallelization (optional)  
Real-time Audio - Pitch/RMS detection with modulation routes

### Advanced Features
Higher-Order Interactions - 3-way/4-way THRML coupling  
Heterogeneous Nodes - Spin, continuous, and categorical sampler backends  
Conditional Sampling - Clamping via REST + THRML Advanced panels  
Custom Energy Factors - Photonic, audio, ML regularization hooks  
Advanced Sampling - Adaptive/annealed schedules, multi-chain CD-k  
ML Integration - PyTorch, TensorFlow, HuggingFace support with 12 model architectures  
Benchmarking Suite - THRML sampler and throughput diagnostics  
Plugin System - Extensible algorithm/node registry  
Persistence - Session save/load, checkpointing, API-driven state export

### Production Features
80+ REST API Endpoints - Complete control via HTTP  
WebSocket Streaming - Binary msgpack for real-time data  
Next.js Frontend - Modern React UI with cyberpunk styling  
Docker Support - Containerized deployment  
Comprehensive Testing - 100+ test cases  
Full Documentation - API, algorithms, deployment guides

---

## GMCS Algorithms

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

## Updated Node & THRML Integration (2025-11)

- Frontend node actions now round-trip through `/node/add`, `/node/update`, `/node/{id}`, and `/nodes`, capturing backend-assigned IDs so the visual canvas matches the live simulation. The Control Panel removal button queries `/nodes` before issuing deletes.
- THRML bridge nodes seed and advance their own PRNG keys, push bias updates into `THRMLWrapper`, and guard fallbacks if sampling fails. Advanced THRML endpoints (`/thrml/heterogeneous/configure`, `/thrml/clamp-nodes`, `/thrml/sample-conditional`) are wired to the `THRMLAdvancedPanels` UI.
- Backend serialization (`_serialize_node`) returns position, oscillator state, GMCS parameters, and chain data, enabling topology reconciliation and HUD overlays. WebSocket frames continue to stream state updates at ~30 Hz.
- Documentation across `ARCHITECTURE.md`, `CONTRIBUTING.md`, and other handbooks reflects the current workflows, testing expectations, and feature matrix.

---

## Available Presets

The system includes 26 pre-configured presets demonstrating various applications:

### Application Presets (16 JSON)

#### Research & Science
**Neural Sim** - Neural network simulation with THRML coupling  
**Neuromapping** - EEG analysis and brain state decoding  
**Quantum Opt** - Quantum optimization using chaotic dynamics  
**Photonic Sim** - Optical computing simulation with photonic algorithms  
**Molecular Design** - Molecular structure exploration and optimization  
**Solar Opt** - Solar cell efficiency optimization

#### Machine Learning
**NAS Discovery** - Neural architecture search with chaotic exploration  
**RL Boost** - Reinforcement learning with chaos-based exploration  
**Emergent Logic** - Logic gate detection and emergent computation  
**Game Code** - Code generation using chaos-gradient hybrid training

#### Media & Creative
**Live Music** - Real-time music analysis and synthesis  
**Music Analysis** - Chord detection, beat tracking, and harmony analysis  
**Live Video** - Video processing with chaotic effects  
**Pixel Art** - Generative pixel art using GANs  
**World Gen** - Procedural world generation

#### Security
**Chaos Crypto** - Complete chaos-based cryptography suite with encryption, hashing, and randomness testing

### Graph Presets (10 In-Code)

**Chaotic Attractor Visualization** - Lorenz attractor with 3D phase space and waveform visualizers  
**THRML Spin Glass Experiment** - Spin Glass EBM with spin state matrix and correlation analysis  
**Audio Processing Chain** - Chua oscillator through waveshaper, compressor, and spectrogram  
**P-Bit Dynamics Exploration** - Spin Glass with P-Bit manipulation nodes  
**Multi-Oscillator Synthesis** - Chua, Lorenz, and Van der Pol oscillators mixed together  
**THRML Training Pipeline** - Ising model with training, moment accumulation, and convergence detection  
**Photonic Processing** - Lorenz oscillator through optical Kerr effect and four-wave mixing  
**ML Predictor Training** - Lorenz attractor feeding MLP predictor for time series forecasting  
**Chaos Analysis Suite** - Chua oscillator with Lyapunov calculator, attractor analyzer, and FFT  
**Feedback Control System** - Van der Pol oscillator with PID controller for stabilization

Load application presets via API: `POST /presets/{preset_name}/load` or through the frontend preset browser.

---

## Node Types (77 Total)

### System Nodes (2)
**Audio Settings** - Audio I/O configuration and routing  
**Sampler Config** - THRML sampler backend and blocking strategy configuration

### Oscillators (3)
**Chua Oscillator** - Classic chaotic oscillator with three equilibrium points  
**Lorenz Attractor** - Atmospheric convection model with butterfly effect  
**Van der Pol** - Non-conservative oscillator

### Wave Processing (7)
**Waveshaper** - Nonlinear waveshaping and harmonic distortion  
**Resonator** - Resonant bandpass filter  
**Hilbert Transform** - Phase shift and envelope extraction  
**Compressor** - Dynamic range compression  
**Limiter** - Hard limiting  
**Expander** - Dynamic range expansion  
**Gate** - Threshold gate with sidechain

### Photonic Algorithms (2)
**Optical Kerr Effect** - Nonlinear refractive index modulation  
**Four-Wave Mixing** - Parametric wavelength conversion

### P-Bit Dynamics (5)
**P-Bit Compressor** - Probability distribution compression  
**P-Bit Limiter** - Hard probability limits  
**P-Bit Expander** - Thermal fluctuation amplification  
**P-Bit Gate** - Threshold-based p-bit routing  
**P-Bit Threshold** - Binary conversion with analog output

### Energy-Based Models (4)
**Spin Glass EBM** - Ising model with spin glass dynamics  
**Continuous EBM** - Continuous-valued energy-based model  
**Heterogeneous Model** - Mixed discrete and continuous nodes  
**Categorical EBM** - Multi-class categorical distributions

### THRML Sampling (11)
**Block Sampling Config** - Configure block Gibbs sampling strategy  
**Conditional Sampler** - Sampling with clamped observations  
**Multi-Chain Sampler** - Parallel chain sampling with convergence diagnostics  
**Annealing Scheduler** - Simulated annealing temperature scheduling  
**Moment Accumulator** - Online statistics (mean, variance, correlations)  
**Weighted Factor** - Custom energy factor with modulation  
**Higher-Order Interactions** - 3-way and 4-way coupling  
**Ising Trainer** - Contrastive divergence training  
**Persistent CD Trainer** - Persistent contrastive divergence with fantasy particles  
**Sampling Profiler** - Performance metrics (ESS, autocorrelation, acceptance rate)  
**Graph Coloring Optimizer** - Optimal block assignment for parallelism  
**State Validator** - Constraint validation

### Visualizers (9)
**Oscilloscope** - Time-domain waveform display  
**Spectrogram** - Frequency-domain visualization  
**Phase Space 3D** - 3D attractor visualization  
**Energy Graph** - Energy landscape over time  
**Spin State Matrix** - 2D spin configuration heatmap  
**Correlation Matrix** - Node correlation visualization  
**Waveform Monitor** - Broadcast-standard waveform display  
**XY Plot** - 2D parametric curves  
**P-Bit Mapper** - Real-time p-bit state history

### Machine Learning (6)
**MLP Predictor** - Multi-layer perceptron with training  
**CNN Classifier** - Convolutional neural network for time series  
**Transformer Encoder** - Attention-based sequence encoder  
**Diffusion Generator** - Denoising diffusion model  
**GAN Generator** - Generative adversarial network  
**RL Controller** - Reinforcement learning policy (PPO)  
**Autoencoder** - Dimensionality reduction and reconstruction

### Analysis Nodes (6)
**FFT Analyzer** - Fast Fourier transform and spectrum analysis  
**Pattern Recognizer** - Template matching and similarity detection  
**Lyapunov Calculator** - Chaos quantification via Lyapunov exponents  
**Attractor Analyzer** - Attractor dimension and structure analysis  
**Energy Surface Scanner** - Parameter space energy landscape mapping  
**Convergence Detector** - Fixed point and limit cycle detection

### Control Nodes (5)
**Parameter Optimizer** - Gradient-based parameter optimization  
**Chaos Controller** - OGY chaos control method  
**PID Controller** - Proportional-integral-derivative feedback control  
**Adaptive Gibbs Steps** - Dynamic Gibbs step adjustment based on convergence  
**Noise Generator** - White, pink, and colored noise generation

### Generator Nodes (3)
**Pattern Generator** - Periodic and chaotic pattern generation  
**Sequence Generator** - Time-series sequence synthesis  
**Audio File Upload** - Load audio files for processing

### I/O Nodes (2)
**Audio Input** - Real-time audio capture  
**Audio Output** - Real-time audio playback

### Music Processing Nodes
Chord Detector, Beat Tracker, Harmony Analyzer, Rhythm Generator, Music Transformer embeddings

### Cryptography Nodes
Stream Cipher, Hash Function, Key Derivation, Random Number Generator, Crypto Analyzer (NIST tests)

### Additional Nodes
Physics Simulators, Molecular Dynamics, EEG Processing, and more

For complete node documentation, see the frontend node browser or `frontend/components/NodeGraph/NodePresets.ts`.

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

# Install dependencies (includes all ML frameworks, testing tools, and dev dependencies)
pip install -r requirements.txt

# Run server
python src/main.py --host 0.0.0.0 --port 8000
```

Note: All dependencies including PyTorch, TensorFlow, testing tools (pytest, httpx), and development utilities are included in requirements.txt.

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

Full API documentation available through OpenAPI spec at `/docs` when server is running.

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

See `src/plugins/examples/` for more plugin examples.

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

## Chaos-Based Cryptography

GMCS includes a complete chaos-based cryptography suite demonstrating practical applications of chaotic dynamics in security:

### Implemented Nodes
**ChaosStreamCipher** - Stream encryption using Chua oscillator as keystream generator  
**HashFunction** - Chaos-based hashing with avalanche effect  
**KeyDerivation** - Cryptographic key generation from chaotic iteration  
**RandomNumberGenerator** - CSPRNG using coupled chaotic oscillators  
**CryptoAnalyzer** - NIST SP 800-22 statistical randomness tests

### Features
Sensitive dependence on initial conditions for strong cryptography  
Chaotic mixing for secure hashing  
Synchronized chaos for secure communication  
Statistical validation of randomness quality

### Example
```python
from src.processor.crypto_nodes import ChaosStreamCipher

# Initialize cipher with key
cipher = ChaosStreamCipher(config)
cipher.set_key(b"my_secret_key")

# Encrypt data
result = cipher.process(b"Hello World", mode='encrypt')
ciphertext = result['output']

# Decrypt data
cipher.set_key(b"my_secret_key")  # Reset with same key
result = cipher.process(ciphertext, mode='decrypt')
plaintext = result['output']
```

Run the chaos cryptography demonstration:
```bash
python demos/run_preset.py chaos_crypto
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
See `docs/thrml_docs/` for THRML integration documentation  
See `src/` directory for inline code documentation  
API documentation available at `/docs` when server is running

### Project Information
[Implementation Guide](WHAT_TO_IMPLEMENT.md) - Future enhancements and optional features  
[Implementation Status](WHAT_IS_MISSING.md) - Current implementation audit

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

## Connection Reliability Configuration

Fine-tune WebSocket behaviour with the following environment variables. Frontend values must be exported with the `NEXT_PUBLIC_` prefix so that Next.js can embed them in the client bundle.

| Variable | Scope | Default | Description |
|----------|-------|---------|-------------|
| `NEXT_PUBLIC_WS_HEARTBEAT_MS` | Frontend | `10000` | Interval (ms) between client heartbeat pings. |
| `NEXT_PUBLIC_WS_HEARTBEAT_TIMEOUT_MS` | Frontend | `20000` | Close and reconnect if no heartbeat ACK within this window. |
| `NEXT_PUBLIC_WS_RECONNECT_BASE_MS` | Frontend | `2000` | Initial reconnect delay; grows exponentially until the max. |
| `NEXT_PUBLIC_WS_RECONNECT_MAX_MS` | Frontend | `15000` | Maximum delay (ms) between reconnect attempts. |
| `NEXT_PUBLIC_WS_STALE_TIMEOUT_MS` | Frontend | `3000` | Mark the stream stale if no packets arrive within this timeout. |
| `GMCS_WS_HEARTBEAT_TIMEOUT` | Backend | `30` | Drop WebSocket clients (seconds) that stop sending heartbeats. |
| `GMCS_WS_HEALTH_CHECK_INTERVAL` | Backend | `5` | Interval (seconds) for pruning stale connections. |
| `GMCS_WS_STATUS_INTERVAL` | Backend | `1` | How often (seconds) to push STATUS packets while the sim is paused. |

Recommended production settings:

- Keep the frontend heartbeat interval between 5–10 s and set the timeout to ~2× the interval.
- When proxies aggressively close idle sockets, lower `NEXT_PUBLIC_WS_HEARTBEAT_MS` to 5000 and backend `GMCS_WS_HEARTBEAT_TIMEOUT` to 15.
- For slower demo hardware, raise `NEXT_PUBLIC_WS_STALE_TIMEOUT_MS` (e.g. 5000) to avoid false positives while paused.

---

## Contributing

We welcome contributions. Development dependencies are included in requirements.txt.

### Development Setup
```bash
# Install all dependencies (includes dev tools)
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run linters
black src/
ruff check src/
mypy src/
```

**Local environment tips:**
- Set `GMCS_RATE_LIMIT_ENABLED=false` during local frontend dev to avoid CORS-style 429s when the UI polls every few seconds. Tune `GMCS_RATE_LIMIT_RPM`/`GMCS_RATE_LIMIT_WINDOW` when load testing.
- Adjust frontend polling via `NEXT_PUBLIC_STATUS_POLL_MS` (default 2000 ms) and `NEXT_PUBLIC_STATUS_REQUEST_TIMEOUT_MS` if you need faster/slower HUD refreshes.

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
**Last Updated**: November 1, 2025
