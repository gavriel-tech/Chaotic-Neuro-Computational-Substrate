# Repository Structure

## Overview

This repository is Gavriel Technologies' fork of THRML that has been extended to become the GMCS (Generalized Modular Control System) - Universal Chaotic-Neuro Computational Substrate Platform v0.1.

**Repository**: https://github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate

---

## Complete Directory Structure

```
Chaotic-Neuro-Computational-Substrate/
│
├── 📁 src/                          # GMCS Core Implementation
│   ├── core/                        # Core simulation modules
│   │   ├── state.py                 # SystemState PyTree definition
│   │   ├── integrators.py           # RK4 Chua oscillator solver
│   │   ├── wave_pde.py              # 2D FDTD wave equation (real)
│   │   ├── wave_pde_complex.py      # Complex-valued wave (photonics)
│   │   ├── gmcs_pipeline.py         # 21 signal processing algorithms
│   │   ├── ebm.py                   # EBM learning with THRML
│   │   ├── simulation.py            # Main simulation step
│   │   ├── thrml_integration.py     # THRML wrapper (994 lines)
│   │   ├── thrml_higher_order.py    # 3-way, 4-way interactions
│   │   ├── thrml_schedules.py       # Advanced sampling schedules
│   │   ├── modulation_matrix.py     # Universal modulation system
│   │   └── multi_gpu.py             # Multi-GPU support
│   │
│   ├── api/                         # REST API & WebSocket
│   │   ├── server.py                # FastAPI application (80+ endpoints)
│   │   ├── routes.py                # Node management endpoints
│   │   ├── serializers.py           # Binary msgpack encoding
│   │   ├── node_configs.py          # Configuration management
│   │   ├── external_models.py       # External GPU/model connections
│   │   ├── ml_endpoints.py          # ML integration API
│   │   └── plugin_endpoints.py      # Plugin system API
│   │
│   ├── audio/                       # Real-time audio processing
│   │   ├── audio_thread.py          # Audio capture loop
│   │   └── pitch_utils.py           # YIN pitch detection
│   │
│   ├── ml/                          # ML Framework Integration
│   │   ├── pytorch_integration.py   # PyTorch wrapper
│   │   ├── tensorflow_integration.py # TensorFlow wrapper
│   │   ├── huggingface_integration.py # HuggingFace wrapper
│   │   └── model_registry.py        # Model management
│   │
│   ├── plugins/                     # Plugin System
│   │   ├── plugin_base.py           # Base classes
│   │   ├── plugin_registry.py       # Plugin manager
│   │   └── examples/                # Example plugins
│   │       ├── waveshaper_plugin.py
│   │       └── pattern_detector_plugin.py
│   │
│   ├── config/                      # Configuration
│   │   ├── defaults.py              # Default parameters
│   │   └── thrml_config.py          # THRML performance modes
│   │
│   ├── tools/                       # Utilities
│   │   └── perf.py                  # Benchmarking suite
│   │
│   └── main.py                      # CLI entry point
│
├── 📁 frontend/                     # Next.js Frontend (TypeScript + Tailwind)
│   ├── app/                         # Next.js App Router
│   │   ├── layout.tsx               # Root layout
│   │   ├── page.tsx                 # Main page
│   │   └── globals.css              # Global styles
│   │
│   ├── components/                  # React Components
│   │   ├── Scene/                   # 3D Visualization
│   │   │   ├── MainScene.tsx        # Three.js scene manager
│   │   │   ├── FieldVisualization.tsx # Wave field renderer
│   │   │   ├── OscillatorParticles.tsx # Oscillator particles
│   │   │   └── PostProcessing.tsx   # Bloom, chromatic aberration
│   │   │
│   │   ├── Controls/                # Control Panels
│   │   │   ├── ControlPanel.tsx     # Main control interface
│   │   │   └── THRMLControls.tsx    # THRML-specific controls
│   │   │
│   │   ├── UI/                      # UI Components
│   │   │   ├── LoadingScreen.tsx    # Loading state
│   │   │   ├── ErrorToast.tsx       # Error notifications
│   │   │   ├── Tooltip.tsx          # Tooltips
│   │   │   └── KeyboardShortcutsPanel.tsx # Keyboard help
│   │   │
│   │   ├── HUD/                     # Heads-Up Display
│   │   │   └── HUD.tsx              # Real-time stats
│   │   │
│   │   ├── providers/               # Context Providers
│   │   │   ├── ColorSchemeProvider.tsx # Color management
│   │   │   └── WebSocketBridge.tsx  # WebSocket connection
│   │   │
│   │   ├── AlgorithmSelector.tsx    # Algorithm browser
│   │   ├── ModulationMatrixEditor.tsx # Modulation routing
│   │   ├── PluginBrowser.tsx        # Plugin management
│   │   ├── ObserverDashboard.tsx    # THRML diagnostics
│   │   └── SessionManager.tsx       # Session save/load
│   │
│   ├── lib/                         # Libraries & Utilities
│   │   ├── stores/                  # Zustand state management
│   │   │   ├── simulation.ts        # Simulation state
│   │   │   ├── controls.ts          # Control state
│   │   │   └── colorScheme.ts       # Color state
│   │   │
│   │   ├── hooks/                   # React hooks
│   │   │   └── useKeyboardShortcuts.ts
│   │   │
│   │   ├── websocket.ts             # WebSocket client
│   │   └── color-schemes.ts         # Color palettes
│   │
│   ├── shaders/                     # WebGL Shaders
│   │   └── field.ts                 # Field visualization shader
│   │
│   ├── types/                       # TypeScript types
│   ├── public/                      # Static assets
│   ├── package.json                 # Node dependencies
│   ├── tsconfig.json                # TypeScript config
│   ├── tailwind.config.ts           # Tailwind config
│   ├── next.config.mjs              # Next.js config
│   └── postcss.config.cjs           # PostCSS config
│
├── 📁 tests/                        # Test Suite (100+ tests)
│   ├── test_state.py                # State management tests
│   ├── test_integrators.py          # Oscillator integration tests
│   ├── test_wave_stability.py       # Wave PDE stability tests
│   ├── test_gmcs_pipeline.py        # GMCS algorithm tests
│   ├── test_gmcs_extended.py        # Extended algorithm tests
│   ├── test_ebm.py                  # EBM learning tests
│   ├── test_thrml_integration.py    # THRML integration tests (50+ cases)
│   ├── test_integration.py          # End-to-end integration tests
│   ├── test_serialization.py        # Binary serialization tests
│   ├── test_routes.py               # API endpoint tests
│   ├── test_api_integration.py      # Complete API tests
│   ├── test_ml_integration.py       # ML framework tests
│   ├── test_plugin_system.py        # Plugin system tests
│   ├── test_pitch.py                # Audio pitch detection tests
│   └── test_stability.py            # Long-run stability tests
│
├── 📁 examples/                     # Example Scripts
│   ├── quickstart_demo.py           # Quick start example
│   └── thrml_integration_demo.py    # THRML usage examples
│
├── 📁 docs/                         # Documentation
│   ├── architecture.md              # System architecture (990 lines)
│   ├── ALGORITHM_REFERENCE.md       # All 21 algorithms detailed
│   ├── API_REFERENCE.md             # Complete API docs
│   ├── thrml_integration.md         # THRML integration guide (260 lines)
│   ├── THRML_COMPLETE_INTEGRATION.md # THRML verification
│   ├── PLUGIN_DEVELOPMENT.md        # Plugin development guide
│   └── DEPLOYMENT.md                # Deployment guide
│
├── 📁 docker/                       # Docker Configuration
│   └── (Docker-related files)
│
├── 📁 thrml-main/                   # Original THRML Library (Fork)
│   └── thrml-main/                  # THRML source code
│       ├── thrml/                   # THRML Python package
│       ├── docs/                    # THRML documentation
│       ├── examples/                # THRML examples
│       └── tests/                   # THRML tests
│
├── 📄 README.md                     # Main project README (comprehensive)
├── 📄 CONTRIBUTING.md               # Contribution guidelines
├── 📄 LICENSE                       # MIT License
├── 📄 SECURITY.md                   # Security policy
├── 📄 PROJECT_INFO.md               # Project details
├── 📄 REPOSITORY_STRUCTURE.md       # This file
├── 📄 LAUNCH_CHECKLIST.md           # Pre-launch verification
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 pyproject.toml                # Python project config
├── 📄 Dockerfile                    # Docker image definition
├── 📄 docker-compose.yml            # Docker Compose config
│
└── 📄 .gitignore                    # Git ignore rules
```

---

## Key Components

### Core Modules (src/core/)

| Module | Lines | Purpose |
|--------|-------|---------|
| `thrml_integration.py` | 994 | Complete THRML wrapper with all features |
| `gmcs_pipeline.py` | 600+ | 21 signal processing algorithms |
| `simulation.py` | 405 | Main simulation orchestration |
| `state.py` | 267 | SystemState PyTree definition |
| `ebm.py` | 347 | EBM learning with THRML |
| `wave_pde.py` | 300+ | 2D wave equation FDTD |
| `modulation_matrix.py` | 400+ | Universal modulation system |

### API Layer (src/api/)

| Module | Endpoints | Purpose |
|--------|-----------|---------|
| `server.py` | 80+ | Main FastAPI application |
| `routes.py` | 10+ | Node management |
| `node_configs.py` | 5+ | Configuration management |
| `external_models.py` | 8+ | External GPU/model API |
| `ml_endpoints.py` | 10+ | ML integration API |
| `plugin_endpoints.py` | 6+ | Plugin system API |

### Frontend (frontend/)

| Component | Purpose |
|-----------|---------|
| `MainScene.tsx` | Three.js 3D visualization |
| `AlgorithmSelector.tsx` | Browse 21 algorithms |
| `ModulationMatrixEditor.tsx` | Visual modulation routing |
| `ObserverDashboard.tsx` | Real-time THRML monitoring |
| `PluginBrowser.tsx` | Plugin management UI |
| `SessionManager.tsx` | Save/load sessions |
| `THRMLControls.tsx` | THRML parameter control |

### Tests (tests/)

| Test Suite | Cases | Coverage |
|------------|-------|----------|
| THRML Integration | 50+ | Complete |
| ML Integration | 20+ | Complete |
| Plugin System | 15+ | Complete |
| API Integration | 30+ | Complete |
| Core Modules | 40+ | Complete |

---

## Relationship to THRML

### Original THRML (Extropic AI)
- **Repository**: https://github.com/extropic-ai/thrml
- **Purpose**: JAX library for probabilistic graphical models
- **Features**: Block Gibbs sampling, EBM utilities, hardware prototyping

### This Fork (Gavriel Technologies)
- **Forked from**: extropic-ai/thrml
- **Extended with**: GMCS platform (chaotic oscillators, wave field, 21 algorithms)
- **Integration**: THRML fully integrated as EBM backend
- **Location**: `thrml-main/` contains original THRML, `src/` contains GMCS additions

---

## File Statistics

### Python Code
- **Total Lines**: ~15,000+
- **Core Modules**: ~5,000 lines
- **API Layer**: ~3,000 lines
- **ML Integration**: ~1,500 lines
- **Plugin System**: ~800 lines
- **Tests**: ~4,000 lines

### Frontend Code
- **Total Lines**: ~8,000+
- **Components**: ~5,000 lines
- **Stores & Hooks**: ~1,000 lines
- **Shaders**: ~500 lines
- **Config**: ~500 lines

### Documentation
- **Total Lines**: ~5,000+
- **Architecture**: 990 lines
- **API Reference**: 800+ lines
- **Algorithm Reference**: 600+ lines
- **THRML Docs**: 1,000+ lines

---

## Technology Stack

### Backend
- **Language**: Python 3.10+
- **Framework**: FastAPI
- **Computation**: JAX (GPU-accelerated)
- **ML**: THRML, PyTorch, TensorFlow, HuggingFace
- **Audio**: sounddevice, librosa
- **WebSocket**: FastAPI WebSocket with msgpack

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **3D**: Three.js + React Three Fiber
- **State**: Zustand
- **WebSocket**: Native WebSocket API

### DevOps
- **Containerization**: Docker, Docker Compose
- **Testing**: pytest, pytest-cov
- **Linting**: ruff, black, mypy
- **CI/CD**: GitHub Actions (planned)

---

## Dependencies

### Python (requirements.txt)
```
jax[cuda12]>=0.4.20
jaxlib>=0.4.20
thrml>=0.1.3
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
numpy>=1.24.0
sounddevice>=0.4.6
librosa>=0.10.0
msgpack>=1.0.7
torch>=2.0.0 (optional)
tensorflow>=2.13.0 (optional)
transformers>=4.30.0 (optional)
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

### Frontend (package.json)
```json
{
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.2.0",
    "three": "^0.158.0",
    "@react-three/fiber": "^8.15.0",
    "zustand": "^4.4.0",
    "tailwindcss": "^3.3.0"
  }
}
```

---

## Development Workflow

### 1. Setup
```bash
# Clone repository
git clone https://github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate
cd Chaotic-Neuro-Computational-Substrate

# Backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### 2. Development
```bash
# Backend (terminal 1)
python src/main.py --host 0.0.0.0 --port 8000 --reload

# Frontend (terminal 2)
cd frontend
npm run dev
```

### 3. Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_thrml_integration.py -v
```

### 4. Linting
```bash
# Format code
black src/
ruff check src/ --fix

# Type checking
mypy src/
```

---

## Production Deployment

### Docker
```bash
# Build
docker build -t gmcs:latest .

# Run
docker-compose up -d

# With GPU
docker-compose up -d --gpus all
```

### Manual
```bash
# Backend
gunicorn src.api.server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000

# Frontend
cd frontend
npm run build
npm start
```

---

## Repository Metrics

| Metric | Value |
|--------|-------|
| **Total Files** | 200+ |
| **Python Files** | 80+ |
| **TypeScript Files** | 50+ |
| **Test Files** | 15+ |
| **Documentation Files** | 10+ |
| **Total Lines of Code** | 30,000+ |
| **Test Coverage** | 85%+ |
| **API Endpoints** | 80+ |
| **Algorithms** | 21 |
| **Components** | 30+ |

---

## License

MIT License - Copyright © 2025 Gavriel Technologies

This repository is a fork of [THRML](https://github.com/extropic-ai/thrml) by Extropic AI, which is also MIT licensed.

---

## Contact

- **Organization**: Gavriel Technologies
- **Website**: https://gavriel.tech
- **GitHub**: https://github.com/gavriel-tech
- **Twitter**: [@GavrielTech](https://twitter.com/GavrielTech)

---

**Last Updated**: October 30, 2025  
**Version**: 0.1  
**Status**: In Development
