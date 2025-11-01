# GMCS System Architecture

This document provides comprehensive technical documentation for the GMCS (Generalized Modular Control System) platform architecture. It covers both high-level design and detailed implementation.

## Table of Contents

- [High-Level Overview](#high-level-overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Technical Deep-Dive](#technical-deep-dive)

---

## High-Level Overview

GMCS is a GPU-accelerated chaotic-neuro computational platform built on JAX. The system combines three main computational layers:

1. **Chaotic Oscillators** - 1024 Chua circuits generating nonlinear dynamics
2. **Wave Field** - 256x256 FDTD wave equation coupling oscillators spatially
3. **THRML Energy-Based Model** - Learning oscillator interactions through probabilistic sampling

These layers interact through a signal processing pipeline (GMCS algorithms) that transforms oscillator outputs before feeding back into the system. The entire architecture follows functional programming principles with immutable state and JIT compilation for performance.

### Key Design Decisions

**Immutable State with JAX PyTree**

All simulation state is stored in an immutable `SystemState` NamedTuple. This enables JAX's JIT compilation, automatic differentiation, and functional transformations (vmap, pmap). State updates create new state objects rather than mutating in place.

**Pre-allocated Arrays**

Arrays are pre-allocated to maximum capacity (`N_MAX=1024`, `GRID_W=GRID_H=256`). Active nodes are controlled via a boolean mask (`node_active_mask`). This avoids dynamic array resizing which breaks JIT compilation.

**Layered Architecture**

The system separates concerns into distinct layers:
- Core simulation (JIT-compiled JAX)
- High-level node graph (Python)
- REST API (FastAPI)
- Frontend (Next.js/React)

Each layer has well-defined interfaces and minimal coupling.

**Plugin-Based Extensibility**

Custom algorithms, nodes, and models can be added via plugins without modifying core code. Plugins are discovered dynamically and registered at runtime.

### Technology Stack

- **JAX** - Automatic differentiation, JIT compilation, GPU acceleration
- **THRML** - Energy-based model sampling and learning
- **FastAPI** - REST API with automatic OpenAPI docs
- **WebSocket** - Real-time binary streaming (custom header + payload)
- **Next.js 14** - React frontend with server components
- **React Flow** - Node graph visualization
- **SQLAlchemy** - Database ORM for session persistence
- **PyTorch/TensorFlow** - ML model integration
- **Pydantic** - Request/response validation

---

## System Components

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Node Graph  │  │  Visualizers │  │  Control Panels  │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/WebSocket
┌────────────────────────┼────────────────────────────────────┐
│                   API Layer (FastAPI)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Routes    │  │  WebSocket   │  │   Serializers    │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────┐
│                 Node System (Python)                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Executor  │  │   Factory    │  │     Plugins      │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────┐
│            Core Simulation (JAX - JIT Compiled)             │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Oscillators │  │   Wave PDE   │  │  GMCS Pipeline   │  │
│  │   (Chua)    │  │    (FDTD)    │  │  (21 Algos)      │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           THRML Integration (EBM Learning)          │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Key Files |
|-----------|---------------|-----------|
| **Oscillators** | Integrate Chua circuit ODEs | `src/core/integrators.py` |
| **Wave PDE** | FDTD wave equation solver | `src/core/wave_pde.py`, `src/core/wave_pde_complex.py` |
| **GMCS Pipeline** | Signal processing algorithms | `src/core/gmcs_pipeline.py` |
| **THRML** | Energy-based model sampling | `src/core/thrml_integration.py`, `src/core/ebm.py` |
| **State Management** | Immutable state container | `src/core/state.py` |
| **Simulation Loop** | Orchestrate simulation step | `src/core/simulation.py` |
| **API Endpoints** | REST API handlers | `src/api/routes.py`, `src/api/*.py` |
| **WebSocket** | Real-time data streaming | `src/api/server.py` |
| **Node System** | High-level node execution | `src/nodes/node_executor.py` |
| **ML Integration** | PyTorch/TF wrappers | `src/ml/*.py` |
| **Frontend** | User interface | `frontend/components/*.tsx` |

---

## Data Flow

### Simulation Step Flow

Each simulation step follows this sequence:

```
1. Modulation Matrix
   └─> Apply parameter modulations from audio/THRML/ML sources

2. Wave Field Sampling
   └─> Sample wave field at each oscillator position → h_i

3. GMCS Pipeline
   └─> Process h_i through algorithm chain → F_i (continuous), B_i (discrete)

4. THRML Sampling
   └─> Use B_i as bias, sample p-bits → feedback_i

5. Oscillator Integration
   └─> Integrate Chua ODEs with F_i + feedback_i → new (x,y,z)

6. Wave PDE Step
   └─> Use oscillator states as source term, integrate FDTD → new field

7. State Update
   └─> Create new immutable SystemState with updated values

8. Data Serialization
   └─> Downsample and serialize for WebSocket streaming
```

### API Request Flow

```
HTTP Request
    │
    ├─> FastAPI Route Handler
    │       │
    │       ├─> Pydantic Validation
    │       │
    │       ├─> Business Logic
    │       │   └─> Modify Global State
    │       │
    │       └─> Response Serialization
    │
    └─> HTTP Response
```

### WebSocket Data Flow

```
Simulation Loop (async)
    │
    ├─> simulation_step() → new_state
    │
    ├─> serialize_for_frontend(state, simulation_running)
    │   └─> Downsample arrays
    │   └─> Assemble binary payload
    │   └─> Pack 36-byte header (8×uint32 + 1×float)
    │       • includes simulation_running flag
    │       • includes latest simulation timestamp
    │
    └─> websocket.send_bytes()
         │
         ├─> Frontend receives binary packet
         │    └─> Parse header via DataView
         │    └─> Decompress payload if flagged
         │    └─> Update Zustand simulation store
         │
         └─> Stale monitor marks stream offline after >3 s without updates
```

When a client connects, the server immediately sends a lightweight JSON `STATUS` message on the same socket so the UI can synchronize without waiting for the first binary frame. Clients periodically send `PING` messages; the server replies with `PONG` status snapshots, which the frontend uses to refresh heartbeat timers.

---

## Technical Deep-Dive

### Core Layer

#### Oscillators

The oscillator subsystem implements Chua circuit dynamics using 4th-order Runge-Kutta integration.

**Key Files:**
- `src/core/integrators.py` - RK4 integrator for Chua equations
- `src/core/differentiable_chua.py` - Differentiable optimizer with gradient flow

**Data Structures:**

```python
oscillator_state: jnp.ndarray  # Shape (N_MAX, 3) - [x, y, z] for each oscillator
```

**Equations:**

```python
dx/dt = α(y - x - f(x))
dy/dt = x - y + z  
dz/dt = -βy

f(x) = m₁x + 0.5(m₀ - m₁)(|x + 1| - |x - 1|)
```

**Implementation:**

The integrator uses JAX's `@jax.jit` decorator for compilation and batches all oscillators using `jax.vmap`:

```python
@jax.jit
def rk4_step_chua(
    state_vec: jnp.ndarray,  # (N_MAX, 3)
    driving_F: jnp.ndarray,   # (N_MAX,)
    ebm_bias: jnp.ndarray,    # (N_MAX,)
    dt: float
) -> jnp.ndarray:
    """RK4 integration for all oscillators simultaneously."""
    # Vectorized RK4 implementation
    # ...
```

**Driving Forces:**

Oscillators are driven by three sources:
1. Wave field samples (via GMCS pipeline)
2. THRML feedback (learned coupling)
3. Audio reactivity (RMS energy)

#### Wave Field PDE

The wave field implements a 2D wave equation using Finite-Difference Time-Domain (FDTD) method.

**Key Files:**
- `src/core/wave_pde.py` - Real-valued wave equation
- `src/core/wave_pde_complex.py` - Complex-valued for photonics

**Equation:**

```
∂²P/∂t² = c² ∇²P + S(r,t)
```

Where:
- `P(r,t)` is the wave field
- `c` is wave speed (audio-reactive)
- `S(r,t)` is the source term from oscillators

**Implementation:**

```python
@jax.jit
def fdtd_step_wave(
    field_p: jnp.ndarray,      # (GRID_W, GRID_H) current field
    field_p_prev: jnp.ndarray, # (GRID_W, GRID_H) previous field  
    source_k: jnp.ndarray,     # (N_MAX,) source strengths
    positions: jnp.ndarray,    # (N_MAX, 2) positions
    c: float,                   # wave speed
    dt: float
) -> jnp.ndarray:
    """FDTD step for wave equation."""
    # Compute Laplacian using finite differences
    laplacian = (
        jnp.roll(field_p, 1, axis=0) + jnp.roll(field_p, -1, axis=0) +
        jnp.roll(field_p, 1, axis=1) + jnp.roll(field_p, -1, axis=1) -
        4 * field_p
    )
    
    # Add source term
    source_grid = compute_pde_source(source_k, positions, grid_shape)
    
    # FDTD update (leapfrog integration)
    field_p_new = 2*field_p - field_p_prev + (c*dt)**2 * (laplacian + source_grid)
    
    return field_p_new
```

**Boundary Conditions:**

The system uses periodic boundary conditions (implemented via `jnp.roll`), creating a toroidal topology.

#### GMCS Pipeline

The GMCS (Generalized Modular Control System) applies 21 signal processing algorithms in sequence.

**Key Files:**
- `src/core/gmcs_pipeline.py` - Algorithm implementations

**Algorithm Categories:**

1. **Basic (0-6):** Limiter, Compressor, Expander, Threshold, Phase Mod, Fold
2. **Audio/Signal (7-13):** Resonator, Hilbert, Rectifier, Quantizer, Slew Limiter, Cross Mod, Bipolar Fold  
3. **Photonic (14-20):** Optical Kerr, Electro-Optic, Optical Switch, FWM, Raman, Saturation, Optical Gain

**Architecture:**

Each node has an algorithm chain (up to 8 algorithms applied sequentially). The pipeline produces two outputs:
- **Continuous (F):** Used as Chua forcing
- **Discrete (B):** Used as THRML bias

```python
@jax.jit
def gmcs_pipeline_dual(
    h_inputs: jnp.ndarray,     # (N_MAX,) sampled field values
    chains: jnp.ndarray,       # (N_MAX, MAX_CHAIN_LEN) algorithm IDs
    params: jnp.ndarray,       # (N_MAX, 16) parameters
    t: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply algorithm chains to all nodes, return continuous and discrete."""
    # Vectorized algorithm application
    # ...
    return F_continuous, B_discrete
```

**Example Algorithm:**

```python
@jax.jit
def algo_limiter(h: float, A_max: float, *_) -> float:
    """Soft clipping using tanh."""
    return A_max * jnp.tanh(h / (A_max + 1e-12))
```

#### THRML Integration

THRML (Thermodynamic Reversible Markov Language) provides energy-based model sampling through block Gibbs sampling.

- Wrapper creation now runs asynchronously via `schedule_thrml_rebuild()`, which serializes rebuild requests, offloads heavy work to a background executor, and publishes results back into `SystemState`. Node add/remove endpoints simply mark the wrapper dirty and enqueue a rebuild instead of blocking API calls. Detailed derivations and coupling strategies live in `docs/thrml_docs/`.

**Key Files:**
- `src/core/thrml_integration.py` - THRML wrapper and model creation
- `src/core/ebm.py` - Energy-based model learning
- `src/core/sampler_backend.py` - Generic sampler interface

**Architecture:**

```python
class THRMLWrapper:
    """Wrapper around THRML IsingEBM model."""
    
    def __init__(self, model, n_nodes: int, beta: float):
        self.model = model          # IsingEBM instance
        self.n_nodes = n_nodes
        self.beta = beta
        
    def sample(self, 
               n_steps: int,
               bias: Optional[jnp.ndarray] = None,
               key: jax.random.PRNGKey = None) -> jnp.ndarray:
        """Sample from model using block Gibbs sampling."""
        # THRML sampling implementation
        # ...
```

**Learning Algorithm:**

GMCS uses Contrastive Divergence (CD-1) to learn oscillator coupling:

```python
def ebm_cd1_update(
    weights: jnp.ndarray,      # (N_MAX, N_MAX) current weights
    states: jnp.ndarray,       # (N_MAX, 3) oscillator states
    mask: jnp.ndarray,         # (N_MAX,) active mask
    key: jax.random.PRNGKey,
    eta: float = 0.01          # learning rate
) -> Tuple[jnp.ndarray, jax.random.PRNGKey]:
    """
    CD-1 weight update.
    
    1. Compute positive phase: v_data
    2. Sample negative phase using THRML: v_model
    3. Update: w += η(v_data v_data^T - v_model v_model^T)
    """
    # Implementation
    # ...
```

**Performance Modes:**

THRML supports three performance configurations:
- **Speed:** Fewer Gibbs steps, single chain
- **Accuracy:** More Gibbs steps, multiple chains
- **Research:** Maximum steps, extensive diagnostics

**Alignment with `docs/thrml_docs/` (2025-11):**

- The wrapper tracks the block/program concepts from the upstream THRML architecture (see `api/block_sampling.md`). Creation and rebuilds apply two-colour blocking, matching the documentation.
- Advanced capabilities described in the docs—heterogeneous node types, conditional clamping, custom factor systems—are implemented through `HeterogeneousTHRMLWrapper`, the clamping helpers, and `THRMLFactorSystem`, and are reachable via `/thrml/heterogeneous/configure`, `/thrml/clamp-nodes`, `/thrml/interactions/*`, and `/thrml/factors/*`.
- Observers are conditionally imported to mirror the THRML diagnostics API. When THRML is installed without optional extras, compatibility shims under `src/core/thrml_compat.py` are used instead.
- Docs emphasise programmes rebuilding from serialized state; the server honours this by persisting `thrml_model_data` inside `SystemState` so reconstruction can occur lazily when a packet or energy request arrives.
- Frontend `THRMLAdvancedPanels` now exercise `/thrml/heterogeneous/configure`, `/thrml/clamp-nodes`, `/thrml/sample-conditional`, and related endpoints, giving operators UI-driven access to mixed node types, clamping, higher-order interactions, and custom energy factors documented in THRML.

#### State Management

All simulation state lives in an immutable `SystemState` NamedTuple.

**Key Files:**
- `src/core/state.py` - State definition and initialization

**Structure:**

```python
class SystemState(NamedTuple):
    # Time
    key: jax.random.PRNGKey
    t: jnp.ndarray            # Current time
    dt: float
    
    # Oscillators
    oscillator_state: jnp.ndarray    # (N_MAX, 3)
    ebm_weights: jnp.ndarray         # (N_MAX, N_MAX)
    
    # Wave field
    field_p: jnp.ndarray             # (GRID_W, GRID_H)
    field_p_prev: jnp.ndarray        # (GRID_W, GRID_H)
    
    # GMCS parameters (16 parameter arrays)
    gmcs_chain: jnp.ndarray          # (N_MAX, MAX_CHAIN_LEN)
    gmcs_A_max: jnp.ndarray          # (N_MAX,)
    gmcs_R_comp: jnp.ndarray         # (N_MAX,)
    # ... 13 more parameter arrays
    
    # Topology
    node_active_mask: jnp.ndarray    # (N_MAX,)
    node_positions: jnp.ndarray      # (N_MAX, 2)
    
    # Audio control
    k_strengths: jnp.ndarray         # (N_MAX,)
    c_val: jnp.ndarray               # (1,)
    
    # THRML state (serialized for JAX compatibility)
    thrml_model_data: Optional[Dict[str, Any]]
    thrml_enabled: bool
    # ... other THRML fields
    
    # Modulation
    modulation_routes: Optional[Dict[str, Any]]
    audio_pitch: float
    audio_rms: float
    thrml_temperature: float
    # ... other modulation fields
```

**Design Rationale:**

- **Immutable:** Enables JAX transformations (grad, jit, vmap, pmap)
- **Pre-allocated:** All arrays at maximum size, JIT-friendly
- **Flat structure:** Direct field access, no nested objects
- **Serializable:** Can be pickled for checkpoints

### Node System

The node system provides a high-level graph-based interface on top of the core simulation.

**Key Files:**
- `src/nodes/node_executor.py` - Graph execution engine
- `src/nodes/node_factory.py` - Node instantiation
- `src/nodes/*.py` - Node type implementations

**Node Types:**

1. **Input Nodes:** Audio capture, MIDI input, file readers
2. **Processor Nodes:** Music analysis, crypto, effects
3. **ML Nodes:** Model inference, training, feature extraction
4. **Generator Nodes:** Pattern generators, sequencers
5. **Control Nodes:** Modulators, envelopes, LFOs
6. **Analysis Nodes:** Spectrum analysis, feature extraction
7. **Output Nodes:** File writers, network senders
8. **Visualizer Nodes:** Oscilloscopes, spectrograms, phase plots

**Implementation notes (2025-11):**

- Many factory helpers still return placeholders. The NodeGraph highlights them with a "Stub – backend execution pending" badge so operators understand which nodes are non-functional until real implementations arrive.
- The simulation bridge nodes inside `src/nodes/simulation_bridge.py` run lightweight numpy integrations that are disconnected from the authoritative `SystemState`. They remain demo shims and do not yet drive the live JAX simulation.
- `THRMLNode` now seeds and advances its own PRNG key, updates wrapper biases when provided, and calls `THRMLWrapper.sample_gibbs`. When THRML is unavailable it falls back to numpy-based sampling while keeping the interface consistent.
- Newly created nodes persist their backend slot IDs (displayed in the card header) while still using human-readable local IDs for layout. Backend CRUD operations therefore round-trip the correct slot ID and the NodeGraph stays in sync with `/nodes`.

**Node Execution:**

```python
class NodeGraph:
    def __init__(self):
        self.nodes: Dict[str, Any] = {}
        self.connections: List[Dict[str, str]] = []
        self.execution_order: List[str] = []
    
    def build_execution_order(self):
        """Topological sort using Kahn's algorithm."""
        # Build dependency graph
        # ...
        
    def execute_step(self) -> Dict[str, Dict[str, Any]]:
        """Execute all nodes in topological order."""
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            inputs = self._gather_inputs(node_id)
            outputs = node.process(**inputs)
            self.node_outputs[node_id] = outputs
        
        return self.node_outputs
```

**Data Routing:**

Connections define data flow between nodes:

```python
graph.add_connection(
    from_node="oscillator_1",
    from_output="x",
    to_node="spectrum_analyzer",
    to_input="signal"
)
```

### API Layer

The API layer exposes the system via REST and WebSocket protocols.

**Key Files:**
- `src/api/server.py` - FastAPI application and WebSocket
- `src/api/routes.py` - Request/response models
- `src/api/*_endpoints.py` - Endpoint implementations

**Endpoint Categories:**

| Category | Endpoints | Purpose |
|----------|-----------|---------|
| System | `/health`, `/status` | Health checks, system info |
| Nodes | `/node/add`, `/node/update`, `/node/{id}`, `/nodes` | Node management & topology sync |
| Algorithms | `/algorithms/list`, `/algorithms/{id}` | Algorithm catalog |
| THRML | `/thrml/performance-mode`, `/thrml/temperature` | THRML configuration |
| Modulation | `/modulation/routes`, `/modulation/presets/{name}` | Modulation matrix |
| ML | `/ml/models/list`, `/ml/models/{id}/forward` | ML model operations |
| Plugins | `/plugins/list`, `/plugins/{id}/execute` | Plugin management |
| Sessions | `/session/save`, `/session/list`, `/session/{id}/load` | Session persistence |

**Request Validation:**

All requests use Pydantic models for validation:

```python
class AddNodeRequest(BaseModel):
    position: List[float] = Field(..., min_items=2, max_items=2)
    config: Optional[Dict[str, float]] = Field(default_factory=dict)
    chain: Optional[List[int]] = Field(default_factory=list)
    initial_perturbation: float = Field(default=0.1, ge=-1.0, le=1.0)
    
    @field_validator('position')
    def validate_position(cls, v):
        x, y = v
        if not (0 <= x <= GRID_W):
            raise ValueError(f"x must be in [0, {GRID_W}]")
        if not (0 <= y <= GRID_H):
            raise ValueError(f"y must be in [0, {GRID_H}]")
        return v
```

**WebSocket Protocol:**

Real-time data streams via a custom binary envelope with an accompanying status channel:

```text
Header (36 bytes, little-endian)
    uint32 downsampled_width
    uint32 downsampled_height
    uint32 active_node_count
    uint32 capacity (N_MAX)
    uint32 flags (bit0 = compressed payload)
    uint32 payload_size_uncompressed
    uint32 payload_size_stored
    uint32 simulation_running (0/1)
    float32 simulation_time_seconds

Payload (variable)
    float32[]   field data (down_w × down_h)
    float32[]   oscillator state (active_count × 3)
    float32[]   node positions (active_count × 3)
    float32[]   active mask (active_count)
```

Server responsibilities:

- Include `simulation_running` in every packet so the UI can distinguish “connected but paused” from “actively running”.
- Emit an initial JSON `STATUS` envelope on connection to fast-sync the UI with the latest state.
- Respond to client `PING` messages with `PONG` JSON replies containing the same status fields.
- Acknowledge client `HEARTBEAT` messages and update per-socket last-seen timestamps.
- Broadcast binary packets at ~30 Hz while the simulation is running and emit lightweight `STATUS` JSON payloads every `GMCS_WS_STATUS_INTERVAL` seconds while paused.
- Drop websocket clients whose heartbeats exceed `GMCS_WS_HEARTBEAT_TIMEOUT` seconds (health monitor runs every `GMCS_WS_HEALTH_CHECK_INTERVAL`).

Client responsibilities:

- Parse the header via `DataView`, update local simulation time, and detect compression flags before reading the payload.
- Feed parsed arrays into the shared Zustand simulation store.
- Emit a JSON `HEARTBEAT` every `NEXT_PUBLIC_WS_HEARTBEAT_MS` milliseconds and close/retry if no ACK is observed within `NEXT_PUBLIC_WS_HEARTBEAT_TIMEOUT_MS`.
- Maintain a stale timer (`NEXT_PUBLIC_WS_STALE_TIMEOUT_MS`) to surface UI warnings when data has not arrived in time, even if the WebSocket remains open.
- Use exponential backoff (`NEXT_PUBLIC_WS_RECONNECT_BASE_MS` → `NEXT_PUBLIC_WS_RECONNECT_MAX_MS`) during reconnect attempts and expose the current phase (`connecting`, `reconnecting`, `stale`, etc.) to the UI.

### Frontend Architecture

The frontend is a Next.js 14 application. The node editor is a bespoke canvas implementation (`NodeGraphSimple.tsx`) rather than React Flow, which means drag/selection logic, connection routing, and state persistence are all custom to this project.

**Key Files:**
- `frontend/app/page.tsx` - Main application page
- `frontend/components/NodeGraph/` - Node graph visualization
- `frontend/components/Visualizers/` - Real-time visualizers
- `frontend/lib/websocket.ts` - WebSocket client
- `frontend/lib/stores/` - Zustand state management
- `frontend/lib/stores/backendData.ts` - Shared backend polling + status/reconnect state

**Component Structure:**

```
App (page.tsx)
├── WebSocketProvider
├── NodeGraphWorkspace (custom canvas)
│   ├── AlgorithmSelector
│   ├── PluginBrowser
│   └── SessionManager
├── VisualizerPanel
│   ├── Oscilloscope
│   ├── Spectrogram
│   ├── PhaseSpace
│   └── EnergyGraph
└── ControlPanel
    ├── THRMLControls
    ├── AudioControls
    └── ModulationMatrix
```

**WebSocket Client:**

```typescript
class GMCSWebSocket {
  private ws: WebSocket | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;

  connect(url: string) {
    this.ws = new WebSocket(url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      const store = useSimulationStore.getState();
      store.setConnected(true);
      store.setConnectionState('connected');
      store.updateLastUpdateTime(Date.now());
      this.startStaleMonitor();
      this.startHeartbeat(); // HEARTBEAT every NEXT_PUBLIC_WS_HEARTBEAT_MS
    };

    this.ws.onclose = () => {
      const store = useSimulationStore.getState();
      store.markStale();
      store.setConnectionState('reconnecting');
      this.stopStaleMonitor();
      this.stopHeartbeat();
      this.scheduleReconnect();
    };
  }
}
```

_See `frontend/lib/websocket.ts` for the full implementation with exponential backoff, toast debouncing, and status reconciliation._

The client centralizes connection, stale detection, and toast notifications so every UI surface (HUD, control panels, visualizers) consumes a single source of truth from the simulation store.

**Backend Status Poller:**

- `frontend/lib/stores/backendData.ts` maintains a single polling loop (ref-counted) that queries `/simulation/status`, `/thrml/energy`, `/sampler/benchmarks`, and `/processor/list`, updates shared Zustand state, and surfaces rate-limited/error flags consumed by HUD/SystemControls/NodeGraph.
- Polling cadence and timeout thresholds are controlled by `NEXT_PUBLIC_STATUS_POLL_MS` (default 2000 ms) and `NEXT_PUBLIC_STATUS_REQUEST_TIMEOUT_MS` (default 4000 ms).
- The store exposes `rateLimited`, `statusError`, etc., allowing the UI to display actionable messaging instead of duplicating fetch intervals per component. SystemControls and the node graph subscribe to the store rather than issuing their own intervals.

**State Management:**

Zustand stores manage global state:

```typescript
interface SimulationStore {
    state: SimulationState | null;
    nodes: Node[];
    connections: Connection[];
    addNode: (node: Node) => void;
    updateNode: (id: string, updates: Partial<Node>) => void;
}

const useSimulationStore = create<SimulationStore>((set) => ({
    state: null,
    nodes: [],
    connections: [],
    
    addNode: (node) => set((state) => ({
        nodes: [...state.nodes, node]
    })),
    
    updateNode: (id, updates) => set((state) => ({
        nodes: state.nodes.map(n => n.id === id ? { ...n, ...updates } : n)
    })),
}));
```

#### Node Graph Integration Status (2025-11 audit)

- The Control Panel now calls `POST /node/add` with the canonical `AddNodeRequest` payload and resolves deletions through `GET /nodes` + `DELETE /node/{id}`. Users receive success/failure toasts sourced from backend responses.
- `NodeGraphSimple` retains the bespoke layout but tags cards with backend IDs sourced from the `/nodes` endpoint. When data is available, the component merges backend config into the local node model; badges indicate when a node is still a stub placeholder.
- `NodeCreatorPanel` exposes an optional `onCreated` callback that forwards the backend response (including the assigned slot ID) so parent components can persist the mapping.
- `THRMLNode` within `src/nodes/simulation_bridge.py` now seeds and advances its own PRNG key, updates wrapper biases, and falls back gracefully on sampling errors. The prior `self.time` crash has been removed.
- A new `GET /nodes` endpoint surfaces active node metadata (position, chain, config). The frontend uses it for topology reconciliation, while `/nodes` plus the websocket stream keep the HUD aligned with the live simulation.

### ML Integration Layer

The ML layer provides wrappers for PyTorch, TensorFlow, and HuggingFace models.

**Key Files:**
- `src/ml/pytorch_integration.py` - PyTorch wrapper
- `src/ml/tensorflow_integration.py` - TensorFlow wrapper
- `src/ml/huggingface_integration.py` - HuggingFace wrapper
- `src/ml/concrete_models.py` - Pre-defined model architectures
- `src/ml/trainer.py` - Unified training interface
- `src/ml/hybrid_training.py` - Chaos-gradient hybrid training

**Model Wrapper Pattern:**

```python
class PyTorchModelWrapper:
    """Wrapper for PyTorch models in GMCS."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def forward(self, gmcs_state: jnp.ndarray) -> jnp.ndarray:
        """Run forward pass, converting between JAX and PyTorch."""
        # JAX → NumPy → PyTorch
        x_torch = torch.from_numpy(np.array(gmcs_state)).float().to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(x_torch)
        
        # PyTorch → NumPy → JAX
        return jnp.array(output.cpu().numpy())
```

**Model Registry:**

```python
class ModelRegistry:
    """Central registry for all ML models."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
    
    def register(self, name: str, model: Any, metadata: ModelMetadata):
        """Register a model."""
        self.models[name] = model
        self.metadata[name] = metadata
    
    def get(self, name: str) -> Any:
        """Retrieve a registered model."""
        return self.models.get(name)
```

**Implemented Models:**

The system includes 12 pre-defined model architectures:

1. **GenreClassifier** - CNN for music genre classification
2. **MusicTransformer** - Transformer for music embeddings
3. **PPOAgent** - RL policy learning
4. **ValueFunction** - RL value estimation
5. **PixelArtGAN** - Generative adversarial network
6. **CodeGenerator** - Code generation transformer
7. **LogicGateDetector** - Logic gate classifier
8. **PerformancePredictor** - Neural architecture search
9. **EfficiencyPredictor** - Solar cell efficiency
10. **CognitiveStateDecoder** - EEG/brain state classification
11. **BindingPredictor** - Molecular binding affinity
12. **MLPerformanceSelector** - Algorithm performance prediction

### Database Layer

Session and configuration persistence uses SQLAlchemy.

**Key Files:**
- `src/db/database.py` - Database connection and session management
- `src/db/models.py` - SQLAlchemy model definitions

**Schema:**

```python
class Session(Base):
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Serialized state
    state_data = Column(JSON, nullable=False)
    
    # Metadata
    n_active_nodes = Column(Integer)
    description = Column(Text)
    tags = Column(JSON)  # List of tags

class APIKey(Base):
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    key_hash = Column(String, unique=True, nullable=False)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    rate_limit = Column(Integer, default=100)
```

**Session Serialization:**

```python
def save_session(state: SystemState, name: str, description: str = ""):
    """Save simulation state to database."""
    # Serialize state to JSON-compatible dict
    state_dict = {
        't': float(state.t[0]),
        'oscillator_state': state.oscillator_state.tolist(),
        'ebm_weights': state.ebm_weights.tolist(),
        # ... all fields
    }
    
    # Create database entry
    session = Session(
        name=name,
        state_data=state_dict,
        n_active_nodes=int(jnp.sum(state.node_active_mask)),
        description=description
    )
    
    db.add(session)
    db.commit()
```

### Plugin System

Plugins extend the system without modifying core code.

**Key Files:**
- `src/plugins/plugin_base.py` - Base classes and interfaces
- `src/plugins/plugin_registry.py` - Plugin discovery and registration
- `src/plugins/examples/` - Example plugins

**Plugin Interface:**

```python
class AlgorithmPlugin(ABC):
    """Base class for algorithm plugins."""
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        pass
    
    @abstractmethod
    def compute(self, 
                input_signal: jnp.ndarray,
                params: List[float],
                **kwargs) -> jnp.ndarray:
        """Compute plugin output."""
        pass
```

**Plugin Discovery:**

```python
class PluginRegistry:
    """Discover and manage plugins."""
    
    def discover_plugins(self, directory: Path = Path("src/plugins/custom")):
        """Scan directory for plugin modules."""
        for file in directory.glob("*.py"):
            if file.name.startswith("_"):
                continue
            
            # Import module
            spec = importlib.util.spec_from_file_location(file.stem, file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, AlgorithmPlugin) and 
                    obj != AlgorithmPlugin):
                    self.register_plugin(obj)
```

**Example Plugin:**

```python
class WaveshaperPlugin(AlgorithmPlugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Waveshaper",
            version="1.0.0",
            author="GMCS Team",
            description="Nonlinear waveshaping",
            category="effects",
            parameters=[
                {"name": "drive", "type": "float", "range": [0, 10], "default": 1.0},
                {"name": "mix", "type": "float", "range": [0, 1], "default": 1.0}
            ]
        )
    
    def compute(self, input_signal, params, **kwargs):
        drive, mix = params[0], params[1]
        shaped = jnp.tanh(drive * input_signal)
        return mix * shaped + (1 - mix) * input_signal
```

### Authentication & Security

JWT-based authentication with rate limiting.

**Key Files:**
- `src/auth/jwt_auth.py` - JWT token generation and validation
- `src/auth/api_keys.py` - API key management
- `src/auth/rate_limiting.py` - Rate limiting middleware
- `src/auth/permissions.py` - Role-based access control

**JWT Authentication:**

```python
def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=1))
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

**Rate Limiting:**

```python
class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: int = 100, period: int = 60):
        self.rate = rate
        self.period = period
        self.buckets: Dict[str, TokenBucket] = {}
    
    async def check_limit(self, key: str) -> bool:
        """Check if request is within rate limit."""
        if key not in self.buckets:
            self.buckets[key] = TokenBucket(self.rate, self.period)
        
        return self.buckets[key].consume()
```

### Monitoring & Performance

Prometheus metrics and performance profiling.

**Key Files:**
- `src/monitoring/metrics.py` - Metric definitions
- `src/monitoring/prometheus_exporter.py` - Prometheus integration
- `src/performance/profiler.py` - Performance profiler
- `src/performance/bottleneck_analyzer.py` - Bottleneck detection

**Metrics Collection:**

```python
# Define metrics
simulation_step_duration = Histogram(
    'gmcs_simulation_step_duration_seconds',
    'Duration of simulation step'
)

node_execution_duration = Histogram(
    'gmcs_node_execution_duration_seconds',
    'Node execution time',
    ['node_type']
)

# Instrument code
@simulation_step_duration.time()
def simulation_step(state, **kwargs):
    # Simulation logic
    pass
```

**Performance Profiling:**

```python
class Profiler:
    """Profile code execution."""
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_memory = get_memory_usage()
        return self
    
    def __exit__(self, *args):
        self.duration = time.perf_counter() - self.start_time
        self.memory_delta = get_memory_usage() - self.start_memory
        
        # Log results
        logger.info(f"Duration: {self.duration:.3f}s, Memory: {self.memory_delta}MB")
```

### Configuration System

Centralized configuration management.

**Key Files:**
- `src/config/config_schema.py` - Pydantic configuration schemas
- `src/config/config_manager.py` - Configuration loading
- `src/config/defaults.py` - Default values
- `src/config/thrml_config.py` - THRML-specific configuration

**Configuration Schema:**

```python
class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    reload: bool = False

class THRMLConfig(BaseModel):
    performance_mode: str = "accuracy"
    temperature: float = 1.0
    gibbs_steps: int = 10
    cd_k: int = 1

class GMCSConfig(BaseModel):
    server: ServerConfig
    thrml: THRMLConfig
    # ... other sections
```

**Loading Configuration:**

```python
def load_config(config_file: Optional[Path] = None) -> GMCSConfig:
    """Load configuration from file or use defaults."""
    if config_file and config_file.exists():
        with open(config_file) as f:
            data = yaml.safe_load(f)
        return GMCSConfig(**data)
    else:
        return GMCSConfig()  # Use defaults
```

**Environment Overrides:**

- `GMCS_RATE_LIMIT_ENABLED` (default `false`) toggles HTTP rate limiting middleware; use `true` in production to enforce limits.
- `GMCS_RATE_LIMIT_RPM` (default `600`) and `GMCS_RATE_LIMIT_WINDOW` (seconds, default `60`) tune the per-IP quota without code changes.
- Frontend polling cadence can be adjusted via `NEXT_PUBLIC_STATUS_POLL_MS` and `NEXT_PUBLIC_STATUS_REQUEST_TIMEOUT_MS`, allowing dev environments to relax load while keeping production responsive.

### Error Handling & Recovery

Comprehensive error handling with recovery strategies.

**Key Files:**
- `src/recovery/error_handler.py` - Global error handler
- `src/recovery/recovery_strategies.py` - Recovery logic

**Error Hierarchy:**

```python
class GMCSException(Exception):
    """Base exception for GMCS."""
    pass

class SimulationError(GMCSException):
    """Error during simulation step."""
    pass

class NodeExecutionError(GMCSException):
    """Error executing node."""
    pass

class THRMLError(GMCSException):
    """Error in THRML integration."""
    pass
```

**Recovery Strategy:**

```python
async def simulation_loop_with_recovery():
    """Simulation loop with automatic recovery."""
    error_count = 0
    max_errors = 10
    
    while True:
        try:
            new_state, thrml_wrapper = simulation_step(state, ...)
            state = new_state
            error_count = 0  # Reset on success
            
        except SimulationError as e:
            error_count += 1
            logger.error(f"Simulation error: {e}")
            
            if error_count > max_errors:
                raise
            
            # Recovery: Reset to last good state
            state = restore_checkpoint()
            thrml_wrapper = reconstruct_thrml_wrapper(state)
```

---

## Design Patterns

### Functional Core, Imperative Shell

Core simulation functions are pure and JIT-compiled. Side effects (I/O, state mutation) happen at the boundaries (API layer, database).

### Repository Pattern

Database access abstracted behind repository interfaces:

```python
class SessionRepository:
    def get(self, id: int) -> Session:
        pass
    
    def save(self, session: Session) -> int:
        pass
    
    def list(self, filters: Dict[str, Any]) -> List[Session]:
        pass
```

### Factory Pattern

Nodes created via factory for consistent initialization:

```python
class NodeFactory:
    @staticmethod
    def create(node_type: str, config: Dict[str, Any]) -> Node:
        if node_type == "oscillator":
            return OscillatorNode(config)
        elif node_type == "ml":
            return MLNode(config)
        # ...
```

### Observer Pattern

Components subscribe to simulation events:

```python
class SimulationObserver(ABC):
    @abstractmethod
    def on_step_complete(self, state: SystemState) -> None:
        pass

class MetricsObserver(SimulationObserver):
    def on_step_complete(self, state):
        # Record metrics
        pass
```

---

## Performance Considerations

### JIT Compilation

Core functions use `@jax.jit` for compilation. Avoid:
- Python control flow (use `jax.lax.cond`, `jax.lax.scan`)
- Dynamic array shapes
- Non-JAX libraries in jitted functions

### Memory Management

- Pre-allocated arrays avoid allocations in hot loops
- Use `jax.device_put` to explicitly manage GPU transfers
- Clear unused arrays with `del` to free GPU memory

### Parallelization

- `jax.vmap` for batch operations (oscillators, nodes)
- `jax.pmap` for multi-GPU parallelism
- Async/await for I/O operations

### Profiling

```bash
# JAX profiling
JAX_ENABLE_PROFILING=1 python src/main.py

# Python profiling
python -m cProfile -o profile.stats src/main.py
python -m pstats profile.stats
```

### Validation (2025-11)

- Verified THRML dependency availability (`pip show thrml` → 0.1.3).
- `python -m pytest tests/test_thrml_integration.py` &mdash; 40 passed / 3 skipped in ~50&nbsp;s.

---

## Extension Points

### Adding a New Algorithm

1. Define algorithm function in `src/core/gmcs_pipeline.py`
2. Add algorithm ID constant
3. Add to algorithm dispatcher
4. Update frontend algorithm list

### Adding a New Node Type

1. Create node class in `src/nodes/`
2. Implement required interface methods
3. Register in `NodeFactory`
4. Add frontend component

### Adding a New ML Model

1. Define model architecture in `src/ml/concrete_models.py`
2. Register in model registry
3. Add training script in `examples/ml/training/`
4. Create frontend node component

### Adding a New Visualizer

1. Create visualizer component in `frontend/components/Visualizers/`
2. Connect to WebSocket data stream
3. Add to visualizer panel

---

## Further Reading

- JAX Documentation: https://jax.readthedocs.io/
- THRML Documentation: `docs/thrml_docs/`
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Next.js Documentation: https://nextjs.org/docs

---

**Last Updated:** November 1, 2025  
**Version:** 0.1

