"""
FastAPI server with simulation loop and WebSocket support.

This module creates the main server application, manages global simulation state,
runs the async simulation loop, and handles WebSocket connections for real-time
visualization.
"""

import asyncio
import json
import os
from typing import Set, Dict, List, Optional, Any
from contextlib import asynccontextmanager
from collections import defaultdict

import jax
import jax.numpy as jnp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.core.state import initialize_system_state, SystemState, N_MAX, MAX_CHAIN_LEN
from src.core.simulation import (
    simulation_step,
    simulation_step_with_thrml_learning,
    add_node_to_state,
    remove_node_from_state,
)
from src.core.ebm import (
    ebm_update_with_thrml,
    compute_ebm_energy_thrml,
    get_last_thrml_sample,
    get_thrml_sample_history,
    update_thrml_sample,
    get_last_thrml_feedback_norm
)
from src.core.thrml_integration import (
    THRMLWrapper,
    create_thrml_model,
    reconstruct_thrml_wrapper
)
from src.core.gmcs_pipeline import ALGO_NOP
from src.core.performance_config import (
    get_perf_config,
    apply_optimal_settings,
    print_device_info
)
from src.config.thrml_config import (
    PerformanceMode,
    get_performance_config,
    PERFORMANCE_PRESETS,
    list_performance_modes
)
from src.api.serializers import serialize_for_frontend, _auto_downsample_resolution
from src.api.routes import (
    AddNodeRequest,
    UpdateNodeRequest,
    NodeResponse,
    SystemStatusResponse,
    NodeInfoResponse,
    NodeListResponse,
    NodeListItem,
    merge_config_with_defaults,
    pad_chain_to_max_length,
    validate_parameter_ranges,
)
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np


# ============================================================================
# Global State
# ============================================================================

# Simulation state (mutable, managed by simulation loop)
sim_state: SystemState = None

# THRML wrapper (mutable, managed by simulation loop)
thrml_wrapper: THRMLWrapper = None

# Multi-instance processor manager
processor_instances: Dict[str, Dict[str, Any]] = {}  # {node_id: {type: 'thrml', instance: wrapper, config: {...}}}

# Mapping of config names to SystemState attributes for node serialization
NODE_CONFIG_ATTRIBUTES: Dict[str, str] = {
    "A_max": "gmcs_A_max",
    "R_comp": "gmcs_R_comp",
    "T_comp": "gmcs_T_comp",
    "R_exp": "gmcs_R_exp",
    "T_exp": "gmcs_T_exp",
    "Phi": "gmcs_Phi",
    "omega": "gmcs_omega",
    "gamma": "gmcs_gamma",
    "beta": "gmcs_beta",
    "f0": "gmcs_f0",
    "Q": "gmcs_Q",
    "levels": "gmcs_levels",
    "rate_limit": "gmcs_rate_limit",
    "n2": "gmcs_n2",
    "V": "gmcs_V",
    "V_pi": "gmcs_V_pi",
    "k_strength": "k_strengths",
}


def _ensure_simulation_initialized() -> None:
    if sim_state is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")


def _assert_node_id(node_id: int) -> None:
    if node_id < 0 or node_id >= N_MAX:
        raise ValueError(f"Node ID {node_id} must be in range [0, {N_MAX})")


def _collect_node_config(state: SystemState, node_id: int) -> Dict[str, float]:
    config: Dict[str, float] = {}
    for name, attr in NODE_CONFIG_ATTRIBUTES.items():
        array = getattr(state, attr, None)
        if array is None:
            continue
        config[name] = float(np.asarray(array[node_id]))
    return config


def _serialize_node(state: SystemState, node_id: int) -> Dict[str, Any]:
    config = _collect_node_config(state, node_id)
    position = np.asarray(state.node_positions[node_id]).astype(float).tolist()
    oscillator_state = np.asarray(state.oscillator_state[node_id]).astype(float).tolist()
    chain_values = np.asarray(state.gmcs_chain[node_id]).astype(int).tolist()
    if len(chain_values) < MAX_CHAIN_LEN:
        chain_values.extend([int(ALGO_NOP)] * (MAX_CHAIN_LEN - len(chain_values)))
    else:
        chain_values = chain_values[:MAX_CHAIN_LEN]

    active_flag = bool(np.asarray(state.node_active_mask[node_id]) > 0.5)

    return {
        "node_id": int(node_id),
        "active": active_flag,
        "position": position,
        "oscillator_state": oscillator_state,
        "config": config,
        "chain": chain_values,
    }


def _list_active_node_ids(state: SystemState) -> List[int]:
    mask = np.asarray(state.node_active_mask)
    return [int(idx) for idx in np.nonzero(mask > 0.5)[0].tolist()]

# Audio parameters (updated by audio thread)
audio_params: Dict[str, float] = {
    'rms': 0.0,
    'pitch': 440.0
}

# WebSocket clients
websocket_clients: Set[WebSocket] = set()
websocket_last_seen: Dict[WebSocket, float] = {}
websocket_health_task: Optional[asyncio.Task] = None

# THRML rebuild coordination
thrml_rebuild_lock: Optional[asyncio.Lock] = None
thrml_rebuild_task: Optional[asyncio.Task] = None
thrml_rebuild_requested: bool = False

# Configuration helpers
def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default

def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default

# Heartbeat / status configuration (seconds)
HEARTBEAT_TIMEOUT_SECONDS = _get_env_float("GMCS_WS_HEARTBEAT_TIMEOUT", 30.0)
HEALTH_CHECK_INTERVAL_SECONDS = _get_env_float("GMCS_WS_HEALTH_CHECK_INTERVAL", 5.0)
PAUSED_STATUS_INTERVAL_SECONDS = _get_env_float("GMCS_WS_STATUS_INTERVAL", 1.0)

# Simulation control flags
simulation_running: bool = False  # Don't start automatically - wait for user to click Start
simulation_task: asyncio.Task = None

# Rate limiting state
rate_limit_state = defaultdict(lambda: {'count': 0, 'reset_time': datetime.now()})
RATE_LIMIT_REQUESTS_PER_MINUTE = _get_env_int("GMCS_RATE_LIMIT_RPM", 600)
RATE_LIMIT_WINDOW_SECONDS = _get_env_float("GMCS_RATE_LIMIT_WINDOW", 60.0)
RATE_LIMIT_ENABLED = _get_env_bool("GMCS_RATE_LIMIT_ENABLED", False)


# ============================================================================
# Pydantic Models for THRML Endpoints
# ============================================================================

class THRMLPerformanceModeRequest(BaseModel):
    """Request to change THRML performance mode."""
    mode: Literal["speed", "accuracy", "research"]

class THRMLTemperatureRequest(BaseModel):
    """Request to change THRML sampling temperature."""
    temperature: float = Field(gt=0.0, le=10.0, description="Sampling temperature")

class THRMLEnergyResponse(BaseModel):
    """Response containing current THRML energy."""
    energy: float
    timestamp: float

class THRMLStatusResponse(BaseModel):
    """Response containing THRML status."""
    enabled: bool
    performance_mode: str
    temperature: float
    gibbs_steps: int
    cd_k: int
    update_freq: int
    n_nodes: int


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    global sim_state, thrml_wrapper, simulation_task, websocket_health_task
    global thrml_rebuild_lock, thrml_rebuild_task, thrml_rebuild_requested
    
    print("Initializing GMCS simulation with THRML...")
    
    # Initialise locks / tasks
    thrml_rebuild_lock = asyncio.Lock()
    thrml_rebuild_task = None
    thrml_rebuild_requested = False
    
    # Initialize JAX state
    key = jax.random.PRNGKey(42)
    sim_state = initialize_system_state(key, dt=0.01)
    # Note: Don't use jax.device_put on SystemState as it contains non-JAX types (strings)
    
    print(f"Simulation state initialized (N_MAX={N_MAX})")
    
    # Initialize THRML wrapper (will be created when nodes are added)
    thrml_wrapper = None
    print("THRML wrapper ready for initialization")
    
    # JIT warmup (components still jitted, but simulation_step is not)
    print("Warming up JIT compilation...")
    for _ in range(5):
        sim_state, thrml_wrapper = simulation_step(sim_state, enable_ebm_feedback=False, thrml_wrapper=thrml_wrapper)
    sim_state.oscillator_state.block_until_ready()
    print("JIT warmup complete")
    
    # Start simulation loop
    print("Starting simulation loop...")
    simulation_task = asyncio.create_task(simulation_loop())
    websocket_health_task = asyncio.create_task(websocket_health_monitor())
    
    print("GMCS server with THRML ready!")
    
    yield
    
    # Shutdown
    print("\nShutting down GMCS server...")
    global simulation_running
    simulation_running = False
    if simulation_task:
        simulation_task.cancel()
        try:
            await simulation_task
        except asyncio.CancelledError:
            pass
    if websocket_health_task:
        websocket_health_task.cancel()
        try:
            await websocket_health_task
        except asyncio.CancelledError:
            pass
    if thrml_rebuild_task:
        thrml_rebuild_task.cancel()
        try:
            await thrml_rebuild_task
        except asyncio.CancelledError:
            pass
    print("Goodbye!")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="GMCS API",
    description="Generative Chaotic-Neuro Cymatics System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (if they exist)
try:
    app.mount("/static", StaticFiles(directory="src/frontend"), name="static")
except RuntimeError:
    pass  # Directory doesn't exist yet


# ============================================================================
# Simulation Loop
# ============================================================================

async def websocket_health_monitor():
    """Close websocket connections that stop sending heartbeats."""
    try:
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL_SECONDS)
            now = asyncio.get_running_loop().time()
            stale_clients: List[WebSocket] = []

            for client in list(websocket_clients):
                last_seen = websocket_last_seen.get(client)
                if last_seen is None or now - last_seen > HEARTBEAT_TIMEOUT_SECONDS:
                    stale_clients.append(client)

            for client in stale_clients:
                websocket_last_seen.pop(client, None)
                websocket_clients.discard(client)
                try:
                    await client.close(code=status.WS_1008_POLICY_VIOLATION, reason="Heartbeat timeout")
                except Exception:
                    # Ignore errors while closing stale sockets
                    pass
    except asyncio.CancelledError:
        # Task cancelled during shutdown
        pass


async def broadcast_status_update():
    """Send a lightweight status JSON packet to all connected clients."""
    if not websocket_clients:
        return

    if sim_state is not None:
        active_nodes = int(jnp.sum(sim_state.node_active_mask))
        sim_time = float(sim_state.t[0])
    else:
        active_nodes = 0
        sim_time = 0.0

    payload = {
        "type": "STATUS",
        "simulation_running": simulation_running,
        "active_nodes": active_nodes,
        "sim_time": sim_time,
        "timestamp": asyncio.get_running_loop().time(),
    }

    disconnected: Set[WebSocket] = set()
    for client in list(websocket_clients):
        try:
            await client.send_json(payload)
        except Exception:
            disconnected.add(client)

    if disconnected:
        for client in disconnected:
            websocket_clients.discard(client)
            websocket_last_seen.pop(client, None)


async def simulation_loop():
    """
    Main simulation loop running at high internal rate with THRML.
    
    Responsibilities:
    - Poll audio parameters
    - Step simulation (with THRML)
    - Run THRML learning (every N steps based on state.thrml_update_freq)
    - Broadcast visualization packets (to WebSocket clients)
    """
    global sim_state, thrml_wrapper, audio_params, simulation_running
    
    sim_step_count = 0
    viz_step_count = 0
    thrml_step_count = 0
    
    # Timing parameters
    INTERNAL_RATE = 0.01        # 100 Hz simulation (10ms per step)
    VIZ_RATE = 0.033            # ~30 Hz visualization (~33ms)
    
    sim_timer = 0.0
    viz_timer = 0.0
    status_timer = 0.0
    was_running = False
    last_loop_time = asyncio.get_event_loop().time()
    
    try:
        while True:
            if not simulation_running:
                if was_running:
                    # Reset timers so they don't accumulate during pause
                    sim_timer = 0.0
                    viz_timer = 0.0
                    status_timer = 0.0
                    was_running = False
                    print("[SIM] Simulation paused")
                await asyncio.sleep(0.05)
                status_timer += 0.05
                if status_timer >= PAUSED_STATUS_INTERVAL_SECONDS:
                    await broadcast_status_update()
                    status_timer = 0.0
                last_loop_time = asyncio.get_event_loop().time()
                continue

            if not was_running:
                was_running = True
                sim_timer = 0.0
                viz_timer = 0.0
                status_timer = 0.0
                last_loop_time = asyncio.get_event_loop().time()
                print("[SIM] Simulation resumed")

            loop_start = asyncio.get_event_loop().time()
            delta_time = max(0.0, loop_start - last_loop_time)

            # Yield control to allow other tasks to run promptly
            await asyncio.sleep(0)

            # Reconstruct THRML wrapper if needed
            if thrml_wrapper is None and sim_state.thrml_model_data is not None:
                print("[THRML] Reconstructing THRML wrapper from state...")
                thrml_wrapper = reconstruct_thrml_wrapper(sim_state.thrml_model_data)
                print(f"[THRML] Wrapper reconstructed with {thrml_wrapper.n_nodes} nodes")

            # 1. Get audio control parameters and update state
            c_val = map_pitch_to_c(audio_params['pitch'])
            k_val = map_rms_to_k(audio_params['rms'])

            # Update state with audio params
            sim_state = sim_state._replace(
                c_val=jnp.array([c_val]),
                # Broadcast k_val to all active nodes
                k_strengths=sim_state.k_strengths * 0 + k_val
            )

            # 2. Step simulation with THRML
            sim_state, thrml_wrapper = simulation_step(
                sim_state,
                enable_ebm_feedback=True,
                thrml_wrapper=thrml_wrapper
            )
            sim_step_count += 1
            sim_timer += delta_time
            viz_timer += delta_time
            status_timer += delta_time

            # Log THRML activity every 100 steps
            if sim_step_count % 100 == 0:
                if thrml_wrapper is not None:
                    print(f"[THRML] Step {sim_step_count}: THRML active with {thrml_wrapper.n_nodes} nodes, enabled={sim_state.thrml_enabled}")
                else:
                    print(f"[THRML] Step {sim_step_count}: No THRML wrapper, enabled={sim_state.thrml_enabled}")

            # 3. THRML learning update (based on state.thrml_update_freq)
            if sim_step_count % sim_state.thrml_update_freq == 0 and sim_step_count > 0 and thrml_wrapper is not None:
                # Get learning rate from performance config
                config = get_performance_config(sim_state.thrml_performance_mode)

                print(f"[THRML] Learning update at step {sim_step_count}, eta={config.learning_rate}")
                sim_state, thrml_wrapper = simulation_step_with_thrml_learning(
                    sim_state,
                    thrml_wrapper,
                    eta=config.learning_rate
                )
                thrml_step_count += 1
                print(f"[THRML] Learning complete, thrml_step_count={thrml_step_count}")

            # 4. Visualization broadcast (slower rate)
            if viz_timer >= VIZ_RATE and len(websocket_clients) > 0:
                # Force computation to complete
                sim_state.oscillator_state.block_until_ready()

                # Adaptive resolution based on client count
                down_w, down_h = get_adaptive_downsample_resolution(len(websocket_clients))

                # Serialize (CPU operation)
                binary_packet = serialize_for_frontend(
                    sim_state,
                    down_w=down_w,
                    down_h=down_h,
                    compression="auto",
                    max_nodes=512,
                    mask_threshold=0.1,
                    simulation_running=simulation_running,
                )

                # Broadcast to all clients
                disconnected = set()
                for client in websocket_clients:
                    try:
                        await client.send_bytes(binary_packet)
                    except Exception as e:
                        disconnected.add(client)

                # Clean up disconnected clients
                websocket_clients.difference_update(disconnected)
                for client in disconnected:
                    websocket_last_seen.pop(client, None)

                viz_timer = 0.0
                viz_step_count += 1

            if status_timer >= PAUSED_STATUS_INTERVAL_SECONDS:
                await broadcast_status_update()
                status_timer = 0.0

            # 5. Sleep to maintain rate
            elapsed = asyncio.get_event_loop().time() - loop_start
            sleep_time = max(0.001, INTERNAL_RATE - elapsed)
            await asyncio.sleep(sleep_time)

            # Update last loop time for pause/resume calculations
            last_loop_time = loop_start

    except asyncio.CancelledError:
        print("Simulation loop cancelled")
    except Exception as e:
        print(f"Error in simulation loop: {e}")
        import traceback
        traceback.print_exc()


def map_pitch_to_c(pitch_hz: float) -> float:
    """Map audio pitch to wave speed (placeholder)."""
    # Normalize pitch to [0, 1]
    pitch_norm = (pitch_hz - 80.0) / (800.0 - 80.0)
    pitch_norm = max(0.0, min(1.0, pitch_norm))
    
    # Map to c range [0.5, 2.0]
    c = 0.5 + pitch_norm * 1.5
    return float(c)


def map_rms_to_k(rms: float) -> float:
    """Map audio RMS to source strength (placeholder)."""
    # Apply power law for perceptual scaling
    k = 10.0 * (rms ** 2.0)
    return float(max(0.0, min(10.0, k)))


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve frontend HTML."""
    try:
        return FileResponse("src/frontend/threejs/index.html")
    except FileNotFoundError:
        return {"message": "GMCS API Server", "status": "ready", "docs": "/docs"}


@app.get("/status", response_model=SystemStatusResponse)
async def get_status():
    """Get system status."""
    if sim_state is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    return SystemStatusResponse(
        active_nodes=int(jnp.sum(sim_state.node_active_mask)),
        total_capacity=N_MAX,
        simulation_time=float(sim_state.t[0]),
        dt=float(sim_state.dt),
        grid_size=[sim_state.field_p.shape[0], sim_state.field_p.shape[1]]
    )


@app.post("/node/add", response_model=NodeResponse)
async def add_node(request: AddNodeRequest):
    """Add a new node to the simulation."""
    global sim_state, thrml_wrapper, thrml_rebuild_lock
    
    try:
        # Merge config with defaults
        config = merge_config_with_defaults(request.config)
        config = validate_parameter_ranges(config)
        
        # Pad chain
        chain = pad_chain_to_max_length(request.chain)
        
        # Add node
        sim_state, node_id = add_node_to_state(
            sim_state,
            tuple(request.position),
            initial_perturbation=request.initial_perturbation,
            gmcs_chain=chain
        )
        
        # Update GMCS parameters
        for param_name, param_value in config.items():
            attr_name = f"gmcs_{param_name}"
            if hasattr(sim_state, attr_name):
                old_array = getattr(sim_state, attr_name)
                new_array = old_array.at[node_id].set(param_value)
                sim_state = sim_state._replace(**{attr_name: new_array})
        
        # Push to device
        sim_state = jax.device_put(sim_state)
        
        # Invalidate and rebuild THRML wrapper asynchronously
        if thrml_rebuild_lock is None:
            thrml_rebuild_lock = asyncio.Lock()
        async with thrml_rebuild_lock:
            thrml_wrapper = None
            sim_state = sim_state._replace(thrml_model_data=None)
        schedule_thrml_rebuild()

        node_payload = _serialize_node(sim_state, node_id)
        return NodeResponse(
            status="success",
            node_id=node_id,
            message=f"Node {node_id} added at position {request.position}",
            data={"node": node_payload}
        )
        
    except Exception as e:
        return NodeResponse(
            status="error",
            message=str(e)
        )


@app.post("/node/update", response_model=NodeResponse)
async def update_node(request: UpdateNodeRequest):
    """Update existing node(s)."""
    global sim_state
    
    try:
        # Validate node IDs
        for node_id in request.node_ids:
            _assert_node_id(node_id)
        
        # Validate and clamp parameters
        config_updates = validate_parameter_ranges(request.config_updates)
        
        # Update GMCS parameters
        for param_name, param_value in config_updates.items():
            attr_name = f"gmcs_{param_name}"
            if hasattr(sim_state, attr_name):
                old_array = getattr(sim_state, attr_name)
                for node_id in request.node_ids:
                    old_array = old_array.at[node_id].set(param_value)
                sim_state = sim_state._replace(**{attr_name: old_array})
        
        # Update chain if provided
        if request.chain_update is not None:
            chain = pad_chain_to_max_length(request.chain_update)
            for node_id in request.node_ids:
                sim_state = sim_state._replace(
                    gmcs_chain=sim_state.gmcs_chain.at[node_id].set(chain)
                )
        
        # Update position if provided
        if request.position_update is not None:
            pos = jnp.array(request.position_update)
            for node_id in request.node_ids:
                sim_state = sim_state._replace(
                    node_positions=sim_state.node_positions.at[node_id].set(pos)
                )
        
        # Push to device
        sim_state = jax.device_put(sim_state)

        updated_nodes = [_serialize_node(sim_state, node_id) for node_id in request.node_ids]

        return NodeResponse(
            status="success",
            message=f"Updated {len(request.node_ids)} node(s)",
            data={"nodes": updated_nodes}
        )
        
    except Exception as e:
        return NodeResponse(
            status="error",
            message=str(e)
        )


@app.delete("/node/{node_id}", response_model=NodeResponse)
async def remove_node(node_id: int):
    """Remove (deactivate) a node."""
    global sim_state, thrml_wrapper, thrml_rebuild_lock
    
    try:
        _assert_node_id(node_id)
        
        sim_state = remove_node_from_state(sim_state, node_id)
        sim_state = jax.device_put(sim_state)
        
        if thrml_rebuild_lock is None:
            thrml_rebuild_lock = asyncio.Lock()
        async with thrml_rebuild_lock:
            thrml_wrapper = None
            sim_state = sim_state._replace(thrml_model_data=None)
        schedule_thrml_rebuild()

        node_payload = _serialize_node(sim_state, node_id)
        return NodeResponse(
            status="success",
            node_id=node_id,
            message=f"Node {node_id} removed",
            data={"node": node_payload}
        )
        
    except Exception as e:
        return NodeResponse(
            status="error",
            message=str(e)
        )


@app.get("/node/{node_id}", response_model=NodeInfoResponse)
async def get_node_info(node_id: int):
    """Get information about a specific node."""
    _ensure_simulation_initialized()
    try:
        _assert_node_id(node_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload = _serialize_node(sim_state, node_id)
    return NodeInfoResponse(**payload)


@app.get("/nodes", response_model=NodeListResponse)
async def list_nodes():
    """List all active nodes in the simulation."""
    _ensure_simulation_initialized()
    node_ids = _list_active_node_ids(sim_state)
    nodes = [NodeListItem(**_serialize_node(sim_state, node_id)) for node_id in node_ids]
    return NodeListResponse(nodes=nodes)


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Bidirectional WebSocket for visualization and control.
    
    Server → Client: Binary state packets
    Client → Server: JSON control messages
    """
    await websocket.accept()
    websocket_clients.add(websocket)
    websocket_last_seen[websocket] = asyncio.get_running_loop().time()

    # Send initial status snapshot so clients can sync immediately
    try:
        await websocket.send_json(
            {
                "type": "STATUS",
                "simulation_running": simulation_running,
                "active_nodes": int(jnp.sum(sim_state.node_active_mask)) if sim_state else 0,
                "sim_time": float(sim_state.t[0]) if sim_state is not None else 0.0,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )
    except Exception as exc:
        # If initial status delivery fails, close connection gracefully
        print(f"[WS] Failed to send initial status: {exc}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        websocket_clients.discard(websocket)
        websocket_last_seen.pop(websocket, None)
        return
    
    try:
        while True:
            # Listen for control messages from client
            message = await websocket.receive()
            websocket_last_seen[websocket] = asyncio.get_running_loop().time()
            if 'text' in message:
                try:
                    data = json.loads(message['text'])
                    msg_type = data.get('type')
                    if msg_type == 'PING':
                        await websocket.send_json({
                            'type': 'PONG',
                            'active_nodes': int(jnp.sum(sim_state.node_active_mask)),
                            'sim_time': float(sim_state.t[0]),
                            'simulation_running': simulation_running
                        })
                    elif msg_type == 'HEARTBEAT':
                        websocket_last_seen[websocket] = asyncio.get_running_loop().time()
                        await websocket.send_json({
                            'type': 'HEARTBEAT_ACK',
                            'timestamp': asyncio.get_running_loop().time(),
                            'simulation_running': simulation_running
                        })
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        websocket_clients.discard(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        websocket_clients.discard(websocket)
    finally:
        websocket_last_seen.pop(websocket, None)


# ============================================================================
# Production Hardening: Health Check, Rate Limiting, Error Handling
# ============================================================================

import time
import logging
from datetime import datetime, timedelta

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def rate_limiting_middleware(request, call_next):
    """Rate limiting middleware to prevent abuse."""
    if not RATE_LIMIT_ENABLED:
        return await call_next(request)
    
    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    now = datetime.now()
    state = rate_limit_state[client_ip]
    
    if now >= state['reset_time']:
        # Reset window
        state['count'] = 0
        state['reset_time'] = now + timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)
    
    state['count'] += 1
    
    if state['count'] > RATE_LIMIT_REQUESTS_PER_MINUTE:
        retry_after = max(1, int((state['reset_time'] - now).total_seconds()))
        headers = _build_cors_headers(request)
        headers["Retry-After"] = str(retry_after)
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": retry_after
            },
            headers=headers
        )
    
    response = await call_next(request)
    return response


@app.middleware("http")
async def error_handling_middleware(request, call_next):
    """Global error handler with structured logging."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(
            f"Unhandled exception for {request.method} {request.url.path}",
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(e) if app.debug else "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/health", response_model=dict)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns system status, uptime, and resource availability.
    """
    import psutil
    
    global sim_state, websocket_clients, simulation_running
    
    # Compute uptime (approximate via simulation time)
    uptime_seconds = float(sim_state.t[0] / sim_state.dt) * sim_state.dt if sim_state else 0
    
    # Check GPU availability
    gpu_available = False
    gpu_memory_used = 0
    gpu_memory_total = 0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_available = True
        gpu_memory_used = mem_info.used / (1024**3)  # GB
        gpu_memory_total = mem_info.total / (1024**3)  # GB
        pynvml.nvmlShutdown()
    except:
        pass
    
    # System resources
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    status = {
        "status": "healthy" if sim_state is not None else "initializing",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": uptime_seconds,
        "simulation": {
            "running": simulation_running,
            "time": float(sim_state.t[0]) if sim_state else 0.0,
            "active_nodes": int(jnp.sum(sim_state.node_active_mask)) if sim_state else 0,
            "max_nodes": N_MAX
        },
        "websocket": {
            "connections": len(websocket_clients)
        },
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3)
        },
        "gpu": {
            "available": gpu_available,
            "memory_used_gb": gpu_memory_used if gpu_available else None,
            "memory_total_gb": gpu_memory_total if gpu_available else None
        },
        "version": app.version
    }
    
    return status


@app.get("/metrics", response_model=dict)
async def metrics_endpoint():
    """
    Prometheus-compatible metrics endpoint.
    
    Returns key performance indicators for monitoring.
    """
    global sim_state, simulation_running
    
    metrics = {
        "gmcs_simulation_time": float(sim_state.t[0]) if sim_state else 0.0,
        "gmcs_active_nodes": int(jnp.sum(sim_state.node_active_mask)) if sim_state else 0,
        "gmcs_websocket_clients": len(websocket_clients),
        "gmcs_simulation_running": 1 if simulation_running else 0,
        "gmcs_ebm_max_weight": float(jnp.max(jnp.abs(sim_state.ebm_weights))) if sim_state else 0.0,
        "gmcs_field_max_amplitude": float(jnp.max(jnp.abs(sim_state.field_p))) if sim_state else 0.0
    }
    
    return metrics


# ============================================================================
# Simulation Control Endpoints
# ============================================================================

@app.post("/simulation/start")
async def start_simulation():
    """Start the simulation loop and activate all processor instances."""
    global simulation_running, processor_instances
    
    if simulation_running:
        return {
            "status": "already_running",
            "message": "Simulation is already running",
            "running": simulation_running
        }
    
    simulation_running = True
    
    # Activate all processor instances
    activated = 0
    for node_id, proc in processor_instances.items():
        proc["active"] = True
        activated += 1
        print(f"[PROCESSOR] Activated {proc['type']} instance for node {node_id}")
    
    if activated > 0:
        print(f"[SIMULATION] Started with {activated} active processors")
    
    return {
        "status": "started",
        "message": f"Simulation started with {activated} active processors",
        "running": simulation_running,
        "active_processors": activated
    }


@app.post("/simulation/stop")
async def stop_simulation():
    """Stop the simulation loop and deactivate all processor instances."""
    global simulation_running, processor_instances
    
    if not simulation_running:
        return {
            "status": "already_stopped",
            "message": "Simulation is already stopped",
            "running": simulation_running
        }
    
    simulation_running = False
    
    # Deactivate all processor instances (but keep them in memory for resume)
    deactivated = 0
    for node_id, proc in processor_instances.items():
        proc["active"] = False
        deactivated += 1
        print(f"[PROCESSOR] Deactivated {proc['type']} instance for node {node_id}")
    
    if deactivated > 0:
        print(f"[SIMULATION] Stopped, {deactivated} processors deactivated")
    
    return {
        "status": "stopped",
        "message": f"Simulation stopped, {deactivated} processors deactivated",
        "running": simulation_running,
        "deactivated_processors": deactivated
    }


@app.post("/simulation/toggle")
async def toggle_simulation():
    """Toggle simulation running state."""
    global simulation_running
    
    simulation_running = not simulation_running
    
    return {
        "status": "toggled",
        "message": f"Simulation {'started' if simulation_running else 'stopped'}",
        "running": simulation_running
    }


@app.get("/simulation/status")
async def get_simulation_status():
    """Get current simulation status."""
    global simulation_running, sim_state
    
    if sim_state is None:
        return {
            "running": simulation_running,
            "time": 0.0,
            "active_nodes": 0,
            "dt": 0.0
        }
    
    return {
        "running": simulation_running,
        "time": float(sim_state.t[0]),
        "active_nodes": int(jnp.sum(sim_state.node_active_mask)),
        "dt": float(sim_state.dt)
    }


# ============================================================================
# THRML Control Endpoints
# ============================================================================

@app.post("/processor/create")
async def create_processor_instance(
    node_id: str,
    processor_type: str = "thrml",
    nodes: int = 64,
    temperature: float = 1.0,
    gibbs_steps: int = 5,
    config: Dict[str, Any] = None
):
    """
    Create a processor instance for a specific node.
    
    Supports multiple processor types:
    - thrml: THRML Energy-Based Model sampler
    - photonic: Photonic Ising Machine (future)
    - neuromorphic: Neuromorphic hardware (future)
    
    Each node gets its own isolated processor instance that can be
    independently controlled and queried.
    """
    global processor_instances, sim_state
    
    try:
        print(f"[PROCESSOR] Creating {processor_type} instance for node {node_id}: nodes={nodes}, temp={temperature}")
        
        if processor_type == "thrml":
            # Create THRML wrapper with specified parameters
            instance = create_thrml_model(
                n_nodes=nodes,
                weights=np.random.randn(nodes, nodes) * 0.1,  # Small random weights
                biases=np.zeros(nodes),
                beta=1.0 / temperature
            )
            
            processor_instances[node_id] = {
                "type": "thrml",
                "instance": instance,
                "config": {
                    "nodes": nodes,
                    "temperature": temperature,
                    "gibbs_steps": gibbs_steps,
                    **(config or {})
                },
                "active": False  # Will be activated when simulation starts
            }
            
            print(f"[PROCESSOR] THRML instance created for node {node_id}: {instance.n_nodes} nodes")
            
            return {
                "status": "success",
                "node_id": node_id,
                "processor_type": processor_type,
                "n_nodes": instance.n_nodes,
                "temperature": temperature,
                "gibbs_steps": gibbs_steps
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported processor type: {processor_type}. Supported: thrml, photonic (future), neuromorphic (future)"
            )
        
    except Exception as e:
        print(f"[PROCESSOR ERROR] Failed to create instance: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create processor: {str(e)}")


@app.delete("/processor/{node_id}")
async def delete_processor_instance(node_id: str):
    """Delete a processor instance when a node is removed."""
    global processor_instances
    
    if node_id in processor_instances:
        proc = processor_instances[node_id]
        print(f"[PROCESSOR] Deleting {proc['type']} instance for node {node_id}")
        del processor_instances[node_id]
        return {"status": "success", "message": f"Processor for node {node_id} deleted"}
    else:
        return {"status": "not_found", "message": f"No processor found for node {node_id}"}


@app.get("/processor/{node_id}/status")
async def get_processor_status(node_id: str):
    """Get status of a specific processor instance."""
    global processor_instances
    
    if node_id not in processor_instances:
        raise HTTPException(status_code=404, detail=f"No processor found for node {node_id}")
    
    proc = processor_instances[node_id]
    
    status = {
        "node_id": node_id,
        "processor_type": proc["type"],
        "active": proc["active"],
        "config": proc["config"]
    }
    
    if proc["type"] == "thrml":
        instance = proc["instance"]
        status["n_nodes"] = instance.n_nodes
        status["n_edges"] = len(instance.edges)
    
    return status


@app.get("/processor/list")
async def list_processors():
    """List all active processor instances."""
    global processor_instances
    
    return {
        "processors": [
            {
                "node_id": node_id,
                "type": proc["type"],
                "active": proc["active"],
                "config": proc["config"]
            }
            for node_id, proc in processor_instances.items()
        ],
        "total": len(processor_instances)
    }


@app.get("/thrml/status", response_model=THRMLStatusResponse)
async def get_thrml_status():
    """Get current THRML configuration and status."""
    global sim_state, thrml_wrapper
    
    n_nodes = thrml_wrapper.n_nodes if thrml_wrapper is not None else 0
    
    return THRMLStatusResponse(
        enabled=sim_state.thrml_enabled,
        performance_mode=sim_state.thrml_performance_mode,
        temperature=sim_state.thrml_temperature,
        gibbs_steps=sim_state.thrml_gibbs_steps,
        cd_k=sim_state.thrml_cd_k,
        update_freq=sim_state.thrml_update_freq,
        n_nodes=n_nodes
    )


@app.get("/thrml/performance-mode")
async def get_thrml_performance_mode():
    """Get current THRML performance mode with configuration details."""
    global sim_state
    
    config = get_performance_config(sim_state.thrml_performance_mode)
    
    return {
        "mode": sim_state.thrml_performance_mode,
        "config": {
            "gibbs_steps": config.gibbs_steps,
            "temperature": config.temperature,
            "learning_rate": config.learning_rate,
            "cd_k_steps": config.cd_k_steps,
            "weight_update_freq": config.weight_update_freq,
            "use_jit": config.use_jit,
            "description": config.description
        }
    }


@app.post("/thrml/performance-mode")
async def set_thrml_performance_mode(request: THRMLPerformanceModeRequest):
    """
    Set THRML performance mode: speed, accuracy, or research.
    
    This updates all THRML sampling and learning parameters based on the selected preset.
    """
    global sim_state, thrml_wrapper
    
    try:
        # Get configuration for new mode
        config = get_performance_config(request.mode)
        
        # Update state with new configuration
        sim_state = sim_state._replace(
            thrml_performance_mode=request.mode,
            thrml_temperature=config.temperature,
            thrml_gibbs_steps=config.gibbs_steps,
            thrml_cd_k=config.cd_k_steps,
            thrml_update_freq=config.weight_update_freq
        )
        
        # Update THRML wrapper temperature if it exists
        if thrml_wrapper is not None:
            thrml_wrapper.set_temperature(config.temperature)
        
        return {
            "status": "success",
            "mode": request.mode,
            "message": f"THRML performance mode set to {request.mode}",
            "config": {
                "gibbs_steps": config.gibbs_steps,
                "temperature": config.temperature,
                "learning_rate": config.learning_rate,
                "cd_k_steps": config.cd_k_steps,
                "weight_update_freq": config.weight_update_freq
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/thrml/performance-modes")
async def list_thrml_performance_modes():
    """List all available THRML performance modes with descriptions."""
    return list_performance_modes()


@app.get("/thrml/energy", response_model=THRMLEnergyResponse)
async def get_thrml_energy():
    """Get current THRML model energy."""
    global sim_state, thrml_wrapper
    
    if sim_state is None:
        return THRMLEnergyResponse(energy=0.0, timestamp=0.0)
    
    if thrml_wrapper is None:
        # Try to reconstruct from state
        if sim_state.thrml_model_data is not None:
            print("[THRML] Energy endpoint: Reconstructing THRML wrapper from state...")
            thrml_wrapper = reconstruct_thrml_wrapper(sim_state.thrml_model_data)
            print(f"[THRML] Energy endpoint: Wrapper reconstructed with {thrml_wrapper.n_nodes} nodes")
        else:
            print("[THRML] Energy endpoint: No THRML wrapper or state data available")
            # Return a response instead of raising an error
            return THRMLEnergyResponse(
                energy=0.0,
                timestamp=float(sim_state.t[0]) if sim_state else 0.0
            )
    
    try:
        energy = compute_ebm_energy_thrml(
            thrml_wrapper,
            sim_state.oscillator_state,
            sim_state.node_active_mask
        )
        
        print(f"[THRML] Energy computed: {energy:.4f}")
        
        return THRMLEnergyResponse(
            energy=float(energy),
            timestamp=float(sim_state.t[0])
        )
        
    except Exception as e:
        print(f"[THRML] Energy endpoint error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/thrml/temperature")
async def get_thrml_temperature():
    """Get current THRML sampling temperature."""
    global sim_state
    
    return {
        "temperature": sim_state.thrml_temperature
    }


@app.post("/thrml/temperature")
async def set_thrml_temperature(request: THRMLTemperatureRequest):
    """
    Manually adjust THRML sampling temperature.
    
    Higher temperature = more stochastic sampling.
    Lower temperature = more deterministic sampling.
    """
    global sim_state, thrml_wrapper
    
    try:
        # Update state
        sim_state = sim_state._replace(
            thrml_temperature=request.temperature
        )
        
        # Update THRML wrapper if it exists
        if thrml_wrapper is not None:
            thrml_wrapper.set_temperature(request.temperature)
        
        return {
            "status": "success",
            "temperature": request.temperature,
            "message": f"THRML temperature set to {request.temperature}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Algorithm Management API
# ============================================================================

@app.get("/algorithms/list")
async def list_algorithms():
    """List all available GMCS algorithms with metadata."""
    from src.core.gmcs_pipeline import (
        ALGO_NOP, ALGO_LIMITER, ALGO_COMPRESSOR, ALGO_EXPANDER,
        ALGO_THRESHOLD, ALGO_PHASEMOD, ALGO_FOLD,
        ALGO_RESONATOR, ALGO_HILBERT, ALGO_RECTIFIER, ALGO_QUANTIZER,
        ALGO_SLEW_LIMITER, ALGO_CROSS_MOD, ALGO_BIPOLAR_FOLD,
        ALGO_OPTICAL_KERR, ALGO_ELECTRO_OPTIC, ALGO_OPTICAL_SWITCH,
        ALGO_FOUR_WAVE_MIXING, ALGO_RAMAN_AMPLIFIER, ALGO_SATURATION,
        ALGO_OPTICAL_GAIN, get_algorithm_name
    )
    
    algorithms = [
        # Basic (0-6)
        {"id": ALGO_NOP, "name": get_algorithm_name(ALGO_NOP), "category": "basic", "description": "Pass-through (no operation)"},
        {"id": ALGO_LIMITER, "name": get_algorithm_name(ALGO_LIMITER), "category": "basic", "description": "Soft clipping limiter using tanh"},
        {"id": ALGO_COMPRESSOR, "name": get_algorithm_name(ALGO_COMPRESSOR), "category": "basic", "description": "Dynamic range compression"},
        {"id": ALGO_EXPANDER, "name": get_algorithm_name(ALGO_EXPANDER), "category": "basic", "description": "Dynamic range expansion"},
        {"id": ALGO_THRESHOLD, "name": get_algorithm_name(ALGO_THRESHOLD), "category": "basic", "description": "Amplitude gate/threshold"},
        {"id": ALGO_PHASEMOD, "name": get_algorithm_name(ALGO_PHASEMOD), "category": "basic", "description": "Phase/amplitude modulation"},
        {"id": ALGO_FOLD, "name": get_algorithm_name(ALGO_FOLD), "category": "basic", "description": "Wave folding nonlinearity"},
        # Audio/Signal (7-13)
        {"id": ALGO_RESONATOR, "name": get_algorithm_name(ALGO_RESONATOR), "category": "audio", "description": "Resonant bandpass filter"},
        {"id": ALGO_HILBERT, "name": get_algorithm_name(ALGO_HILBERT), "category": "audio", "description": "Hilbert transform (90° phase shift)"},
        {"id": ALGO_RECTIFIER, "name": get_algorithm_name(ALGO_RECTIFIER), "category": "audio", "description": "Full-wave rectification"},
        {"id": ALGO_QUANTIZER, "name": get_algorithm_name(ALGO_QUANTIZER), "category": "audio", "description": "Bit-depth reduction"},
        {"id": ALGO_SLEW_LIMITER, "name": get_algorithm_name(ALGO_SLEW_LIMITER), "category": "audio", "description": "Slew rate limiter"},
        {"id": ALGO_CROSS_MOD, "name": get_algorithm_name(ALGO_CROSS_MOD), "category": "audio", "description": "Cross-modulation (ring mod)"},
        {"id": ALGO_BIPOLAR_FOLD, "name": get_algorithm_name(ALGO_BIPOLAR_FOLD), "category": "audio", "description": "Bipolar wave folding"},
        # Photonic (14-20)
        {"id": ALGO_OPTICAL_KERR, "name": get_algorithm_name(ALGO_OPTICAL_KERR), "category": "photonic", "description": "Optical Kerr effect (χ³ nonlinearity)"},
        {"id": ALGO_ELECTRO_OPTIC, "name": get_algorithm_name(ALGO_ELECTRO_OPTIC), "category": "photonic", "description": "Electro-optic modulation (Pockels)"},
        {"id": ALGO_OPTICAL_SWITCH, "name": get_algorithm_name(ALGO_OPTICAL_SWITCH), "category": "photonic", "description": "All-optical switch"},
        {"id": ALGO_FOUR_WAVE_MIXING, "name": get_algorithm_name(ALGO_FOUR_WAVE_MIXING), "category": "photonic", "description": "Four-wave mixing"},
        {"id": ALGO_RAMAN_AMPLIFIER, "name": get_algorithm_name(ALGO_RAMAN_AMPLIFIER), "category": "photonic", "description": "Raman amplification"},
        {"id": ALGO_SATURATION, "name": get_algorithm_name(ALGO_SATURATION), "category": "photonic", "description": "Soft saturation"},
        {"id": ALGO_OPTICAL_GAIN, "name": get_algorithm_name(ALGO_OPTICAL_GAIN), "category": "photonic", "description": "Linear optical gain"},
    ]
    
    return {
        "algorithms": algorithms,
        "total": len(algorithms),
        "categories": ["basic", "audio", "photonic"]
    }


@app.get("/algorithms/{algo_id}")
async def get_algorithm_details(algo_id: int):
    """Get detailed information about a specific algorithm."""
    from src.core.gmcs_pipeline import get_algorithm_name
    
    if algo_id < 0 or algo_id > 20:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    
    # Parameter mappings
    param_info = {
        0: [],  # NOP
        1: ["A_max"],  # Limiter
        2: ["R_comp", "T_comp"],  # Compressor
        3: ["R_exp", "T_exp"],  # Expander
        4: ["T_comp"],  # Threshold
        5: ["Phi", "omega"],  # Phase Mod
        6: ["gamma", "beta"],  # Fold
        7: ["f0", "Q"],  # Resonator
        8: [],  # Hilbert
        9: [],  # Rectifier
        10: ["levels"],  # Quantizer
        11: ["rate_limit"],  # Slew Limiter
        12: ["Phi", "omega"],  # Cross Mod
        13: ["T_comp"],  # Bipolar Fold
        14: ["n2", "beta"],  # Optical Kerr
        15: ["V", "V_pi"],  # Electro-Optic
        16: ["T_comp", "gamma"],  # Optical Switch
        17: ["gamma", "n2"],  # Four-Wave Mixing
        18: ["gamma", "n2", "beta"],  # Raman
        19: ["A_max"],  # Saturation
        20: ["gamma"],  # Optical Gain
    }
    
    return {
        "id": algo_id,
        "name": get_algorithm_name(algo_id),
        "parameters": param_info.get(algo_id, [])
    }


@app.get("/algorithms/categories")
async def get_algorithm_categories():
    """List algorithm categories."""
    return {
        "categories": [
            {"name": "basic", "description": "Basic signal processing", "count": 7},
            {"name": "audio", "description": "Audio/signal processing", "count": 7},
            {"name": "photonic", "description": "Photonic simulation", "count": 7}
        ]
    }


# ============================================================================
# Modulation Matrix API
# ============================================================================

from src.api.routes import ModulationRouteRequest, ModulationRouteResponse

# Initialize global modulation matrix
modulation_matrix = None

@app.post("/modulation/routes")
async def add_modulation_route(request: ModulationRouteRequest):
    """Add a new modulation route."""
    global modulation_matrix
    
    from src.core.modulation_matrix import ModulationMatrix, ModulationRoute, ModulationSource, ModulationTarget
    
    # Initialize if needed
    if modulation_matrix is None:
        modulation_matrix = ModulationMatrix()
    
    try:
        route = ModulationRoute(
            source_type=ModulationSource(request.source_type),
            source_node_id=request.source_node_id,
            target_type=ModulationTarget(request.target_type),
            target_node_id=request.target_node_id,
            strength=request.strength,
            mode=request.mode,
            condition_node_id=request.condition_node_id,
            condition_value=request.condition_value
        )
        
        route_id = modulation_matrix.add_route(route)
        
        return ModulationRouteResponse(
            route_id=route_id,
            route=route.to_dict(),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/modulation/routes")
async def list_modulation_routes():
    """List all modulation routes."""
    global modulation_matrix
    
    if modulation_matrix is None:
        return {"routes": [], "total": 0}
    
    routes = [
        {"id": i, "route": route.to_dict()}
        for i, route in enumerate(modulation_matrix.routes)
    ]
    
    return {
        "routes": routes,
        "total": len(routes)
    }


@app.delete("/modulation/routes/{route_id}")
async def delete_modulation_route(route_id: int):
    """Delete a modulation route."""
    global modulation_matrix
    
    if modulation_matrix is None:
        raise HTTPException(status_code=404, detail="No routes configured")
    
    success = modulation_matrix.remove_route(route_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Route not found")
    
    return {"status": "success", "message": f"Route {route_id} deleted"}


@app.post("/modulation/presets/{preset_name}")
async def load_modulation_preset(preset_name: str):
    """Load a modulation preset."""
    global modulation_matrix, sim_state
    
    from src.core.modulation_matrix import create_audio_reactive_preset, create_feedback_preset
    
    n_active = int(jnp.sum(sim_state.node_active_mask))
    
    if preset_name == "audio-reactive":
        modulation_matrix = create_audio_reactive_preset(n_active)
    elif preset_name == "feedback":
        modulation_matrix = create_feedback_preset(n_active)
    else:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    return {
        "status": "success",
        "preset": preset_name,
        "routes": len(modulation_matrix.routes)
    }


@app.get("/modulation/sources")
async def list_modulation_sources():
    """List available modulation source types."""
    from src.core.modulation_matrix import ModulationSource
    
    sources = [
        {"id": ModulationSource.OSCILLATOR_X, "name": "Oscillator X"},
        {"id": ModulationSource.OSCILLATOR_Y, "name": "Oscillator Y"},
        {"id": ModulationSource.OSCILLATOR_Z, "name": "Oscillator Z"},
        {"id": ModulationSource.GMCS_OUTPUT, "name": "GMCS Output"},
        {"id": ModulationSource.THRML_PBIT, "name": "THRML P-bit"},
        {"id": ModulationSource.WAVE_FIELD, "name": "Wave Field"},
        {"id": ModulationSource.AUDIO_PITCH, "name": "Audio Pitch"},
        {"id": ModulationSource.AUDIO_RMS, "name": "Audio RMS"},
        {"id": ModulationSource.EXTERNAL_MODEL, "name": "External Model"},
        {"id": ModulationSource.CONSTANT, "name": "Constant"},
    ]
    
    return {"sources": sources}


@app.get("/modulation/targets")
async def list_modulation_targets():
    """List available modulation target types."""
    from src.core.modulation_matrix import ModulationTarget
    
    targets = [
        # GMCS parameters
        {"id": ModulationTarget.GMCS_PARAM_0, "name": "GMCS A_max"},
        {"id": ModulationTarget.GMCS_PARAM_1, "name": "GMCS R_comp"},
        {"id": ModulationTarget.GMCS_PARAM_2, "name": "GMCS T_comp"},
        {"id": ModulationTarget.GMCS_PARAM_3, "name": "GMCS R_exp"},
        {"id": ModulationTarget.GMCS_PARAM_4, "name": "GMCS T_exp"},
        {"id": ModulationTarget.GMCS_PARAM_5, "name": "GMCS Phi"},
        {"id": ModulationTarget.GMCS_PARAM_6, "name": "GMCS omega"},
        {"id": ModulationTarget.GMCS_PARAM_7, "name": "GMCS gamma"},
        {"id": ModulationTarget.GMCS_PARAM_8, "name": "GMCS beta"},
        {"id": ModulationTarget.GMCS_PARAM_9, "name": "GMCS f0"},
        {"id": ModulationTarget.GMCS_PARAM_10, "name": "GMCS Q"},
        {"id": ModulationTarget.GMCS_PARAM_11, "name": "GMCS levels"},
        {"id": ModulationTarget.GMCS_PARAM_12, "name": "GMCS rate_limit"},
        {"id": ModulationTarget.GMCS_PARAM_13, "name": "GMCS n2"},
        {"id": ModulationTarget.GMCS_PARAM_14, "name": "GMCS V"},
        {"id": ModulationTarget.GMCS_PARAM_15, "name": "GMCS V_pi"},
        # Other targets
        {"id": ModulationTarget.THRML_BIAS, "name": "THRML Bias"},
        {"id": ModulationTarget.THRML_TEMPERATURE, "name": "THRML Temperature"},
        {"id": ModulationTarget.OSCILLATOR_FORCE, "name": "Oscillator Force"},
        {"id": ModulationTarget.WAVE_SOURCE_STRENGTH, "name": "Wave Source Strength"},
        {"id": ModulationTarget.WAVE_SPEED, "name": "Wave Speed"},
    ]
    
    return {"targets": targets}


# ============================================================================
# THRML Advanced Features API
# ============================================================================

from src.api.routes import THRMLHeterogeneousRequest, THRMLClampNodesRequest, THRMLFactorRequest

# Global higher-order interaction manager
higher_order_manager = None
# Global factor system
factor_system = None

@app.post("/thrml/heterogeneous/configure")
async def configure_heterogeneous_thrml(request: THRMLHeterogeneousRequest):
    """Configure heterogeneous THRML model with mixed node types."""
    global thrml_wrapper, sim_state
    
    _ensure_simulation_initialized()
    
    from src.core.thrml_integration import HeterogeneousTHRMLWrapper
    import numpy as np
    
    try:
        n_nodes = len(request.node_types)
        node_types = np.array(request.node_types, dtype=np.int32)
        
        # Create heterogeneous model
        thrml_wrapper = HeterogeneousTHRMLWrapper(
            n_nodes=n_nodes,
            node_types=node_types
        )
        
        # Update state
        sim_state = sim_state._replace(
            thrml_model_data=thrml_wrapper.serialize()
        )
        
        return {
            "status": "success",
            "n_nodes": n_nodes,
            "node_types": request.node_types,
            "message": "Heterogeneous THRML model configured"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/thrml/clamp-nodes")
async def clamp_thrml_nodes(request: THRMLClampNodesRequest):
    """Clamp specific nodes for conditional sampling."""
    global thrml_wrapper
    
    _ensure_simulation_initialized()
    
    if thrml_wrapper is None:
        raise HTTPException(status_code=400, detail="THRML not initialized")
    
    try:
        # Check if wrapper supports clamping
        if hasattr(thrml_wrapper, 'set_clamped_nodes'):
            thrml_wrapper.set_clamped_nodes(request.node_ids, request.values)
        else:
            raise HTTPException(
                status_code=400,
                detail="Current THRML wrapper does not support clamping. Use heterogeneous model."
            )
        
        return {
            "status": "success",
            "clamped_nodes": request.node_ids,
            "values": request.values,
            "message": f"Clamped {len(request.node_ids)} nodes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/thrml/sample-conditional")
async def sample_conditional():
    """Sample THRML with clamped nodes (conditional sampling)."""
    global thrml_wrapper
    
    if thrml_wrapper is None:
        raise HTTPException(status_code=400, detail="THRML not initialized")
    
    try:
        key = jax.random.PRNGKey(int(datetime.now().timestamp()))
        
        if hasattr(thrml_wrapper, 'sample_conditional'):
            sample = thrml_wrapper.sample_conditional(key)
            return {
                "status": "success",
                "sample": sample.tolist(),
                "message": "Conditional sample generated"
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Current THRML wrapper does not support conditional sampling"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/thrml/clamped-nodes")
async def get_clamped_nodes():
    """Get currently clamped nodes."""
    global thrml_wrapper
    
    if thrml_wrapper is None:
        return {"clamped_nodes": [], "values": []}
    
    if hasattr(thrml_wrapper, 'clamped_nodes'):
        return {
            "clamped_nodes": thrml_wrapper.clamped_nodes,
            "values": thrml_wrapper.clamped_values
        }
    
    return {"clamped_nodes": [], "values": []}


@app.post("/thrml/interactions/add")
async def add_higher_order_interaction(
    order: int,
    node_ids: List[int],
    strength: float
):
    """Add higher-order interaction (3-way or 4-way)."""
    global higher_order_manager
    
    from src.core.thrml_higher_order import HigherOrderInteractionManager
    
    if higher_order_manager is None:
        higher_order_manager = HigherOrderInteractionManager()
    
    try:
        if order == 3 and len(node_ids) == 3:
            interaction_id = higher_order_manager.add_three_way_interaction(
                node_ids[0], node_ids[1], node_ids[2], strength
            )
        elif order == 4 and len(node_ids) == 4:
            interaction_id = higher_order_manager.add_four_way_interaction(
                node_ids[0], node_ids[1], node_ids[2], node_ids[3], strength
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interaction: order={order}, nodes={len(node_ids)}"
            )
        
        return {
            "status": "success",
            "interaction_id": interaction_id,
            "order": order,
            "node_ids": node_ids,
            "strength": strength
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/thrml/interactions/list")
async def list_interactions():
    """List all higher-order interactions."""
    global higher_order_manager
    
    if higher_order_manager is None:
        return {"interactions": [], "total": 0}
    
    interactions = [
        {"id": i, "interaction": interaction.to_dict()}
        for i, interaction in enumerate(higher_order_manager.interactions)
    ]
    
    return {
        "interactions": interactions,
        "total": len(interactions),
        "three_way_count": higher_order_manager.three_way_count,
        "four_way_count": higher_order_manager.four_way_count
    }


@app.delete("/thrml/interactions/{interaction_id}")
async def remove_interaction(interaction_id: int):
    """Remove a higher-order interaction."""
    global higher_order_manager
    
    if higher_order_manager is None:
        raise HTTPException(status_code=404, detail="No interactions configured")
    
    success = higher_order_manager.remove_interaction(interaction_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Interaction not found")
    
    return {"status": "success", "message": f"Interaction {interaction_id} removed"}


@app.post("/thrml/factors/add")
async def add_custom_factor(request: THRMLFactorRequest):
    """Add custom energy factor."""
    global factor_system
    
    from src.core.thrml_integration import THRMLFactorSystem
    
    if factor_system is None:
        factor_system = THRMLFactorSystem()
    
    try:
        if request.factor_type == "photonic_coupling":
            wavelength = request.parameters.get("wavelength", 1550e-9)
            factor_system.add_photonic_coupling_factor(
                request.node_ids,
                request.strength,
                wavelength
            )
        elif request.factor_type == "audio_harmony":
            fundamental = request.parameters.get("fundamental", 440.0)
            factor_system.add_audio_harmony_factor(
                request.node_ids,
                fundamental
            )
        elif request.factor_type == "ml_regularization":
            reg_type = request.parameters.get("reg_type", "l2")
            factor_system.add_ml_regularization_factor(
                request.node_ids,
                reg_type,
                request.strength
            )
        else:
            raise HTTPException(status_code=400, detail="Unknown factor type")
        
        return {
            "status": "success",
            "factor_type": request.factor_type,
            "node_ids": request.node_ids,
            "strength": request.strength
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/thrml/factors/list")
async def list_factors():
    """List all custom factors."""
    global factor_system
    
    if factor_system is None:
        return {"factors": [], "total": 0}
    
    return {
        "factors": factor_system.factors,
        "total": len(factor_system.factors)
    }


@app.get("/thrml/factors/library")
async def get_factor_library():
    """Get available factor types."""
    return {
        "factor_types": [
            {
                "type": "photonic_coupling",
                "description": "Optical coupling energy",
                "parameters": ["wavelength"]
            },
            {
                "type": "audio_harmony",
                "description": "Musical harmony constraints",
                "parameters": ["fundamental"]
            },
            {
                "type": "ml_regularization",
                "description": "ML regularization (L1/L2/elastic)",
                "parameters": ["reg_type"]
            }
        ]
    }


# ============================================================================
# Session Persistence API
# ============================================================================

import pickle
from pathlib import Path

SESSION_DIR = Path("saved_sessions")
SESSION_DIR.mkdir(exist_ok=True)


@app.post("/session/save")
async def save_session(name: str):
    """Save current simulation session."""
    global sim_state, thrml_wrapper, modulation_matrix
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    session_data = {
        "name": name,
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "state": {
            # Serialize state (simplified - would need full serialization)
            "n_active_nodes": int(jnp.sum(sim_state.node_active_mask)),
            "time": float(sim_state.t[0]),
        },
        "modulation_routes": modulation_matrix.to_dict() if modulation_matrix else None,
    }
    
    filepath = SESSION_DIR / f"{session_id}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(session_data, f)
    
    return {
        "status": "success",
        "session_id": session_id,
        "name": name,
        "message": "Session saved"
    }


@app.get("/session/list")
async def list_sessions():
    """List all saved sessions."""
    sessions = []
    for filepath in SESSION_DIR.glob("*.pkl"):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "name": data["name"],
                    "created_at": data["created_at"]
                })
        except Exception as e:
            print(f"Error loading session {filepath}: {e}")
            continue
    
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    return {"sessions": sessions, "total": len(sessions)}


@app.post("/session/{session_id}/load")
async def load_session(session_id: str):
    """Load a saved session."""
    filepath = SESSION_DIR / f"{session_id}.pkl"
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        with open(filepath, 'rb') as f:
            session_data = pickle.load(f)
        
        return {
            "status": "success",
            "session": session_data,
            "message": f"Session '{session_data['name']}' loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a saved session."""
    filepath = SESSION_DIR / f"{session_id}.pkl"
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    filepath.unlink()
    return {"status": "success", "message": f"Session {session_id} deleted"}


# ============================================================================
# Visualizer Data Endpoints
# ============================================================================

@app.get("/visualizer/oscilloscope/{node_id}")
async def get_oscilloscope_data(node_id: int, samples: int = 1000):
    """
    Get oscilloscope data for a specific node.
    Returns the last N samples of x, y, z states.
    """
    global sim_state
    
    if node_id < 0 or node_id >= N_MAX:
        raise HTTPException(status_code=400, detail="Invalid node_id")
    
    if not sim_state.node_active_mask[node_id]:
        raise HTTPException(status_code=404, detail="Node not active")
    
    # Get current oscillator states
    osc_state = sim_state.oscillator_state
    x = float(osc_state.x[node_id])
    y = float(osc_state.y[node_id])
    z = float(osc_state.z[node_id])
    
    # For now, return current state
    # TODO: Implement history buffer for time series
    return {
        "node_id": node_id,
        "current": {
            "x": x,
            "y": y,
            "z": z
        },
        "time": float(sim_state.t[0]),
        "samples_requested": samples
    }


@app.get("/visualizer/fft/{node_id}")
async def get_fft_data(node_id: int, fft_size: int = 1024):
    """
    Get FFT data for a specific node's output.
    Computes frequency spectrum of the oscillator's x state.
    """
    global sim_state
    
    if node_id < 0 or node_id >= N_MAX:
        raise HTTPException(status_code=400, detail="Invalid node_id")
    
    if not sim_state.node_active_mask[node_id]:
        raise HTTPException(status_code=404, detail="Node not active")
    
    # For now, return placeholder
    # TODO: Implement FFT computation on history buffer
    import numpy as np
    freqs = np.fft.rfftfreq(fft_size, d=float(sim_state.dt))
    magnitudes = np.zeros_like(freqs)
    
    return {
        "node_id": node_id,
        "fft_size": fft_size,
        "frequencies": freqs[:100].tolist(),  # Limit to first 100 bins
        "magnitudes": magnitudes[:100].tolist(),
        "sample_rate": 1.0 / float(sim_state.dt)
    }


@app.get("/visualizer/thrml/pbits")
async def get_thrml_pbit_states(max_history: int = 128, node_id: str = None):
    """
    Get current THRML p-bit states and optional history for visualizers.
    
    If node_id is provided, returns data for that specific processor instance.
    Otherwise, returns data from the global wrapper (legacy mode).
    """
    global thrml_wrapper, sim_state, processor_instances

    if sim_state is None:
        return {
            "current": [],
            "history": [],
            "grid_size": 0,
            "history_length": 0,
            "timestamp": 0.0
        }

    # Try to get from processor instance first
    active_wrapper = None
    if node_id and node_id in processor_instances:
        proc = processor_instances[node_id]
        if proc["type"] == "thrml" and proc["active"]:
            active_wrapper = proc["instance"]
            print(f"[THRML] Using processor instance for node {node_id}")
    
    # Fall back to global wrapper if no processor instance
    if active_wrapper is None:
        if thrml_wrapper is None:
            # Attempt to reconstruct from serialized state
            if sim_state.thrml_model_data is not None:
                print("[THRML] P-bit endpoint: Reconstructing wrapper from state...")
                thrml_wrapper = reconstruct_thrml_wrapper(sim_state.thrml_model_data)
            else:
                print("[THRML] P-bit endpoint: No wrapper available")
                return {
                    "current": [],
                    "history": [],
                    "grid_size": 0,
                    "history_length": 0,
                    "timestamp": float(sim_state.t[0])
                }
        active_wrapper = thrml_wrapper

    sample = get_last_thrml_sample()

    # If no sample recorded yet, generate one so the visualizer has data
    if sample is None and active_wrapper is not None:
        try:
            key = jax.random.PRNGKey(np.random.randint(0, 2**31 - 1))
            
            # Get config from processor instance if available
            if node_id and node_id in processor_instances:
                proc_config = processor_instances[node_id]["config"]
                temperature = proc_config.get("temperature", 1.0)
                gibbs_steps = proc_config.get("gibbs_steps", 5)
            else:
                temperature = sim_state.thrml_temperature
                gibbs_steps = sim_state.thrml_gibbs_steps
            
            sample = active_wrapper.sample_gibbs(
                n_steps=max(5, gibbs_steps),
                temperature=temperature,
                key=key
            )
            update_thrml_sample(sample)
            print(f"[THRML] P-bit endpoint: Generated fresh sample for visualizer (node_id={node_id})")
        except Exception as exc:
            print(f"[THRML] P-bit endpoint error while sampling: {exc}")
            sample = None

    if sample is None:
        return {
            "current": [],
            "history": [],
            "grid_size": 0,
            "history_length": 0,
            "timestamp": float(sim_state.t[0])
        }

    current_list = np.array(sample, dtype=np.float32).tolist()
    total_spins = len(current_list)
    grid_size = int(np.ceil(np.sqrt(total_spins)))

    history = get_thrml_sample_history(max_history=max_history)

    return {
        "current": current_list,
        "history": history,
        "grid_size": grid_size,
        "history_length": len(history),
        "timestamp": float(sim_state.t[0])
    }


@app.get("/thrml/status")
async def get_thrml_status():
    """Summarize current THRML simulation state for diagnostics."""
    global sim_state, thrml_wrapper

    base_response = {
        "has_state": sim_state is not None,
        "thrml_enabled": bool(sim_state.thrml_enabled) if sim_state else False,
        "active_nodes": int(jnp.sum(sim_state.node_active_mask)) if sim_state is not None else 0,
        "temperature": float(sim_state.thrml_temperature) if sim_state else 0.0,
        "gibbs_steps": int(sim_state.thrml_gibbs_steps) if sim_state else 0,
        "has_wrapper": thrml_wrapper is not None,
        "last_feedback_norm": get_last_thrml_feedback_norm(),
        "timestamp": float(sim_state.t[0]) if sim_state else 0.0
    }

    if thrml_wrapper is not None:
        base_response.update({
            "thrml_nodes": thrml_wrapper.n_nodes,
            "beta": float(thrml_wrapper.beta),
        })
    else:
        base_response.update({
            "thrml_nodes": 0,
            "beta": 0.0,
        })

    return base_response


@app.get("/visualizer/thrml/energy")
async def get_thrml_energy_history(samples: int = 1000):
    """
    Get THRML energy time series.
    """
    global thrml_wrapper
    
    if thrml_wrapper is None:
        return {
            "energy_history": [],
            "current_energy": 0.0,
            "samples": 0
        }
    
    # Get current energy
    try:
        current_energy = float(thrml_wrapper.compute_energy())
    except:
        current_energy = 0.0
    
    # TODO: Implement energy history buffer
    return {
        "energy_history": [current_energy],
        "current_energy": current_energy,
        "samples": 1,
        "time": float(sim_state.t[0]) if sim_state else 0.0
    }


@app.get("/visualizer/thrml/spins")
async def get_thrml_spin_states():
    """
    Get current THRML spin state matrix.
    """
    global thrml_wrapper
    
    if thrml_wrapper is None:
        return {
            "spins": [],
            "shape": [0, 0],
            "num_nodes": 0
        }
    
    # Get spin states from THRML
    try:
        spins = thrml_wrapper.get_state()
        spins_list = np.array(spins).tolist()
        
        # Reshape to square matrix if possible
        n = len(spins_list)
        grid_size = int(np.sqrt(n))
        if grid_size * grid_size == n:
            spins_matrix = np.array(spins_list).reshape(grid_size, grid_size).tolist()
        else:
            spins_matrix = [spins_list]
        
        return {
            "spins": spins_matrix,
            "shape": [len(spins_matrix), len(spins_matrix[0])] if spins_matrix else [0, 0],
            "num_nodes": n
        }
    except Exception as e:
        return {
            "spins": [],
            "shape": [0, 0],
            "num_nodes": 0,
            "error": str(e)
        }


@app.get("/visualizer/thrml/correlations")
async def get_thrml_correlations():
    """
    Get THRML correlation matrix.
    Computes pairwise correlations between spin states.
    """
    global thrml_wrapper
    
    if thrml_wrapper is None:
        return {
            "correlations": [],
            "shape": [0, 0]
        }
    
    try:
        spins = np.array(thrml_wrapper.get_state())
        n = len(spins)
        
        # Compute correlation matrix (simplified)
        # For a proper implementation, we'd need spin history
        correlations = np.eye(n)  # Identity for now
        
        return {
            "correlations": correlations.tolist(),
            "shape": [n, n],
            "num_nodes": n
        }
    except Exception as e:
        return {
            "correlations": [],
            "shape": [0, 0],
            "error": str(e)
        }


@app.get("/visualizer/wave_field")
async def get_wave_field_data(downsample: int = 4):
    """
    Get current wave field state for visualization.
    """
    global sim_state
    
    if sim_state is None:
        return {
            "field": [],
            "shape": [0, 0]
        }
    
    # Get wave field
    field = np.array(sim_state.wave_state.p)
    
    # Downsample for performance
    if downsample > 1:
        field = field[::downsample, ::downsample]
    
    return {
        "field": field.tolist(),
        "shape": list(field.shape),
        "time": float(sim_state.t[0]),
        "downsample_factor": downsample
    }


@app.get("/visualizer/node_positions")
async def get_node_positions():
    """
    Get positions of all active nodes for visualization.
    """
    global sim_state
    
    if sim_state is None:
        return {"nodes": []}
    
    nodes = []
    for i in range(N_MAX):
        if sim_state.node_active_mask[i]:
            nodes.append({
                "id": i,
                "x": float(sim_state.node_x_pos[i]),
                "y": float(sim_state.node_y_pos[i]),
                "state": {
                    "x": float(sim_state.oscillator_state.x[i]),
                    "y": float(sim_state.oscillator_state.y[i]),
                    "z": float(sim_state.oscillator_state.z[i])
                }
            })
    
    return {"nodes": nodes, "count": len(nodes)}


# ============================================================================
# Sampler Backend API Endpoints (New Generic Interface)
# ============================================================================

@app.get("/sampler/benchmarks")
async def get_sampler_benchmarks():
    """
    Get current benchmark diagnostics from the active sampler backend.
    
    Returns samples/sec, ESS/sec, autocorrelation, and other performance metrics.
    """
    global thrml_wrapper
    
    if thrml_wrapper is None:
        return {
            "error": "No sampler backend active",
            "samples_per_sec": 0.0,
            "ess_per_sec": 0.0,
            "lag1_autocorr": 0.0,
            "tau_int": 0.0
        }
    
    # Get diagnostics from THRML wrapper
    diagnostics = thrml_wrapper.get_benchmark_diagnostics()
    
    return {
        "samples_per_sec": diagnostics.get("samples_per_sec", 0.0),
        "ess_per_sec": diagnostics.get("ess_per_sec", 0.0),
        "lag1_autocorr": diagnostics.get("lag1_autocorr", 0.0),
        "tau_int": diagnostics.get("tau_int", 0.0),
        "total_samples": diagnostics.get("total_samples", 0),
        "mean_magnetization": diagnostics.get("mean_magnetization", 0.0),
        "timestamp": float(sim_state.t[0]) if sim_state else 0.0
    }


@app.get("/sampler/benchmarks/history")
async def get_sampler_benchmark_history(max_samples: int = 100):
    """
    Get benchmark history over time.
    
    Args:
        max_samples: Maximum number of historical samples to return
        
    Returns:
        List of benchmark samples with timestamps
    """
    global thrml_wrapper
    
    if thrml_wrapper is None:
        return {
            "history": [],
            "count": 0
        }
    
    history = thrml_wrapper.get_benchmark_history(max_samples=max_samples)
    
    return {
        "history": history,
        "count": len(history)
    }


@app.get("/sampler/benchmarks/export")
async def export_sampler_benchmarks(format: str = "json"):
    """
    Export benchmark data in JSON or CSV format.
    
    Args:
        format: 'json' or 'csv'
        
    Returns:
        Benchmark data in requested format
    """
    global thrml_wrapper
    
    if thrml_wrapper is None:
        return {"error": "No sampler backend active"}
    
    if format == "json":
        return thrml_wrapper.get_benchmark_json()
    elif format == "csv":
        # For CSV, we'll return a text response
        import io
        csv_buffer = io.StringIO()
        thrml_wrapper.export_benchmark_csv(csv_buffer)
        csv_data = csv_buffer.getvalue()
        
        from fastapi.responses import Response
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=benchmark_data.csv"}
        )
    else:
        return {"error": f"Unknown format: {format}. Use 'json' or 'csv'."}


# ============================================================================
# Multi-Chain Control Endpoints
# ============================================================================

@app.get("/sampler/chains")
async def get_chain_config():
    """
    Get current multi-chain configuration.
    
    Returns:
        Dict with current chain count, auto-detected optimal, and hardware info
    """
    global thrml_wrapper, sim_state
    
    if thrml_wrapper is None:
        return {
            "current_chains": 1,
            "auto_detected_optimal": 1,
            "has_gpu": False,
            "enabled": False
        }
    
    chain_diag = thrml_wrapper.get_chain_diagnostics()
    
    return {
        "current_chains": chain_diag.get("current_n_chains", 1),
        "auto_detected_optimal": chain_diag.get("auto_detected_optimal", 1),
        "has_gpu": chain_diag.get("has_gpu", False),
        "enabled": sim_state.sampler_num_chains != 1 if sim_state else False
    }


@app.post("/sampler/chains")
async def set_chain_config(num_chains: int = -1):
    """
    Set number of parallel sampling chains.
    
    Args:
        num_chains: Number of chains (1 for single, -1 for auto-detect, >1 for specific count)
        
    Returns:
        Updated configuration
    """
    global sim_state
    
    if sim_state is None:
        return {"error": "System not initialized"}
    
    # Update system state
    sim_state = sim_state._replace(sampler_num_chains=num_chains)
    
    return {
        "num_chains": num_chains,
        "message": f"Set to {'auto-detect' if num_chains == -1 else f'{num_chains} chains'}"
    }


# ============================================================================
# Conditional Sampling (Clamped Nodes) Endpoints
# ============================================================================

@app.post("/sampler/clamp")
async def clamp_nodes(node_ids: List[int], values: List[float]):
    """
    Clamp (fix) specific nodes for conditional sampling.
    
    Args:
        node_ids: List of node indices to clamp
        values: List of values to clamp to
        
    Returns:
        Status of clamping operation
    """
    global thrml_wrapper, sim_state
    
    if thrml_wrapper is None:
        return {"error": "No sampler backend active"}
    
    if len(node_ids) != len(values):
        return {"error": "node_ids and values must have same length"}
    
    # Clamp nodes in THRML wrapper
    thrml_wrapper.set_clamped_nodes(
        node_ids=node_ids,
        values=values,
        node_positions=None  # Could extract from sim_state if needed
    )
    
    # Update system state
    if sim_state:
        sim_state = sim_state._replace(
            sampler_clamped_nodes={'node_ids': node_ids, 'values': values}
        )
    
    return {
        "clamped_count": len(node_ids),
        "node_ids": node_ids,
        "values": values,
        "message": f"Clamped {len(node_ids)} nodes"
    }


@app.get("/sampler/clamp")
async def get_clamped_nodes():
    """
    Get currently clamped nodes.
    
    Returns:
        Dict with clamped node IDs and values
    """
    global thrml_wrapper
    
    if thrml_wrapper is None:
        return {
            "clamped_count": 0,
            "node_ids": [],
            "values": []
        }
    
    node_ids, values = thrml_wrapper.get_clamped_nodes()
    
    return {
        "clamped_count": len(node_ids),
        "node_ids": node_ids,
        "values": values
    }


@app.delete("/sampler/clamp")
async def clear_clamped_nodes():
    """
    Clear all clamped nodes, returning to unconstrained sampling.
    
    Returns:
        Status message
    """
    global thrml_wrapper, sim_state
    
    if thrml_wrapper is None:
        return {"error": "No sampler backend active"}
    
    thrml_wrapper.clear_clamped_nodes(node_positions=None)
    
    # Update system state
    if sim_state:
        sim_state = sim_state._replace(sampler_clamped_nodes=None)
    
    return {
        "message": "Cleared all clamped nodes",
        "clamped_count": 0
    }


# ============================================================================
# THRML Blocking Strategy Endpoints
# ============================================================================

@app.get("/thrml/blocking-strategy")
async def get_blocking_strategy():
    """
    Get current blocking strategy.
    
    Returns:
        Dict with current strategy name and available strategies
    """
    global thrml_wrapper
    
    if thrml_wrapper is None:
        return {
            "current": "checkerboard",
            "available": ["checkerboard", "random", "stripes", "supercell", "graph-coloring"]
        }
    
    return {
        "current": thrml_wrapper.get_blocking_strategy(),
        "available": thrml_wrapper.get_supported_strategies()
    }


@app.post("/thrml/blocking-strategy")
async def set_blocking_strategy(strategy_name: str):
    """
    Set blocking strategy for parallel sampling.
    
    Args:
        strategy_name: Name of strategy ('checkerboard', 'random', 'stripes', 'supercell', 'graph-coloring')
        
    Returns:
        Status of operation
    """
    global thrml_wrapper, sim_state
    
    if thrml_wrapper is None:
        return {"error": "No sampler backend active"}
    
    available = thrml_wrapper.get_supported_strategies()
    if strategy_name not in available:
        return {
            "error": f"Unknown strategy: {strategy_name}",
            "available": available
        }
    
    # Set strategy
    thrml_wrapper.set_blocking_strategy(
        strategy_name=strategy_name,
        node_positions=None,  # Could extract from sim_state if needed
        connectivity=None
    )
    
    # Update system state
    if sim_state:
        sim_state = sim_state._replace(sampler_blocking_strategy=strategy_name)
    
    # Validate
    validation = thrml_wrapper.validate_current_blocks()
    
    return {
        "strategy": strategy_name,
        "valid": validation.get("valid", False),
        "balance_score": validation.get("balance_score", 0.0),
        "message": f"Set blocking strategy to {strategy_name}"
    }


@app.post("/thrml/sample")
async def trigger_manual_sample(n_steps: int = 10, temperature: float = 1.0):
    """
    Manually trigger a THRML sample.
    
    Args:
        n_steps: Number of Gibbs steps
        temperature: Sampling temperature
        
    Returns:
        Sampled binary states and diagnostics
    """
    global thrml_wrapper, sim_state
    
    if thrml_wrapper is None:
        raise HTTPException(status_code=404, detail="THRML wrapper not initialized")
    
    try:
        # Generate random key
        import jax.random
        key = jax.random.PRNGKey(int(np.random.randint(0, 2**31)))
        
        # Sample
        samples = thrml_wrapper.sample_gibbs(
            n_steps=n_steps,
            temperature=temperature,
            key=key,
            return_all_samples=False
        )
        
        # Compute energy
        energy = thrml_wrapper.compute_energy(samples)
        
        return {
            "status": "success",
            "samples": samples.tolist(),
            "energy": float(energy),
            "n_steps": n_steps,
            "temperature": temperature,
            "n_nodes": thrml_wrapper.n_nodes
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sampling failed: {str(e)}")


@app.get("/thrml/diagnostics")
async def get_thrml_diagnostics():
    """
    Get comprehensive THRML diagnostics including benchmark metrics.
    
    Returns:
        Dictionary with samples/sec, ESS/sec, autocorr, tau_int, energy, etc.
    """
    global thrml_wrapper, sim_state
    
    if thrml_wrapper is None:
        return {
            "available": False,
            "error": "THRML wrapper not initialized"
        }
    
    try:
        # Get benchmark diagnostics
        benchmark_diag = thrml_wrapper.get_benchmark_diagnostics()
        
        # Get health status
        health = thrml_wrapper.get_health_status()
        
        # Get chain diagnostics
        chain_diag = thrml_wrapper.get_chain_diagnostics()
        
        # Get device info
        devices = jax.devices()
        device_info = {
            'device_count': len(devices),
            'device_type': "gpu" if any("gpu" in str(d).lower() for d in devices) else "cpu",
            'devices': [str(d) for d in devices]
        }
        
        # Combine all diagnostics
        return {
            "available": True,
            "benchmark": benchmark_diag,
            "health": health,
            "chains": chain_diag,
            "device": device_info,
            "blocking": {
                "current_strategy": thrml_wrapper.get_blocking_strategy(),
                "supported_strategies": thrml_wrapper.get_supported_strategies()
            },
            "clamped_nodes": {
                "count": len(thrml_wrapper._clamped_node_ids),
                "node_ids": thrml_wrapper._clamped_node_ids
            }
        }
        
    except Exception as e:
        return {
            "available": False,
            "error": f"Failed to get diagnostics: {str(e)}"
        }


@app.get("/performance/info")
async def get_performance_info():
    """
    Get current performance configuration and device info.
    
    Returns:
        Performance configuration and device information
    """
    perf_config = get_perf_config()
    device_info = perf_config.get_device_info()
    
    return {
        "device_info": device_info,
        "available_modes": ["gpu_high_performance", "gpu_low_memory", "cpu_optimized", "debug"],
        "current_mode": "unknown"  # We don't track this globally yet
    }


@app.post("/performance/mode")
async def set_performance_mode(mode: str):
    """
    Set performance optimization mode.
    
    Args:
        mode: Performance mode ('gpu_high_performance', 'gpu_low_memory', 'cpu_optimized', 'debug')
        
    Returns:
        Confirmation message
    """
    try:
        apply_optimal_settings(mode)
        return {
            "status": "success",
            "mode": mode,
            "message": f"Performance mode set to '{mode}'"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set mode: {str(e)}")


@app.get("/thrml/health")
async def get_thrml_health():
    """
    Get THRML health status for monitoring.
    
    Returns:
        Health status dictionary
    """
    global thrml_wrapper
    
    if thrml_wrapper is None:
        return {
            "healthy": False,
            "reason": "THRML wrapper not initialized"
        }
    
    try:
        health = thrml_wrapper.get_health_status()
        return health
        
    except Exception as e:
        return {
            "healthy": False,
            "reason": f"Health check failed: {str(e)}"
        }


# ============================================================================
# Backend Selection Endpoints
# ============================================================================

@app.get("/sampler/backends")
async def get_available_backends():
    """
    Get list of available sampler backends with their capabilities.
    
    Returns:
        List of backend info dicts
    """
    from src.core.sampler_backend import SamplerBackendRegistry
    
    backends = SamplerBackendRegistry.get_available_backends()
    
    return {
        "backends": backends,
        "count": len(backends)
    }


@app.post("/sampler/backend")
async def set_sampler_backend(backend_type: str):
    """
    Switch to a different sampler backend.
    
    Args:
        backend_type: Type of backend ('thrml', 'photonic', 'neuromorphic', 'quantum')
        
    Returns:
        Status of backend switch
    """
    global sim_state, thrml_wrapper
    
    if sim_state is None:
        return {"error": "System not initialized"}
    
    from src.core.sampler_backend import SamplerBackendRegistry
    
    backend_class = SamplerBackendRegistry.get_backend(backend_type)
    if backend_class is None:
        available = SamplerBackendRegistry.list_backends()
        return {
            "error": f"Unknown backend: {backend_type}",
            "available": available
        }
    
    # For now, we only support THRML backend fully
    # Other backends would require more integration work
    if backend_type != "thrml":
        return {
            "error": f"Backend '{backend_type}' not yet fully integrated",
            "message": "Only 'thrml' backend is currently supported",
            "available": ["thrml"]
        }
    
    # Update system state
    sim_state = sim_state._replace(sampler_backend_type=backend_type)
    
    return {
        "backend": backend_type,
        "message": f"Switched to {backend_type} backend"
    }


# ============================================================================
# Include routers from separate files
# ============================================================================

from src.api.node_configs import router as configs_router
from src.api.external_models import router as external_router
from src.api.ml_endpoints import router as ml_router
from src.api.plugin_endpoints import router as plugin_router
from src.api.file_endpoints import router as file_router
# Temporarily disabled due to email-validator dependency issue
# from src.api.auth_endpoints import router as auth_router
from src.api.monitoring_endpoints import router as monitoring_router
from src.api.performance_endpoints import router as performance_router
from src.api.preset_endpoints import router as preset_router

app.include_router(configs_router)
app.include_router(external_router)
app.include_router(ml_router)
app.include_router(plugin_router)
app.include_router(file_router)
# app.include_router(auth_router)
app.include_router(monitoring_router)
app.include_router(performance_router)
app.include_router(preset_router)


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def schedule_thrml_rebuild() -> None:
    """Trigger (or queue) an asynchronous THRML wrapper rebuild."""
    global thrml_rebuild_task, thrml_rebuild_requested

    loop = asyncio.get_running_loop()
    if thrml_rebuild_task and not thrml_rebuild_task.done():
        thrml_rebuild_requested = True
        return

    thrml_rebuild_requested = False
    thrml_rebuild_task = loop.create_task(rebuild_thrml_wrapper_async())


async def rebuild_thrml_wrapper_async() -> None:
    """Rebuild the THRML wrapper off the event loop to avoid long blocking operations."""
    global sim_state, thrml_wrapper, thrml_rebuild_task, thrml_rebuild_requested, thrml_rebuild_lock

    if thrml_rebuild_lock is None:
        thrml_rebuild_lock = asyncio.Lock()

    async with thrml_rebuild_lock:
        current_state = sim_state
        if current_state is None:
            thrml_rebuild_task = None
            return

        n_active = int(jnp.sum(current_state.node_active_mask))
        if n_active <= 0:
            thrml_wrapper = None
            sim_state = current_state._replace(thrml_model_data=None)
            restart = thrml_rebuild_requested
            thrml_rebuild_requested = False
            thrml_rebuild_task = None
            if restart:
                schedule_thrml_rebuild()
            return

        weights = np.array(current_state.ebm_weights[:n_active, :n_active])
        temperature = float(current_state.thrml_temperature)

    loop = asyncio.get_running_loop()

    def build_wrapper():
        wrapper = create_thrml_model(
            n_nodes=n_active,
            weights=weights,
            biases=np.zeros(n_active),
            beta=1.0 / temperature,
        )
        return wrapper, wrapper.serialize()

    try:
        wrapper, serialized = await loop.run_in_executor(None, build_wrapper)
    except Exception as exc:
        print(f"[THRML] Failed to rebuild wrapper: {exc}")
        wrapper = None
        serialized = None

    restart = False
    async with thrml_rebuild_lock:
        if wrapper is not None:
            current_active = int(jnp.sum(sim_state.node_active_mask))
            if current_active == n_active:
                thrml_wrapper = wrapper
                sim_state = sim_state._replace(thrml_model_data=serialized)
                print(f"[THRML] Wrapper rebuilt asynchronously with {n_active} nodes")
            else:
                restart = True
        else:
            restart = True

        if thrml_rebuild_requested:
            restart = True
        thrml_rebuild_requested = False
        thrml_rebuild_task = None

    if restart:
        schedule_thrml_rebuild()


def _build_cors_headers(request) -> Dict[str, str]:
    origin = request.headers.get("origin")
    headers = {
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": request.headers.get("access-control-request-method", "GET,POST,PUT,PATCH,DELETE,OPTIONS"),
        "Access-Control-Allow-Headers": request.headers.get("access-control-request-headers", "*")
    }
    if origin:
        headers["Access-Control-Allow-Origin"] = origin
    else:
        headers["Access-Control-Allow-Origin"] = "*"
    return headers

