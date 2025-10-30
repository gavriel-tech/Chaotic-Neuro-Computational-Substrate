"""
FastAPI server with simulation loop and WebSocket support.

This module creates the main server application, manages global simulation state,
runs the async simulation loop, and handles WebSocket connections for real-time
visualization.
"""

import asyncio
import json
from typing import Set, Dict, List, Optional, Any
from contextlib import asynccontextmanager

import jax
import jax.numpy as jnp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.core.state import initialize_system_state, SystemState, N_MAX
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

# Audio parameters (updated by audio thread)
audio_params: Dict[str, float] = {
    'rms': 0.0,
    'pitch': 440.0
}

# WebSocket clients
websocket_clients: Set[WebSocket] = set()

# Simulation control flags
simulation_running: bool = False  # Don't start automatically - wait for user to click Start
simulation_task: asyncio.Task = None


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
    global sim_state, thrml_wrapper, simulation_task
    
    print("Initializing GMCS simulation with THRML...")
    
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
    was_running = False
    last_loop_time = asyncio.get_event_loop().time()
    
    try:
        while True:
            if not simulation_running:
                if was_running:
                    # Reset timers so they don't accumulate during pause
                    sim_timer = 0.0
                    viz_timer = 0.0
                    was_running = False
                    print("[SIM] Simulation paused")
                await asyncio.sleep(0.05)
                last_loop_time = asyncio.get_event_loop().time()
                continue

            if not was_running:
                was_running = True
                sim_timer = 0.0
                viz_timer = 0.0
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

                viz_timer = 0.0
                viz_step_count += 1

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
    global sim_state, thrml_wrapper
    
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
        
        # Recreate THRML wrapper with new node count
        n_active = int(jnp.sum(sim_state.node_active_mask))
        if n_active > 0:
            print(f"[THRML] Creating THRML wrapper with {n_active} active nodes...")
            print(f"[THRML] THRML enabled: {sim_state.thrml_enabled}, temperature: {sim_state.thrml_temperature}")
            thrml_wrapper = create_thrml_model(
                n_nodes=n_active,
                weights=np.array(sim_state.ebm_weights[:n_active, :n_active]),
                biases=np.zeros(n_active),
                beta=1.0 / sim_state.thrml_temperature
            )
            print(f"[THRML] Wrapper created successfully with {thrml_wrapper.n_nodes} nodes")
            print(f"[THRML] Wrapper beta: {thrml_wrapper.beta}, n_edges: {len(thrml_wrapper.edges)}")
            
            # Serialize to state
            thrml_data = thrml_wrapper.serialize()
            sim_state = sim_state._replace(thrml_model_data=thrml_data)
            print(f"[THRML] Wrapper serialized to state")
        
        return NodeResponse(
            status="success",
            node_id=node_id,
            message=f"Node {node_id} added at position {request.position}"
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
            if node_id >= N_MAX:
                raise ValueError(f"Invalid node ID: {node_id}")
        
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
        
        return NodeResponse(
            status="success",
            message=f"Updated {len(request.node_ids)} node(s)"
        )
        
    except Exception as e:
        return NodeResponse(
            status="error",
            message=str(e)
        )


@app.delete("/node/{node_id}", response_model=NodeResponse)
async def remove_node(node_id: int):
    """Remove (deactivate) a node."""
    global sim_state
    
    try:
        if node_id < 0 or node_id >= N_MAX:
            raise ValueError(f"Invalid node ID: {node_id}")
        
        sim_state = remove_node_from_state(sim_state, node_id)
        sim_state = jax.device_put(sim_state)
        
        return NodeResponse(
            status="success",
            node_id=node_id,
            message=f"Node {node_id} removed"
        )
        
    except Exception as e:
        return NodeResponse(
            status="error",
            message=str(e)
        )


@app.get("/node/{node_id}", response_model=NodeInfoResponse)
async def get_node_info(node_id: int):
    """Get information about a specific node."""
    if sim_state is None:
        raise HTTPException(status_code=503, detail="Simulation not initialized")
    
    if node_id < 0 or node_id >= N_MAX:
        raise HTTPException(status_code=400, detail=f"Invalid node ID: {node_id}")
    
    import numpy as np
    
    active = bool(sim_state.node_active_mask[node_id] > 0.5)
    position = [float(sim_state.node_positions[node_id, 0]), float(sim_state.node_positions[node_id, 1])]
    osc_state = [float(sim_state.oscillator_state[node_id, i]) for i in range(3)]
    chain = [int(x) for x in np.array(sim_state.gmcs_chain[node_id])]
    
    config = {
        'A_max': float(sim_state.gmcs_A_max[node_id]),
        'R_comp': float(sim_state.gmcs_R_comp[node_id]),
        'T_comp': float(sim_state.gmcs_T_comp[node_id]),
        'Phi': float(sim_state.gmcs_Phi[node_id]),
        'omega': float(sim_state.gmcs_omega[node_id]),
        'gamma': float(sim_state.gmcs_gamma[node_id]),
        'beta': float(sim_state.gmcs_beta[node_id]),
    }
    
    return NodeInfoResponse(
        node_id=node_id,
        active=active,
        position=position,
        oscillator_state=osc_state,
        config=config,
        chain=chain
    )


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
    
    try:
        while True:
            # Listen for control messages from client
            message = await websocket.receive()
            
            if 'text' in message:
                # JSON control message
                try:
                    data = json.loads(message['text'])
                    
                    if data.get('type') == 'PING':
                        # Send pong with stats
                        await websocket.send_json({
                            'type': 'PONG',
                            'active_nodes': int(jnp.sum(sim_state.node_active_mask)),
                            'sim_time': float(sim_state.t[0]),
                            'simulation_running': simulation_running
                        })
                    
                except json.JSONDecodeError:
                    pass
    
    except WebSocketDisconnect:
        websocket_clients.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        websocket_clients.discard(websocket)


# ============================================================================
# Production Hardening: Health Check, Rate Limiting, Error Handling
# ============================================================================

import time
import logging
from collections import defaultdict
from datetime import datetime, timedelta

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting state
rate_limit_state = defaultdict(lambda: {'count': 0, 'reset_time': datetime.now()})
RATE_LIMIT_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_ENABLED = True  # Set via env var in production


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
    
    if now > state['reset_time']:
        # Reset window
        state['count'] = 0
        state['reset_time'] = now + timedelta(minutes=1)
    
    state['count'] += 1
    
    if state['count'] > RATE_LIMIT_REQUESTS_PER_MINUTE:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": int((state['reset_time'] - now).total_seconds())
            }
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
    import os
    
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
    """Start the simulation loop."""
    global simulation_running
    
    if simulation_running:
        return {
            "status": "already_running",
            "message": "Simulation is already running",
            "running": simulation_running
        }
    
    simulation_running = True
    return {
        "status": "started",
        "message": "Simulation started",
        "running": simulation_running
    }


@app.post("/simulation/stop")
async def stop_simulation():
    """Stop the simulation loop."""
    global simulation_running
    
    if not simulation_running:
        return {
            "status": "already_stopped",
            "message": "Simulation is already stopped",
            "running": simulation_running
        }
    
    simulation_running = False
    return {
        "status": "stopped",
        "message": "Simulation stopped",
        "running": simulation_running
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
async def get_thrml_pbit_states(max_history: int = 128):
    """
    Get current THRML p-bit states and optional history for visualizers.
    """
    global thrml_wrapper, sim_state

    if sim_state is None:
        return {
            "current": [],
            "history": [],
            "grid_size": 0,
            "history_length": 0,
            "timestamp": 0.0
        }

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

    sample = get_last_thrml_sample()

    # If no sample recorded yet, generate one so the visualizer has data
    if sample is None and thrml_wrapper is not None:
        try:
            key = jax.random.PRNGKey(np.random.randint(0, 2**31 - 1))
            sample = thrml_wrapper.sample_gibbs(
                n_steps=max(5, sim_state.thrml_gibbs_steps),
                temperature=sim_state.thrml_temperature,
                key=key
            )
            update_thrml_sample(sample)
            print("[THRML] P-bit endpoint: Generated fresh sample for visualizer")
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
# Include routers from separate files
# ============================================================================

from src.api.node_configs import router as configs_router
from src.api.external_models import router as external_router
from src.api.ml_endpoints import router as ml_router
from src.api.plugin_endpoints import router as plugin_router

app.include_router(configs_router)
app.include_router(external_router)
app.include_router(ml_router)
app.include_router(plugin_router)


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

