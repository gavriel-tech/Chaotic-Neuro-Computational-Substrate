"""
SystemState definition and initialization for GMCS.

This module defines the immutable PyTree structure that holds all simulation state,
following JAX's functional programming paradigm.
"""

from typing import NamedTuple, Optional, Dict, Any
import jax
import jax.numpy as jnp

# System constants
N_MAX = 1024  # Maximum number of oscillator nodes
GRID_W = 256  # PDE grid width
GRID_H = 256  # PDE grid height
MAX_CHAIN_LEN = 8  # Maximum algorithms per node


class SystemState(NamedTuple):
    """
    Complete simulation state as immutable PyTree.
    
    All arrays are pre-allocated to maximum capacity. Active nodes are controlled
    via node_active_mask. This design allows JAX JIT compilation without dynamic
    array resizing.
    
    Attributes:
        key: PRNG key for random operations
        t: Current simulation time (scalar in array)
        dt: Time step size
        
        oscillator_state: (N_MAX, 3) Chua oscillator [x, y, z] states
        ebm_weights: (N_MAX, N_MAX) Learned coupling weights
        
        field_p: (GRID_W, GRID_H) Current wave field
        field_p_prev: (GRID_W, GRID_H) Previous wave field (for FDTD)
        
        gmcs_chain: (N_MAX, MAX_CHAIN_LEN) Algorithm IDs per node
        gmcs_A_max: (N_MAX,) Limiter parameter
        gmcs_R_comp: (N_MAX,) Compressor ratio
        gmcs_T_comp: (N_MAX,) Compressor threshold
        gmcs_R_exp: (N_MAX,) Expander ratio
        gmcs_T_exp: (N_MAX,) Expander threshold
        gmcs_Phi: (N_MAX,) Phase modulation depth
        gmcs_omega: (N_MAX,) Phase modulation frequency
        gmcs_gamma: (N_MAX,) Fold amplitude
        gmcs_beta: (N_MAX,) Fold frequency
        
        node_active_mask: (N_MAX,) Boolean mask (1.0=active, 0.0=inactive)
        node_positions: (N_MAX, 2) [x, y] positions in grid coordinates
        
        k_strengths: (N_MAX,) Source strength per node (audio controlled)
        c_val: (1,) Wave speed (audio controlled)
    """
    
    # PRNG and time
    key: jax.random.PRNGKey
    t: jnp.ndarray  # Scalar in array for consistency
    dt: float
    
    # Layer I: Oscillators & EBM
    oscillator_state: jnp.ndarray  # (N_MAX, 3)
    ebm_weights: jnp.ndarray  # (N_MAX, N_MAX)
    
    # Layer II: Wave Field
    field_p: jnp.ndarray  # (GRID_W, GRID_H)
    field_p_prev: jnp.ndarray  # (GRID_W, GRID_H)
    
    # Layer III: GMCS Control (Basic parameters 0-8)
    gmcs_chain: jnp.ndarray  # (N_MAX, MAX_CHAIN_LEN) int32
    gmcs_A_max: jnp.ndarray  # (N_MAX,)
    gmcs_R_comp: jnp.ndarray  # (N_MAX,)
    gmcs_T_comp: jnp.ndarray  # (N_MAX,)
    gmcs_R_exp: jnp.ndarray  # (N_MAX,)
    gmcs_T_exp: jnp.ndarray  # (N_MAX,)
    gmcs_Phi: jnp.ndarray  # (N_MAX,)
    gmcs_omega: jnp.ndarray  # (N_MAX,)
    gmcs_gamma: jnp.ndarray  # (N_MAX,)
    gmcs_beta: jnp.ndarray  # (N_MAX,)
    
    # Layer III: GMCS Extended Parameters (9-15)
    gmcs_f0: jnp.ndarray  # (N_MAX,) Resonator frequency
    gmcs_Q: jnp.ndarray  # (N_MAX,) Resonator Q factor
    gmcs_levels: jnp.ndarray  # (N_MAX,) Quantizer levels
    gmcs_rate_limit: jnp.ndarray  # (N_MAX,) Slew limiter rate
    gmcs_n2: jnp.ndarray  # (N_MAX,) Kerr coefficient
    gmcs_V: jnp.ndarray  # (N_MAX,) Electro-optic voltage
    gmcs_V_pi: jnp.ndarray  # (N_MAX,) Electro-optic half-wave voltage
    
    # Topology
    node_active_mask: jnp.ndarray  # (N_MAX,)
    node_positions: jnp.ndarray  # (N_MAX, 2)
    
    # Audio control
    k_strengths: jnp.ndarray  # (N_MAX,)
    c_val: jnp.ndarray  # (1,) scalar in array
    
    # THRML Integration (Legacy - kept for backward compatibility)
    thrml_model_data: Optional[Dict[str, Any]]  # Serialized THRML model (for JAX compatibility)
    thrml_enabled: bool  # Always True now (THRML required)
    
    # Sampler Backend (Universal)
    sampler_backend_type: str  # 'thrml', 'photonic', 'neuromorphic', 'quantum'
    sampler_backend_data: Optional[Dict[str, Any]]  # Serialized backend state
    sampler_num_chains: int  # Number of parallel chains (-1 for auto-detect)
    sampler_blocking_strategy: str  # 'checkerboard', 'random', 'stripes', 'supercell', 'graph-coloring', 'auto'
    sampler_auto_adapt_strategy: bool  # Enable adaptive strategy selection
    sampler_clamped_nodes: Optional[Dict[str, Any]]  # Serialized clamped node data
    sampler_performance_history: Optional[Dict[str, Any]]  # Performance metrics history
    
    # Modulation Matrix
    modulation_routes: Optional[Dict[str, Any]]  # Serialized modulation routes
    audio_pitch: float  # Current audio pitch for modulation
    audio_rms: float  # Current audio RMS for modulation
    thrml_temperature: float  # Sampling temperature
    thrml_gibbs_steps: int  # Steps per simulation update
    thrml_cd_k: int  # CD-k for learning
    thrml_performance_mode: str  # 'speed', 'accuracy', 'research'
    thrml_update_freq: int  # How often to update weights (in steps)


def initialize_system_state(
    key: jax.random.PRNGKey,
    n_max: int = N_MAX,
    grid_w: int = GRID_W,
    grid_h: int = GRID_H,
    dt: float = 0.01,
) -> SystemState:
    """
    Create initial system state with all nodes inactive.
    
    Args:
        key: JAX PRNG key
        n_max: Maximum number of nodes
        grid_w: Grid width
        grid_h: Grid height
        dt: Time step size
        
    Returns:
        SystemState with pre-allocated arrays initialized to defaults
    """
    # Split keys for different random initializations
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Initialize time
    t = jnp.array([0.0])
    
    # Initialize oscillator states (all zeros, inactive)
    oscillator_state = jnp.zeros((n_max, 3), dtype=jnp.float32)
    
    # Initialize EBM weights (small random values)
    ebm_weights = jax.random.normal(subkey1, (n_max, n_max), dtype=jnp.float32) * 0.01
    # Zero diagonal (no self-connections)
    ebm_weights = ebm_weights.at[jnp.diag_indices(n_max)].set(0.0)
    
    # Initialize wave fields
    field_p = jnp.zeros((grid_w, grid_h), dtype=jnp.float32)
    field_p_prev = jnp.zeros((grid_w, grid_h), dtype=jnp.float32)
    
    # Initialize GMCS chains (all zeros = NOP)
    gmcs_chain = jnp.zeros((n_max, MAX_CHAIN_LEN), dtype=jnp.int32)
    
    # Initialize GMCS basic parameters (0-8) with sensible defaults
    gmcs_A_max = jnp.full((n_max,), 0.8, dtype=jnp.float32)  # Limiter ceiling
    gmcs_R_comp = jnp.full((n_max,), 2.0, dtype=jnp.float32)  # Compressor ratio
    gmcs_T_comp = jnp.full((n_max,), 0.5, dtype=jnp.float32)  # Compressor threshold
    gmcs_R_exp = jnp.full((n_max,), 2.0, dtype=jnp.float32)  # Expander ratio
    gmcs_T_exp = jnp.full((n_max,), 0.2, dtype=jnp.float32)  # Expander threshold
    gmcs_Phi = jnp.zeros((n_max,), dtype=jnp.float32)  # Phase mod depth
    gmcs_omega = jnp.full((n_max,), 1.0, dtype=jnp.float32)  # Phase mod freq
    gmcs_gamma = jnp.full((n_max,), 1.0, dtype=jnp.float32)  # Fold amplitude
    gmcs_beta = jnp.full((n_max,), 1.0, dtype=jnp.float32)  # Fold frequency
    
    # Initialize GMCS extended parameters (9-15)
    gmcs_f0 = jnp.full((n_max,), 440.0, dtype=jnp.float32)  # Resonator freq (Hz)
    gmcs_Q = jnp.full((n_max,), 10.0, dtype=jnp.float32)  # Resonator Q
    gmcs_levels = jnp.full((n_max,), 16.0, dtype=jnp.float32)  # Quantizer levels
    gmcs_rate_limit = jnp.full((n_max,), 1.0, dtype=jnp.float32)  # Slew rate
    gmcs_n2 = jnp.full((n_max,), 2.6e-20, dtype=jnp.float32)  # Kerr coeff (m²/W)
    gmcs_V = jnp.zeros((n_max,), dtype=jnp.float32)  # Electro-optic voltage
    gmcs_V_pi = jnp.full((n_max,), 5.0, dtype=jnp.float32)  # Half-wave voltage
    
    # Initialize topology (all inactive, centered positions)
    node_active_mask = jnp.zeros((n_max,), dtype=jnp.float32)
    node_positions = jnp.stack([
        jnp.full((n_max,), grid_w / 2.0, dtype=jnp.float32),
        jnp.full((n_max,), grid_h / 2.0, dtype=jnp.float32)
    ], axis=1)
    
    # Initialize audio control
    k_strengths = jnp.ones((n_max,), dtype=jnp.float32)  # Source strengths
    c_val = jnp.array([1.0], dtype=jnp.float32)  # Wave speed
    
    # Initialize THRML settings (defaults to 'speed' mode)
    thrml_model_data = None  # Will be created when nodes are added
    thrml_enabled = True
    thrml_temperature = 1.0
    thrml_gibbs_steps = 5  # Speed mode default
    thrml_cd_k = 1  # CD-1 for speed
    thrml_performance_mode = 'speed'
    thrml_update_freq = 20  # Update every 20 steps
    
    # Initialize sampler backend (universal)
    sampler_backend_type = 'thrml'  # Default to THRML
    sampler_backend_data = None  # Will be set when backend is created
    sampler_num_chains = -1  # Auto-detect optimal chain count
    sampler_blocking_strategy = 'checkerboard'  # Default strategy
    sampler_auto_adapt_strategy = False  # Disabled by default
    sampler_clamped_nodes = None  # No clamped nodes initially
    sampler_performance_history = None  # Will accumulate during simulation
    
    # Initialize modulation matrix
    modulation_routes = None  # Will be set by modulation matrix
    audio_pitch = 440.0  # Default A4
    audio_rms = 0.0  # No audio initially
    
    return SystemState(
        key=key,
        t=t,
        dt=dt,
        oscillator_state=oscillator_state,
        ebm_weights=ebm_weights,
        field_p=field_p,
        field_p_prev=field_p_prev,
        gmcs_chain=gmcs_chain,
        gmcs_A_max=gmcs_A_max,
        gmcs_R_comp=gmcs_R_comp,
        gmcs_T_comp=gmcs_T_comp,
        gmcs_R_exp=gmcs_R_exp,
        gmcs_T_exp=gmcs_T_exp,
        gmcs_Phi=gmcs_Phi,
        gmcs_omega=gmcs_omega,
        gmcs_gamma=gmcs_gamma,
        gmcs_beta=gmcs_beta,
        gmcs_f0=gmcs_f0,
        gmcs_Q=gmcs_Q,
        gmcs_levels=gmcs_levels,
        gmcs_rate_limit=gmcs_rate_limit,
        gmcs_n2=gmcs_n2,
        gmcs_V=gmcs_V,
        gmcs_V_pi=gmcs_V_pi,
        node_active_mask=node_active_mask,
        node_positions=node_positions,
        k_strengths=k_strengths,
        c_val=c_val,
        thrml_model_data=thrml_model_data,
        thrml_enabled=thrml_enabled,
        sampler_backend_type=sampler_backend_type,
        sampler_backend_data=sampler_backend_data,
        sampler_num_chains=sampler_num_chains,
        sampler_blocking_strategy=sampler_blocking_strategy,
        sampler_auto_adapt_strategy=sampler_auto_adapt_strategy,
        sampler_clamped_nodes=sampler_clamped_nodes,
        sampler_performance_history=sampler_performance_history,
        thrml_temperature=thrml_temperature,
        thrml_gibbs_steps=thrml_gibbs_steps,
        thrml_cd_k=thrml_cd_k,
        thrml_performance_mode=thrml_performance_mode,
        thrml_update_freq=thrml_update_freq,
        modulation_routes=modulation_routes,
        audio_pitch=audio_pitch,
        audio_rms=audio_rms,
    )


def get_active_node_count(state: SystemState) -> int:
    """Get number of currently active nodes."""
    return int(jnp.sum(state.node_active_mask))


def validate_state(state: SystemState) -> bool:
    """
    Validate state for NaN/Inf values.
    
    Returns:
        True if state is valid (no NaN/Inf)
    """
    # Check oscillator states
    if jnp.any(jnp.isnan(state.oscillator_state)) or jnp.any(jnp.isinf(state.oscillator_state)):
        return False
    
    # Check wave fields
    if jnp.any(jnp.isnan(state.field_p)) or jnp.any(jnp.isinf(state.field_p)):
        return False
    
    # Check EBM weights
    if jnp.any(jnp.isnan(state.ebm_weights)) or jnp.any(jnp.isinf(state.ebm_weights)):
        return False
    
    return True

