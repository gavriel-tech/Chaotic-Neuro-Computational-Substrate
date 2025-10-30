"""
Complete simulation step integration.

This module ties together all core modules (oscillators, wave PDE, GMCS pipeline,
EBM learning with THRML) into a single simulation step.

Note: The main simulation_step function is NOT jitted because it now uses THRML,
which requires Python-level operations. Individual components (Chua, wave PDE, GMCS)
remain jitted for performance.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Any

from .state import SystemState, N_MAX, GRID_W, GRID_H
from .integrators import integrate_all_oscillators
from .wave_pde import fdtd_step_wave, compute_pde_source, sample_field_at_nodes
from .gmcs_pipeline import gmcs_pipeline_dual
from .ebm import (
    compute_ebm_bias,
    compute_thrml_feedback,
    ebm_update_with_thrml,
    compute_ebm_energy_thrml
)
from .thrml_integration import THRMLWrapper, reconstruct_thrml_wrapper


def simulation_step(
    state: SystemState,
    enable_ebm_feedback: bool = True,
    thrml_wrapper: Optional[THRMLWrapper] = None,
    modulation_matrix: Optional[Any] = None  # ModulationMatrix instance
) -> Tuple[SystemState, Optional[THRMLWrapper]]:
    """
    Complete simulation step with THRML integration and modulation.
    
    Note: No longer jitted due to THRML Python operations. Individual
    components (Chua, wave PDE, GMCS) remain jitted.
    
    Integration order:
    0. Apply modulation matrix (if provided) → modulate parameters
    1. Sample wave field at node positions → h_i
    2. Run GMCS dual pipeline on sampled values → F_i (Chua), B_i (EBM bias)
    3. Compute THRML feedback (if enabled) → feedback_i
    4. Integrate Chua ODEs with driving forces → new x_i, y_i, z_i
    5. Compute PDE source term from oscillator states → S(r,t)
    6. Integrate wave PDE → new P(r,t)
    7. Apply node mask to zero inactive nodes
    8. Update time
    
    Args:
        state: Current SystemState
        enable_ebm_feedback: Whether to apply THRML feedback to oscillators
        thrml_wrapper: THRMLWrapper instance (reconstructed if None)
        modulation_matrix: ModulationMatrix instance for parameter modulation
        
    Returns:
        (new_state, thrml_wrapper) tuple
    """
    # === Step 0: Apply Modulation Matrix ===
    if modulation_matrix is not None and len(modulation_matrix.routes) > 0:
        state = modulation_matrix.apply_modulation(
            state,
            audio_pitch=state.audio_pitch,
            audio_rms=state.audio_rms
        )
    
    dt = state.dt
    t_scalar = state.t[0] if state.t.ndim > 0 else state.t
    
    # === Step 1: Sample field at node positions ===
    field_samples = sample_field_at_nodes(state.field_p, state.node_positions)
    
    # Apply gain for field coupling
    field_gain = 1.0
    h_inputs = field_samples * field_gain
    
    # === Step 2: GMCS Dual Pipeline ===
    # Stack all GMCS parameters into matrix (N_MAX, 16)
    params_matrix = jnp.stack([
        state.gmcs_A_max,
        state.gmcs_R_comp,
        state.gmcs_T_comp,
        state.gmcs_R_exp,
        state.gmcs_T_exp,
        state.gmcs_Phi,
        state.gmcs_omega,
        state.gmcs_gamma,
        state.gmcs_beta,
        state.gmcs_f0,
        state.gmcs_Q,
        state.gmcs_levels,
        state.gmcs_rate_limit,
        state.gmcs_n2,
        state.gmcs_V,
        state.gmcs_V_pi
    ], axis=1)
    
    # Get both continuous (F) and discrete (B) outputs
    F_all, B_all = gmcs_pipeline_dual(
        h_inputs,
        state.gmcs_chain,
        params_matrix,
        t_scalar
    )
    
    driving_forces = F_all  # Continuous output for Chua
    gmcs_ebm_bias = B_all   # Discrete output for EBM
    
    # === Step 3: THRML P-bit Sampling & Feedback ===
    if enable_ebm_feedback and state.thrml_enabled and thrml_wrapper is not None:
        # Split key for THRML sampling
        key, subkey = jax.random.split(state.key)
        
        # Sample THRML and compute feedback
        ebm_feedback = compute_thrml_feedback(
            thrml_wrapper,
            gmcs_ebm_bias,
            state.thrml_temperature,
            state.thrml_gibbs_steps,
            subkey
        )
        
        driving_forces = driving_forces + ebm_feedback
        
        # Update state key
        state = state._replace(key=key)
    else:
        ebm_feedback = jnp.zeros(N_MAX)
    
    # === Step 4: Integrate Chua ODEs ===
    new_oscillator_state = integrate_all_oscillators(
        state.oscillator_state,
        driving_forces,
        ebm_feedback,
        dt
    )
    
    # Apply mask to zero inactive nodes
    mask_3d = state.node_active_mask[:, None]  # (N_MAX, 1)
    new_oscillator_state = new_oscillator_state * mask_3d
    
    # === Step 5: Compute PDE Source Term ===
    source_term_S = compute_pde_source(
        new_oscillator_state[:, 0],  # x values only
        state.node_positions,
        state.node_active_mask,
        state.k_strengths,
        alpha=0.05,
        grid_w=GRID_W,
        grid_h=GRID_H
    )
    
    # === Step 6: Integrate Wave PDE ===
    c_val = state.c_val[0]
    new_field_p = fdtd_step_wave(
        state.field_p,
        state.field_p_prev,
        source_term_S,
        dt,
        c_val,
        dx=1.0
    )
    
    # === Step 7: Update Time ===
    new_t = state.t + dt
    
    # === Step 8: Return Updated State ===
    new_state = state._replace(
        t=new_t,
        oscillator_state=new_oscillator_state,
        field_p=new_field_p,
        field_p_prev=state.field_p
    )
    
    return new_state, thrml_wrapper


def simulation_step_with_thrml_learning(
    state: SystemState,
    thrml_wrapper: THRMLWrapper,
    eta: float = 0.01
) -> Tuple[SystemState, THRMLWrapper]:
    """
    Simulation step with THRML learning update.
    
    This combines simulation_step with THRML CD-k learning.
    Should be called less frequently than simulation_step (e.g., every N steps
    based on state.thrml_update_freq).
    
    Args:
        state: Current SystemState
        thrml_wrapper: THRMLWrapper instance
        eta: Learning rate (overrides state config if provided)
        
    Returns:
        (new_state, updated_thrml_wrapper) tuple
    """
    # Perform simulation step
    new_state, thrml_wrapper = simulation_step(
        state,
        enable_ebm_feedback=True,
        thrml_wrapper=thrml_wrapper
    )
    
    # Split key for learning
    key, subkey = jax.random.split(new_state.key)
    
    # Update THRML weights using CD-k
    thrml_wrapper = ebm_update_with_thrml(
        thrml_wrapper,
        new_state.oscillator_state,
        new_state.node_active_mask,
        eta,
        k_steps=state.thrml_cd_k,
        key=subkey,
        x_threshold=0.0
    )
    
    # Update EBM weights in state from THRML
    new_weights = thrml_wrapper.get_weights()
    
    # Pad to N_MAX if necessary
    weights_padded = np.zeros((N_MAX, N_MAX), dtype=np.float32)
    n_nodes = new_weights.shape[0]
    weights_padded[:n_nodes, :n_nodes] = new_weights
    
    # Update state with new weights and key
    new_state = new_state._replace(
        ebm_weights=jnp.array(weights_padded),
        key=key
    )
    
    # Serialize THRML wrapper back to state
    thrml_data = thrml_wrapper.serialize()
    new_state = new_state._replace(thrml_model_data=thrml_data)
    
    return new_state, thrml_wrapper


def run_simulation(
    state: SystemState,
    n_steps: int,
    thrml_wrapper: Optional[THRMLWrapper] = None,
    ebm_learning_interval: Optional[int] = None,
    ebm_learning_rate: float = 0.01,
    enable_ebm: bool = True
) -> Tuple[SystemState, THRMLWrapper, dict]:
    """
    Run simulation for multiple steps with THRML (non-jitted, for testing).
    
    Args:
        state: Initial SystemState
        n_steps: Number of steps to run
        thrml_wrapper: THRMLWrapper instance (created if None)
        ebm_learning_interval: Steps between THRML learning updates (uses state.thrml_update_freq if None)
        ebm_learning_rate: THRML learning rate
        enable_ebm: Whether to use THRML learning
        
    Returns:
        (final_state, thrml_wrapper, diagnostics) tuple
    """
    # Create THRML wrapper if not provided
    if thrml_wrapper is None and enable_ebm:
        # Reconstruct from state if available
        if state.thrml_model_data is not None:
            thrml_wrapper = reconstruct_thrml_wrapper(state.thrml_model_data)
        else:
            # Create new wrapper
            from .thrml_integration import create_thrml_model
            n_active = int(jnp.sum(state.node_active_mask))
            if n_active > 0:
                thrml_wrapper = create_thrml_model(
                    n_nodes=n_active,
                    weights=np.array(state.ebm_weights[:n_active, :n_active]),
                    biases=np.zeros(n_active),
                    beta=1.0 / state.thrml_temperature
                )
    
    # Use state's update frequency if not specified
    if ebm_learning_interval is None:
        ebm_learning_interval = state.thrml_update_freq
    
    diagnostics = {
        'max_osc_values': [],
        'max_field_values': [],
        'ebm_energies': [],
        'times': [],
    }
    
    current_state = state
    
    for step in range(n_steps):
        # Regular simulation step
        current_state, thrml_wrapper = simulation_step(
            current_state,
            enable_ebm_feedback=enable_ebm,
            thrml_wrapper=thrml_wrapper
        )
        
        # THRML learning (every ebm_learning_interval steps)
        if enable_ebm and thrml_wrapper is not None and step % ebm_learning_interval == 0 and step > 0:
            current_state, thrml_wrapper = simulation_step_with_thrml_learning(
                current_state,
                thrml_wrapper,
                eta=ebm_learning_rate
            )
            
            # Compute energy for diagnostics
            energy = compute_ebm_energy_thrml(
                thrml_wrapper,
                current_state.oscillator_state,
                current_state.node_active_mask
            )
            diagnostics['ebm_energies'].append(float(energy))
        
        # Collect diagnostics
        if step % 10 == 0:
            max_osc = float(jnp.max(jnp.abs(current_state.oscillator_state)))
            max_field = float(jnp.max(jnp.abs(current_state.field_p)))
            current_time = float(current_state.t[0])
            
            diagnostics['max_osc_values'].append(max_osc)
            diagnostics['max_field_values'].append(max_field)
            diagnostics['times'].append(current_time)
    
    return current_state, thrml_wrapper, diagnostics


def add_node_to_state(
    state: SystemState,
    position: Tuple[float, float],
    initial_perturbation: float = 0.1,
    gmcs_chain: jnp.ndarray = None
) -> Tuple[SystemState, int]:
    """
    Add a new active node to the simulation.
    
    Args:
        state: Current SystemState
        position: (x, y) position in grid coordinates
        initial_perturbation: Initial x value for oscillator
        gmcs_chain: Algorithm chain (None for default)
        
    Returns:
        (new_state, node_id) tuple
    """
    # Find first inactive slot
    mask_cpu = jnp.array(state.node_active_mask)
    inactive_indices = jnp.where(mask_cpu < 0.5)[0]
    
    if len(inactive_indices) == 0:
        raise ValueError("No inactive slots available (all nodes active)")
    
    idx = int(inactive_indices[0])
    
    # Set position
    new_positions = state.node_positions.at[idx].set(jnp.array(position))
    
    # Set initial oscillator state
    new_osc_state = state.oscillator_state.at[idx].set(
        jnp.array([initial_perturbation, 0.0, 0.0])
    )
    
    # Activate node
    new_mask = state.node_active_mask.at[idx].set(1.0)
    
    # Set GMCS chain if provided
    if gmcs_chain is not None:
        new_chain = state.gmcs_chain.at[idx].set(gmcs_chain)
    else:
        new_chain = state.gmcs_chain
    
    new_state = state._replace(
        node_positions=new_positions,
        oscillator_state=new_osc_state,
        node_active_mask=new_mask,
        gmcs_chain=new_chain
    )
    
    return new_state, idx


def remove_node_from_state(state: SystemState, node_id: int) -> SystemState:
    """
    Remove (deactivate) a node from the simulation.
    
    Args:
        state: Current SystemState
        node_id: Node index to deactivate
        
    Returns:
        New SystemState with node deactivated
    """
    if node_id < 0 or node_id >= N_MAX:
        raise ValueError(f"Invalid node_id: {node_id}")
    
    # Deactivate node
    new_mask = state.node_active_mask.at[node_id].set(0.0)
    
    return state._replace(node_active_mask=new_mask)

