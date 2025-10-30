"""
Tests for SystemState initialization and management.
"""

import pytest
import jax
import jax.numpy as jnp
from src.core.state import (
    SystemState,
    initialize_system_state,
    get_active_node_count,
    validate_state,
    N_MAX,
    GRID_W,
    GRID_H,
    MAX_CHAIN_LEN,
)


def test_initialize_system_state():
    """Test that system state initializes with correct shapes and types."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Check types
    assert isinstance(state, SystemState)
    assert isinstance(state.key, jax.random.PRNGKeyArray)
    
    # Check time initialization
    assert state.t.shape == (1,)
    assert state.t[0] == 0.0
    assert state.dt == 0.01
    
    # Check oscillator state shape
    assert state.oscillator_state.shape == (N_MAX, 3)
    assert state.oscillator_state.dtype == jnp.float32
    
    # Check EBM weights shape
    assert state.ebm_weights.shape == (N_MAX, N_MAX)
    assert state.ebm_weights.dtype == jnp.float32
    # Check diagonal is zero (no self-connections)
    assert jnp.all(jnp.diag(state.ebm_weights) == 0.0)
    
    # Check wave field shapes
    assert state.field_p.shape == (GRID_W, GRID_H)
    assert state.field_p_prev.shape == (GRID_W, GRID_H)
    assert state.field_p.dtype == jnp.float32
    
    # Check GMCS chain shape and type
    assert state.gmcs_chain.shape == (N_MAX, MAX_CHAIN_LEN)
    assert state.gmcs_chain.dtype == jnp.int32
    
    # Check GMCS parameter shapes
    assert state.gmcs_A_max.shape == (N_MAX,)
    assert state.gmcs_R_comp.shape == (N_MAX,)
    assert state.gmcs_T_comp.shape == (N_MAX,)
    assert state.gmcs_Phi.shape == (N_MAX,)
    assert state.gmcs_omega.shape == (N_MAX,)
    assert state.gmcs_gamma.shape == (N_MAX,)
    assert state.gmcs_beta.shape == (N_MAX,)
    
    # Check topology shapes
    assert state.node_active_mask.shape == (N_MAX,)
    assert state.node_positions.shape == (N_MAX, 2)
    
    # Check audio control shapes
    assert state.k_strengths.shape == (N_MAX,)
    assert state.c_val.shape == (1,)
    
    # Check that all nodes start inactive
    assert jnp.sum(state.node_active_mask) == 0.0


def test_initialize_with_custom_params():
    """Test initialization with custom parameters."""
    key = jax.random.PRNGKey(123)
    n_max = 512
    grid_w = 128
    grid_h = 128
    dt = 0.005
    
    state = initialize_system_state(
        key, 
        n_max=n_max, 
        grid_w=grid_w, 
        grid_h=grid_h, 
        dt=dt
    )
    
    assert state.oscillator_state.shape == (n_max, 3)
    assert state.field_p.shape == (grid_w, grid_h)
    assert state.dt == dt


def test_state_immutability():
    """Test that SystemState is immutable (NamedTuple behavior)."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Attempt to modify should raise an error
    with pytest.raises(AttributeError):
        state.t = jnp.array([1.0])
    
    # Instead, use _replace for immutable updates
    new_state = state._replace(t=jnp.array([1.0]))
    assert new_state.t[0] == 1.0
    assert state.t[0] == 0.0  # Original unchanged


def test_state_device_placement():
    """Test that state can be placed on device."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Place on default device
    state_on_device = jax.device_put(state)
    
    # Check that it's still a valid SystemState
    assert isinstance(state_on_device, SystemState)
    assert state_on_device.oscillator_state.shape == (N_MAX, 3)


def test_get_active_node_count():
    """Test counting active nodes."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Initially no active nodes
    assert get_active_node_count(state) == 0
    
    # Activate 5 nodes
    mask = state.node_active_mask.at[:5].set(1.0)
    state = state._replace(node_active_mask=mask)
    assert get_active_node_count(state) == 5
    
    # Activate all nodes
    mask = jnp.ones((N_MAX,), dtype=jnp.float32)
    state = state._replace(node_active_mask=mask)
    assert get_active_node_count(state) == N_MAX


def test_validate_state_clean():
    """Test validation of clean state."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    assert validate_state(state) is True


def test_validate_state_with_nan():
    """Test validation catches NaN values."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Introduce NaN in oscillator state
    osc_state = state.oscillator_state.at[0, 0].set(jnp.nan)
    state = state._replace(oscillator_state=osc_state)
    
    assert validate_state(state) is False


def test_validate_state_with_inf():
    """Test validation catches Inf values."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Introduce Inf in wave field
    field = state.field_p.at[0, 0].set(jnp.inf)
    state = state._replace(field_p=field)
    
    assert validate_state(state) is False


def test_default_parameter_values():
    """Test that default GMCS parameters are sensible."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Check limiter parameter
    assert jnp.all(state.gmcs_A_max == 0.8)
    
    # Check compressor parameters
    assert jnp.all(state.gmcs_R_comp == 2.0)
    assert jnp.all(state.gmcs_T_comp == 0.5)
    
    # Check wave speed
    assert state.c_val[0] == 1.0
    
    # Check source strengths
    assert jnp.all(state.k_strengths == 1.0)


def test_node_positions_centered():
    """Test that inactive nodes default to center positions."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # All positions should be centered
    expected_x = GRID_W / 2.0
    expected_y = GRID_H / 2.0
    
    assert jnp.all(state.node_positions[:, 0] == expected_x)
    assert jnp.all(state.node_positions[:, 1] == expected_y)


def test_jit_compilation_with_state():
    """Test that functions using SystemState can be JIT compiled."""
    
    @jax.jit
    def increment_time(state: SystemState) -> SystemState:
        return state._replace(t=state.t + state.dt)
    
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # JIT compile and run
    new_state = increment_time(state)
    
    assert new_state.t[0] == pytest.approx(0.01)
    assert state.t[0] == 0.0  # Original unchanged


def test_state_pytree_structure():
    """Test that SystemState is a valid JAX PyTree."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Test tree_flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(state)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    
    assert isinstance(reconstructed, SystemState)
    assert reconstructed.t[0] == state.t[0]
    assert reconstructed.oscillator_state.shape == state.oscillator_state.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

