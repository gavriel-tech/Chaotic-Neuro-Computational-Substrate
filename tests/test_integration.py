"""
Integration tests for complete simulation step.
"""

import pytest
import jax
import jax.numpy as jnp
from src.core.state import initialize_system_state, N_MAX, GRID_W, GRID_H
from src.core.simulation import (
    simulation_step,
    simulation_step_with_ebm_learning,
    run_simulation,
    add_node_to_state,
    remove_node_from_state,
)
from src.core.gmcs_pipeline import create_default_chain


def test_simulation_step_shapes():
    """Test that simulation_step preserves shapes."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    new_state = simulation_step(state)
    
    # All shapes should be preserved
    assert new_state.oscillator_state.shape == (N_MAX, 3)
    assert new_state.field_p.shape == (GRID_W, GRID_H)
    assert new_state.field_p_prev.shape == (GRID_W, GRID_H)
    assert new_state.ebm_weights.shape == (N_MAX, N_MAX)


def test_simulation_step_time_advances():
    """Test that time advances correctly."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key, dt=0.01)
    
    assert state.t[0] == 0.0
    
    new_state = simulation_step(state)
    
    assert new_state.t[0] == pytest.approx(0.01)


def test_simulation_step_no_nan():
    """Test that simulation doesn't produce NaN."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Add one active node
    state, _ = add_node_to_state(state, (GRID_W/2, GRID_H/2))
    
    # Run 10 steps
    for _ in range(10):
        state = simulation_step(state)
        
        assert not jnp.any(jnp.isnan(state.oscillator_state))
        assert not jnp.any(jnp.isnan(state.field_p))
        assert not jnp.any(jnp.isnan(state.ebm_weights))


def test_simulation_step_with_active_node():
    """Test simulation with one active node."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Add active node at center
    state, node_id = add_node_to_state(state, (GRID_W/2, GRID_H/2), initial_perturbation=0.5)
    
    assert node_id == 0
    assert state.node_active_mask[node_id] == 1.0
    
    # Run simulation
    new_state = simulation_step(state)
    
    # Oscillator should have evolved
    assert not jnp.allclose(new_state.oscillator_state[node_id], state.oscillator_state[node_id])
    
    # Field should have changed (source term from oscillator)
    assert not jnp.allclose(new_state.field_p, state.field_p)


def test_simulation_step_inactive_nodes_stay_zero():
    """Test that inactive nodes don't evolve."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # All nodes inactive initially
    new_state = simulation_step(state)
    
    # Inactive nodes should stay at zero
    assert jnp.allclose(new_state.oscillator_state, 0.0, atol=1e-6)


def test_simulation_step_with_ebm_learning():
    """Test simulation step with EBM learning."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Add two active nodes
    state, _ = add_node_to_state(state, (GRID_W/2 - 10, GRID_H/2))
    state, _ = add_node_to_state(state, (GRID_W/2 + 10, GRID_H/2))
    
    initial_weights = state.ebm_weights.copy()
    
    # Run simulation with learning
    new_state = simulation_step_with_ebm_learning(state, eta=0.01)
    
    # Weights should have changed
    assert not jnp.allclose(new_state.ebm_weights, initial_weights, atol=1e-6)


def test_run_simulation_10_steps():
    """Test running simulation for 10 steps."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Add active node
    state, _ = add_node_to_state(state, (GRID_W/2, GRID_H/2), initial_perturbation=0.1)
    
    final_state, diagnostics = run_simulation(state, n_steps=10, enable_ebm=False)
    
    # Time should have advanced
    assert final_state.t[0] == pytest.approx(10 * state.dt)
    
    # Diagnostics should be collected
    assert len(diagnostics['times']) > 0
    assert len(diagnostics['max_osc_values']) > 0
    assert len(diagnostics['max_field_values']) > 0


def test_run_simulation_with_ebm():
    """Test running simulation with EBM learning."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Add two nodes
    state, _ = add_node_to_state(state, (GRID_W/2 - 10, GRID_H/2))
    state, _ = add_node_to_state(state, (GRID_W/2 + 10, GRID_H/2))
    
    initial_weights = state.ebm_weights.copy()
    
    final_state, diagnostics = run_simulation(
        state, 
        n_steps=50, 
        ebm_learning_interval=10,
        enable_ebm=True
    )
    
    # Weights should have changed
    assert not jnp.allclose(final_state.ebm_weights, initial_weights)
    
    # EBM energies should be recorded
    assert len(diagnostics['ebm_energies']) > 0


def test_add_node_to_state():
    """Test adding nodes to simulation."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Initially no active nodes
    assert jnp.sum(state.node_active_mask) == 0.0
    
    # Add first node
    state, node_id1 = add_node_to_state(state, (100.0, 100.0), initial_perturbation=0.2)
    
    assert node_id1 == 0
    assert state.node_active_mask[0] == 1.0
    assert state.node_positions[0, 0] == 100.0
    assert state.node_positions[0, 1] == 100.0
    assert state.oscillator_state[0, 0] == pytest.approx(0.2)
    
    # Add second node
    state, node_id2 = add_node_to_state(state, (150.0, 150.0))
    
    assert node_id2 == 1
    assert jnp.sum(state.node_active_mask) == 2.0


def test_add_node_with_gmcs_chain():
    """Test adding node with custom GMCS chain."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Create custom chain
    chain = create_default_chain("limiter")
    
    state, node_id = add_node_to_state(
        state, 
        (100.0, 100.0), 
        gmcs_chain=chain
    )
    
    # Chain should be set
    assert jnp.allclose(state.gmcs_chain[node_id], chain)


def test_remove_node_from_state():
    """Test removing nodes from simulation."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Add two nodes
    state, node_id1 = add_node_to_state(state, (100.0, 100.0))
    state, node_id2 = add_node_to_state(state, (150.0, 150.0))
    
    assert jnp.sum(state.node_active_mask) == 2.0
    
    # Remove first node
    state = remove_node_from_state(state, node_id1)
    
    assert state.node_active_mask[node_id1] == 0.0
    assert jnp.sum(state.node_active_mask) == 1.0
    
    # Second node should still be active
    assert state.node_active_mask[node_id2] == 1.0


def test_simulation_stability_100_steps():
    """Test simulation stability over 100 steps."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Add 5 nodes at different positions
    positions = [
        (GRID_W/2, GRID_H/2),
        (GRID_W/2 - 30, GRID_H/2),
        (GRID_W/2 + 30, GRID_H/2),
        (GRID_W/2, GRID_H/2 - 30),
        (GRID_W/2, GRID_H/2 + 30),
    ]
    
    for pos in positions:
        state, _ = add_node_to_state(state, pos, initial_perturbation=0.1)
    
    # Run 100 steps
    for i in range(100):
        state = simulation_step(state)
        
        # Check no NaN
        assert not jnp.any(jnp.isnan(state.oscillator_state)), f"NaN at step {i}"
        assert not jnp.any(jnp.isnan(state.field_p)), f"NaN in field at step {i}"
        
        # Check boundedness
        max_osc = jnp.max(jnp.abs(state.oscillator_state))
        max_field = jnp.max(jnp.abs(state.field_p))
        
        assert max_osc < 100.0, f"Oscillator unbounded at step {i}: {max_osc}"
        assert max_field < 100.0, f"Field unbounded at step {i}: {max_field}"


def test_field_wave_propagation():
    """Test that waves propagate in the field."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Add node with strong initial perturbation
    state, node_id = add_node_to_state(state, (GRID_W/2, GRID_H/2), initial_perturbation=1.0)
    
    # Initial field should be mostly zero
    initial_field_energy = jnp.sum(state.field_p ** 2)
    
    # Run simulation
    for _ in range(20):
        state = simulation_step(state)
    
    # Field energy should have increased (waves propagating)
    final_field_energy = jnp.sum(state.field_p ** 2)
    
    assert final_field_energy > initial_field_energy


def test_ebm_feedback_effect():
    """Test that EBM feedback affects oscillators."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Add two nodes
    state, _ = add_node_to_state(state, (GRID_W/2 - 10, GRID_H/2))
    state, _ = add_node_to_state(state, (GRID_W/2 + 10, GRID_H/2))
    
    # Run without EBM feedback
    state_no_ebm = state
    for _ in range(10):
        state_no_ebm = simulation_step(state_no_ebm, enable_ebm_feedback=False)
    
    # Run with EBM feedback
    state_with_ebm = state
    for _ in range(10):
        state_with_ebm = simulation_step(state_with_ebm, enable_ebm_feedback=True)
    
    # Trajectories should differ
    assert not jnp.allclose(
        state_no_ebm.oscillator_state, 
        state_with_ebm.oscillator_state,
        atol=0.01
    )


def test_jit_compilation():
    """Test that simulation_step is JIT compiled."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Should work (already jitted)
    new_state = simulation_step(state)
    
    assert new_state.t[0] > state.t[0]


def test_multiple_nodes_interact():
    """Test that multiple nodes interact through field."""
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key)
    
    # Add two nodes that will interact through wave field
    state, node1 = add_node_to_state(state, (GRID_W/2 - 20, GRID_H/2), initial_perturbation=0.5)
    state, node2 = add_node_to_state(state, (GRID_W/2 + 20, GRID_H/2), initial_perturbation=-0.5)
    
    # Run simulation
    for _ in range(30):
        state = simulation_step(state)
    
    # Both nodes should have evolved
    assert not jnp.allclose(state.oscillator_state[node1], jnp.array([0.5, 0.0, 0.0]))
    assert not jnp.allclose(state.oscillator_state[node2], jnp.array([-0.5, 0.0, 0.0]))
    
    # Field should show interference pattern
    max_field = jnp.max(jnp.abs(state.field_p))
    assert max_field > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

