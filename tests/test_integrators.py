"""
Tests for RK4 integration of Chua oscillators.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from src.core.integrators import (
    chua_derivatives,
    rk4_step_chua,
    integrate_all_oscillators,
    compute_energy,
    compute_attractor_bounds,
    check_stability,
    CHUA_A,
    CHUA_B,
    CHUA_C,
)


def test_chua_derivatives_shape():
    """Test that chua_derivatives returns correct shape."""
    state = jnp.array([0.1, 0.0, 0.0])
    driving_F = jnp.array(0.0)
    ebm_bias = jnp.array(0.0)
    
    derivatives = chua_derivatives(state, driving_F, ebm_bias)
    
    assert derivatives.shape == (3,)
    assert derivatives.dtype == jnp.float32 or derivatives.dtype == jnp.float64


def test_chua_derivatives_values():
    """Test chua_derivatives with known state."""
    x, y, z = 1.0, 0.5, 0.2
    state = jnp.array([x, y, z])
    driving_F = jnp.array(0.1)
    ebm_bias = jnp.array(0.05)
    
    derivatives = chua_derivatives(state, driving_F, ebm_bias)
    
    # Expected values based on equations
    expected_dx = -y - z + driving_F + ebm_bias
    expected_dy = x + CHUA_A * y
    expected_dz = CHUA_B + z * (x - CHUA_C)
    
    assert jnp.allclose(derivatives[0], expected_dx, atol=1e-6)
    assert jnp.allclose(derivatives[1], expected_dy, atol=1e-6)
    assert jnp.allclose(derivatives[2], expected_dz, atol=1e-6)


def test_rk4_step_chua_convergence():
    """Test RK4 step produces reasonable updates."""
    state = jnp.array([0.1, 0.0, 0.0])
    driving_F = jnp.array(0.0)
    ebm_bias = jnp.array(0.0)
    dt = 0.01
    
    new_state = rk4_step_chua(state, driving_F, ebm_bias, dt)
    
    # State should change but not drastically for small dt
    assert new_state.shape == (3,)
    assert not jnp.allclose(new_state, state)  # Should change
    assert jnp.allclose(new_state, state, atol=0.5)  # But not too much


def test_rk4_step_chua_no_nan():
    """Test that RK4 doesn't produce NaN for reasonable inputs."""
    state = jnp.array([0.5, 0.3, 0.1])
    driving_F = jnp.array(0.2)
    ebm_bias = jnp.array(0.1)
    dt = 0.01
    
    # Run multiple steps
    for _ in range(100):
        state = rk4_step_chua(state, driving_F, ebm_bias, dt)
        assert not jnp.any(jnp.isnan(state))
        assert not jnp.any(jnp.isinf(state))


def test_rk4_step_bounded_trajectory():
    """Test that Chua oscillator stays on bounded attractor."""
    state = jnp.array([0.1, 0.0, 0.0])
    driving_F = jnp.array(0.0)
    ebm_bias = jnp.array(0.0)
    dt = 0.01
    
    # Run for 1000 steps (10 time units)
    max_value = 0.0
    for _ in range(1000):
        state = rk4_step_chua(state, driving_F, ebm_bias, dt)
        max_value = max(max_value, float(jnp.max(jnp.abs(state))))
    
    # Chua attractor is bounded (typically within ±20)
    assert max_value < 50.0, f"Trajectory unbounded: max={max_value}"


def test_integrate_all_oscillators_shape():
    """Test vectorized integration of multiple oscillators."""
    n_osc = 10
    oscillator_states = jnp.zeros((n_osc, 3))
    # Add small perturbations
    oscillator_states = oscillator_states.at[:, 0].set(jnp.linspace(0.1, 0.5, n_osc))
    
    driving_forces = jnp.zeros(n_osc)
    ebm_biases = jnp.zeros(n_osc)
    dt = 0.01
    
    new_states = integrate_all_oscillators(
        oscillator_states, driving_forces, ebm_biases, dt
    )
    
    assert new_states.shape == (n_osc, 3)
    assert not jnp.any(jnp.isnan(new_states))


def test_integrate_all_oscillators_independence():
    """Test that oscillators evolve independently (no cross-talk)."""
    # Two oscillators with different initial conditions
    oscillator_states = jnp.array([
        [0.1, 0.0, 0.0],
        [0.5, 0.0, 0.0]
    ])
    
    driving_forces = jnp.array([0.0, 0.0])
    ebm_biases = jnp.array([0.0, 0.0])
    dt = 0.01
    
    new_states = integrate_all_oscillators(
        oscillator_states, driving_forces, ebm_biases, dt
    )
    
    # Evolve them separately
    new_state_0 = rk4_step_chua(oscillator_states[0], driving_forces[0], ebm_biases[0], dt)
    new_state_1 = rk4_step_chua(oscillator_states[1], driving_forces[1], ebm_biases[1], dt)
    
    # Should match (independence test)
    assert jnp.allclose(new_states[0], new_state_0, atol=1e-6)
    assert jnp.allclose(new_states[1], new_state_1, atol=1e-6)


def test_compute_energy_single():
    """Test energy computation for single oscillator."""
    state = jnp.array([1.0, 2.0, 3.0])
    energy = compute_energy(state)
    
    expected = 1.0**2 + 2.0**2 + 3.0**2
    assert jnp.allclose(energy, expected, atol=1e-6)


def test_compute_energy_multiple():
    """Test energy computation for multiple oscillators."""
    states = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0]
    ])
    
    energies = compute_energy(states)
    
    expected = jnp.array([1.0, 4.0, 9.0])
    assert jnp.allclose(energies, expected, atol=1e-6)


def test_compute_attractor_bounds():
    """Test bounding box computation."""
    oscillator_states = jnp.array([
        [1.0, 2.0, 3.0],
        [-1.0, -2.0, -3.0],
        [0.5, 1.0, 1.5]
    ])
    
    min_vals, max_vals = compute_attractor_bounds(oscillator_states)
    
    assert jnp.allclose(min_vals, jnp.array([-1.0, -2.0, -3.0]))
    assert jnp.allclose(max_vals, jnp.array([1.0, 2.0, 3.0]))


def test_check_stability_stable():
    """Test stability check for bounded states."""
    oscillator_states = jnp.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    
    assert check_stability(oscillator_states, max_value=10.0) is True


def test_check_stability_unstable():
    """Test stability check for unbounded states."""
    oscillator_states = jnp.array([
        [1.0, 2.0, 3.0],
        [150.0, 5.0, 6.0]  # One value exceeds limit
    ])
    
    assert check_stability(oscillator_states, max_value=100.0) is False


def test_rk4_driving_force_effect():
    """Test that driving force affects trajectory."""
    state = jnp.array([0.1, 0.0, 0.0])
    dt = 0.01
    
    # Run with no driving force
    state_no_drive = state
    for _ in range(100):
        state_no_drive = rk4_step_chua(state_no_drive, jnp.array(0.0), jnp.array(0.0), dt)
    
    # Run with driving force
    state_with_drive = state
    for _ in range(100):
        state_with_drive = rk4_step_chua(state_with_drive, jnp.array(1.0), jnp.array(0.0), dt)
    
    # Should produce different trajectories
    assert not jnp.allclose(state_no_drive, state_with_drive, atol=0.1)


def test_jit_compilation():
    """Test that functions are JIT compilable."""
    state = jnp.array([0.1, 0.0, 0.0])
    driving_F = jnp.array(0.0)
    ebm_bias = jnp.array(0.0)
    dt = 0.01
    
    # These should already be jitted, but test they work
    new_state = rk4_step_chua(state, driving_F, ebm_bias, dt)
    assert new_state.shape == (3,)
    
    derivatives = chua_derivatives(state, driving_F, ebm_bias)
    assert derivatives.shape == (3,)


def test_time_step_size_effect():
    """Test that smaller time steps give more accurate results."""
    state_init = jnp.array([0.1, 0.0, 0.0])
    driving_F = jnp.array(0.0)
    ebm_bias = jnp.array(0.0)
    
    # Integrate with large time step
    state_large = state_init
    dt_large = 0.1
    for _ in range(10):  # Total time = 1.0
        state_large = rk4_step_chua(state_large, driving_F, ebm_bias, dt_large)
    
    # Integrate with small time step
    state_small = state_init
    dt_small = 0.01
    for _ in range(100):  # Total time = 1.0
        state_small = rk4_step_chua(state_small, driving_F, ebm_bias, dt_small)
    
    # Smaller time step should give similar but more accurate result
    # They should be close but not identical
    assert jnp.allclose(state_large, state_small, atol=0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

