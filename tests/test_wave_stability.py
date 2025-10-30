"""
Tests for FDTD wave equation solver stability and correctness.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from src.core.wave_pde import (
    laplacian_2d,
    fdtd_step_wave,
    compute_pde_source,
    check_cfl_condition,
    get_cfl_number,
    apply_damping,
    compute_field_energy,
    initialize_gaussian_pulse,
    sample_field_at_point,
    sample_field_at_nodes,
)


def test_laplacian_2d_constant_field():
    """Test Laplacian of constant field is zero."""
    field = jnp.ones((10, 10))
    laplacian = laplacian_2d(field, dx=1.0)
    
    # Interior points should have zero Laplacian
    # (boundaries may differ due to fill value)
    assert jnp.allclose(laplacian[2:-2, 2:-2], 0.0, atol=1e-5)


def test_laplacian_2d_quadratic():
    """Test Laplacian of quadratic field."""
    x = jnp.arange(10, dtype=jnp.float32)
    y = jnp.arange(10, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    
    # f(x,y) = x² + y²
    # ∇²f = 2 + 2 = 4
    field = xx ** 2 + yy ** 2
    laplacian = laplacian_2d(field, dx=1.0)
    
    # Interior points should be close to 4
    assert jnp.allclose(laplacian[2:-2, 2:-2], 4.0, atol=0.5)


def test_fdtd_step_no_source():
    """Test FDTD step with no source term."""
    grid_size = 64
    p_t = jnp.zeros((grid_size, grid_size))
    p_tm1 = jnp.zeros((grid_size, grid_size))
    source_S = jnp.zeros((grid_size, grid_size))
    
    dt = 0.01
    c = 1.0
    dx = 1.0
    
    p_tp1 = fdtd_step_wave(p_t, p_tm1, source_S, dt, c, dx)
    
    # With zero initial conditions, should stay zero
    assert jnp.allclose(p_tp1, 0.0, atol=1e-6)


def test_fdtd_step_gaussian_pulse():
    """Test FDTD step with Gaussian pulse."""
    grid_size = 64
    
    # Initialize Gaussian pulse
    p_t = initialize_gaussian_pulse(grid_size, grid_size, grid_size/2, grid_size/2, sigma=5.0)
    p_tm1 = p_t.copy()
    source_S = jnp.zeros((grid_size, grid_size))
    
    dt = 0.01
    c = 1.0
    dx = 1.0
    
    # Check CFL condition
    assert check_cfl_condition(c, dt, dx)
    
    p_tp1 = fdtd_step_wave(p_t, p_tm1, source_S, dt, c, dx)
    
    # Field should change (pulse spreads)
    assert not jnp.allclose(p_tp1, p_t, atol=0.01)
    # Should not explode
    assert jnp.max(jnp.abs(p_tp1)) < 10.0


def test_fdtd_stability_100_steps():
    """Test FDTD stability over 100 steps."""
    grid_size = 64
    
    p_t = initialize_gaussian_pulse(grid_size, grid_size, grid_size/2, grid_size/2)
    p_tm1 = p_t.copy()
    source_S = jnp.zeros((grid_size, grid_size))
    
    dt = 0.01
    c = 1.0
    dx = 1.0
    
    # Verify CFL condition
    assert check_cfl_condition(c, dt, dx), f"CFL violation: C={get_cfl_number(c, dt, dx)}"
    
    # Run 100 steps
    for i in range(100):
        p_tp1 = fdtd_step_wave(p_t, p_tm1, source_S, dt, c, dx)
        
        # Check for NaN/Inf
        assert not jnp.any(jnp.isnan(p_tp1)), f"NaN at step {i}"
        assert not jnp.any(jnp.isinf(p_tp1)), f"Inf at step {i}"
        
        # Check boundedness
        max_val = jnp.max(jnp.abs(p_tp1))
        assert max_val < 10.0, f"Unbounded at step {i}: max={max_val}"
        
        # Update for next step
        p_tm1 = p_t
        p_t = p_tp1


def test_check_cfl_condition_stable():
    """Test CFL condition checker for stable parameters."""
    c = 1.0
    dt = 0.005
    dx = 1.0
    
    C = get_cfl_number(c, dt, dx)
    assert C < 0.707
    assert check_cfl_condition(c, dt, dx)


def test_check_cfl_condition_unstable():
    """Test CFL condition checker for unstable parameters."""
    c = 1.0
    dt = 0.9  # Too large
    dx = 1.0
    
    C = get_cfl_number(c, dt, dx)
    assert C > 0.707
    assert not check_cfl_condition(c, dt, dx)


def test_compute_pde_source_single_node():
    """Test PDE source computation for single active node."""
    n_max = 10
    grid_w, grid_h = 32, 32
    
    x_states = jnp.zeros(n_max)
    x_states = x_states.at[0].set(1.0)  # One oscillator with amplitude 1
    
    positions = jnp.full((n_max, 2), 16.0)  # All at center
    
    mask = jnp.zeros(n_max)
    mask = mask.at[0].set(1.0)  # Only first node active
    
    k_strengths = jnp.ones(n_max)
    
    alpha = 0.05
    
    source = compute_pde_source(x_states, positions, mask, k_strengths, alpha, grid_w, grid_h)
    
    assert source.shape == (grid_w, grid_h)
    
    # Source should be maximum at center
    max_pos = jnp.unravel_index(jnp.argmax(source), source.shape)
    assert abs(max_pos[0] - 16) < 2
    assert abs(max_pos[1] - 16) < 2
    
    # Source should be non-negative
    assert jnp.all(source >= 0.0)


def test_compute_pde_source_multiple_nodes():
    """Test PDE source with multiple active nodes."""
    n_max = 10
    grid_w, grid_h = 32, 32
    
    # Three active oscillators
    x_states = jnp.zeros(n_max)
    x_states = x_states.at[:3].set(jnp.array([1.0, 0.5, 0.8]))
    
    # At different positions
    positions = jnp.zeros((n_max, 2))
    positions = positions.at[0].set(jnp.array([10.0, 10.0]))
    positions = positions.at[1].set(jnp.array([20.0, 20.0]))
    positions = positions.at[2].set(jnp.array([15.0, 15.0]))
    
    mask = jnp.zeros(n_max)
    mask = mask.at[:3].set(1.0)
    
    k_strengths = jnp.ones(n_max)
    alpha = 0.05
    
    source = compute_pde_source(x_states, positions, mask, k_strengths, alpha, grid_w, grid_h)
    
    # Should have multiple peaks
    assert source.shape == (grid_w, grid_h)
    assert jnp.max(source) > 0.0


def test_apply_damping():
    """Test damping application."""
    field = jnp.ones((10, 10)) * 2.0
    
    damped = apply_damping(field, damping=0.5)
    
    assert jnp.allclose(damped, 1.0, atol=1e-6)


def test_compute_field_energy():
    """Test field energy computation."""
    grid_size = 32
    
    p_t = initialize_gaussian_pulse(grid_size, grid_size, grid_size/2, grid_size/2)
    p_tm1 = jnp.zeros((grid_size, grid_size))
    dt = 0.01
    
    energy = compute_field_energy(p_t, p_tm1, dt)
    
    # Energy should be positive
    assert energy > 0.0
    # Energy should be finite
    assert jnp.isfinite(energy)


def test_sample_field_at_point():
    """Test bilinear interpolation sampling."""
    field = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    
    # Sample at grid point
    pos = jnp.array([0.0, 0.0])
    val = sample_field_at_point(field, pos)
    assert jnp.allclose(val, 0.0)
    
    # Sample at midpoint
    pos = jnp.array([0.5, 0.5])
    val = sample_field_at_point(field, pos)
    expected = (0.0 + 1.0 + 2.0 + 3.0) / 4.0
    assert jnp.allclose(val, expected, atol=1e-5)


def test_sample_field_at_nodes():
    """Test sampling at multiple positions."""
    grid_size = 32
    field = initialize_gaussian_pulse(grid_size, grid_size, grid_size/2, grid_size/2)
    
    # Sample at 5 positions
    positions = jnp.array([
        [16.0, 16.0],  # Center
        [10.0, 10.0],
        [20.0, 20.0],
        [15.0, 17.0],
        [17.0, 15.0]
    ])
    
    samples = sample_field_at_nodes(field, positions)
    
    assert samples.shape == (5,)
    # Center should have highest value
    assert samples[0] == jnp.max(samples)


def test_initialize_gaussian_pulse():
    """Test Gaussian pulse initialization."""
    grid_w, grid_h = 64, 64
    center_x, center_y = 32.0, 32.0
    sigma = 5.0
    amplitude = 2.0
    
    pulse = initialize_gaussian_pulse(grid_w, grid_h, center_x, center_y, sigma, amplitude)
    
    assert pulse.shape == (grid_w, grid_h)
    # Maximum should be at center
    max_pos = jnp.unravel_index(jnp.argmax(pulse), pulse.shape)
    assert abs(max_pos[0] - center_x) < 1
    assert abs(max_pos[1] - center_y) < 1
    # Maximum value should be close to amplitude
    assert jnp.allclose(jnp.max(pulse), amplitude, atol=0.1)


def test_wave_propagation_speed():
    """Test that wave propagates at correct speed."""
    grid_size = 128
    
    # Initialize pulse at center
    p_t = initialize_gaussian_pulse(grid_size, grid_size, grid_size/2, grid_size/2, sigma=3.0)
    p_tm1 = p_t.copy()
    source_S = jnp.zeros((grid_size, grid_size))
    
    dt = 0.01
    c = 1.0
    dx = 1.0
    
    # Run for enough steps to see propagation
    for _ in range(20):
        p_tp1 = fdtd_step_wave(p_t, p_tm1, source_S, dt, c, dx)
        p_tm1 = p_t
        p_t = p_tp1
    
    # Wave should have spread from center
    # Check that energy has moved away from center
    center_val = p_t[grid_size//2, grid_size//2]
    off_center_val = p_t[grid_size//2 + 10, grid_size//2]
    
    # Off-center should have some amplitude (wave has propagated)
    assert abs(off_center_val) > 0.01


def test_absorbing_boundaries():
    """Test that boundaries absorb waves."""
    grid_size = 64
    
    # Initialize pulse near boundary
    p_t = initialize_gaussian_pulse(grid_size, grid_size, 5.0, grid_size/2, sigma=3.0)
    p_tm1 = p_t.copy()
    source_S = jnp.zeros((grid_size, grid_size))
    
    dt = 0.01
    c = 1.0
    dx = 1.0
    
    # Run for many steps
    for _ in range(50):
        p_tp1 = fdtd_step_wave(p_t, p_tm1, source_S, dt, c, dx)
        p_tm1 = p_t
        p_t = p_tp1
    
    # Boundary regions should be zero (absorbed)
    assert jnp.allclose(p_t[:2, :], 0.0, atol=1e-6)
    assert jnp.allclose(p_t[-2:, :], 0.0, atol=1e-6)
    assert jnp.allclose(p_t[:, :2], 0.0, atol=1e-6)
    assert jnp.allclose(p_t[:, -2:], 0.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

