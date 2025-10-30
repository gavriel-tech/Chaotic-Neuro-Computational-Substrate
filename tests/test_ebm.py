"""
Tests for EBM (Energy-Based Model) learning with CD-1.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from src.core.ebm import (
    binary_state_from_x,
    ebm_cd1_update,
    compute_ebm_bias,
    compute_ebm_energy,
    normalize_weights,
    compute_weight_statistics,
    thrml_sample,
)


def test_binary_state_from_x():
    """Test binary state conversion."""
    x_vec = jnp.array([1.0, -0.5, 0.0, 2.0, -1.0])
    threshold = 0.0
    
    binary = binary_state_from_x(x_vec, threshold)
    
    expected = jnp.array([1.0, -1.0, -1.0, 1.0, -1.0])
    assert jnp.allclose(binary, expected)


def test_binary_state_with_threshold():
    """Test binary state conversion with non-zero threshold."""
    x_vec = jnp.array([1.0, 0.5, 0.0, -0.5, -1.0])
    threshold = 0.5
    
    binary = binary_state_from_x(x_vec, threshold)
    
    # Only values > 0.5 should be +1
    expected = jnp.array([1.0, -1.0, -1.0, -1.0, -1.0])
    assert jnp.allclose(binary, expected)


def test_ebm_cd1_update_shape():
    """Test CD-1 update produces correct shape."""
    n_max = 10
    
    ebm_weights = jnp.zeros((n_max, n_max))
    oscillator_states = jnp.random.normal(jax.random.PRNGKey(0), (n_max, 3))
    mask = jnp.ones(n_max)
    key = jax.random.PRNGKey(42)
    eta = 0.01
    
    new_weights, new_key = ebm_cd1_update(
        ebm_weights, oscillator_states, mask, key, eta
    )
    
    assert new_weights.shape == (n_max, n_max)
    assert isinstance(new_key, jax.random.PRNGKeyArray)


def test_ebm_cd1_update_diagonal_zero():
    """Test that diagonal remains zero after update."""
    n_max = 10
    
    ebm_weights = jax.random.normal(jax.random.PRNGKey(0), (n_max, n_max)) * 0.1
    oscillator_states = jax.random.normal(jax.random.PRNGKey(1), (n_max, 3))
    mask = jnp.ones(n_max)
    key = jax.random.PRNGKey(42)
    eta = 0.01
    
    new_weights, _ = ebm_cd1_update(
        ebm_weights, oscillator_states, mask, key, eta
    )
    
    # Diagonal should be zero (no self-connections)
    assert jnp.allclose(jnp.diag(new_weights), 0.0, atol=1e-6)


def test_ebm_cd1_update_mask_effect():
    """Test that mask zeros inactive node weights."""
    n_max = 10
    
    ebm_weights = jax.random.normal(jax.random.PRNGKey(0), (n_max, n_max)) * 0.1
    oscillator_states = jax.random.normal(jax.random.PRNGKey(1), (n_max, 3))
    
    # Only first 5 nodes active
    mask = jnp.zeros(n_max)
    mask = mask.at[:5].set(1.0)
    
    key = jax.random.PRNGKey(42)
    eta = 0.01
    
    new_weights, _ = ebm_cd1_update(
        ebm_weights, oscillator_states, mask, key, eta
    )
    
    # Rows and columns for inactive nodes should be zero
    assert jnp.allclose(new_weights[5:, :], 0.0, atol=1e-6)
    assert jnp.allclose(new_weights[:, 5:], 0.0, atol=1e-6)


def test_ebm_cd1_update_learning():
    """Test that weights change during learning."""
    n_max = 5
    
    ebm_weights = jnp.zeros((n_max, n_max))
    
    # Create correlated oscillator states
    oscillator_states = jnp.zeros((n_max, 3))
    oscillator_states = oscillator_states.at[:, 0].set(jnp.array([1.0, 1.0, -1.0, -1.0, 0.0]))
    
    mask = jnp.ones(n_max)
    key = jax.random.PRNGKey(42)
    eta = 0.1  # High learning rate for testing
    
    new_weights, _ = ebm_cd1_update(
        ebm_weights, oscillator_states, mask, key, eta
    )
    
    # Weights should have changed
    assert not jnp.allclose(new_weights, ebm_weights, atol=1e-6)
    
    # Weights between correlated nodes (0,1) and (2,3) should increase
    # (both positive or both negative)
    assert new_weights[0, 1] > 0.0 or new_weights[1, 0] > 0.0


def test_compute_ebm_bias_shape():
    """Test EBM bias computation shape."""
    n_max = 10
    
    ebm_weights = jax.random.normal(jax.random.PRNGKey(0), (n_max, n_max)) * 0.1
    oscillator_states = jax.random.normal(jax.random.PRNGKey(1), (n_max, 3))
    mask = jnp.ones(n_max)
    beta = 0.05
    
    bias = compute_ebm_bias(ebm_weights, oscillator_states, mask, beta)
    
    assert bias.shape == (n_max,)


def test_compute_ebm_bias_magnitude():
    """Test that EBM bias is scaled by beta."""
    n_max = 5
    
    ebm_weights = jnp.ones((n_max, n_max)) * 0.1
    oscillator_states = jnp.ones((n_max, 3))
    mask = jnp.ones(n_max)
    
    # Test with different beta values
    bias_small = compute_ebm_bias(ebm_weights, oscillator_states, mask, beta=0.01)
    bias_large = compute_ebm_bias(ebm_weights, oscillator_states, mask, beta=0.1)
    
    # Larger beta should give larger bias
    assert jnp.mean(jnp.abs(bias_large)) > jnp.mean(jnp.abs(bias_small))


def test_compute_ebm_bias_mask_effect():
    """Test that mask zeros bias for inactive nodes."""
    n_max = 10
    
    ebm_weights = jax.random.normal(jax.random.PRNGKey(0), (n_max, n_max)) * 0.1
    oscillator_states = jax.random.normal(jax.random.PRNGKey(1), (n_max, 3))
    
    # Only first 5 nodes active
    mask = jnp.zeros(n_max)
    mask = mask.at[:5].set(1.0)
    
    beta = 0.05
    
    bias = compute_ebm_bias(ebm_weights, oscillator_states, mask, beta)
    
    # Bias for inactive nodes should be zero
    assert jnp.allclose(bias[5:], 0.0, atol=1e-6)


def test_compute_ebm_energy():
    """Test EBM energy computation."""
    n_max = 5
    
    # Simple symmetric weights
    ebm_weights = jnp.eye(n_max) * 0.0  # Zero diagonal
    ebm_weights = ebm_weights.at[0, 1].set(1.0)
    ebm_weights = ebm_weights.at[1, 0].set(1.0)
    
    # States where nodes 0 and 1 have same sign (low energy)
    oscillator_states_aligned = jnp.zeros((n_max, 3))
    oscillator_states_aligned = oscillator_states_aligned.at[:, 0].set(jnp.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    
    # States where nodes 0 and 1 have opposite sign (high energy)
    oscillator_states_opposed = jnp.zeros((n_max, 3))
    oscillator_states_opposed = oscillator_states_opposed.at[:, 0].set(jnp.array([1.0, -1.0, 0.0, 0.0, 0.0]))
    
    mask = jnp.ones(n_max)
    
    energy_aligned = compute_ebm_energy(ebm_weights, oscillator_states_aligned, mask)
    energy_opposed = compute_ebm_energy(ebm_weights, oscillator_states_opposed, mask)
    
    # Aligned state should have lower energy
    assert energy_aligned < energy_opposed


def test_normalize_weights():
    """Test weight normalization."""
    weights = jnp.array([
        [0.0, 2.0, -3.0],
        [2.0, 0.0, 0.5],
        [-3.0, 0.5, 0.0]
    ])
    
    normalized = normalize_weights(weights, max_weight=1.0)
    
    # All weights should be within [-1, 1]
    assert jnp.all(normalized >= -1.0)
    assert jnp.all(normalized <= 1.0)


def test_compute_weight_statistics():
    """Test weight statistics computation."""
    n_max = 10
    
    ebm_weights = jax.random.normal(jax.random.PRNGKey(0), (n_max, n_max)) * 0.5
    mask = jnp.ones(n_max)
    
    stats = compute_weight_statistics(ebm_weights, mask)
    
    assert 'mean' in stats
    assert 'std' in stats
    assert 'max' in stats
    assert 'min' in stats
    
    # Statistics should be reasonable
    assert jnp.isfinite(stats['mean'])
    assert stats['std'] >= 0.0


def test_compute_weight_statistics_inactive_nodes():
    """Test weight statistics with inactive nodes."""
    n_max = 10
    
    ebm_weights = jax.random.normal(jax.random.PRNGKey(0), (n_max, n_max)) * 0.5
    
    # Only first 5 nodes active
    mask = jnp.zeros(n_max)
    mask = mask.at[:5].set(1.0)
    
    stats = compute_weight_statistics(ebm_weights, mask)
    
    # Should compute stats only for active nodes
    assert jnp.isfinite(stats['mean'])


def test_thrml_sample():
    """Test THRML-style Gibbs sampling."""
    n_max = 5
    
    # Simple weights favoring alignment
    ebm_weights = jnp.ones((n_max, n_max)) * 0.5
    ebm_weights = ebm_weights.at[jnp.arange(n_max), jnp.arange(n_max)].set(0.0)
    
    initial_state = jnp.ones(n_max) * -1.0
    key = jax.random.PRNGKey(42)
    
    sampled_state = thrml_sample(ebm_weights, initial_state, key, n_steps=10)
    
    assert sampled_state.shape == (n_max,)
    # Should be binary {-1, +1}
    assert jnp.all((sampled_state == -1.0) | (sampled_state == 1.0))


def test_ebm_cd1_convergence():
    """Test that CD-1 learns correlations over multiple steps."""
    n_max = 4
    
    ebm_weights = jnp.zeros((n_max, n_max))
    mask = jnp.ones(n_max)
    key = jax.random.PRNGKey(42)
    eta = 0.05
    
    # Create highly correlated data: nodes 0,1 positive, nodes 2,3 negative
    oscillator_states = jnp.zeros((n_max, 3))
    oscillator_states = oscillator_states.at[:, 0].set(jnp.array([1.0, 1.0, -1.0, -1.0]))
    
    # Run multiple learning steps
    for _ in range(20):
        ebm_weights, key = ebm_cd1_update(
            ebm_weights, oscillator_states, mask, key, eta
        )
    
    # Weights between correlated nodes should be positive
    assert ebm_weights[0, 1] > 0.0
    assert ebm_weights[2, 3] > 0.0
    
    # Weights between anti-correlated nodes should be negative
    assert ebm_weights[0, 2] < 0.0
    assert ebm_weights[1, 3] < 0.0


def test_jit_compilation():
    """Test that EBM functions are JIT compilable."""
    n_max = 10
    
    ebm_weights = jnp.zeros((n_max, n_max))
    oscillator_states = jax.random.normal(jax.random.PRNGKey(0), (n_max, 3))
    mask = jnp.ones(n_max)
    key = jax.random.PRNGKey(42)
    
    # Should work (already jitted)
    new_weights, new_key = ebm_cd1_update(ebm_weights, oscillator_states, mask, key, 0.01)
    
    assert new_weights.shape == (n_max, n_max)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

