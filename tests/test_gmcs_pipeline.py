"""
Tests for GMCS pipeline signal processing.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from src.core.gmcs_pipeline import (
    algo_nop,
    algo_limiter,
    algo_compressor,
    algo_expander,
    algo_threshold,
    algo_phasemod,
    algo_fold,
    gmcs_pipeline_single_node,
    gmcs_pipeline,
    create_default_chain,
    get_algorithm_name,
    describe_chain,
    ALGO_NOP,
    ALGO_LIMITER,
    ALGO_COMPRESSOR,
    ALGO_EXPANDER,
    ALGO_THRESHOLD,
    ALGO_PHASEMOD,
    ALGO_FOLD,
)
from src.core.state import MAX_CHAIN_LEN


def test_algo_nop():
    """Test no-op algorithm passes through."""
    h = 5.0
    result = algo_nop(h)
    assert result == h


def test_algo_limiter_soft_clipping():
    """Test limiter provides soft clipping."""
    A_max = 1.0
    
    # Below limit - should be nearly linear
    h_small = 0.5
    result_small = algo_limiter(h_small, A_max)
    assert abs(result_small - h_small) < 0.1
    
    # Far above limit - should clip to near A_max
    h_large = 10.0
    result_large = algo_limiter(h_large, A_max)
    assert result_large < A_max * 1.1
    assert result_large > A_max * 0.9


def test_algo_limiter_range():
    """Test limiter output range."""
    A_max = 2.0
    
    # Test various inputs
    for h in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        result = algo_limiter(h, A_max)
        assert abs(result) <= A_max * 1.1  # Allow small margin for tanh


def test_algo_compressor_above_threshold():
    """Test compressor reduces signals above threshold."""
    R = 2.0  # Compression ratio
    T = 1.0  # Threshold
    
    h = 3.0  # Above threshold
    result = algo_compressor(h, R, T)
    
    # Output should be: T + (h - T) / R = 1.0 + (3.0 - 1.0) / 2.0 = 2.0
    assert jnp.allclose(result, 2.0, atol=1e-6)


def test_algo_compressor_below_threshold():
    """Test compressor passes through signals below threshold."""
    R = 2.0
    T = 1.0
    
    h = 0.5  # Below threshold
    result = algo_compressor(h, R, T)
    
    # Should pass through unchanged
    assert jnp.allclose(result, h, atol=1e-6)


def test_algo_expander_below_threshold():
    """Test expander expands signals below threshold."""
    R = 2.0  # Expansion ratio
    T = 1.0  # Threshold
    
    h = 0.5  # Below threshold
    result = algo_expander(h, R, T)
    
    # Output should be: T - (T - h) / R = 1.0 - (1.0 - 0.5) / 2.0 = 0.75
    assert jnp.allclose(result, 0.75, atol=1e-6)


def test_algo_expander_above_threshold():
    """Test expander passes through signals above threshold."""
    R = 2.0
    T = 1.0
    
    h = 2.0  # Above threshold
    result = algo_expander(h, R, T)
    
    # Should pass through unchanged
    assert jnp.allclose(result, h, atol=1e-6)


def test_algo_threshold_gate():
    """Test threshold gate behavior."""
    T = 1.0
    V_low = 0.0
    
    # Above threshold
    h_above = 2.0
    result_above = algo_threshold(h_above, T, V_low)
    assert result_above == h_above
    
    # Below threshold
    h_below = 0.5
    result_below = algo_threshold(h_below, T, V_low)
    assert result_below == V_low


def test_algo_phasemod():
    """Test phase modulation."""
    h = 1.0
    t = 0.0
    Phi = 0.5
    omega = 1.0
    
    result = algo_phasemod(h, t, Phi, omega)
    
    # At t=0: sin(0) = 0, so output should be h * (1 + 0.5 * 0) = h
    assert jnp.allclose(result, h, atol=1e-6)
    
    # At t = π/(2*omega): sin(π/2) = 1, so output = h * (1 + Phi)
    t_peak = jnp.pi / (2.0 * omega)
    result_peak = algo_phasemod(h, t_peak, Phi, omega)
    expected_peak = h * (1.0 + Phi)
    assert jnp.allclose(result_peak, expected_peak, atol=1e-5)


def test_algo_fold():
    """Test wave folding."""
    gamma = 1.0
    beta = 1.0
    
    # Small input - should be nearly linear
    h_small = 0.1
    result_small = algo_fold(h_small, gamma, beta)
    assert abs(result_small - h_small) < 0.1
    
    # Large input - should fold back
    h_large = 5.0
    result_large = algo_fold(h_large, gamma, beta)
    assert abs(result_large) <= gamma * 1.1  # arcsin range is [-1, 1]


def test_gmcs_pipeline_single_node_nop_chain():
    """Test pipeline with all NOP."""
    h_in = 2.0
    chain_ids = jnp.zeros(MAX_CHAIN_LEN, dtype=jnp.int32)  # All NOP
    params = jnp.ones(9)  # Dummy parameters
    t_scalar = 0.0
    
    result = gmcs_pipeline_single_node(h_in, chain_ids, params, t_scalar)
    
    # Should pass through unchanged
    assert jnp.allclose(result, h_in, atol=1e-6)


def test_gmcs_pipeline_single_node_limiter():
    """Test pipeline with limiter only."""
    h_in = 5.0
    chain_ids = jnp.zeros(MAX_CHAIN_LEN, dtype=jnp.int32)
    chain_ids = chain_ids.at[0].set(ALGO_LIMITER)
    
    # Parameters: [A_max, R_comp, T_comp, R_exp, T_exp, Phi, omega, gamma, beta]
    params = jnp.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    t_scalar = 0.0
    
    result = gmcs_pipeline_single_node(h_in, chain_ids, params, t_scalar)
    
    # Should be limited to near A_max = 1.0
    assert result < 1.1
    assert result > 0.9


def test_gmcs_pipeline_single_node_chain():
    """Test pipeline with multiple algorithms."""
    h_in = 3.0
    
    # Chain: Compressor -> Limiter
    chain_ids = jnp.zeros(MAX_CHAIN_LEN, dtype=jnp.int32)
    chain_ids = chain_ids.at[0].set(ALGO_COMPRESSOR)
    chain_ids = chain_ids.at[1].set(ALGO_LIMITER)
    
    # Parameters: [A_max, R_comp, T_comp, R_exp, T_exp, Phi, omega, gamma, beta]
    params = jnp.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    t_scalar = 0.0
    
    result = gmcs_pipeline_single_node(h_in, chain_ids, params, t_scalar)
    
    # First: compressor (R=2, T=1): 3.0 -> 1.0 + (3.0-1.0)/2.0 = 2.0
    # Then: limiter (A_max=1): 2.0 -> tanh(2.0/1.0) * 1.0 ≈ 0.964
    assert result < 1.1
    assert result > 0.9


def test_gmcs_pipeline_vectorized():
    """Test vectorized pipeline for multiple nodes."""
    n_nodes = 10
    
    all_h_in = jnp.linspace(0.0, 5.0, n_nodes)
    all_chains = jnp.zeros((n_nodes, MAX_CHAIN_LEN), dtype=jnp.int32)
    all_chains = all_chains.at[:, 0].set(ALGO_LIMITER)
    
    # Same parameters for all nodes
    params_single = jnp.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    all_params = jnp.tile(params_single, (n_nodes, 1))
    
    t_scalar = 0.0
    
    results = gmcs_pipeline(all_h_in, all_chains, all_params, t_scalar)
    
    assert results.shape == (n_nodes,)
    # All results should be limited
    assert jnp.all(results <= 1.1)


def test_create_default_chain_limiter():
    """Test creating default limiter chain."""
    chain = create_default_chain("limiter")
    
    assert chain.shape == (MAX_CHAIN_LEN,)
    assert chain[0] == ALGO_LIMITER
    assert chain[1] == ALGO_NOP


def test_create_default_chain_full():
    """Test creating full chain."""
    chain = create_default_chain("full")
    
    assert chain.shape == (MAX_CHAIN_LEN,)
    assert chain[0] == ALGO_THRESHOLD
    assert chain[1] == ALGO_COMPRESSOR
    assert chain[2] == ALGO_LIMITER


def test_get_algorithm_name():
    """Test algorithm name lookup."""
    assert get_algorithm_name(ALGO_NOP) == "No-Op"
    assert get_algorithm_name(ALGO_LIMITER) == "Limiter"
    assert get_algorithm_name(ALGO_COMPRESSOR) == "Compressor"
    assert "Unknown" in get_algorithm_name(999)


def test_describe_chain():
    """Test chain description."""
    chain = jnp.zeros(MAX_CHAIN_LEN, dtype=jnp.int32)
    chain = chain.at[0].set(ALGO_LIMITER)
    chain = chain.at[1].set(ALGO_COMPRESSOR)
    
    desc = describe_chain(chain)
    
    assert "Limiter" in desc
    assert "Compressor" in desc
    assert "->" in desc


def test_describe_chain_empty():
    """Test description of empty chain."""
    chain = jnp.zeros(MAX_CHAIN_LEN, dtype=jnp.int32)
    
    desc = describe_chain(chain)
    
    assert "No processing" in desc


def test_pipeline_with_different_chains():
    """Test that different chains produce different outputs."""
    h_in = 2.0
    params = jnp.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    t_scalar = 0.0
    
    # Chain 1: Limiter only
    chain1 = jnp.zeros(MAX_CHAIN_LEN, dtype=jnp.int32)
    chain1 = chain1.at[0].set(ALGO_LIMITER)
    result1 = gmcs_pipeline_single_node(h_in, chain1, params, t_scalar)
    
    # Chain 2: Compressor only
    chain2 = jnp.zeros(MAX_CHAIN_LEN, dtype=jnp.int32)
    chain2 = chain2.at[0].set(ALGO_COMPRESSOR)
    result2 = gmcs_pipeline_single_node(h_in, chain2, params, t_scalar)
    
    # Should produce different results
    assert not jnp.allclose(result1, result2, atol=0.1)


def test_jit_compilation():
    """Test that pipeline functions are JIT compilable."""
    h_in = 2.0
    chain_ids = jnp.zeros(MAX_CHAIN_LEN, dtype=jnp.int32)
    params = jnp.ones(9)
    t_scalar = 0.0
    
    # Should work (already jitted)
    result = gmcs_pipeline_single_node(h_in, chain_ids, params, t_scalar)
    assert jnp.isfinite(result)


def test_parameter_effect():
    """Test that changing parameters affects output."""
    h_in = 3.0
    chain_ids = jnp.zeros(MAX_CHAIN_LEN, dtype=jnp.int32)
    chain_ids = chain_ids.at[0].set(ALGO_LIMITER)
    t_scalar = 0.0
    
    # Parameters with A_max = 0.5
    params1 = jnp.array([0.5, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    result1 = gmcs_pipeline_single_node(h_in, chain_ids, params1, t_scalar)
    
    # Parameters with A_max = 2.0
    params2 = jnp.array([2.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    result2 = gmcs_pipeline_single_node(h_in, chain_ids, params2, t_scalar)
    
    # Different A_max should produce different results
    assert not jnp.allclose(result1, result2, atol=0.1)
    assert result1 < result2  # Smaller A_max should give smaller output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

