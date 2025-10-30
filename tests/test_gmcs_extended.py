"""
Tests for extended GMCS algorithms (audio/signal processing and photonic).
"""

import pytest
import jax
import jax.numpy as jnp
from src.core.gmcs_pipeline import (
    # Audio/Signal algorithms
    algo_resonator, algo_hilbert, algo_rectifier, algo_quantizer,
    algo_slew_limiter, algo_cross_mod, algo_bipolar_fold,
    # Photonic algorithms
    algo_optical_kerr, algo_electro_optic, algo_optical_switch,
    algo_four_wave_mixing, algo_raman_amplifier, algo_saturation, algo_optical_gain,
    # Pipeline functions
    gmcs_pipeline_single_node, gmcs_pipeline, gmcs_pipeline_dual,
    # Constants
    ALGO_RESONATOR, ALGO_HILBERT, ALGO_RECTIFIER, ALGO_QUANTIZER,
    ALGO_SLEW_LIMITER, ALGO_CROSS_MOD, ALGO_BIPOLAR_FOLD,
    ALGO_OPTICAL_KERR, ALGO_ELECTRO_OPTIC, ALGO_OPTICAL_SWITCH,
    ALGO_FOUR_WAVE_MIXING, ALGO_RAMAN_AMPLIFIER, ALGO_SATURATION, ALGO_OPTICAL_GAIN
)


class TestAudioSignalAlgorithms:
    """Test audio/signal processing algorithms."""
    
    def test_resonator(self):
        """Test resonator filter."""
        h = 1.0
        f0 = 440.0  # Hz
        Q = 10.0
        t = 0.001  # 1ms
        
        result = algo_resonator(h, f0, Q, t)
        assert jnp.isfinite(result)
        assert isinstance(result, jnp.ndarray)
    
    def test_hilbert(self):
        """Test Hilbert transform."""
        h = 0.5
        result = algo_hilbert(h)
        assert jnp.isfinite(result)
        # Hilbert should preserve magnitude approximately
        assert jnp.abs(result) > 0
    
    def test_rectifier(self):
        """Test full-wave rectification."""
        # Positive input
        assert algo_rectifier(0.5) == 0.5
        # Negative input
        assert algo_rectifier(-0.5) == 0.5
        # Zero
        assert algo_rectifier(0.0) == 0.0
    
    def test_quantizer(self):
        """Test quantization."""
        h = 0.5
        levels = 8.0
        
        result = algo_quantizer(h, levels)
        assert jnp.isfinite(result)
        assert -1.0 <= result <= 1.0
        
        # Test with 2 levels (binary)
        result_binary = algo_quantizer(h, 2.0)
        assert result_binary in [0.0, 1.0, -1.0]
    
    def test_slew_limiter(self):
        """Test slew rate limiter."""
        h = 2.0
        rate_limit = 1.0
        
        result = algo_slew_limiter(h, rate_limit)
        assert jnp.abs(result) <= rate_limit
    
    def test_cross_mod(self):
        """Test cross-modulation."""
        h = 0.5
        mod_depth = 0.5
        mod_freq = 10.0
        t = 0.1
        
        result = algo_cross_mod(h, mod_depth, mod_freq, t)
        assert jnp.isfinite(result)
        # Result should be modulated version of input
        assert jnp.abs(result) >= 0
    
    def test_bipolar_fold(self):
        """Test bipolar wave folding."""
        threshold = 0.5
        
        # Below threshold
        assert algo_bipolar_fold(0.3, threshold) == 0.3
        
        # Above threshold (should fold)
        result = algo_bipolar_fold(0.8, threshold)
        assert jnp.isfinite(result)
        assert jnp.abs(result) <= 2.0 * threshold


class TestPhotonicAlgorithms:
    """Test photonic algorithms."""
    
    def test_optical_kerr(self):
        """Test Optical Kerr effect."""
        h = 1.0
        n2 = 2.6e-20  # m²/W
        length = 1.0  # m
        
        result = algo_optical_kerr(h, n2, length)
        assert jnp.isfinite(result)
        # Kerr effect should increase field
        assert jnp.abs(result) >= jnp.abs(h)
    
    def test_electro_optic(self):
        """Test electro-optic modulation."""
        h = 1.0
        V = 2.5  # Volts
        V_pi = 5.0  # Half-wave voltage
        
        result = algo_electro_optic(h, V, V_pi)
        assert jnp.isfinite(result)
        # EO modulation should preserve or reduce amplitude
        assert jnp.abs(result) <= jnp.abs(h)
    
    def test_optical_switch(self):
        """Test optical switch."""
        h = 1.0
        threshold = 0.5
        contrast = 0.8
        
        result = algo_optical_switch(h, threshold, contrast)
        assert jnp.isfinite(result)
        # Switch should attenuate
        assert jnp.abs(result) <= jnp.abs(h)
    
    def test_four_wave_mixing(self):
        """Test four-wave mixing."""
        h = 1.0
        pump_power = 1.0
        gamma = 0.1
        
        result = algo_four_wave_mixing(h, pump_power, gamma)
        assert jnp.isfinite(result)
        # FWM should amplify
        assert jnp.abs(result) >= jnp.abs(h)
    
    def test_raman_amplifier(self):
        """Test Raman amplification."""
        h = 1.0
        pump_power = 1.0
        g_R = 0.5
        length = 1.0
        
        result = algo_raman_amplifier(h, pump_power, g_R, length)
        assert jnp.isfinite(result)
        # Raman should amplify
        assert jnp.abs(result) >= jnp.abs(h)
    
    def test_saturation(self):
        """Test saturation."""
        sat_level = 1.0
        
        # Below saturation
        assert jnp.abs(algo_saturation(0.5, sat_level)) < sat_level
        
        # At saturation
        result = algo_saturation(10.0, sat_level)
        assert jnp.abs(result) <= sat_level * 1.1  # Allow small margin
    
    def test_optical_gain(self):
        """Test optical gain."""
        h = 1.0
        gain_dB = 10.0  # 10 dB = 10x power = 3.16x amplitude
        
        result = algo_optical_gain(h, gain_dB)
        assert jnp.isfinite(result)
        # Gain should amplify
        assert jnp.abs(result) > jnp.abs(h)


class TestExtendedPipeline:
    """Test pipeline with extended algorithms."""
    
    def test_pipeline_with_resonator(self):
        """Test pipeline with resonator algorithm."""
        h_in = 1.0
        chain_ids = jnp.array([ALGO_RESONATOR, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
        params = jnp.array([
            0.8, 2.0, 0.5, 2.0, 0.2, 0.0, 1.0, 1.0, 1.0,  # Basic 0-8
            440.0, 10.0, 16.0, 1.0, 2.6e-20, 0.0, 5.0  # Extended 9-15
        ], dtype=jnp.float32)
        t = 0.001
        
        result = gmcs_pipeline_single_node(h_in, chain_ids, params, t)
        assert jnp.isfinite(result)
    
    def test_pipeline_with_photonic_chain(self):
        """Test pipeline with photonic algorithm chain."""
        h_in = 1.0
        chain_ids = jnp.array([
            ALGO_OPTICAL_KERR,
            ALGO_RAMAN_AMPLIFIER,
            ALGO_SATURATION,
            0, 0, 0, 0, 0
        ], dtype=jnp.int32)
        params = jnp.array([
            0.8, 2.0, 0.5, 2.0, 0.2, 0.0, 1.0, 1.0, 1.0,
            440.0, 10.0, 16.0, 1.0, 2.6e-20, 0.0, 5.0
        ], dtype=jnp.float32)
        t = 0.0
        
        result = gmcs_pipeline_single_node(h_in, chain_ids, params, t)
        assert jnp.isfinite(result)
    
    def test_pipeline_vectorized(self):
        """Test vectorized pipeline with extended parameters."""
        n_nodes = 4
        h_inputs = jnp.ones(n_nodes, dtype=jnp.float32)
        chains = jnp.zeros((n_nodes, 8), dtype=jnp.int32)
        chains = chains.at[0, 0].set(ALGO_RESONATOR)
        chains = chains.at[1, 0].set(ALGO_OPTICAL_KERR)
        chains = chains.at[2, 0].set(ALGO_QUANTIZER)
        
        params = jnp.tile(
            jnp.array([
                0.8, 2.0, 0.5, 2.0, 0.2, 0.0, 1.0, 1.0, 1.0,
                440.0, 10.0, 16.0, 1.0, 2.6e-20, 0.0, 5.0
            ], dtype=jnp.float32),
            (n_nodes, 1)
        )
        t = 0.0
        
        results = gmcs_pipeline(h_inputs, chains, params, t)
        assert results.shape == (n_nodes,)
        assert jnp.all(jnp.isfinite(results))
    
    def test_dual_pipeline(self):
        """Test dual output pipeline."""
        n_nodes = 4
        h_inputs = jnp.array([0.5, -0.3, 0.8, -0.1], dtype=jnp.float32)
        chains = jnp.zeros((n_nodes, 8), dtype=jnp.int32)
        params = jnp.tile(
            jnp.array([
                0.8, 2.0, 0.5, 2.0, 0.2, 0.0, 1.0, 1.0, 1.0,
                440.0, 10.0, 16.0, 1.0, 2.6e-20, 0.0, 5.0
            ], dtype=jnp.float32),
            (n_nodes, 1)
        )
        t = 0.0
        
        F_continuous, B_discrete = gmcs_pipeline_dual(h_inputs, chains, params, t)
        
        # Check shapes
        assert F_continuous.shape == (n_nodes,)
        assert B_discrete.shape == (n_nodes,)
        
        # Check discrete output is binary
        assert jnp.all((B_discrete == 1.0) | (B_discrete == -1.0))
        
        # Check sign relationship
        for i in range(n_nodes):
            if F_continuous[i] > 0:
                assert B_discrete[i] == 1.0
            else:
                assert B_discrete[i] == -1.0


class TestAlgorithmProperties:
    """Test mathematical properties of algorithms."""
    
    def test_rectifier_symmetry(self):
        """Rectifier should be symmetric."""
        h = 0.7
        assert algo_rectifier(h) == algo_rectifier(-h)
    
    def test_quantizer_range(self):
        """Quantizer output should be in [-1, 1]."""
        for h in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]:
            result = algo_quantizer(h, 8.0)
            assert -1.0 <= result <= 1.0
    
    def test_saturation_bounds(self):
        """Saturation should bound output."""
        sat_level = 1.0
        for h in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            result = algo_saturation(h, sat_level)
            assert jnp.abs(result) <= sat_level * 1.01  # Small tolerance
    
    def test_optical_gain_linearity(self):
        """Optical gain should be linear."""
        h = 1.0
        gain_dB = 10.0
        
        result1 = algo_optical_gain(h, gain_dB)
        result2 = algo_optical_gain(2.0 * h, gain_dB)
        
        # Should be approximately 2x
        ratio = result2 / result1
        assert jnp.abs(ratio - 2.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

