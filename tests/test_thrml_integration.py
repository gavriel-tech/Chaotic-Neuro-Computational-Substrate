"""
Tests for THRML Integration

Comprehensive test suite for THRML features:
- Core wrapper functionality
- Blocking strategies
- Multi-GPU support
- Heterogeneous nodes
- Energy factors
- Benchmarking
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

# Import modules to test
from src.core.thrml_integration import (
    THRMLWrapper,
    create_thrml_model,
    THRML_AVAILABLE
)

if THRML_AVAILABLE:
    try:
        from src.core.blocking_strategies import (
            get_strategy,
            list_strategies,
            CheckerboardStrategy,
            RandomStrategy
        )
        from src.core.multi_gpu import (
            get_device_info,
            create_multi_gpu_thrml_sampler
        )
        from src.core.heterogeneous_nodes import (
            HeterogeneousNodeSpec,
            NodeType,
            create_heterogeneous_model,
            create_domain_specific_model
        )
        from src.core.energy_factors import (
            EnergyFactorSystem,
            PhotonicCouplingFactor,
            AudioHarmonyFactor,
            MLRegularizationFactor,
            add_photonic_coupling
        )
        from src.tools.thrml_benchmark import THRMLBenchmark
        from src.visualization.thrml_visualizers import THRMLVisualizer
        from src.core.thrml_compat import (
            SpinNodes,
            ContinuousNodes,
            DiscreteNodes,
            EnergyObserver,
            CorrelationObserver,
        )
        from thrml import Block
        from thrml.block_sampling import sample_with_observation, SamplingSchedule
        from thrml.models import IsingSamplingProgram, hinton_init
    except ImportError as e:
        # Some modules may have additional dependencies
        import warnings
        warnings.warn(f"Could not import all THRML modules: {e}")


# Skip all tests if THRML not available
pytestmark = pytest.mark.skipif(
    not THRML_AVAILABLE,
    reason="THRML not available"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_weights():
    """Create simple weight matrix."""
    n_nodes = 8
    weights = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        weights[i, j] = weights[j, i] = 0.1
    return weights


@pytest.fixture
def simple_biases():
    """Create simple bias vector."""
    return np.zeros(8)


@pytest.fixture
def thrml_wrapper(simple_weights, simple_biases):
    """Create THRML wrapper for testing."""
    return create_thrml_model(
        n_nodes=8,
        weights=simple_weights,
        biases=simple_biases,
        beta=1.0
    )


@pytest.fixture
def jax_key():
    """Create JAX random key."""
    return random.PRNGKey(42)


# ============================================================================
# Core Integration Tests
# ============================================================================

class TestTHRMLWrapper:
    """Test THRMLWrapper core functionality."""
    
    def test_initialization(self, simple_weights, simple_biases):
        """Test wrapper initialization."""
        wrapper = THRMLWrapper(
            n_nodes=8,
            weights=simple_weights,
            biases=simple_biases,
            beta=1.0
        )
        
        assert wrapper.n_nodes == 8
        assert wrapper.beta == 1.0
        assert len(wrapper.nodes) == 8
        assert wrapper.model is not None
    
    def test_initialization_invalid_weights(self, simple_biases):
        """Test initialization with invalid weights."""
        # Non-finite weights
        weights = np.full((8, 8), np.nan)
        
        with pytest.raises(ValueError, match="non-finite"):
            THRMLWrapper(8, weights, simple_biases, 1.0)
    
    def test_initialization_wrong_shape(self, simple_biases):
        """Test initialization with wrong shape."""
        weights = np.zeros((8, 10))  # Wrong shape
        
        with pytest.raises(ValueError, match="shape"):
            THRMLWrapper(8, weights, simple_biases, 1.0)
    
    def test_sample_gibbs(self, thrml_wrapper, jax_key):
        """Test Gibbs sampling."""
        samples = thrml_wrapper.sample_gibbs(
            n_steps=10,
            temperature=1.0,
            key=jax_key
        )
        
        assert samples.shape == (thrml_wrapper.n_nodes,)
        assert np.all((samples == -1) | (samples == 1))  # Binary values
    
    def test_sample_gibbs_invalid_temperature(self, thrml_wrapper, jax_key):
        """Test sampling with invalid temperature."""
        with pytest.raises(ValueError, match="temperature"):
            thrml_wrapper.sample_gibbs(
                n_steps=10,
                temperature=0.0,  # Invalid
                key=jax_key
            )
    
    def test_compute_energy(self, thrml_wrapper):
        """Test energy computation."""
        states = np.ones(thrml_wrapper.n_nodes)
        energy = thrml_wrapper.compute_energy(states)
        
        assert isinstance(energy, float)
        assert np.isfinite(energy)
    
    def test_update_biases(self, thrml_wrapper):
        """Test bias update."""
        new_biases = np.random.randn(thrml_wrapper.n_nodes)
        thrml_wrapper.update_biases(new_biases)
        
        assert np.allclose(thrml_wrapper.biases, new_biases)
    
    def test_update_weights_cd(self, thrml_wrapper, jax_key):
        """Test CD weight update."""
        data_states = np.random.choice([-1.0, 1.0], size=thrml_wrapper.n_nodes)
        
        diagnostics = thrml_wrapper.update_weights_cd(
            data_states=data_states,
            eta=0.01,
            k_steps=1,
            key=jax_key
        )
        
        assert 'gradient_norm' in diagnostics
        assert 'energy_diff' in diagnostics
        assert isinstance(diagnostics['gradient_norm'], float)
    
    def test_serialize_deserialize(self, thrml_wrapper):
        """Test serialization and deserialization."""
        # Serialize
        state_dict = thrml_wrapper.serialize()
        
        assert isinstance(state_dict, dict)
        assert 'n_nodes' in state_dict
        assert 'beta' in state_dict
        
        # Deserialize
        new_wrapper = THRMLWrapper.deserialize(state_dict)
        
        assert new_wrapper.n_nodes == thrml_wrapper.n_nodes
        assert np.isclose(new_wrapper.beta, thrml_wrapper.beta)
    
    def test_health_status(self, thrml_wrapper):
        """Test health status reporting."""
        health = thrml_wrapper.get_health_status()
        
        assert isinstance(health, dict)
        assert 'healthy' in health
        assert 'n_nodes' in health
        assert 'n_edges' in health
        assert health['healthy'] is True


# ============================================================================
# Blocking Strategies Tests
# ============================================================================

class TestBlockingStrategies:
    """Test blocking strategy implementations."""
    
    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = list_strategies()
        
        assert isinstance(strategies, list)
        strategy_names = [s.name for s in strategies]
        assert 'checkerboard' in strategy_names
        assert 'random' in strategy_names
    
    def test_get_strategy(self):
        """Test getting a strategy."""
        strategy = get_strategy('checkerboard')
        
        assert strategy is not None
        assert isinstance(strategy, CheckerboardStrategy)
    
    def test_get_invalid_strategy(self):
        """Test getting invalid strategy."""
        strategy = get_strategy('nonexistent')
        assert strategy is None
    
    def test_checkerboard_strategy(self):
        """Test checkerboard blocking."""
        strategy = CheckerboardStrategy()
        n_nodes = 16
        # Create 2D grid positions for checkerboard
        positions = np.array([[i // 4, i % 4] for i in range(n_nodes)], dtype=np.float32)
        
        blocks = strategy.build_blocks(n_nodes, positions=positions)
        
        assert len(blocks) == 2
        # Blocks should be non-empty
        assert all(len(block.nodes) > 0 for block in blocks)
    
    def test_random_strategy(self):
        """Test random blocking."""
        strategy = RandomStrategy()
        n_nodes = 16
        
        blocks = strategy.build_blocks(n_nodes, seed=42)
        
        assert len(blocks) >= 2
        # Check all nodes covered - blocks contain actual node objects in THRML, not IDs
        total_nodes = sum(len(block.nodes) for block in blocks)
        assert total_nodes == n_nodes
    
    def test_wrapper_set_blocking_strategy(self, thrml_wrapper):
        """Test setting blocking strategy on wrapper."""
        thrml_wrapper.set_blocking_strategy('checkerboard')
        
        assert thrml_wrapper.current_strategy_name == 'checkerboard'
        assert len(thrml_wrapper.free_blocks) > 0
    
    def test_wrapper_validate_blocks(self, thrml_wrapper):
        """Test block validation."""
        thrml_wrapper.set_blocking_strategy('checkerboard')
        validation = thrml_wrapper.validate_current_blocks()
        
        assert 'valid' in validation
        assert 'balance_score' in validation
        assert validation['valid'] is True


# ============================================================================
# Multi-GPU Tests
# ============================================================================

class TestMultiGPU:
    """Test multi-GPU functionality."""
    
    def test_get_device_info(self):
        """Test device info retrieval."""
        info = get_device_info()
        
        assert 'n_devices' in info
        assert 'default_backend' in info
        assert 'devices' in info
        assert isinstance(info['n_devices'], int)
    
    def test_create_multi_gpu_sampler(self, thrml_wrapper):
        """Test multi-GPU sampler creation."""
        gpu_sampler = create_multi_gpu_thrml_sampler(thrml_wrapper)
        
        # May be None if only 1 device
        if gpu_sampler is not None:
            assert gpu_sampler.n_devices >= 2
            assert gpu_sampler.base_wrapper == thrml_wrapper
    
    @pytest.mark.skipif(
        len(jax.devices()) < 2,
        reason="Requires 2+ GPUs"
    )
    def test_parallel_chain_sampling(self, thrml_wrapper, jax_key):
        """Test parallel chain sampling (requires 2+ GPUs)."""
        gpu_sampler = create_multi_gpu_thrml_sampler(thrml_wrapper)
        
        assert gpu_sampler is not None
        
        samples, diagnostics = gpu_sampler.sample_parallel_chains(
            n_chains=4,
            n_steps=10,
            temperature=1.0,
            key=jax_key
        )
        
        assert samples.shape[0] == 4
        assert samples.shape[1] == thrml_wrapper.n_nodes
        assert len(diagnostics) == 4


# ============================================================================
# Heterogeneous Nodes Tests
# ============================================================================

class TestHeterogeneousNodes:
    """Test heterogeneous node types."""
    
    def test_node_spec_creation(self):
        """Test node spec creation."""
        spec = HeterogeneousNodeSpec(
            node_id=0,
            node_type=NodeType.SPIN,
            properties={}
        )
        
        assert spec.node_id == 0
        assert spec.node_type == NodeType.SPIN
    
    def test_node_spec_serialization(self):
        """Test node spec serialization."""
        spec = HeterogeneousNodeSpec(
            node_id=0,
            node_type=NodeType.CONTINUOUS,
            properties={'min_value': -1.0, 'max_value': 1.0}
        )
        
        data = spec.to_dict()
        new_spec = HeterogeneousNodeSpec.from_dict(data)
        
        assert new_spec.node_id == spec.node_id
        assert new_spec.node_type == spec.node_type
    
    def test_create_heterogeneous_model(self):
        """Test heterogeneous model creation."""
        node_types = np.array([0, 0, 1, 1, 2, 2])  # Mix of types
        weights = np.random.randn(6, 6) * 0.1
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        
        wrapper = create_heterogeneous_model(
            node_type_array=node_types,
            weights=weights,
            beta=1.0
        )
        
        assert wrapper.n_nodes == 6
        assert len(wrapper.node_groups[NodeType.SPIN]) == 2
        assert len(wrapper.node_groups[NodeType.CONTINUOUS]) == 2
        assert len(wrapper.node_groups[NodeType.DISCRETE]) == 2
        assert NodeType.SPIN in wrapper.node_wrappers
        assert isinstance(wrapper.node_wrappers[NodeType.DISCRETE], DiscreteNodes)
    
    def test_domain_specific_model(self):
        """Test domain-specific model creation."""
        wrapper = create_domain_specific_model(
            n_photonic=4,
            n_audio=4,
            n_ml=4
        )
        
        assert wrapper.n_nodes == 12
        info = wrapper.get_info()
        assert info['type_counts']['SPIN'] == 4
        assert info['type_counts']['CONTINUOUS'] == 4
        assert info['type_counts']['DISCRETE'] == 4
        assert NodeType.CONTINUOUS in wrapper.node_wrappers
        continuous_group = wrapper.node_wrappers[NodeType.CONTINUOUS]
        assert isinstance(continuous_group, ContinuousNodes)
        assert 'min_value' in continuous_group.metadata


# ============================================================================
# Energy Factors Tests
# ============================================================================

class TestEnergyFactors:
    """Test custom energy factors."""
    
    def test_photonic_factor(self):
        """Test photonic coupling factor."""
        factor = PhotonicCouplingFactor(
            factor_id='test_photonic',
            node_ids=[0, 1, 2, 3],
            strength=0.5,
            wavelength=1550e-9
        )
        
        states = np.ones(10)
        energy = factor.compute(states)
        
        assert isinstance(energy, float)
        assert np.isfinite(energy)
    
    def test_audio_factor(self):
        """Test audio harmony factor."""
        factor = AudioHarmonyFactor(
            factor_id='test_audio',
            node_ids=[0, 1, 2, 3],
            strength=0.3,
            fundamental_freq=440.0
        )
        
        states = np.ones(10)
        energy = factor.compute(states)
        
        assert isinstance(energy, float)
        assert np.isfinite(energy)
    
    def test_ml_regularization_factor(self):
        """Test ML regularization factor."""
        factor = MLRegularizationFactor(
            factor_id='test_ml',
            node_ids=[0, 1, 2, 3],
            strength=0.01,
            regularization_type='l2'
        )
        
        states = np.random.randn(10)
        energy = factor.compute(states)
        
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert energy >= 0  # L2 is always positive
    
    def test_factor_system(self):
        """Test energy factor system."""
        system = EnergyFactorSystem()
        
        add_photonic_coupling(system, [0, 1, 2, 3], strength=0.5)
        
        assert len(system.factors) == 1
        
        states = np.ones(10)
        total_energy = system.compute_total_energy(states, base_energy=0.0)
        
        assert isinstance(total_energy, float)
        assert np.isfinite(total_energy)
    
    def test_factor_system_serialization(self):
        """Test factor system serialization."""
        system = EnergyFactorSystem()
        add_photonic_coupling(system, [0, 1, 2, 3], strength=0.5)
        
        data = system.serialize()
        new_system = EnergyFactorSystem.deserialize(data)
        
        assert len(new_system.factors) == len(system.factors)


# ============================================================================
# Benchmarking Tests
# ============================================================================

class TestBenchmarking:
    """Test benchmarking functionality."""
    
    def test_benchmark_creation(self):
        """Test benchmark creation."""
        benchmark = THRMLBenchmark(verbose=False)
        
        assert benchmark is not None
    
    def test_single_benchmark(self):
        """Test single benchmark run."""
        benchmark = THRMLBenchmark(verbose=False)
        
        result = benchmark.run_single(
            n_nodes=16,
            strategy='checkerboard',
            n_samples=10,
            seed=0
        )
        
        assert result.samples_per_sec > 0
        assert result.ess_per_sec > 0
        assert -1 <= result.lag1_autocorr <= 1  # Autocorrelation can be negative
    
    def test_compare_strategies(self):
        """Test strategy comparison."""
        benchmark = THRMLBenchmark(verbose=False)
        
        results = benchmark.compare_strategies(
            strategies=['checkerboard', 'random'],
            n_nodes=16,
            n_samples=10
        )
        
        assert len(results) == 2
        strategy_names = [result.strategy for result in results]
        assert 'checkerboard' in strategy_names
        assert 'random' in strategy_names


# ============================================================================
# Compatibility Helpers Tests
# ============================================================================

class TestCompatibilityWrappers:
    """Validate compatibility node group wrappers."""

    def test_spin_nodes_wrapper(self):
        group = SpinNodes('spin', [0, 1, 2])
        assert len(group) == 3
        block = group.as_block()
        assert isinstance(block, Block)
        assert len(block.nodes) == 3

    def test_continuous_nodes_metadata(self):
        group = ContinuousNodes('cont', [0, 1], min_value=-0.5, max_value=0.5)
        assert group.metadata['min_value'] == -0.5
        assert group.metadata['max_value'] == 0.5

    def test_discrete_nodes_metadata(self):
        group = DiscreteNodes('disc', [0, 1, 2], n_values=7)
        assert group.metadata['n_values'] == 7


class TestCompatibilityObservers:
    """Ensure compatibility observers operate with THRML sampling."""

    def test_energy_observer(self, thrml_wrapper, jax_key):
        thrml_wrapper.set_blocking_strategy('checkerboard')
        program = IsingSamplingProgram(
            thrml_wrapper.model,
            thrml_wrapper.free_blocks,
            []
        )
        schedule = SamplingSchedule(n_warmup=1, n_samples=3, steps_per_sample=1)
        key_init, key_run = random.split(jax_key)
        init_state = hinton_init(key_init, thrml_wrapper.model, thrml_wrapper.free_blocks, ())
        observer = EnergyObserver(thrml_wrapper.model)
        carry = observer.init()
        _, energies = sample_with_observation(
            key_run,
            program,
            schedule,
            init_state,
            [],
            carry,
            observer,
        )
        energies = jnp.asarray(energies)
        assert energies.shape[0] == schedule.n_samples

    def test_correlation_observer(self, thrml_wrapper, jax_key):
        thrml_wrapper.set_blocking_strategy('checkerboard')
        program = IsingSamplingProgram(
            thrml_wrapper.model,
            thrml_wrapper.free_blocks,
            []
        )
        schedule = SamplingSchedule(n_warmup=1, n_samples=3, steps_per_sample=1)
        key_init, key_run = random.split(jax_key)
        init_state = hinton_init(key_init, thrml_wrapper.model, thrml_wrapper.free_blocks, ())
        observer = CorrelationObserver(thrml_wrapper.model)
        carry = observer.init()
        _, correlations = sample_with_observation(
            key_run,
            program,
            schedule,
            init_state,
            [],
            carry,
            observer,
        )
        correlations = jnp.asarray(correlations)
        assert correlations.shape[0] == schedule.n_samples
        if thrml_wrapper.model.edges:
            assert correlations.shape[1] == len(thrml_wrapper.model.edges)


# ============================================================================
# Visualization Tests
# ============================================================================

class TestVisualization:
    """Test THRML visualizers."""
    
    @pytest.mark.skipif(
        True,  # Skip matplotlib tests by default
        reason="Matplotlib tests require display"
    )
    def test_visualizer_creation(self):
        """Test visualizer creation."""
        viz = THRMLVisualizer()
        assert viz is not None
    
    @pytest.mark.skipif(
        True,
        reason="Matplotlib tests require display"
    )
    def test_pbit_grid_plot(self):
        """Test P-bit grid visualization."""
        viz = THRMLVisualizer()
        states = np.random.choice([-1, 1], size=64)
        
        fig = viz.plot_pbit_grid(states, grid_shape=(8, 8))
        assert fig is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_workflow(self, jax_key):
        """Test complete workflow."""
        # 1. Create model
        n_nodes = 16
        weights = np.random.randn(n_nodes, n_nodes) * 0.1
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        
        wrapper = create_thrml_model(
            n_nodes=n_nodes,
            weights=weights,
            biases=np.zeros(n_nodes),
            beta=1.0
        )
        
        # 2. Set blocking strategy
        wrapper.set_blocking_strategy('checkerboard')
        
        # 3. Sample
        samples = wrapper.sample_gibbs(n_steps=10, temperature=1.0, key=jax_key)
        
        # 4. Compute energy
        energy = wrapper.compute_energy(samples)
        
        # 5. Update weights
        key, subkey = random.split(jax_key)
        diagnostics = wrapper.update_weights_cd(
            data_states=samples,
            eta=0.01,
            k_steps=1,
            key=subkey
        )
        
        # Verify all steps succeeded
        assert samples.shape == (n_nodes,)
        assert isinstance(energy, float)
        assert 'gradient_norm' in diagnostics
    
    def test_error_recovery(self, thrml_wrapper):
        """Test error recovery."""
        # Try with invalid temperature (should handle gracefully)
        try:
            thrml_wrapper.sample_gibbs(
                n_steps=10,
                temperature=-1.0,  # Invalid
                key=random.PRNGKey(0)
            )
        except ValueError:
            pass  # Expected
        
        # Wrapper should still be functional
        health = thrml_wrapper.get_health_status()
        assert health is not None


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance regression tests."""
    
    def test_sampling_speed(self, thrml_wrapper, jax_key):
        """Test sampling speed meets minimum threshold."""
        import time
        
        n_samples = 100
        t_start = time.time()
        
        for _ in range(n_samples):
            key, subkey = random.split(jax_key)
            thrml_wrapper.sample_gibbs(n_steps=1, temperature=1.0, key=subkey)
        
        elapsed = time.time() - t_start
        samples_per_sec = n_samples / elapsed
        
        # Minimum threshold: 1 samples/sec (very conservative, accounts for JIT compilation)
        assert samples_per_sec > 1
    
    def test_memory_usage(self, jax_key):
        """Test memory usage stays reasonable."""
        # Create large model
        n_nodes = 256
        weights = np.random.randn(n_nodes, n_nodes) * 0.01
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        
        wrapper = create_thrml_model(
            n_nodes=n_nodes,
            weights=weights,
            biases=np.zeros(n_nodes),
            beta=1.0
        )
        
        # Sample multiple times
        for _ in range(10):
            key, subkey = random.split(jax_key)
            samples = wrapper.sample_gibbs(n_steps=10, temperature=1.0, key=subkey)
        
        # Should not crash with OOM
        assert True


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
