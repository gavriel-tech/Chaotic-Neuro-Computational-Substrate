"""
Comprehensive tests for THRML integration.

Tests cover:
- THRMLWrapper creation and management
- Gibbs sampling
- CD-k learning
- Energy computation
- Performance mode switching
- State serialization
- Integration with simulation loop
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.core.thrml_integration import (
    THRMLWrapper,
    create_thrml_model,
    thrml_to_jax_weights,
    jax_to_thrml_weights,
    reconstruct_thrml_wrapper
)
from src.core.ebm import (
    ebm_update_with_thrml,
    compute_thrml_feedback,
    compute_ebm_energy_thrml,
    binary_state_from_x
)
from src.core.state import initialize_system_state
from src.core.simulation import (
    simulation_step,
    simulation_step_with_thrml_learning,
    run_simulation
)
from src.config.thrml_config import (
    PerformanceMode,
    get_performance_config,
    PERFORMANCE_PRESETS
)


class TestTHRMLWrapper:
    """Test THRMLWrapper class functionality."""
    
    def test_wrapper_creation(self):
        """Test creating a THRML wrapper."""
        n_nodes = 5
        weights = np.random.randn(n_nodes, n_nodes) * 0.1
        weights = (weights + weights.T) / 2  # Make symmetric
        np.fill_diagonal(weights, 0)
        biases = np.zeros(n_nodes)
        
        wrapper = THRMLWrapper(n_nodes, weights, biases, beta=1.0)
        
        assert wrapper.n_nodes == n_nodes
        assert len(wrapper.nodes) == n_nodes
        assert wrapper.beta == 1.0
        
    def test_wrapper_sampling(self):
        """Test Gibbs sampling produces valid binary states."""
        n_nodes = 4
        weights = np.array([
            [0.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 0.5, 0.0]
        ])
        biases = np.zeros(n_nodes)
        
        wrapper = THRMLWrapper(n_nodes, weights, biases)
        key = jax.random.PRNGKey(42)
        
        # Sample
        samples = wrapper.sample_gibbs(n_steps=10, temperature=1.0, key=key)
        
        # Check shape and values
        assert samples.shape == (n_nodes,)
        assert np.all(np.isin(samples, [-1.0, 1.0]))
        
    def test_wrapper_energy_computation(self):
        """Test energy computation matches Ising formula."""
        n_nodes = 3
        weights = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])
        biases = np.array([0.1, -0.2, 0.1])
        
        wrapper = THRMLWrapper(n_nodes, weights, biases)
        
        # Test with known state
        states = np.array([1.0, -1.0, 1.0])
        energy = wrapper.compute_energy(states)
        
        # Manual calculation: E = -0.5 * s^T @ W @ s - b^T @ s
        expected_interaction = -0.5 * np.dot(states, np.dot(weights, states))
        expected_bias = -np.dot(biases, states)
        expected_energy = expected_interaction + expected_bias
        
        assert np.isclose(energy, expected_energy, atol=1e-5)
        
    def test_wrapper_cd_learning(self):
        """Test CD-k learning updates weights."""
        n_nodes = 4
        weights = np.zeros((n_nodes, n_nodes))
        biases = np.zeros(n_nodes)
        
        wrapper = THRMLWrapper(n_nodes, weights, biases)
        
        # Data states (all aligned)
        data_states = np.array([1.0, 1.0, 1.0, 1.0])
        
        # Get initial weights
        initial_weights = wrapper.get_weights().copy()
        
        # Run CD-1
        key = jax.random.PRNGKey(0)
        wrapper.update_weights_cd(data_states, eta=0.1, k_steps=1, key=key)
        
        # Weights should have changed
        new_weights = wrapper.get_weights()
        assert not np.allclose(initial_weights, new_weights)
        
        # Weights should still be symmetric
        assert np.allclose(new_weights, new_weights.T)
        
        # Diagonal should still be zero
        assert np.allclose(np.diag(new_weights), 0)
        
    def test_wrapper_serialization(self):
        """Test wrapper can be serialized and reconstructed."""
        n_nodes = 3
        weights = np.random.randn(n_nodes, n_nodes) * 0.1
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        biases = np.random.randn(n_nodes) * 0.1
        beta = 1.5
        
        wrapper = THRMLWrapper(n_nodes, weights, biases, beta)
        
        # Serialize
        data = wrapper.serialize()
        
        # Reconstruct
        wrapper2 = THRMLWrapper.deserialize(data)
        
        # Check equivalence
        assert wrapper2.n_nodes == wrapper.n_nodes
        assert np.allclose(wrapper2.get_weights(), wrapper.get_weights())
        assert np.allclose(wrapper2.biases_jax, wrapper.biases_jax)
        assert wrapper2.beta == wrapper.beta


class TestEBMFunctions:
    """Test EBM functions with THRML."""
    
    def test_ebm_update_with_thrml(self):
        """Test THRML-based EBM update."""
        n_nodes = 4
        weights = np.zeros((n_nodes, n_nodes))
        biases = np.zeros(n_nodes)
        
        wrapper = THRMLWrapper(n_nodes, weights, biases)
        
        # Create oscillator states
        oscillator_states = jnp.array([
            [0.5, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [-0.2, 0.0, 0.0],
            [-0.4, 0.0, 0.0]
        ])
        mask = jnp.ones(n_nodes)
        
        key = jax.random.PRNGKey(0)
        
        # Update
        wrapper = ebm_update_with_thrml(
            wrapper,
            oscillator_states,
            mask,
            eta=0.01,
            k_steps=1,
            key=key
        )
        
        # Weights should have been updated
        new_weights = wrapper.get_weights()
        assert not np.allclose(new_weights, 0)
        
    def test_compute_thrml_feedback(self):
        """Test THRML feedback computation."""
        n_nodes = 3
        weights = np.array([
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0]
        ])
        biases = np.zeros(n_nodes)
        
        wrapper = THRMLWrapper(n_nodes, weights, biases)
        
        gmcs_biases = jnp.array([0.1, -0.1, 0.1])
        key = jax.random.PRNGKey(42)
        
        feedback = compute_thrml_feedback(
            wrapper,
            gmcs_biases,
            temperature=1.0,
            gibbs_steps=5,
            key=key
        )
        
        # Feedback should be a JAX array
        assert isinstance(feedback, jnp.ndarray)
        assert feedback.shape[0] >= n_nodes
        
    def test_compute_ebm_energy_thrml(self):
        """Test energy computation via THRML."""
        n_nodes = 3
        weights = np.eye(n_nodes) * 0 + 0.1  # Small coupling
        np.fill_diagonal(weights, 0)
        biases = np.zeros(n_nodes)
        
        wrapper = THRMLWrapper(n_nodes, weights, biases)
        
        oscillator_states = jnp.array([
            [0.5, 0.0, 0.0],
            [-0.3, 0.0, 0.0],
            [0.2, 0.0, 0.0]
        ])
        mask = jnp.ones(n_nodes)
        
        energy = compute_ebm_energy_thrml(wrapper, oscillator_states, mask)
        
        # Energy should be a float
        assert isinstance(energy, float)


class TestPerformanceModes:
    """Test performance mode configuration."""
    
    def test_all_modes_available(self):
        """Test all three performance modes exist."""
        assert PerformanceMode.SPEED in PERFORMANCE_PRESETS
        assert PerformanceMode.ACCURACY in PERFORMANCE_PRESETS
        assert PerformanceMode.RESEARCH in PERFORMANCE_PRESETS
        
    def test_mode_configurations(self):
        """Test each mode has valid configuration."""
        for mode in PerformanceMode:
            config = PERFORMANCE_PRESETS[mode]
            
            assert config.gibbs_steps > 0
            assert config.temperature > 0
            assert config.learning_rate > 0
            assert config.cd_k_steps > 0
            assert config.weight_update_freq > 0
            assert isinstance(config.use_jit, bool)
            assert len(config.description) > 0
            
    def test_mode_ordering(self):
        """Test modes are ordered by computational cost."""
        speed_config = PERFORMANCE_PRESETS[PerformanceMode.SPEED]
        accuracy_config = PERFORMANCE_PRESETS[PerformanceMode.ACCURACY]
        research_config = PERFORMANCE_PRESETS[PerformanceMode.RESEARCH]
        
        # Speed should have fewest steps
        assert speed_config.gibbs_steps < accuracy_config.gibbs_steps
        assert accuracy_config.gibbs_steps < research_config.gibbs_steps
        
        # Speed should have lowest CD-k
        assert speed_config.cd_k_steps <= accuracy_config.cd_k_steps
        assert accuracy_config.cd_k_steps <= research_config.cd_k_steps
        
    def test_get_performance_config(self):
        """Test getting config by mode name."""
        config = get_performance_config('speed')
        assert config.mode == PerformanceMode.SPEED
        
        config = get_performance_config('accuracy')
        assert config.mode == PerformanceMode.ACCURACY
        
        config = get_performance_config('research')
        assert config.mode == PerformanceMode.RESEARCH
        
        # Test invalid mode
        with pytest.raises(ValueError):
            get_performance_config('invalid')


class TestSimulationIntegration:
    """Test THRML integration with simulation loop."""
    
    def test_simulation_step_with_thrml(self):
        """Test simulation step with THRML wrapper."""
        key = jax.random.PRNGKey(0)
        state = initialize_system_state(key, dt=0.01)
        
        # Add a node
        from src.core.simulation import add_node_to_state
        state, node_id = add_node_to_state(state, (128, 128), initial_perturbation=0.1)
        
        # Create THRML wrapper
        n_active = 1
        wrapper = create_thrml_model(
            n_nodes=n_active,
            weights=np.zeros((n_active, n_active)),
            biases=np.zeros(n_active),
            beta=1.0
        )
        
        # Run simulation step
        new_state, new_wrapper = simulation_step(
            state,
            enable_ebm_feedback=True,
            thrml_wrapper=wrapper
        )
        
        # State should have advanced
        assert float(new_state.t[0]) > float(state.t[0])
        
    def test_simulation_step_with_learning(self):
        """Test simulation step with THRML learning."""
        key = jax.random.PRNGKey(0)
        state = initialize_system_state(key, dt=0.01)
        
        # Add nodes
        from src.core.simulation import add_node_to_state
        state, _ = add_node_to_state(state, (100, 128), initial_perturbation=0.1)
        state, _ = add_node_to_state(state, (156, 128), initial_perturbation=-0.1)
        
        # Create THRML wrapper
        n_active = 2
        wrapper = create_thrml_model(
            n_nodes=n_active,
            weights=np.zeros((n_active, n_active)),
            biases=np.zeros(n_active),
            beta=1.0
        )
        
        # Get initial weights
        initial_weights = wrapper.get_weights().copy()
        
        # Run learning step
        new_state, new_wrapper = simulation_step_with_thrml_learning(
            state,
            wrapper,
            eta=0.01
        )
        
        # Weights should have changed
        new_weights = new_wrapper.get_weights()
        assert not np.allclose(initial_weights, new_weights, atol=1e-6)
        
    def test_run_simulation_with_thrml(self):
        """Test multi-step simulation with THRML."""
        key = jax.random.PRNGKey(0)
        state = initialize_system_state(key, dt=0.01)
        
        # Add a node
        from src.core.simulation import add_node_to_state
        state, _ = add_node_to_state(state, (128, 128), initial_perturbation=0.1)
        
        # Create THRML wrapper
        n_active = 1
        wrapper = create_thrml_model(
            n_nodes=n_active,
            weights=np.zeros((n_active, n_active)),
            biases=np.zeros(n_active),
            beta=1.0
        )
        
        # Run simulation
        final_state, final_wrapper, diagnostics = run_simulation(
            state,
            n_steps=10,
            thrml_wrapper=wrapper,
            ebm_learning_interval=5,
            enable_ebm=True
        )
        
        # Check diagnostics
        assert len(diagnostics['times']) > 0
        assert len(diagnostics['max_osc_values']) > 0
        assert len(diagnostics['max_field_values']) > 0
        
        # Time should have advanced
        assert float(final_state.t[0]) > float(state.t[0])


class TestStateSerialization:
    """Test THRML state serialization."""
    
    def test_thrml_model_data_in_state(self):
        """Test THRML model data can be stored in SystemState."""
        key = jax.random.PRNGKey(0)
        state = initialize_system_state(key)
        
        # Create wrapper
        n_nodes = 2
        wrapper = create_thrml_model(
            n_nodes=n_nodes,
            weights=np.random.randn(n_nodes, n_nodes) * 0.1,
            biases=np.zeros(n_nodes),
            beta=1.0
        )
        
        # Serialize to state
        thrml_data = wrapper.serialize()
        state = state._replace(thrml_model_data=thrml_data)
        
        # Reconstruct
        wrapper2 = reconstruct_thrml_wrapper(state.thrml_model_data)
        
        # Check equivalence
        assert wrapper2.n_nodes == wrapper.n_nodes
        assert np.allclose(wrapper2.get_weights(), wrapper.get_weights())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

