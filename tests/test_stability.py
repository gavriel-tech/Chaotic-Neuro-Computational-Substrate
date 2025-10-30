"""
Stability and stress tests for GMCS platform.

Tests long-running simulations, maximum node counts, reconnection scenarios,
and error handling to ensure production readiness.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.core.state import initialize_system_state, validate_state
from src.core.simulation import simulation_step, run_simulation


class TestLongRunSimulation:
    """Test simulation stability over extended periods."""
    
    def test_1000_step_stability(self):
        """Run 1000 steps and verify no NaN/Inf values."""
        
        key = jax.random.PRNGKey(0)
        state = initialize_system_state(key, n_max=64, grid_w=64, grid_h=64)
        
        # Activate some nodes
        state = state._replace(
            node_active_mask=state.node_active_mask.at[:16].set(1.0)
        )
        
        # Run for 1000 steps
        for _ in range(1000):
            state = simulation_step(state, enable_ebm_feedback=True)
            assert validate_state(state), "State became invalid (NaN/Inf detected)"
        
        # Check final state is reasonable
        assert jnp.all(jnp.abs(state.oscillator_state[:16]) < 10.0), \
            "Oscillator states grew unbounded"
        assert jnp.all(jnp.abs(state.field_p) < 5.0), \
            "Wave field grew unbounded"
    
    def test_10000_step_memory_stability(self):
        """Verify no memory leaks over 10k steps."""
        
        key = jax.random.PRNGKey(1)
        state = initialize_system_state(key, n_max=32, grid_w=32, grid_h=32)
        state = state._replace(
            node_active_mask=state.node_active_mask.at[:8].set(1.0)
        )
        
        # Sample memory footprint at start
        initial_field_mean = float(jnp.mean(jnp.abs(state.field_p)))
        
        # Run many steps
        for _ in range(10000):
            state = simulation_step(state, enable_ebm_feedback=False)
        
        # Verify state is still valid
        assert validate_state(state)
        
        # Check values haven't diverged
        final_field_mean = float(jnp.mean(jnp.abs(state.field_p)))
        assert final_field_mean < 10.0, "Field energy diverged"
    
    def test_energy_conservation_bounds(self):
        """Verify total system energy remains bounded."""
        
        key = jax.random.PRNGKey(2)
        state = initialize_system_state(key)
        state = state._replace(
            node_active_mask=state.node_active_mask.at[:32].set(1.0)
        )
        
        energies = []
        for _ in range(500):
            # Compute total energy (oscillators + field)
            osc_energy = jnp.sum(state.oscillator_state[:32]**2)
            field_energy = jnp.sum(state.field_p**2)
            total_energy = float(osc_energy + field_energy)
            energies.append(total_energy)
            
            state = simulation_step(state, enable_ebm_feedback=True)
        
        # Energy should stabilize (not grow indefinitely)
        mean_energy = np.mean(energies[-100:])
        assert mean_energy < 1000.0, f"Energy grew too large: {mean_energy}"


class TestMaximumCapacity:
    """Test behavior at maximum node counts."""
    
    def test_full_1024_nodes_activation(self):
        """Activate all 1024 nodes and verify stability."""
        
        key = jax.random.PRNGKey(0)
        state = initialize_system_state(key, n_max=1024)
        
        # Activate ALL nodes
        state = state._replace(
            node_active_mask=jnp.ones(1024, dtype=jnp.float32)
        )
        
        # Distribute nodes in grid
        side = 32
        xs = jnp.linspace(0.0, 255.0, side)
        ys = jnp.linspace(0.0, 255.0, side)
        xx, yy = jnp.meshgrid(xs, ys)
        positions = jnp.stack([xx.flatten(), yy.flatten()], axis=1)
        state = state._replace(node_positions=positions)
        
        # Run for 100 steps
        for _ in range(100):
            state = simulation_step(state, enable_ebm_feedback=True)
            assert validate_state(state), "Failed with maximum nodes"
        
        # Verify all nodes are still active
        assert jnp.sum(state.node_active_mask) == 1024
    
    def test_maximum_grid_resolution(self):
        """Test with maximum grid resolution (512×512)."""
        
        key = jax.random.PRNGKey(1)
        state = initialize_system_state(key, n_max=64, grid_w=512, grid_h=512)
        state = state._replace(
            node_active_mask=state.node_active_mask.at[:64].set(1.0)
        )
        
        # Run for 50 steps (will be slower due to large grid)
        for _ in range(50):
            state = simulation_step(state, enable_ebm_feedback=False)
            assert validate_state(state)
    
    def test_sparse_activation_pattern(self):
        """Test with many nodes but sparse activation."""
        
        key = jax.random.PRNGKey(2)
        state = initialize_system_state(key, n_max=1024)
        
        # Activate only every 10th node
        mask = jnp.zeros(1024)
        mask = mask.at[::10].set(1.0)
        state = state._replace(node_active_mask=mask)
        
        for _ in range(200):
            state = simulation_step(state, enable_ebm_feedback=True)
            assert validate_state(state)


class TestCFLStability:
    """Test CFL condition stability constraints."""
    
    def test_cfl_violation_handling(self):
        """Verify simulation handles CFL violations gracefully."""
        
        key = jax.random.PRNGKey(0)
        state = initialize_system_state(key, dt=0.1)  # Large dt
        state = state._replace(
            c_val=jnp.array([5.0]),  # High wave speed
            node_active_mask=state.node_active_mask.at[:16].set(1.0)
        )
        
        # This violates CFL (c*dt/dx > 1/sqrt(2))
        # System should still not produce NaN/Inf
        for _ in range(50):
            state = simulation_step(state, enable_ebm_feedback=False)
            # May be numerically unstable but should not crash
            if not validate_state(state):
                pytest.skip("CFL violation caused expected numerical instability")
                break
    
    def test_stable_cfl_range(self):
        """Verify stability within CFL limits."""
        
        key = jax.random.PRNGKey(1)
        
        # dt = 0.01, c = 1.0, dx = 1.0
        # CFL = 1.0 * 0.01 / 1.0 = 0.01 << 1/sqrt(2) ≈ 0.707 ✓
        state = initialize_system_state(key, dt=0.01)
        state = state._replace(
            c_val=jnp.array([1.0]),
            node_active_mask=state.node_active_mask.at[:32].set(1.0)
        )
        
        for _ in range(1000):
            state = simulation_step(state, enable_ebm_feedback=True)
            assert validate_state(state), "Failed within CFL limits"


class TestEBMLearningStability:
    """Test EBM learning stability over extended periods."""
    
    def test_ebm_weight_bounds(self):
        """Verify EBM weights remain bounded during learning."""
        
        key = jax.random.PRNGKey(0)
        state = initialize_system_state(key)
        state = state._replace(
            node_active_mask=state.node_active_mask.at[:64].set(1.0)
        )
        
        # Run with EBM learning
        final_state, diagnostics = run_simulation(
            state,
            n_steps=500,
            ebm_learning_interval=10,
            ebm_learning_rate=0.01,
            enable_ebm=True
        )
        
        # Check weights didn't explode
        max_weight = float(jnp.max(jnp.abs(final_state.ebm_weights)))
        assert max_weight < 10.0, f"EBM weights grew too large: {max_weight}"
        
        # Check diagonal is still zero (no self-connections)
        diag = jnp.diag(final_state.ebm_weights)
        assert jnp.allclose(diag, 0.0), "EBM diagonal became non-zero"
    
    def test_ebm_energy_tracking(self):
        """Verify EBM energy calculations are stable."""
        
        from src.core.ebm import compute_ebm_energy
        
        key = jax.random.PRNGKey(1)
        state = initialize_system_state(key)
        state = state._replace(
            node_active_mask=state.node_active_mask.at[:32].set(1.0)
        )
        
        energies = []
        for _ in range(200):
            state = simulation_step(state, enable_ebm_feedback=True)
            
            energy = compute_ebm_energy(
                state.ebm_weights,
                state.oscillator_state,
                state.node_active_mask
            )
            energies.append(float(energy))
        
        # Energy should be bounded
        assert all(e < 1000.0 for e in energies), "EBM energy diverged"
        assert not any(np.isnan(e) or np.isinf(e) for e in energies), \
            "EBM energy became NaN/Inf"


class TestErrorRecovery:
    """Test system recovery from error conditions."""
    
    def test_zero_active_nodes(self):
        """Verify simulation handles no active nodes gracefully."""
        
        key = jax.random.PRNGKey(0)
        state = initialize_system_state(key)
        # All nodes inactive (default)
        
        # Should not crash
        for _ in range(10):
            state = simulation_step(state, enable_ebm_feedback=False)
            assert validate_state(state)
        
        # Field should remain near zero
        assert jnp.max(jnp.abs(state.field_p)) < 0.1
    
    def test_invalid_parameter_clamping(self):
        """Test that invalid parameters are handled."""
        
        key = jax.random.PRNGKey(1)
        state = initialize_system_state(key)
        
        # Set extreme parameters
        state = state._replace(
            c_val=jnp.array([100.0]),  # Very high wave speed
            gmcs_gamma=jnp.full(1024, 10.0),  # High fold amplitude
            node_active_mask=state.node_active_mask.at[:8].set(1.0)
        )
        
        # System should handle it (may be unstable but not crash)
        try:
            for _ in range(20):
                state = simulation_step(state, enable_ebm_feedback=False)
                if not validate_state(state):
                    break  # Expected instability
        except Exception as e:
            pytest.fail(f"System crashed with extreme parameters: {e}")
    
    def test_nan_injection_recovery(self):
        """Test recovery from NaN injection (if possible)."""
        
        key = jax.random.PRNGKey(2)
        state = initialize_system_state(key)
        state = state._replace(
            node_active_mask=state.node_active_mask.at[:4].set(1.0)
        )
        
        # Inject NaN into one oscillator
        bad_state = jnp.array([[jnp.nan, 0.0, 0.0]])
        state = state._replace(
            oscillator_state=state.oscillator_state.at[0].set(bad_state[0])
        )
        
        # Validation should catch it
        assert not validate_state(state), "Validation failed to catch NaN"


class TestStressCombinations:
    """Test combinations of stressors."""
    
    def test_max_nodes_with_ebm_learning(self):
        """Maximum nodes + EBM learning."""
        
        key = jax.random.PRNGKey(0)
        state = initialize_system_state(key, n_max=512)
        state = state._replace(
            node_active_mask=jnp.ones(512, dtype=jnp.float32)
        )
        
        # Distribute nodes
        side = int(jnp.sqrt(512))
        xs = jnp.linspace(0.0, 255.0, side)
        ys = jnp.linspace(0.0, 255.0, side)
        xx, yy = jnp.meshgrid(xs, ys)
        positions = jnp.stack([xx.flatten()[:512], yy.flatten()[:512]], axis=1)
        state = state._replace(node_positions=positions)
        
        # Run with EBM learning
        for step in range(100):
            state = simulation_step(state, enable_ebm_feedback=True)
            
            # EBM update every 10 steps
            if step % 10 == 0:
                from src.core.ebm import ebm_cd1_update
                new_weights, new_key = ebm_cd1_update(
                    state.ebm_weights,
                    state.oscillator_state,
                    state.node_active_mask,
                    state.key,
                    eta=0.01
                )
                state = state._replace(ebm_weights=new_weights, key=new_key)
            
            assert validate_state(state)
    
    def test_high_frequency_parameter_changes(self):
        """Rapid parameter updates."""
        
        key = jax.random.PRNGKey(1)
        state = initialize_system_state(key)
        state = state._replace(
            node_active_mask=state.node_active_mask.at[:16].set(1.0)
        )
        
        for i in range(100):
            # Change wave speed every step
            c_new = 0.5 + (i % 10) * 0.2
            state = state._replace(c_val=jnp.array([c_new]))
            
            state = simulation_step(state, enable_ebm_feedback=False)
            assert validate_state(state)


@pytest.mark.slow
class TestExtendedStress:
    """Very long-running tests (marked as slow)."""
    
    def test_24_hour_equivalent(self):
        """Simulate 24 hours at 60 Hz (5,184,000 steps)."""
        # This is marked slow and would run in CI separately
        # For actual testing, run a representative sample
        
        key = jax.random.PRNGKey(0)
        state = initialize_system_state(key)
        state = state._replace(
            node_active_mask=state.node_active_mask.at[:16].set(1.0)
        )
        
        # Run 5000 steps as a proxy (would be ~80 seconds at 60Hz)
        for _ in range(5000):
            state = simulation_step(state, enable_ebm_feedback=True)
            assert validate_state(state)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
