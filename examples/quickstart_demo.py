"""
GMCS Quickstart Demo

This script demonstrates the basic usage of GMCS:
1. Initialize simulation
2. Add nodes
3. Run simulation steps
4. Extract outputs for different domains

Run: python examples/quickstart_demo.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from src.core.state import initialize_system_state
from src.core.simulation import simulation_step
from src.core.ebm import ebm_cd1_update, compute_ebm_energy


def demo_basic_simulation():
    """Demonstrate basic simulation with a few nodes."""
    print("=" * 60)
    print("GMCS Basic Simulation Demo")
    print("=" * 60)
    
    # Initialize
    print("\n1. Initializing system...")
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key, n_max=64, grid_w=128, grid_h=128, dt=0.01)
    
    # Activate 16 nodes in a grid
    print("2. Activating 16 nodes...")
    state = state._replace(node_active_mask=state.node_active_mask.at[:16].set(1.0))
    
    # Distribute nodes in grid
    positions = []
    for i in range(4):
        for j in range(4):
            x = 32.0 + i * 21.0
            y = 32.0 + j * 21.0
            positions.append([x, y])
    positions = jnp.array(positions)
    state = state._replace(node_positions=state.node_positions.at[:16].set(positions))
    
    # Run simulation
    print("3. Running simulation for 100 steps...")
    for step in range(100):
        state = simulation_step(state, enable_ebm_feedback=True)
        
        # EBM learning every 10 steps
        if step % 10 == 0 and step > 0:
            new_weights, new_key = ebm_cd1_update(
                state.ebm_weights,
                state.oscillator_state,
                state.node_active_mask,
                state.key,
                eta=0.01
            )
            state = state._replace(ebm_weights=new_weights, key=new_key)
        
        if step % 20 == 0:
            max_field = float(jnp.max(jnp.abs(state.field_p)))
            print(f"   Step {step:3d}: t={float(state.t[0]):.3f}, max_field={max_field:.3f}")
    
    # Extract metrics
    print("\n4. Final state:")
    print(f"   Simulation time: {float(state.t[0]):.3f}s")
    print(f"   Active nodes: {int(jnp.sum(state.node_active_mask))}")
    print(f"   Max field amplitude: {float(jnp.max(jnp.abs(state.field_p))):.3f}")
    print(f"   Max oscillator state: {float(jnp.max(jnp.abs(state.oscillator_state))):.3f}")
    
    energy = compute_ebm_energy(state.ebm_weights, state.oscillator_state, state.node_active_mask)
    print(f"   EBM energy: {float(energy):.3f}")
    
    print("\n✓ Basic simulation complete!")
    return state


def demo_terrain_generation(state):
    """Demonstrate terrain generation use case."""
    print("\n" + "=" * 60)
    print("GMCS Terrain Generation Demo")
    print("=" * 60)
    
    print("\n1. Generating terrain heightmap...")
    
    # Use wave field as heightmap
    heightmap = np.array(state.field_p)
    
    print(f"2. Heightmap shape: {heightmap.shape}")
    print(f"   Min height: {heightmap.min():.3f}")
    print(f"   Max height: {heightmap.max():.3f}")
    print(f"   Mean height: {heightmap.mean():.3f}")
    
    # Normalize to [0, 1]
    normalized = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)
    
    print(f"3. Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print("\n✓ Terrain generation complete!")
    print("   (Use this heightmap in your rendering engine)")
    
    return normalized


def demo_encryption_keystream(state):
    """Demonstrate encryption keystream generation."""
    print("\n" + "=" * 60)
    print("GMCS Encryption Keystream Demo")
    print("=" * 60)
    
    print("\n1. Extracting binary states from oscillators...")
    
    # Convert oscillator x-coordinates to binary
    x_states = state.oscillator_state[:, 0]
    binary_stream = (x_states > 0.0).astype(jnp.int32)
    
    active_bits = binary_stream[state.node_active_mask > 0.5]
    
    print(f"2. Generated {len(active_bits)} bits")
    print(f"   Bit string (first 32): {' '.join(map(str, active_bits[:32]))}")
    
    # Compute entropy (simple estimate)
    ones = int(jnp.sum(active_bits))
    zeros = len(active_bits) - ones
    p1 = ones / len(active_bits) if len(active_bits) > 0 else 0
    p0 = 1 - p1
    entropy = -p0 * np.log2(p0 + 1e-10) - p1 * np.log2(p1 + 1e-10)
    
    print(f"3. Entropy: {entropy:.4f} bits/bit (ideal: 1.0)")
    print(f"   Ones: {ones}, Zeros: {zeros}")
    
    print("\n✓ Keystream generation complete!")
    print("   (XOR this with plaintext for encryption)")
    
    return np.array(active_bits)


def demo_anomaly_detection():
    """Demonstrate anomaly detection use case."""
    print("\n" + "=" * 60)
    print("GMCS Anomaly Detection Demo")
    print("=" * 60)
    
    print("\n1. Setting up anomaly detection system...")
    
    key = jax.random.PRNGKey(123)
    state = initialize_system_state(key, n_max=32)
    state = state._replace(node_active_mask=state.node_active_mask.at[:8].set(1.0))
    
    # Train on "normal" data
    print("2. Training on normal data (50 steps)...")
    for _ in range(50):
        state = simulation_step(state)
        if _ % 10 == 0:
            new_weights, new_key = ebm_cd1_update(
                state.ebm_weights,
                state.oscillator_state,
                state.node_active_mask,
                state.key,
                eta=0.02
            )
            state = state._replace(ebm_weights=new_weights, key=new_key)
    
    # Compute baseline energy
    baseline_energy = float(compute_ebm_energy(
        state.ebm_weights,
        state.oscillator_state,
        state.node_active_mask
    ))
    
    print(f"   Baseline energy: {baseline_energy:.3f}")
    
    # Inject anomaly (perturb one oscillator)
    print("\n3. Injecting anomaly...")
    anomaly_state = state.oscillator_state.at[0].set(jnp.array([5.0, 5.0, 5.0]))
    state = state._replace(oscillator_state=anomaly_state)
    
    # Measure energy after anomaly
    anomaly_energy = float(compute_ebm_energy(
        state.ebm_weights,
        state.oscillator_state,
        state.node_active_mask
    ))
    
    print(f"   Anomaly energy: {anomaly_energy:.3f}")
    print(f"   Energy increase: {anomaly_energy - baseline_energy:.3f}")
    
    threshold = baseline_energy * 1.5
    is_anomaly = anomaly_energy > threshold
    
    print(f"\n4. Anomaly detected: {is_anomaly}")
    print(f"   Threshold: {threshold:.3f}")
    print(f"   Confidence: {min((anomaly_energy / baseline_energy - 1) * 100, 100):.1f}%")
    
    print("\n✓ Anomaly detection demo complete!")


def main():
    """Run all demos."""
    print("\n" + "🌊" * 30)
    print(" " * 20 + "GMCS DEMO SUITE")
    print("🌊" * 30)
    
    # Basic simulation
    state = demo_basic_simulation()
    
    # Domain-specific demos
    heightmap = demo_terrain_generation(state)
    keystream = demo_encryption_keystream(state)
    demo_anomaly_detection()
    
    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Explore domain presets in src/config/presets.py")
    print("2. Start the API server: python -m src.main")
    print("3. Run the frontend: cd frontend && npm run dev")
    print("4. Check out the API docs: http://localhost:8000/docs")
    print("5. Read CONTRIBUTING.md to add your own domains!")
    print("\nHappy experimenting! 🚀✨")


if __name__ == "__main__":
    main()

