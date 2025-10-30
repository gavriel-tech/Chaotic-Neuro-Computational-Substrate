"""
THRML Integration Demo

Demonstrates how GMCS uses THRML for advanced Energy-Based Model sampling.

This example shows:
1. Creating a THRML IsingEBM model from GMCS weights
2. Using THRML's block Gibbs sampling
3. Comparing THRML sampling vs custom CD-1

Run: python examples/thrml_integration_demo.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.core.state import initialize_system_state
from src.core.simulation import simulation_step
from src.core.ebm import (
    ebm_cd1_update,
    create_thrml_ebm_model,
    thrml_block_gibbs_sample,
    THRML_AVAILABLE
)


def demo_thrml_integration():
    """Demonstrate THRML integration with GMCS."""
    
    print("=" * 70)
    print("GMCS + THRML Integration Demo")
    print("=" * 70)
    
    if not THRML_AVAILABLE:
        print("\n⚠️  THRML not installed!")
        print("Install with: pip install thrml")
        print("\nFalling back to custom CD-1 implementation...")
        return demo_custom_cd1()
    
    print("\n✓ THRML is available!")
    print("  Using GPU-accelerated block Gibbs sampling\n")
    
    # Initialize GMCS simulation
    print("1. Initializing GMCS simulation...")
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key, n_max=16, grid_w=64, grid_h=64)
    
    # Activate nodes
    n_active = 8
    state = state._replace(
        node_active_mask=state.node_active_mask.at[:n_active].set(1.0)
    )
    
    # Run simulation to generate some dynamics
    print(f"2. Running simulation for 50 steps with {n_active} nodes...")
    for step in range(50):
        state = simulation_step(state)
        
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
    
    print(f"   Simulation time: {float(state.t[0]):.3f}s")
    
    # Extract learned weights
    weights = state.ebm_weights[:n_active, :n_active]
    biases = jnp.zeros(n_active)
    
    print(f"\n3. Creating THRML IsingEBM model...")
    print(f"   Active nodes: {n_active}")
    print(f"   Weight matrix shape: {weights.shape}")
    print(f"   Non-zero weights: {int(jnp.sum(jnp.abs(weights) > 1e-6))}")
    
    # Create THRML model
    thrml_model = create_thrml_ebm_model(
        n_nodes=n_active,
        ebm_weights=weights,
        biases=biases,
        beta=1.0
    )
    
    if thrml_model is None:
        print("   ✗ Failed to create THRML model")
        return
    
    print(f"   ✓ THRML model created")
    print(f"   Nodes: {len(thrml_model.nodes)}")
    print(f"   Edges: {len(thrml_model.edges)}")
    
    # Perform THRML block Gibbs sampling
    print(f"\n4. Performing THRML block Gibbs sampling...")
    print(f"   Warmup: 50 steps")
    print(f"   Samples: 100")
    
    key_sample = jax.random.PRNGKey(123)
    samples = thrml_block_gibbs_sample(
        model=thrml_model,
        n_samples=100,
        n_warmup=50,
        key=key_sample
    )
    
    print(f"   ✓ Sampling complete")
    print(f"   Sample shape: {samples.shape}")
    
    # Analyze samples
    print(f"\n5. Sample statistics:")
    mean_magnetization = float(jnp.mean(samples))
    std_magnetization = float(jnp.std(samples))
    
    print(f"   Mean magnetization: {mean_magnetization:.4f}")
    print(f"   Std magnetization: {std_magnetization:.4f}")
    
    # Compute correlations
    correlations = jnp.corrcoef(samples.T)
    print(f"   Mean correlation: {float(jnp.mean(jnp.abs(correlations))):.4f}")
    
    print(f"\n✓ THRML integration demo complete!")
    print(f"\nTHRML provides:")
    print(f"  • Efficient block Gibbs sampling")
    print(f"  • GPU acceleration via JAX")
    print(f"  • Support for heterogeneous graphical models")
    print(f"  • Optimized for Extropic hardware prototyping")


def demo_custom_cd1():
    """Fallback demo using custom CD-1 implementation."""
    
    print("\n" + "=" * 70)
    print("Custom CD-1 Implementation Demo (THRML not available)")
    print("=" * 70)
    
    key = jax.random.PRNGKey(42)
    state = initialize_system_state(key, n_max=16)
    state = state._replace(
        node_active_mask=state.node_active_mask.at[:8].set(1.0)
    )
    
    print("\nRunning simulation with custom CD-1...")
    for step in range(100):
        state = simulation_step(state)
        
        if step % 10 == 0 and step > 0:
            new_weights, new_key = ebm_cd1_update(
                state.ebm_weights,
                state.oscillator_state,
                state.node_active_mask,
                state.key,
                eta=0.01
            )
            state = state._replace(ebm_weights=new_weights, key=new_key)
    
    max_weight = float(jnp.max(jnp.abs(state.ebm_weights[:8, :8])))
    print(f"\nSimulation complete!")
    print(f"Max weight magnitude: {max_weight:.4f}")
    print(f"\nInstall THRML for enhanced capabilities:")
    print(f"  pip install thrml")


def main():
    """Run the demo."""
    
    print("\n" + "🌊" * 35)
    print(" " * 20 + "GMCS + THRML")
    print("🌊" * 35)
    
    demo_thrml_integration()
    
    print("\n" + "=" * 70)
    print("For more information:")
    print("  THRML: https://github.com/extropic-ai/thrml")
    print("  THRML Docs: https://docs.thrml.ai")
    print("  Gavriel Fork: https://github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate")
    print("=" * 70)


if __name__ == "__main__":
    main()

