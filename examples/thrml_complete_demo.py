"""
Complete THRML Demo - All Features

This script demonstrates all major THRML features:
- Basic sampling
- Multiple blocking strategies
- Multi-chain sampling
- Contrastive Divergence learning
- Conditional sampling (inpainting)
- Benchmarking
- Multi-GPU (if available)

Run with:
    python examples/thrml_complete_demo.py
"""

import numpy as np
import jax
import jax.random as random
import matplotlib.pyplot as plt
from pathlib import Path

# Import THRML integration
from src.core.thrml_integration import (
    THRMLWrapper,
    create_thrml_model
)
from src.core.blocking_strategies import list_strategies
from src.core.multi_gpu import (
    create_multi_gpu_thrml_sampler,
    get_device_info
)
from src.tools.thrml_benchmark import THRMLBenchmark


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def demo_basic_sampling():
    """Demo 1: Basic THRML sampling."""
    print_section("Demo 1: Basic Sampling")
    
    # Create a simple Ising model (8x8 grid = 64 nodes)
    n_nodes = 64
    
    # Initialize random weights (small values)
    weights = np.random.randn(n_nodes, n_nodes) * 0.1
    weights = (weights + weights.T) / 2  # Make symmetric
    np.fill_diagonal(weights, 0)
    
    # Add nearest-neighbor coupling (grid structure)
    grid_size = 8
    J = 0.5  # Coupling strength
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            # Right neighbor
            if j + 1 < grid_size:
                idx2 = i * grid_size + (j + 1)
                weights[idx, idx2] = J
                weights[idx2, idx] = J
            # Bottom neighbor
            if i + 1 < grid_size:
                idx2 = (i + 1) * grid_size + j
                weights[idx, idx2] = J
                weights[idx2, idx] = J
    
    biases = np.zeros(n_nodes)
    
    # Create wrapper
    wrapper = create_thrml_model(
        n_nodes=n_nodes,
        weights=weights,
        biases=biases,
        beta=1.0
    )
    
    print(f"Created THRML wrapper with {n_nodes} nodes")
    print(f"Edges: {len(wrapper.edges)}")
    
    # Check health
    health = wrapper.get_health_status()
    print(f"Healthy: {health['healthy']}")
    print(f"Current strategy: {health['current_strategy']}")
    
    # Sample
    key = random.PRNGKey(42)
    samples = wrapper.sample_gibbs(
        n_steps=100,
        temperature=1.0,
        key=key
    )
    
    print(f"\nSampled states shape: {samples.shape}")
    print(f"Unique values: {np.unique(samples)}")
    print(f"Magnetization: {np.mean(samples):.3f}")
    
    # Compute energy
    energy = wrapper.compute_energy(samples)
    print(f"Energy: {energy:.3f}")
    
    return wrapper, key


def demo_blocking_strategies(wrapper, key):
    """Demo 2: Compare blocking strategies."""
    print_section("Demo 2: Blocking Strategies")
    
    strategies = ['checkerboard', 'random', 'stripes', 'supercell']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy}...")
        
        # Set strategy
        wrapper.set_blocking_strategy(strategy)
        
        # Validate
        validation = wrapper.validate_current_blocks()
        print(f"  Valid: {validation['valid']}")
        print(f"  Balance: {validation['balance_score']:.3f}")
        
        # Sample and time
        import time
        t_start = time.time()
        
        n_samples = 50
        for _ in range(n_samples):
            key, subkey = random.split(key)
            samples = wrapper.sample_gibbs(
                n_steps=2,
                temperature=1.0,
                key=subkey
            )
        
        elapsed = time.time() - t_start
        samples_per_sec = n_samples / elapsed
        
        results[strategy] = samples_per_sec
        print(f"  Samples/sec: {samples_per_sec:.1f}")
    
    # Print summary
    print("\nSummary:")
    best = max(results, key=results.get)
    for strategy, sps in sorted(results.items(), key=lambda x: -x[1]):
        marker = "←" if strategy == best else ""
        print(f"  {strategy:15s}: {sps:6.1f} samples/sec {marker}")
    
    return results


def demo_multi_chain_sampling(wrapper, key):
    """Demo 3: Multi-chain parallel sampling."""
    print_section("Demo 3: Multi-Chain Sampling")
    
    # Sample multiple chains
    n_chains = 4
    print(f"Sampling {n_chains} chains in parallel...")
    
    all_samples, diagnostics = wrapper.sample_gibbs_parallel(
        n_steps=50,
        temperature=1.0,
        n_chains=n_chains,
        key=key
    )
    
    print(f"\nResults shape: {all_samples.shape}")
    print(f"Expected: ({n_chains}, {wrapper.n_nodes})")
    
    # Show per-chain diagnostics
    print("\nPer-chain diagnostics:")
    for i, diag in enumerate(diagnostics):
        print(f"  Chain {i}: energy={diag['energy']:.3f}, "
              f"mag={diag['magnetization']:.3f}")
    
    # Compute statistics
    energies = [d['energy'] for d in diagnostics]
    mags = [d['magnetization'] for d in diagnostics]
    
    print(f"\nAcross chains:")
    print(f"  Mean energy: {np.mean(energies):.3f} ± {np.std(energies):.3f}")
    print(f"  Mean magnetization: {np.mean(mags):.3f} ± {np.std(mags):.3f}")


def demo_cd_learning(wrapper, key):
    """Demo 4: Contrastive Divergence learning."""
    print_section("Demo 4: Contrastive Divergence Learning")
    
    # Create synthetic "data" (random binary states)
    data_states = np.random.choice([-1.0, 1.0], size=wrapper.n_nodes)
    data_energy = wrapper.compute_energy(data_states)
    
    print(f"Data energy (before): {data_energy:.3f}")
    
    # Run CD-1 learning for a few steps
    print("\nRunning CD-1 learning...")
    for epoch in range(5):
        key, subkey = random.split(key)
        
        diagnostics = wrapper.update_weights_cd(
            data_states=data_states,
            eta=0.01,  # Learning rate
            k_steps=1,  # CD-1
            key=subkey,
            n_chains=1
        )
        
        print(f"  Epoch {epoch}: "
              f"grad_norm={diagnostics['gradient_norm']:.4f}, "
              f"energy_diff={diagnostics['energy_diff']:.4f}")
    
    # Check final energy
    final_data_energy = wrapper.compute_energy(data_states)
    print(f"\nData energy (after): {final_data_energy:.3f}")
    print(f"Change: {final_data_energy - data_energy:.3f}")


def demo_conditional_sampling(wrapper, key):
    """Demo 5: Conditional sampling (inpainting)."""
    print_section("Demo 5: Conditional Sampling")
    
    # Create a known pattern
    full_sample = np.ones(wrapper.n_nodes)
    full_sample[::2] = -1  # Alternating pattern
    
    print("Original pattern (first 16 values):")
    print(full_sample[:16])
    
    # Clamp first half
    n_clamped = wrapper.n_nodes // 2
    clamped_ids = list(range(n_clamped))
    clamped_values = full_sample[:n_clamped].tolist()
    
    wrapper.set_clamped_nodes(
        node_ids=clamped_ids,
        values=clamped_values
    )
    
    print(f"\nClamped {n_clamped} nodes")
    
    # Sample conditional
    conditional_sample = wrapper.sample_conditional(
        n_steps=100,
        temperature=0.5,  # Low temp for more deterministic
        key=key
    )
    
    print("\nConditional sample (first 16 values):")
    print(conditional_sample[:16])
    
    # Verify clamped nodes didn't change
    clamped_match = np.allclose(
        conditional_sample[:n_clamped],
        full_sample[:n_clamped]
    )
    print(f"\nClamped nodes preserved: {clamped_match}")
    
    # Clear clamps
    wrapper.clear_clamped_nodes()


def demo_benchmarking(wrapper):
    """Demo 6: Benchmarking."""
    print_section("Demo 6: Benchmarking")
    
    # Get diagnostics
    diagnostics = wrapper.get_benchmark_diagnostics()
    
    print("Current performance:")
    print(f"  Samples/sec: {diagnostics['samples_per_sec']:.1f}")
    print(f"  ESS/sec: {diagnostics['ess_per_sec']:.1f}")
    print(f"  Lag-1 autocorr: {diagnostics['lag1_autocorr']:.3f}")
    print(f"  τ_int: {diagnostics['tau_int']:.1f}")
    print(f"  Total samples: {diagnostics['total_samples']}")
    
    # Export to leaderboard format
    leaderboard_json = wrapper.get_benchmark_json()
    print(f"\nLeaderboard format:")
    print(f"  Device: {leaderboard_json['device_type']}")
    print(f"  Blocking: {leaderboard_json['blocking']}")
    print(f"  Samples/sec: {leaderboard_json['samples_per_sec']:.1f}")


def demo_multi_gpu(wrapper, key):
    """Demo 7: Multi-GPU sampling (if available)."""
    print_section("Demo 7: Multi-GPU Sampling")
    
    # Check device info
    device_info = get_device_info()
    print(f"Devices available: {device_info['n_devices']}")
    print(f"Default backend: {device_info['default_backend']}")
    
    for device in device_info['devices']:
        print(f"  Device {device['id']}: {device['platform']} - {device['device_kind']}")
    
    # Try creating multi-GPU sampler
    if device_info['n_devices'] < 2:
        print("\nMulti-GPU not available (need 2+ devices)")
        print("Skipping multi-GPU demo")
        return
    
    print(f"\nCreating multi-GPU sampler...")
    gpu_sampler = create_multi_gpu_thrml_sampler(wrapper, n_devices=None)
    
    if gpu_sampler is None:
        print("Failed to create multi-GPU sampler")
        return
    
    # Sample across GPUs
    print(f"Sampling 16 chains across {gpu_sampler.n_devices} GPUs...")
    
    samples, diagnostics = gpu_sampler.sample_parallel_chains(
        n_chains=16,
        n_steps=50,
        temperature=1.0,
        key=key
    )
    
    print(f"\nResults:")
    print(f"  Samples shape: {samples.shape}")
    print(f"  Chains: {len(diagnostics)}")
    
    # Show device distribution
    device_counts = {}
    for diag in diagnostics:
        dev_id = diag['device_id']
        device_counts[dev_id] = device_counts.get(dev_id, 0) + 1
    
    print(f"\nChains per device:")
    for dev_id, count in sorted(device_counts.items()):
        print(f"  Device {dev_id}: {count} chains")


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "THRML Complete Demo" + " " * 21 + "#")
    print("#" * 60)
    
    # Run demos
    wrapper, key = demo_basic_sampling()
    
    key, subkey = random.split(key)
    demo_blocking_strategies(wrapper, subkey)
    
    key, subkey = random.split(key)
    demo_multi_chain_sampling(wrapper, subkey)
    
    key, subkey = random.split(key)
    demo_cd_learning(wrapper, subkey)
    
    key, subkey = random.split(key)
    demo_conditional_sampling(wrapper, subkey)
    
    demo_benchmarking(wrapper)
    
    key, subkey = random.split(key)
    demo_multi_gpu(wrapper, subkey)
    
    print_section("Demo Complete!")
    print("All THRML features demonstrated successfully.")
    print("\nNext steps:")
    print("  - Read docs/THRML_INTEGRATION_GUIDE.md")
    print("  - Try python -m src.tools.thrml_benchmark --help")
    print("  - Explore other examples in examples/")


if __name__ == '__main__':
    main()

