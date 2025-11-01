"""
Conditional Sampling Demo

Demonstrates clamping nodes, multi-chain parallelism, and blocking strategies
for the GMCS platform using the THRML backend.

This example shows:
1. Audio inpainting (fixing corrupted regions)
2. Constrained synthesis (user-controlled patterns)
3. Pattern completion (ML-style inference)
4. Multi-chain sampling for better gradient estimates
5. Blocking strategy comparison

Usage:
    python examples/conditional_sampling_demo.py
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import GMCS sampler backend
from src.core.thrml_sampler_backend import THRMLSamplerBackend


def create_test_model(n_nodes: int = 64, coupling_strength: float = 0.1) -> THRMLSamplerBackend:
    """
    Create a test THRML backend with random coupling.
    
    Args:
        n_nodes: Number of nodes
        coupling_strength: Strength of random coupling
        
    Returns:
        Initialized THRMLSamplerBackend
    """
    # Random coupling matrix (symmetric)
    weights = np.random.randn(n_nodes, n_nodes) * coupling_strength
    weights = (weights + weights.T) / 2  # Symmetrize
    np.fill_diagonal(weights, 0)  # No self-coupling
    
    # Random biases
    biases = np.random.randn(n_nodes) * 0.1
    
    # Create backend
    backend = THRMLSamplerBackend(
        n_nodes=n_nodes,
        initial_weights=weights,
        initial_biases=biases,
        beta=1.0
    )
    
    print(f"Created THRML backend with {n_nodes} nodes")
    print(f"Coupling strength: {coupling_strength:.3f}")
    
    return backend


def demo_audio_inpainting(backend: THRMLSamplerBackend):
    """
    Demo 1: Audio Inpainting
    
    Simulate corrupted audio by zeroing out a region, then use conditional
    sampling to restore it.
    """
    print("\n" + "="*60)
    print("DEMO 1: Audio Inpainting")
    print("="*60)
    
    n_nodes = backend.n_nodes
    
    # Generate "clean" audio pattern
    samples_clean, _ = backend.sample(n_steps=10, temperature=1.0)
    clean_pattern = samples_clean[0]
    
    print(f"Generated clean pattern: {clean_pattern[:8]} ...")
    
    # Corrupt middle region (nodes 24-40)
    corrupted_start = 24
    corrupted_end = 40
    corrupted_pattern = clean_pattern.copy()
    corrupted_pattern[corrupted_start:corrupted_end] = 0  # Simulate corruption
    
    print(f"Corrupted region: nodes {corrupted_start}-{corrupted_end}")
    
    # Clamp known good regions
    known_good_ids = list(range(corrupted_start)) + list(range(corrupted_end, n_nodes))
    known_good_values = corrupted_pattern[known_good_ids].tolist()
    
    backend.set_conditional_nodes(known_good_ids, known_good_values)
    print(f"Clamped {len(known_good_ids)} nodes (known good regions)")
    
    # Sample to inpaint corrupted region
    samples_inpainted, _ = backend.sample(n_steps=20, temperature=0.8)
    inpainted_pattern = samples_inpainted[0]
    
    backend.clear_conditional_nodes()
    
    # Compare
    print(f"\nOriginal (clean):    {clean_pattern[corrupted_start:corrupted_end]}")
    print(f"Corrupted:           {corrupted_pattern[corrupted_start:corrupted_end]}")
    print(f"Inpainted:           {inpainted_pattern[corrupted_start:corrupted_end]}")
    
    # Compute restoration accuracy
    accuracy = np.mean(clean_pattern[corrupted_start:corrupted_end] == inpainted_pattern[corrupted_start:corrupted_end])
    print(f"\nRestoration accuracy: {accuracy*100:.1f}%")


def demo_constrained_synthesis(backend: THRMLSamplerBackend):
    """
    Demo 2: Constrained Synthesis
    
    User pins specific nodes to create a rhythm, then samples variations.
    """
    print("\n" + "="*60)
    print("DEMO 2: Constrained Synthesis")
    print("="*60)
    
    # Pin rhythm nodes (every 8th node = quarter notes)
    rhythm_nodes = [0, 8, 16, 24, 32, 40, 48, 56]
    rhythm_values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # All active
    
    backend.set_conditional_nodes(rhythm_nodes, rhythm_values)
    print(f"Pinned rhythm nodes: {rhythm_nodes}")
    print(f"Rhythm pattern: {rhythm_values}")
    
    # Sample 3 variations
    print("\nGenerating 3 variations:")
    for i in range(3):
        samples, _ = backend.sample(n_steps=10, temperature=1.5)
        pattern = samples[0]
        
        print(f"Variation {i+1}: {pattern[:16]} ...")
        print(f"  Rhythm nodes: {pattern[rhythm_nodes]} (should all be 1.0)")
    
    backend.clear_conditional_nodes()


def demo_pattern_completion(backend: THRMLSamplerBackend):
    """
    Demo 3: Pattern Completion
    
    Observe first half of a pattern, infer the second half.
    """
    print("\n" + "="*60)
    print("DEMO 3: Pattern Completion")
    print("="*60)
    
    n_nodes = backend.n_nodes
    
    # Generate full pattern
    samples_full, _ = backend.sample(n_steps=10, temperature=1.0)
    full_pattern = samples_full[0]
    
    # Observe only first half
    observed_nodes = list(range(n_nodes // 2))
    observed_values = full_pattern[:n_nodes // 2].tolist()
    
    backend.set_conditional_nodes(observed_nodes, observed_values)
    print(f"Observed first {len(observed_nodes)} nodes")
    
    # Infer second half
    samples_completed, _ = backend.sample(n_steps=15, temperature=0.8)
    completed_pattern = samples_completed[0]
    
    backend.clear_conditional_nodes()
    
    # Compare
    print(f"\nOriginal second half: {full_pattern[n_nodes//2:]}")
    print(f"Inferred second half: {completed_pattern[n_nodes//2:]}")
    
    # Compute completion accuracy
    accuracy = np.mean(full_pattern[n_nodes//2:] == completed_pattern[n_nodes//2:])
    print(f"\nCompletion accuracy: {accuracy*100:.1f}%")


def demo_multi_chain_sampling(backend: THRMLSamplerBackend):
    """
    Demo 4: Multi-Chain Sampling
    
    Compare single-chain vs multi-chain sampling for gradient estimation.
    """
    print("\n" + "="*60)
    print("DEMO 4: Multi-Chain Sampling")
    print("="*60)
    
    # Single chain
    print("\nSingle-chain sampling:")
    samples_single, diag_single = backend.sample(
        n_steps=10,
        temperature=1.0,
        num_chains=1
    )
    print(f"  Samples shape: {samples_single.shape}")
    print(f"  Energy: {diag_single.get('energy', 'N/A'):.3f}")
    
    # Multi-chain (auto-detect)
    print("\nMulti-chain sampling (auto-detect):")
    samples_multi, diag_multi = backend.sample(
        n_steps=10,
        temperature=1.0,
        num_chains=-1  # Auto-detect
    )
    print(f"  Samples shape: {samples_multi.shape}")
    print(f"  Num chains: {diag_multi.get('num_chains', 'N/A')}")
    print(f"  Mean energy: {diag_multi.get('mean_energy', 'N/A'):.3f}")
    
    # Per-chain diagnostics
    if 'per_chain' in diag_multi:
        print("\n  Per-chain diagnostics:")
        for i, chain_diag in enumerate(diag_multi['per_chain']):
            print(f"    Chain {i}: energy={chain_diag['energy']:.3f}, mag={chain_diag['magnetization']:.3f}")
    
    # Specific chain count
    print("\nMulti-chain sampling (4 chains):")
    samples_4chain, diag_4chain = backend.sample(
        n_steps=10,
        temperature=1.0,
        num_chains=4
    )
    print(f"  Samples shape: {samples_4chain.shape}")
    print(f"  Mean energy: {diag_4chain.get('mean_energy', 'N/A'):.3f}")


def demo_blocking_strategies(backend: THRMLSamplerBackend):
    """
    Demo 5: Blocking Strategy Comparison
    
    Compare different blocking strategies for sampling performance.
    """
    print("\n" + "="*60)
    print("DEMO 5: Blocking Strategy Comparison")
    print("="*60)
    
    strategies = ["checkerboard", "random", "stripes", "supercell"]
    
    # Generate grid positions for spatial strategies
    n_nodes = backend.n_nodes
    grid_size = int(np.ceil(np.sqrt(n_nodes)))
    positions = np.array([
        [i % grid_size, i // grid_size]
        for i in range(n_nodes)
    ], dtype=np.float32)
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        # Set strategy
        try:
            backend.thrml_wrapper.set_blocking_strategy(
                strategy_name=strategy,
                node_positions=positions
            )
            
            # Validate
            validation = backend.thrml_wrapper.validate_current_blocks()
            print(f"  Valid: {validation['valid']}")
            print(f"  Balance score: {validation['balance_score']:.3f}")
            
            # Sample and measure performance
            import time
            t_start = time.time()
            samples, diagnostics = backend.sample(n_steps=10, temperature=1.0)
            wall_time = time.time() - t_start
            
            results[strategy] = {
                'wall_time': wall_time,
                'samples_per_sec': diagnostics.get('samples_per_sec', 0),
                'ess_per_sec': diagnostics.get('ess_per_sec', 0),
                'valid': validation['valid']
            }
            
            print(f"  Wall time: {wall_time:.4f}s")
            print(f"  Samples/sec: {results[strategy]['samples_per_sec']:.2f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[strategy] = {'error': str(e)}
    
    # Summary
    print("\n" + "-"*60)
    print("STRATEGY COMPARISON SUMMARY")
    print("-"*60)
    for strategy, result in results.items():
        if 'error' not in result:
            print(f"{strategy:15s}: {result['wall_time']:.4f}s, {result['samples_per_sec']:.2f} sps, valid={result['valid']}")
        else:
            print(f"{strategy:15s}: ERROR - {result['error']}")


def main():
    """
    Run all conditional sampling demos.
    """
    print("="*60)
    print("GMCS Conditional Sampling Demo")
    print("="*60)
    print("\nThis demo showcases:")
    print("  1. Audio inpainting (fixing corrupted regions)")
    print("  2. Constrained synthesis (user-controlled patterns)")
    print("  3. Pattern completion (ML-style inference)")
    print("  4. Multi-chain sampling (parallel chains)")
    print("  5. Blocking strategy comparison")
    
    # Create backend
    backend = create_test_model(n_nodes=64, coupling_strength=0.1)
    
    # Run demos
    demo_audio_inpainting(backend)
    demo_constrained_synthesis(backend)
    demo_pattern_completion(backend)
    demo_multi_chain_sampling(backend)
    demo_blocking_strategies(backend)
    
    print("\n" + "="*60)
    print("All demos completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

