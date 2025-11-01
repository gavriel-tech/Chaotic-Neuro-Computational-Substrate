"""
Multi-GPU Training Example

Demonstrates how to use multiple GPUs for THRML training with near-linear speedup.

Run with:
    python examples/multi_gpu_training.py --n-gpus 4 --epochs 100

Requirements:
    - 2+ GPUs
    - JAX with CUDA support
    - THRML >= 0.1.3
"""

import argparse
import numpy as np
import jax
import jax.random as random
import time
from pathlib import Path

from src.core.thrml_integration import create_thrml_model
from src.core.multi_gpu import (
    create_multi_gpu_thrml_sampler,
    get_device_info
)


def create_synthetic_dataset(n_samples, n_nodes, seed=0):
    """
    Create synthetic dataset for training.
    
    Simulates patterns that THRML should learn.
    """
    np.random.seed(seed)
    
    # Create patterns: alternating stripes
    dataset = []
    for i in range(n_samples):
        # Stripe width varies
        stripe_width = np.random.randint(2, 8)
        pattern = np.zeros(n_nodes)
        
        for j in range(n_nodes):
            if (j // stripe_width) % 2 == 0:
                pattern[j] = 1.0
            else:
                pattern[j] = -1.0
        
        # Add noise
        noise_mask = np.random.rand(n_nodes) < 0.1
        pattern[noise_mask] *= -1
        
        dataset.append(pattern)
    
    return np.array(dataset)


def benchmark_single_vs_multi_gpu(
    wrapper,
    data_batch,
    n_chains,
    k_steps,
    key
):
    """
    Compare single-GPU vs multi-GPU training speed.
    """
    print("\n" + "-" * 60)
    print("Benchmarking: Single-GPU vs Multi-GPU")
    print("-" * 60)
    
    # Single-GPU (using standard CD)
    print(f"\nSingle-GPU: {n_chains} chains, CD-{k_steps}")
    t_start = time.time()
    
    for i in range(n_chains):
        key, subkey = random.split(key)
        wrapper.update_weights_cd(
            data_states=data_batch,
            eta=0.01,
            k_steps=k_steps,
            key=subkey,
            n_chains=1
        )
    
    single_gpu_time = time.time() - t_start
    print(f"  Time: {single_gpu_time:.3f}s")
    print(f"  Throughput: {n_chains / single_gpu_time:.1f} updates/sec")
    
    # Multi-GPU
    gpu_sampler = create_multi_gpu_thrml_sampler(wrapper)
    
    if gpu_sampler is None:
        print("\nMulti-GPU not available, skipping comparison")
        return
    
    print(f"\nMulti-GPU ({gpu_sampler.n_devices} GPUs): {n_chains} chains, CD-{k_steps}")
    t_start = time.time()
    
    key, subkey = random.split(key)
    gpu_sampler.parallel_cd_update(
        data_states=data_batch,
        eta=0.01,
        k_steps=k_steps,
        n_chains=n_chains,
        key=subkey
    )
    
    multi_gpu_time = time.time() - t_start
    print(f"  Time: {multi_gpu_time:.3f}s")
    print(f"  Throughput: {n_chains / multi_gpu_time:.1f} updates/sec")
    
    # Speedup
    speedup = single_gpu_time / multi_gpu_time
    efficiency = speedup / gpu_sampler.n_devices * 100
    
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.1f}%")
    print(f"(Ideal for {gpu_sampler.n_devices} GPUs: {gpu_sampler.n_devices:.2f}x)")


def train_multi_gpu(
    n_nodes,
    n_epochs,
    n_chains,
    k_steps,
    learning_rate,
    batch_size,
    n_gpus,
    seed
):
    """
    Train THRML model using multi-GPU.
    """
    print("\n" + "=" * 60)
    print("Multi-GPU Training")
    print("=" * 60)
    
    # Setup
    print(f"\nConfiguration:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Chains per update: {n_chains}")
    print(f"  CD-k steps: {k_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  GPUs: {n_gpus}")
    print(f"  Seed: {seed}")
    
    # Check devices
    device_info = get_device_info()
    print(f"\nAvailable devices: {device_info['n_devices']}")
    for device in device_info['devices']:
        print(f"  {device['id']}: {device['platform']} - {device['device_kind']}")
    
    if device_info['n_devices'] < 2:
        print("\nWARNING: Multi-GPU requires 2+ devices!")
        print("Running on single device...")
    
    # Create dataset
    print(f"\nCreating synthetic dataset...")
    dataset = create_synthetic_dataset(
        n_samples=batch_size * n_epochs,
        n_nodes=n_nodes,
        seed=seed
    )
    print(f"  Dataset shape: {dataset.shape}")
    
    # Initialize model
    print(f"\nInitializing THRML model...")
    weights = np.random.randn(n_nodes, n_nodes) * 0.01
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0)
    biases = np.zeros(n_nodes)
    
    wrapper = create_thrml_model(
        n_nodes=n_nodes,
        weights=weights,
        biases=biases,
        beta=1.0
    )
    
    # Create multi-GPU sampler
    gpu_sampler = create_multi_gpu_thrml_sampler(wrapper, n_devices=n_gpus)
    
    if gpu_sampler is None:
        print("WARNING: Multi-GPU sampler creation failed!")
        print("Falling back to single GPU")
        use_multi_gpu = False
    else:
        print(f"Multi-GPU sampler ready with {gpu_sampler.n_devices} devices")
        use_multi_gpu = True
    
    # Benchmark (optional)
    if use_multi_gpu and n_epochs > 10:
        key = random.PRNGKey(seed)
        benchmark_single_vs_multi_gpu(
            wrapper=wrapper,
            data_batch=dataset[0],
            n_chains=n_chains,
            k_steps=k_steps,
            key=key
        )
    
    # Training loop
    print("\n" + "-" * 60)
    print("Training")
    print("-" * 60)
    
    key = random.PRNGKey(seed + 1)
    
    history = {
        'epoch': [],
        'gradient_norm': [],
        'energy_diff': [],
        'data_energy': [],
        'model_energy': [],
        'wall_time': []
    }
    
    t_total_start = time.time()
    
    for epoch in range(n_epochs):
        # Get batch
        batch_idx = epoch % (len(dataset) // batch_size)
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = dataset[start_idx:end_idx]
        
        # Use first sample in batch as target
        data_sample = batch[0]
        
        key, subkey = random.split(key)
        t_epoch_start = time.time()
        
        # Update
        if use_multi_gpu:
            # Multi-GPU update
            diagnostics = gpu_sampler.parallel_cd_update(
                data_states=data_sample,
                eta=learning_rate,
                k_steps=k_steps,
                n_chains=n_chains,
                key=subkey
            )
        else:
            # Single-GPU update
            diagnostics = wrapper.update_weights_cd(
                data_states=data_sample,
                eta=learning_rate,
                k_steps=k_steps,
                key=subkey,
                n_chains=n_chains
            )
        
        epoch_time = time.time() - t_epoch_start
        
        # Record
        history['epoch'].append(epoch)
        history['gradient_norm'].append(diagnostics['gradient_norm'])
        history['energy_diff'].append(diagnostics['energy_diff'])
        history['data_energy'].append(diagnostics['data_energy'])
        history['model_energy'].append(diagnostics['model_energy'])
        history['wall_time'].append(epoch_time)
        
        # Print progress
        if epoch % max(1, n_epochs // 10) == 0:
            print(f"Epoch {epoch:4d}/{n_epochs}: "
                  f"grad_norm={diagnostics['gradient_norm']:7.4f}, "
                  f"E_diff={diagnostics['energy_diff']:7.3f}, "
                  f"time={epoch_time:.3f}s")
    
    t_total = time.time() - t_total_start
    
    # Summary
    print("\n" + "-" * 60)
    print("Training Complete")
    print("-" * 60)
    print(f"Total time: {t_total:.2f}s")
    print(f"Epochs/sec: {n_epochs / t_total:.2f}")
    print(f"Mean epoch time: {np.mean(history['wall_time']):.3f}s")
    print(f"Final gradient norm: {history['gradient_norm'][-1]:.4f}")
    print(f"Final energy diff: {history['energy_diff'][-1]:.3f}")
    
    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Gradient norm
        axes[0, 0].plot(history['epoch'], history['gradient_norm'])
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].set_title('Gradient Norm vs Epoch')
        axes[0, 0].grid(True)
        
        # Energy difference
        axes[0, 1].plot(history['epoch'], history['energy_diff'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Energy Difference')
        axes[0, 1].set_title('Energy Difference vs Epoch')
        axes[0, 1].grid(True)
        
        # Data vs Model energy
        axes[1, 0].plot(history['epoch'], history['data_energy'], label='Data')
        axes[1, 0].plot(history['epoch'], history['model_energy'], label='Model')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].set_title('Data vs Model Energy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Wall time per epoch
        axes[1, 1].plot(history['epoch'], history['wall_time'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].set_title('Wall Time per Epoch')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'multi_gpu_training.png'
        plt.savefig(output_path, dpi=150)
        print(f"\nPlot saved to {output_path}")
        
        # plt.show()
        
    except ImportError:
        print("\nMatplotlib not available, skipping plot")
    
    return wrapper, history


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU THRML Training')
    parser.add_argument('--n-nodes', type=int, default=256,
                       help='Number of nodes (default: 256)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--n-chains', type=int, default=8,
                       help='Number of parallel chains (default: 8)')
    parser.add_argument('--k-steps', type=int, default=1,
                       help='CD-k steps (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--n-gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Train
    wrapper, history = train_multi_gpu(
        n_nodes=args.n_nodes,
        n_epochs=args.epochs,
        n_chains=args.n_chains,
        k_steps=args.k_steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_gpus=args.n_gpus,
        seed=args.seed
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Visualize results in outputs/multi_gpu_training.png")
    print("  - Try different --n-chains values to find optimal throughput")
    print("  - Experiment with --k-steps for better gradient estimates")
    print("  - Scale up --n-nodes to stress-test GPU memory")


if __name__ == '__main__':
    main()

