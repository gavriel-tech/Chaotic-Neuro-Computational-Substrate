"""
THRML Benchmarking System.

Provides comprehensive benchmarking utilities for THRML sampling performance,
compatible with thrmlbench.com leaderboard format.

Features:
- Samples/sec measurement
- Autocorrelation tracking (magnetization)
- Integrated autocorrelation time (τ_int) via AR(1)
- Effective sample size per second (ESS/sec)
- Energy trajectory tracking
- Leaderboard JSON export
- Comparison across blocking strategies
- Visualization (autocorr plots, mixing curves)
"""

import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import jax
import jax.numpy as jnp

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from src.core.thrml_integration import THRMLWrapper, create_thrml_model
from src.core.blocking_strategies import list_strategies


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    # Configuration
    strategy: str
    n_nodes: int
    grid_shape: Tuple[int, int]
    J: float  # Coupling strength
    beta: float  # Inverse temperature
    n_chains: int
    seed: int
    
    # Schedule
    n_warmup: int
    n_samples: int
    steps_per_sample: int
    
    # Performance metrics
    wall_time: float  # Total time in seconds
    samples_per_sec: float
    lag1_autocorr: float
    tau_int_est: float
    ess_per_sec: float
    
    # Physics
    mean_energy: float
    std_energy: float
    mean_magnetization: float
    std_magnetization: float
    
    # System info
    device_type: str  # 'cpu' or 'gpu'
    device_name: str
    jax_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_leaderboard_json(self) -> Dict[str, Any]:
        """Convert to thrmlbench.com leaderboard format."""
        H, W = self.grid_shape
        return {
            'device': self.device_name,
            'device_type': self.device_type,
            'H': H,
            'W': W,
            'n_nodes': self.n_nodes,
            'blocking': self.strategy,
            'n_chains': self.n_chains,
            'J': self.J,
            'beta': self.beta,
            'samples_per_sec': self.samples_per_sec,
            'lag1_autocorr': self.lag1_autocorr,
            'tau_int_est': self.tau_int_est,
            'ess_per_sec': self.ess_per_sec,
            'mean_energy': self.mean_energy,
            'timestamp': time.time(),
            'warmup': self.n_warmup,
            'n_samples': self.n_samples,
            'steps_per_sample': self.steps_per_sample,
            'seed': self.seed
        }


# ============================================================================
# Benchmark Runner
# ============================================================================

class THRMLBenchmark:
    """
    THRML benchmarking utility.
    
    Runs standardized benchmarks for THRML sampling performance,
    measuring throughput (samples/sec) and mixing (ESS/sec).
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize benchmark runner.
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
    
    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"[Benchmark] {message}")
    
    def run_single(
        self,
        n_nodes: int,
        strategy: str = "checkerboard",
        J: float = 0.35,
        beta: float = 1.0,
        n_warmup: int = 100,
        n_samples: int = 128,
        steps_per_sample: int = 1,
        n_chains: int = 1,
        seed: int = 0
    ) -> BenchmarkResult:
        """
        Run a single benchmark.
        
        Args:
            n_nodes: Number of nodes (will create square grid if possible)
            strategy: Blocking strategy name
            J: Coupling strength
            beta: Inverse temperature
            n_warmup: Warmup steps
            n_samples: Number of samples to collect
            steps_per_sample: Gibbs steps per sample
            n_chains: Number of parallel chains
            seed: Random seed
            
        Returns:
            BenchmarkResult with all metrics
        """
        self._log(f"Running benchmark: {strategy}, n={n_nodes}, chains={n_chains}, seed={seed}")
        
        # Create grid shape
        grid_size = int(np.ceil(np.sqrt(n_nodes)))
        grid_shape = (grid_size, grid_size)
        actual_n_nodes = grid_size * grid_size
        
        # Create model
        weights = np.random.randn(actual_n_nodes, actual_n_nodes) * 0.01
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        # Add nearest-neighbor coupling
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if i + 1 < grid_size:
                    idx2 = (i + 1) * grid_size + j
                    weights[idx, idx2] = J
                    weights[idx2, idx] = J
                if j + 1 < grid_size:
                    idx2 = i * grid_size + (j + 1)
                    weights[idx, idx2] = J
                    weights[idx2, idx] = J
        
        biases = np.zeros(actual_n_nodes)
        
        wrapper = THRMLWrapper(actual_n_nodes, weights, biases, beta=beta)
        
        # Set blocking strategy
        wrapper.set_blocking_strategy(strategy)
        
        # Create keys for sampling
        key = jax.random.PRNGKey(seed)
        
        # Warmup
        self._log(f"Warmup: {n_warmup} steps...")
        wrapper.sample_gibbs(n_steps=n_warmup, temperature=1.0/beta, key=key)
        
        # Main sampling loop
        self._log(f"Sampling: {n_samples} samples × {steps_per_sample} sps × {n_chains} chains...")
        t_start = time.time()
        
        all_samples = []
        magnetizations = []
        energies = []
        
        for i in range(n_samples):
            key, subkey = jax.random.split(key)
            
            if n_chains > 1:
                samples, diags = wrapper.sample_gibbs_parallel(
                    n_steps=steps_per_sample,
                    temperature=1.0/beta,
                    n_chains=n_chains,
                    key=subkey
                )
                # Take first chain for consistency
                sample = samples[0]
            else:
                sample = wrapper.sample_gibbs(
                    n_steps=steps_per_sample,
                    temperature=1.0/beta,
                    key=subkey
                )
            
            all_samples.append(sample)
            
            # Compute magnetization and energy
            m = float(np.mean(sample))
            e = wrapper.compute_energy(sample)
            
            magnetizations.append(m)
            energies.append(e)
        
        wall_time = time.time() - t_start
        
        # Compute metrics
        samples_per_sec = n_samples / wall_time
        
        # Autocorrelation
        m_arr = np.array(magnetizations)
        m_centered = m_arr - np.mean(m_arr)
        if len(m_centered) > 1 and np.std(m_centered) > 1e-10:
            lag1_autocorr = np.corrcoef(m_centered[:-1], m_centered[1:])[0, 1]
        else:
            lag1_autocorr = 0.0
        
        # Integrated autocorrelation time (AR(1) approximation)
        rho1 = np.clip(lag1_autocorr, -0.999, 0.999)
        tau_int = (1.0 + rho1) / (1.0 - rho1) if abs(rho1) < 0.999 else 100.0
        tau_int = max(1.0, tau_int)
        
        # ESS/sec
        ess_per_sec = samples_per_sec / tau_int
        
        # Physics metrics
        mean_energy = float(np.mean(energies))
        std_energy = float(np.std(energies))
        mean_magnetization = float(np.mean(magnetizations))
        std_magnetization = float(np.std(magnetizations))
        
        # Device info
        devices = jax.devices()
        device_type = "gpu" if any("gpu" in str(d).lower() for d in devices) else "cpu"
        device_name = str(devices[0])
        
        result = BenchmarkResult(
            strategy=strategy,
            n_nodes=actual_n_nodes,
            grid_shape=grid_shape,
            J=J,
            beta=beta,
            n_chains=n_chains,
            seed=seed,
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample,
            wall_time=wall_time,
            samples_per_sec=samples_per_sec,
            lag1_autocorr=lag1_autocorr,
            tau_int_est=tau_int,
            ess_per_sec=ess_per_sec,
            mean_energy=mean_energy,
            std_energy=std_energy,
            mean_magnetization=mean_magnetization,
            std_magnetization=std_magnetization,
            device_type=device_type,
            device_name=device_name,
            jax_version=jax.__version__
        )
        
        self._log(f"Result: {samples_per_sec:.1f} samples/sec, ESS/sec={ess_per_sec:.1f}, τ_int={tau_int:.1f}")
        
        self.results.append(result)
        return result
    
    def compare_strategies(
        self,
        strategies: List[str],
        n_nodes: int = 784,  # 28×28
        n_samples: int = 128,
        **kwargs
    ) -> List[BenchmarkResult]:
        """
        Compare multiple blocking strategies.
        
        Args:
            strategies: List of strategy names
            n_nodes: Number of nodes
            n_samples: Samples per strategy
            **kwargs: Additional arguments for run_single
            
        Returns:
            List of BenchmarkResults
        """
        results = []
        for strategy in strategies:
            self._log(f"\n=== Testing strategy: {strategy} ===")
            result = self.run_single(
                n_nodes=n_nodes,
                strategy=strategy,
                n_samples=n_samples,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def export_json(self, filepath: str, leaderboard_format: bool = False):
        """
        Export results to JSON.
        
        Args:
            filepath: Output file path
            leaderboard_format: Use thrmlbench.com format
        """
        if leaderboard_format:
            data = [r.to_leaderboard_json() for r in self.results]
        else:
            data = [r.to_dict() for r in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._log(f"Exported {len(self.results)} results to {filepath}")
    
    def plot_comparison(
        self,
        metric: str = "ess_per_sec",
        filepath: Optional[str] = None
    ):
        """
        Plot comparison of strategies.
        
        Args:
            metric: Metric to plot ('ess_per_sec', 'samples_per_sec', 'tau_int')
            filepath: Save figure to file (optional)
            
        Returns:
            Matplotlib Figure (if matplotlib available)
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for plotting. Install with: pip install matplotlib")
        
        if not self.results:
            raise ValueError("No results to plot")
        
        strategies = [r.strategy for r in self.results]
        values = [getattr(r, metric) for r in self.results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(strategies, values)
        ax.set_xlabel('Blocking Strategy')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'THRML Benchmark: {metric.replace("_", " ").title()}')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if filepath:
            fig.savefig(filepath, dpi=150)
            self._log(f"Saved plot to {filepath}")
        
        return fig


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="THRML Benchmarking Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--strategy',
        default='checkerboard',
        choices=['checkerboard', 'random', 'stripes', 'supercell', 'graph-coloring', 'all'],
        help='Blocking strategy to benchmark'
    )
    parser.add_argument('--n-nodes', type=int, default=784, help='Number of nodes (will create square grid)')
    parser.add_argument('--J', type=float, default=0.35, help='Coupling strength')
    parser.add_argument('--beta', type=float, default=1.0, help='Inverse temperature')
    parser.add_argument('--warmup', type=int, default=100, help='Warmup steps')
    parser.add_argument('--n-samples', type=int, default=128, help='Number of samples')
    parser.add_argument('--sps', type=int, default=1, help='Steps per sample')
    parser.add_argument('--n-chains', type=int, default=1, help='Number of parallel chains')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--output', default='benchmark_results.json', help='Output JSON file')
    parser.add_argument('--leaderboard', action='store_true', help='Use thrmlbench.com format')
    parser.add_argument('--plot', default=None, help='Save comparison plot to file')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Create benchmark runner
    benchmark = THRMLBenchmark(verbose=not args.quiet)
    
    # Run benchmark(s)
    if args.strategy == 'all':
        strategies = ['checkerboard', 'random', 'stripes', 'supercell', 'graph-coloring']
        benchmark.compare_strategies(
            strategies=strategies,
            n_nodes=args.n_nodes,
            J=args.J,
            beta=args.beta,
            n_warmup=args.warmup,
            n_samples=args.n_samples,
            steps_per_sample=args.sps,
            n_chains=args.n_chains,
            seed=args.seed
        )
    else:
        benchmark.run_single(
            n_nodes=args.n_nodes,
            strategy=args.strategy,
            J=args.J,
            beta=args.beta,
            n_warmup=args.warmup,
            n_samples=args.n_samples,
            steps_per_sample=args.sps,
            n_chains=args.n_chains,
            seed=args.seed
        )
    
    # Export results
    benchmark.export_json(args.output, leaderboard_format=args.leaderboard)
    
    # Plot if requested
    if args.plot:
        benchmark.plot_comparison(metric='ess_per_sec', filepath=args.plot)
    
    print("\n=== Benchmark Complete ===")
    print(f"Results saved to: {args.output}")
    if args.plot:
        print(f"Plot saved to: {args.plot}")


if __name__ == '__main__':
    main()

