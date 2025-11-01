"""
Benchmark Comparison Suite

Comprehensive benchmarking for GMCS sampler backends with:
- Blocking strategy performance comparison
- Multi-chain scaling analysis
- Temperature sweep
- Model size scaling
- Leaderboard JSON export

Usage:
    python examples/benchmark_comparison.py --output results.json
"""

import numpy as np
import jax
import time
import json
import argparse
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

# Import GMCS sampler backend
from src.core.thrml_sampler_backend import THRMLSamplerBackend


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    strategy: str
    n_nodes: int
    n_chains: int
    temperature: float
    gibbs_steps: int
    wall_time: float
    samples_per_sec: float
    ess_per_sec: float
    lag1_autocorr: float
    tau_int: float
    mean_energy: float
    mean_magnetization: float
    valid_blocks: bool
    balance_score: float


class BenchmarkSuite:
    """Comprehensive benchmarking suite for GMCS samplers."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def create_backend(self, n_nodes: int, coupling_strength: float = 0.1) -> THRMLSamplerBackend:
        """Create a test backend."""
        weights = np.random.randn(n_nodes, n_nodes) * coupling_strength
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        biases = np.random.randn(n_nodes) * 0.1
        
        return THRMLSamplerBackend(
            n_nodes=n_nodes,
            initial_weights=weights,
            initial_biases=biases,
            beta=1.0
        )
    
    def benchmark_strategy(
        self,
        backend: THRMLSamplerBackend,
        strategy: str,
        n_chains: int = 1,
        temperature: float = 1.0,
        gibbs_steps: int = 10,
        n_runs: int = 5
    ) -> BenchmarkResult:
        """
        Benchmark a single configuration.
        
        Args:
            backend: THRMLSamplerBackend instance
            strategy: Blocking strategy name
            n_chains: Number of parallel chains
            temperature: Sampling temperature
            gibbs_steps: Number of Gibbs steps
            n_runs: Number of runs to average
            
        Returns:
            BenchmarkResult with averaged metrics
        """
        n_nodes = backend.n_nodes
        
        # Generate grid positions
        grid_size = int(np.ceil(np.sqrt(n_nodes)))
        positions = np.array([
            [i % grid_size, i // grid_size]
            for i in range(n_nodes)
        ], dtype=np.float32)
        
        # Set strategy
        try:
            backend.thrml_wrapper.set_blocking_strategy(
                strategy_name=strategy,
                node_positions=positions
            )
        except Exception as e:
            print(f"  Warning: Failed to set strategy {strategy}: {e}")
            return None
        
        # Validate blocks
        validation = backend.thrml_wrapper.validate_current_blocks()
        
        # Run benchmark
        wall_times = []
        energies = []
        magnetizations = []
        
        for run in range(n_runs):
            t_start = time.time()
            samples, diagnostics = backend.sample(
                n_steps=gibbs_steps,
                temperature=temperature,
                num_chains=n_chains,
                blocking_strategy_name=strategy
            )
            wall_time = time.time() - t_start
            
            wall_times.append(wall_time)
            
            if 'mean_energy' in diagnostics:
                energies.append(diagnostics['mean_energy'])
            elif 'energy' in diagnostics:
                energies.append(diagnostics['energy'])
            
            if 'mean_magnetization' in diagnostics:
                magnetizations.append(diagnostics['mean_magnetization'])
            elif samples.ndim == 2:
                magnetizations.append(float(np.mean(samples[0])))
            else:
                magnetizations.append(float(np.mean(samples)))
        
        # Get benchmark diagnostics
        bench_diag = backend.thrml_wrapper.get_benchmark_diagnostics()
        
        # Create result
        result = BenchmarkResult(
            name=f"{strategy}_n{n_nodes}_c{n_chains}_T{temperature:.1f}",
            strategy=strategy,
            n_nodes=n_nodes,
            n_chains=n_chains,
            temperature=temperature,
            gibbs_steps=gibbs_steps,
            wall_time=float(np.mean(wall_times)),
            samples_per_sec=bench_diag.get('samples_per_sec', 0.0),
            ess_per_sec=bench_diag.get('ess_per_sec', 0.0),
            lag1_autocorr=bench_diag.get('lag1_autocorr', 0.0),
            tau_int=bench_diag.get('tau_int', 0.0),
            mean_energy=float(np.mean(energies)) if energies else 0.0,
            mean_magnetization=float(np.mean(magnetizations)) if magnetizations else 0.0,
            valid_blocks=validation.get('valid', False),
            balance_score=validation.get('balance_score', 0.0)
        )
        
        self.results.append(result)
        return result
    
    def run_strategy_comparison(self, n_nodes: int = 64):
        """
        Benchmark 1: Compare all blocking strategies.
        """
        print("\n" + "="*60)
        print(f"BENCHMARK 1: Blocking Strategy Comparison (n={n_nodes})")
        print("="*60)
        
        backend = self.create_backend(n_nodes)
        strategies = ["checkerboard", "random", "stripes", "supercell"]
        
        for strategy in strategies:
            print(f"\nTesting {strategy}...")
            result = self.benchmark_strategy(
                backend,
                strategy=strategy,
                n_chains=1,
                temperature=1.0,
                gibbs_steps=10,
                n_runs=5
            )
            
            if result:
                print(f"  Wall time: {result.wall_time:.4f}s")
                print(f"  Samples/sec: {result.samples_per_sec:.2f}")
                print(f"  ESS/sec: {result.ess_per_sec:.2f}")
                print(f"  Autocorr: {result.lag1_autocorr:.3f}")
                print(f"  Valid: {result.valid_blocks}")
    
    def run_multichain_scaling(self, n_nodes: int = 64):
        """
        Benchmark 2: Multi-chain scaling analysis.
        """
        print("\n" + "="*60)
        print(f"BENCHMARK 2: Multi-Chain Scaling (n={n_nodes})")
        print("="*60)
        
        backend = self.create_backend(n_nodes)
        chain_counts = [1, 2, 4, 8]
        
        for n_chains in chain_counts:
            print(f"\nTesting {n_chains} chains...")
            result = self.benchmark_strategy(
                backend,
                strategy="checkerboard",
                n_chains=n_chains,
                temperature=1.0,
                gibbs_steps=10,
                n_runs=3
            )
            
            if result:
                print(f"  Wall time: {result.wall_time:.4f}s")
                print(f"  Samples/sec: {result.samples_per_sec:.2f}")
                print(f"  Speedup: {result.samples_per_sec / self.results[0].samples_per_sec:.2f}x")
    
    def run_temperature_sweep(self, n_nodes: int = 64):
        """
        Benchmark 3: Temperature sweep for mixing analysis.
        """
        print("\n" + "="*60)
        print(f"BENCHMARK 3: Temperature Sweep (n={n_nodes})")
        print("="*60)
        
        backend = self.create_backend(n_nodes)
        temperatures = [0.5, 1.0, 1.5, 2.0]
        
        for temp in temperatures:
            print(f"\nTesting T={temp:.1f}...")
            result = self.benchmark_strategy(
                backend,
                strategy="checkerboard",
                n_chains=1,
                temperature=temp,
                gibbs_steps=10,
                n_runs=3
            )
            
            if result:
                print(f"  Autocorr: {result.lag1_autocorr:.3f}")
                print(f"  τ_int: {result.tau_int:.2f}")
                print(f"  ESS/sec: {result.ess_per_sec:.2f}")
    
    def run_size_scaling(self):
        """
        Benchmark 4: Model size scaling.
        """
        print("\n" + "="*60)
        print("BENCHMARK 4: Model Size Scaling")
        print("="*60)
        
        sizes = [32, 64, 128, 256]
        
        for n_nodes in sizes:
            print(f"\nTesting n={n_nodes}...")
            backend = self.create_backend(n_nodes)
            result = self.benchmark_strategy(
                backend,
                strategy="checkerboard",
                n_chains=1,
                temperature=1.0,
                gibbs_steps=10,
                n_runs=3
            )
            
            if result:
                print(f"  Wall time: {result.wall_time:.4f}s")
                print(f"  Samples/sec: {result.samples_per_sec:.2f}")
                print(f"  Time per node: {result.wall_time / n_nodes * 1000:.3f}ms")
    
    def generate_leaderboard(self) -> Dict[str, Any]:
        """
        Generate leaderboard JSON with all results.
        
        Returns:
            Dict with leaderboard data
        """
        # Sort by ESS/sec (primary metric)
        sorted_results = sorted(
            self.results,
            key=lambda r: r.ess_per_sec,
            reverse=True
        )
        
        leaderboard = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_benchmarks': len(self.results),
            'hardware': {
                'jax_devices': [str(d) for d in jax.devices()],
                'jax_version': jax.__version__
            },
            'top_10': [
                {
                    'rank': i + 1,
                    **asdict(result)
                }
                for i, result in enumerate(sorted_results[:10])
            ],
            'all_results': [asdict(r) for r in sorted_results],
            'summary': {
                'best_strategy': sorted_results[0].strategy if sorted_results else None,
                'best_ess_per_sec': sorted_results[0].ess_per_sec if sorted_results else 0.0,
                'mean_ess_per_sec': float(np.mean([r.ess_per_sec for r in self.results])),
                'mean_autocorr': float(np.mean([r.lag1_autocorr for r in self.results]))
            }
        }
        
        return leaderboard
    
    def export_json(self, filepath: str):
        """Export leaderboard to JSON file."""
        leaderboard = self.generate_leaderboard()
        
        with open(filepath, 'w') as f:
            json.dump(leaderboard, f, indent=2)
        
        print(f"\nExported leaderboard to: {filepath}")
    
    def print_summary(self):
        """Print summary of all benchmarks."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        if not self.results:
            print("No results to summarize.")
            return
        
        # Top 5 by ESS/sec
        sorted_results = sorted(
            self.results,
            key=lambda r: r.ess_per_sec,
            reverse=True
        )
        
        print("\nTop 5 Configurations (by ESS/sec):")
        print("-" * 60)
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i+1}. {result.name}")
            print(f"   ESS/sec: {result.ess_per_sec:.2f}, Autocorr: {result.lag1_autocorr:.3f}")
        
        # Strategy comparison
        print("\nStrategy Performance:")
        print("-" * 60)
        strategies = {}
        for result in self.results:
            if result.strategy not in strategies:
                strategies[result.strategy] = []
            strategies[result.strategy].append(result.ess_per_sec)
        
        for strategy, ess_values in strategies.items():
            mean_ess = np.mean(ess_values)
            print(f"{strategy:15s}: {mean_ess:.2f} ESS/sec (avg)")
        
        # Overall stats
        print("\nOverall Statistics:")
        print("-" * 60)
        print(f"Total benchmarks: {len(self.results)}")
        print(f"Mean ESS/sec: {np.mean([r.ess_per_sec for r in self.results]):.2f}")
        print(f"Mean autocorr: {np.mean([r.lag1_autocorr for r in self.results]):.3f}")
        print(f"Mean τ_int: {np.mean([r.tau_int for r in self.results]):.2f}")


def main():
    """Run full benchmark suite."""
    parser = argparse.ArgumentParser(description='GMCS Benchmark Suite')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output JSON file for leaderboard')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark (fewer runs)')
    args = parser.parse_args()
    
    print("="*60)
    print("GMCS Comprehensive Benchmark Suite")
    print("="*60)
    print(f"\nJAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    
    # Create suite
    suite = BenchmarkSuite()
    
    # Run benchmarks
    if args.quick:
        print("\nRunning quick benchmark...")
        suite.run_strategy_comparison(n_nodes=32)
    else:
        print("\nRunning full benchmark suite...")
        suite.run_strategy_comparison(n_nodes=64)
        suite.run_multichain_scaling(n_nodes=64)
        suite.run_temperature_sweep(n_nodes=64)
        suite.run_size_scaling()
    
    # Print summary
    suite.print_summary()
    
    # Export leaderboard
    suite.export_json(args.output)
    
    print("\n" + "="*60)
    print("Benchmark suite completed!")
    print("="*60)


if __name__ == "__main__":
    main()

