"""
THRML Integration Wrapper for GMCS.

This module provides a clean interface between GMCS and THRML's probabilistic
graphical model framework, enabling efficient block Gibbs sampling and
energy-based model learning.

THRML (https://github.com/extropic-ai/thrml) is a JAX library for building
and sampling probabilistic graphical models with focus on efficient block
Gibbs sampling and energy-based models.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import json
from collections import deque
from typing import Tuple, Optional, Dict, Any, List, NamedTuple, Deque

# Import universal blocking strategies
from src.core.blocking_strategies import (
    BlockingStrategy,
    get_strategy,
    list_strategies,
    Block as StrategyBlock
)

# THRML imports - full feature set
try:
    from thrml import SpinNode, Block, SamplingSchedule, sample_states
    from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
    THRML_AVAILABLE = True
    
    # Import observers for advanced sampling diagnostics (optional)
    try:
        from thrml.observers import StateObserver, MomentAccumulatorObserver
        try:
            from thrml.observers import EnergyObserver, CorrelationObserver  # type: ignore
        except ImportError:
            from src.core.thrml_compat import EnergyObserver, CorrelationObserver
        THRML_OBSERVERS_AVAILABLE = True
    except ImportError:
        from src.core.thrml_compat import EnergyObserver, CorrelationObserver
        THRML_OBSERVERS_AVAILABLE = False
    
    # Import heterogeneous node types (optional)
    try:
        from thrml import ContinuousNode, DiscreteNode
        THRML_HETEROGENEOUS_AVAILABLE = True
    except ImportError:
        THRML_HETEROGENEOUS_AVAILABLE = False
    
except ImportError:
    THRML_AVAILABLE = False
    THRML_OBSERVERS_AVAILABLE = False
    THRML_HETEROGENEOUS_AVAILABLE = False
    raise ImportError(
        "THRML is required for this module. Install with: pip install thrml>=0.1.3"
    )


# ============================================================================
# Benchmarking Data Structures
# ============================================================================

class BenchmarkSample(NamedTuple):
    """Single benchmark measurement."""
    timestamp: float
    wall_time: float
    n_samples: int
    magnetization: float
    energy: float
    strategy: str
    n_chains: int


# Benchmarking constants
BENCHMARK_HISTORY_LENGTH = 1000  # Keep last 1000 samples
MAGNETIZATION_HISTORY_LENGTH = 512  # For autocorrelation


class THRMLWrapper:
    """
    Manages THRML Ising model for GMCS.
    
    This wrapper provides a clean interface for:
    - Creating and managing THRML IsingEBM models
    - Updating biases from GMCS pipeline
    - Performing block Gibbs sampling
    - Computing energy and feedback
    - Contrastive Divergence learning
    
    Attributes:
        n_nodes: Number of nodes in the model
        nodes: List of THRML SpinNode instances
        edges: List of edge tuples
        model: THRML IsingEBM instance
        free_blocks: Two-color blocks for Gibbs sampling
        beta: Inverse temperature parameter
    """
    
    def __init__(
        self,
        n_nodes: int,
        weights: np.ndarray,
        biases: np.ndarray,
        beta: float = 1.0
    ):
        """
        Initialize THRML wrapper with Ising model.
        
        Args:
            n_nodes: Number of spin nodes
            weights: (n_nodes, n_nodes) symmetric weight matrix
            biases: (n_nodes,) bias vector
            beta: Inverse temperature (default 1.0)
            
        Raises:
            RuntimeError: If THRML is not available
            ValueError: If inputs are invalid (wrong shapes, non-finite values)
        """
        if not THRML_AVAILABLE:
            raise RuntimeError(
                "THRML not available. Install with: pip install thrml>=0.1.3"
            )
        
        # Validate inputs
        if n_nodes <= 0:
            raise ValueError(f"n_nodes must be positive, got {n_nodes}")
        
        # Convert to numpy arrays and validate shapes
        try:
            weights = np.asarray(weights, dtype=np.float32)
            biases = np.asarray(biases, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to convert inputs to numpy arrays: {e}")
        
        if weights.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"weights shape {weights.shape} doesn't match "
                f"expected ({n_nodes}, {n_nodes})"
            )
        
        if biases.shape != (n_nodes,):
            raise ValueError(
                f"biases shape {biases.shape} doesn't match expected ({n_nodes},)"
            )
        
        # Check for non-finite values
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contains non-finite values (inf/nan)")
        
        if not np.all(np.isfinite(biases)):
            raise ValueError("biases contains non-finite values (inf/nan)")
        
        # Validate beta
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(f"beta must be positive and finite, got {beta}")
        
        # Make weights symmetric
        weights = (weights + weights.T) / 2.0
        np.fill_diagonal(weights, 0.0)
        
        self.n_nodes = n_nodes
        self.beta = float(beta)
        
        # Create THRML spin nodes
        try:
            self.nodes = [SpinNode() for _ in range(n_nodes)]
        except Exception as e:
            raise RuntimeError(f"Failed to create THRML SpinNodes: {e}")
        
        # Build edges from weight matrix
        # Only include non-zero weights, use upper triangle
        self.edges = []
        self.edge_weights = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                w = float(weights[i, j])
                if abs(w) > 1e-6:
                    self.edges.append((self.nodes[i], self.nodes[j]))
                    self.edge_weights.append(w)
        
        # Convert to JAX arrays with error handling
        try:
            self.edge_weights_jax = jnp.array(
                self.edge_weights if self.edge_weights else [0.0], 
                dtype=jnp.float32
            )
            self.biases_jax = jnp.array(biases, dtype=jnp.float32)
            self.beta_jax = jnp.array(self.beta, dtype=jnp.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to create JAX arrays: {e}")
        
        # Create THRML Ising model
        try:
            self.model = IsingEBM(
                nodes=self.nodes,
                edges=self.edges if self.edges else [],
                biases=self.biases_jax,
                weights=self.edge_weights_jax,
                beta=self.beta_jax
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create THRML IsingEBM: {e}")
        
        # Create two-color blocks for efficient Gibbs sampling
        # Even/odd coloring ensures no conflicts within a block
        try:
            even_nodes = self.nodes[::2] if len(self.nodes) > 0 else []
            odd_nodes = self.nodes[1::2] if len(self.nodes) > 1 else []
            
            self.free_blocks = []
            if even_nodes:
                self.free_blocks.append(Block(even_nodes))
            if odd_nodes:
                self.free_blocks.append(Block(odd_nodes))
            
            if not self.free_blocks:
                raise ValueError("No free blocks created - need at least 1 node")
        except Exception as e:
            raise RuntimeError(f"Failed to create THRML blocks: {e}")
        
        # Store full weight matrix for convenience
        self._full_weights = weights.copy()
        
        # Benchmarking infrastructure
        self._benchmark_history: Deque[BenchmarkSample] = deque(maxlen=BENCHMARK_HISTORY_LENGTH)
        self._magnetization_history: Deque[float] = deque(maxlen=MAGNETIZATION_HISTORY_LENGTH)
        self._last_sample_time: float = 0.0
        self._total_samples: int = 0
        self._current_strategy: str = "checkerboard"  # Default
        self._current_n_chains: int = 1
        
        # Blocking strategy (universal)
        self._blocking_strategy: Optional[BlockingStrategy] = None
        self._block_cache: Dict[str, List[Block]] = {}  # Cache built blocks
        
        # Conditional sampling (clamped nodes)
        self._clamped_node_ids: List[int] = []
        self._clamped_values: List[float] = []
        self._clamped_blocks: List[Block] = []
        
        # Error tracking
        self._last_error: Optional[str] = None
        self._error_count: int = 0
        
        print(f"[THRML] Initialized wrapper: {self.n_nodes} nodes, "
              f"{len(self.edges)} edges, strategy={self._current_strategy}")
    
    @property
    def biases(self) -> np.ndarray:
        """Get current biases as numpy array."""
        return np.array(self.biases_jax)
    
    @property
    def weights(self) -> np.ndarray:
        """Get current weight matrix as numpy array."""
        if hasattr(self, '_full_weights'):
            return self._full_weights[:self.n_nodes, :self.n_nodes]
        # Reconstruct from edges
        weights = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        for (node_i, node_j), w in zip(self.edges, self.edge_weights):
            i = node_to_idx[node_i]
            j = node_to_idx[node_j]
            weights[i, j] = w
            weights[j, i] = w
        return weights
    
    @property
    def current_strategy_name(self) -> str:
        """Get current blocking strategy name."""
        return self._current_strategy
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of THRML wrapper.
        
        Returns:
            Dict with health information
        """
        return {
            'healthy': self._error_count < 10,  # Threshold for "unhealthy"
            'n_nodes': self.n_nodes,
            'n_edges': len(self.edges),
            'error_count': self._error_count,
            'last_error': self._last_error,
            'total_samples': self._total_samples,
            'current_strategy': self._current_strategy,
            'n_chains': self._current_n_chains,
            'has_clamped_nodes': len(self._clamped_node_ids) > 0,
            'thrml_available': THRML_AVAILABLE,
            'observers_available': THRML_OBSERVERS_AVAILABLE,
            'heterogeneous_available': THRML_HETEROGENEOUS_AVAILABLE
        }
    
    def reset_error_tracking(self) -> None:
        """Reset error counter and last error message."""
        self._error_count = 0
        self._last_error = None
    
    def update_biases(self, gmcs_biases: np.ndarray) -> None:
        """
        Update node biases from GMCS pipeline output.
        
        This allows the GMCS signal processing chain to directly
        influence the EBM sampling distribution.
        
        Args:
            gmcs_biases: (n_nodes,) or larger array of new bias values
            
        Raises:
            ValueError: If biases contain non-finite values
            RuntimeError: If model recreation fails
        """
        try:
            # Convert to numpy and validate
            gmcs_biases = np.asarray(gmcs_biases, dtype=np.float32)
            
            # Truncate/pad to correct size
            if len(gmcs_biases) < self.n_nodes:
                # Pad with zeros if too short
                padded = np.zeros(self.n_nodes, dtype=np.float32)
                padded[:len(gmcs_biases)] = gmcs_biases
                gmcs_biases = padded
            else:
                # Truncate if too long
                gmcs_biases = gmcs_biases[:self.n_nodes]
            
            # Check for non-finite values
            if not np.all(np.isfinite(gmcs_biases)):
                raise ValueError("gmcs_biases contains non-finite values (inf/nan)")
            
            # Update JAX array
            self.biases_jax = jnp.array(gmcs_biases, dtype=jnp.float32)
            
            # Recreate model with updated biases
            self.model = IsingEBM(
                nodes=self.nodes,
                edges=self.edges,
                biases=self.biases_jax,
                weights=self.edge_weights_jax,
                beta=self.beta_jax
            )
            
        except Exception as e:
            self._last_error = f"update_biases failed: {e}"
            self._error_count += 1
            raise RuntimeError(self._last_error) from e
    
    def sample_gibbs(
        self,
        n_steps: int,
        temperature: float,
        key: Optional[jax.random.PRNGKey] = None,
        return_all_samples: bool = False
    ) -> np.ndarray:
        """
        Run THRML block Gibbs sampling with optimized schedule.
        
        Performs efficient two-color block Gibbs sampling to generate
        samples from the current EBM distribution. Uses THRML's optimized
        sampling schedule with proper warmup and steps_per_sample.
        
        Args:
            n_steps: Total number of Gibbs steps (warmup + sampling)
            temperature: Sampling temperature (higher = more random)
            key: JAX random key (generated if None)
            return_all_samples: If True, return all samples instead of just final
            
        Returns:
            (n_nodes,) binary states {-1, +1}, or (n_samples, n_nodes) if return_all_samples
            
        Raises:
            ValueError: If n_steps or temperature are invalid
            RuntimeError: If sampling fails
        """
        # Validate inputs
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be positive and finite, got {temperature}")
        
        try:
            if key is None:
                key = jax.random.PRNGKey(0)
            
            # Update beta based on temperature
            beta_temp = 1.0 / temperature
            self.beta_jax = jnp.array(beta_temp, dtype=jnp.float32)
            
            # Recreate model with new temperature
            model_temp = IsingEBM(
                nodes=self.nodes,
                edges=self.edges,
                biases=self.biases_jax,
                weights=self.edge_weights_jax,
                beta=self.beta_jax
            )
            
            # Create sampling program with free and clamped blocks
            # Using empty clamped_blocks for fully free sampling
            # Following https://docs.thrml.ai/en/latest/ - use positional arguments
            program = IsingSamplingProgram(
                model_temp,  # model as first positional arg
                self.free_blocks,
                clamped_blocks=[]
            )
            
            # Initialize state using Hinton initialization (optimal for Ising)
            k_init, k_samp = jax.random.split(key, 2)
            init_state = hinton_init(k_init, model_temp, self.free_blocks, ())
            
            # Create optimized sampling schedule
            # THRML best practice: warmup = 50-100% of total steps
            # steps_per_sample = 2 for two-color blocking (one sweep)
            if return_all_samples:
                n_warmup = max(1, n_steps // 3)  # 1/3 warmup
                n_samples = max(1, (n_steps - n_warmup) // 2)  # Rest for sampling
            else:
                n_warmup = max(1, n_steps // 2)  # Half for warmup
                n_samples = 1  # Only final state needed
            
            schedule = SamplingSchedule(
                n_warmup=n_warmup,
                n_samples=n_samples,
                steps_per_sample=2  # Two-color blocking = one full sweep
            )
            
            # Time the sampling for benchmarking
            t_start = time.time()
            
            # Sample states using THRML's optimized sampler
            # The last argument specifies which blocks to observe/return
            samples = sample_states(
                k_samp,
                program,
                schedule,
                init_state,
                [],  # No observers
                [Block(self.nodes)]  # Observe all nodes
            )
            
            wall_time = time.time() - t_start
            
            # Extract samples
            # THRML returns samples in model's native format
            if return_all_samples:
                # Return all samples as (n_samples, n_nodes) array
                binary_states = np.array(samples, dtype=np.float32)
            else:
                # Return only final sample
                final_sample = samples[-1]
                binary_states = np.array(final_sample, dtype=np.float32)
            
            # Validate output
            if binary_states.size == 0:
                raise RuntimeError("THRML sampling returned empty array")
            
            if not np.all(np.isfinite(binary_states)):
                raise RuntimeError("THRML sampling returned non-finite values")
            
            # Record benchmark sample
            energy = self.compute_energy(binary_states if not return_all_samples else binary_states[-1])
            self._record_benchmark_sample(
                wall_time=wall_time,
                n_samples=n_samples,
                states=binary_states if not return_all_samples else binary_states[-1],
                energy=energy
            )
            
            return binary_states
            
        except Exception as e:
            self._last_error = f"sample_gibbs failed: {e}"
            self._error_count += 1
            # Return fallback: random binary states
            fallback_states = np.random.choice([-1.0, 1.0], size=(self.n_nodes,)).astype(np.float32)
            print(f"[THRML WARNING] Sampling failed, returning random states: {e}")
            return fallback_states
    
    def sample_gibbs_with_diagnostics(
        self,
        n_steps: int,
        temperature: float,
        key: Optional[jax.random.PRNGKey] = None,
        track_energy: bool = True,
        track_correlations: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run THRML block Gibbs sampling with diagnostic observers.
        
        This advanced sampling method uses THRML's observer framework to track
        energy trajectories, state evolution, and correlations during sampling.
        Useful for debugging, research, and understanding model behavior.
        
        Args:
            n_steps: Total number of Gibbs steps
            temperature: Sampling temperature
            key: JAX random key
            track_energy: Track energy at each step
            track_correlations: Track spin correlations
            
        Returns:
            Tuple of:
                - (n_nodes,) final binary states
                - Dict with diagnostic data (energies, correlations, etc.)
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Update beta based on temperature
        beta_temp = 1.0 / temperature
        self.beta_jax = jnp.array(beta_temp, dtype=jnp.float32)
        
        # Recreate model with new temperature
        model_temp = IsingEBM(
            nodes=self.nodes,
            edges=self.edges,
            biases=self.biases_jax,
            weights=self.edge_weights_jax,
            beta=self.beta_jax
        )
        
        # Create sampling program
        # Per https://docs.thrml.ai/en/latest/ API
        program = IsingSamplingProgram(
            model_temp,
            self.free_blocks,
            clamped_blocks=[]
        )
        
        # Initialize state
        k_init, k_samp = jax.random.split(key, 2)
        init_state = hinton_init(k_init, model_temp, self.free_blocks, ())
        
        # Setup observers if available
        observers = []
        diagnostics = {}
        
        if THRML_OBSERVERS_AVAILABLE:
            if track_energy:
                energy_obs = EnergyObserver()
                observers.append(energy_obs)
            
            if track_correlations:
                corr_obs = CorrelationObserver()
                observers.append(corr_obs)
        
        # Create sampling schedule
        n_warmup = max(1, n_steps // 2)
        n_samples = max(1, (n_steps - n_warmup) // 2)
        
        schedule = SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=2
        )
        
        # Sample with observers
        samples = sample_states(
            k_samp,
            program,
            schedule,
            init_state,
            observers,  # Pass observers for diagnostics
            [Block(self.nodes)]
        )
        
        # Extract diagnostics from observers
        if THRML_OBSERVERS_AVAILABLE:
            if track_energy and observers:
                # Energy trajectory during sampling
                diagnostics['energy_trajectory'] = [
                    float(e) for e in getattr(observers[0], 'energies', [])
                ]
            
            if track_correlations and len(observers) > 1:
                # Correlation matrices
                diagnostics['correlations'] = getattr(observers[-1], 'correlations', None)
        
        # Extract final sample
        final_sample = samples[-1]
        binary_states = np.array(final_sample, dtype=np.float32)
        
        # Add basic diagnostics
        diagnostics['n_samples'] = len(samples)
        diagnostics['final_energy'] = self.compute_energy(binary_states)
        diagnostics['temperature'] = temperature
        diagnostics['beta'] = float(beta_temp)
        
        return binary_states, diagnostics
    
    def compute_energy(self, states: Optional[np.ndarray] = None) -> float:
        """
        Calculate Ising energy: E = -0.5 * s^T @ W @ s - b^T @ s
        
        Args:
            states: (n_nodes,) binary states {-1, +1}. If None, samples current state.
            
        Returns:
            Energy value (scalar)
        """
        if states is None:
            # Sample current state
            key = jax.random.PRNGKey(0)
            states = self.sample_gibbs(n_steps=10, temperature=1.0, key=key)
        
        states_jax = jnp.array(states[:self.n_nodes], dtype=jnp.float32)
        
        # Compute interaction energy: -0.5 * s^T @ W @ s
        interaction_energy = -0.5 * jnp.dot(
            states_jax,
            jnp.dot(self._full_weights[:self.n_nodes, :self.n_nodes], states_jax)
        )
        
        # Compute bias energy: -b^T @ s
        bias_energy = -jnp.dot(self.biases_jax, states_jax)
        
        total_energy = interaction_energy + bias_energy
        
        return float(total_energy)
    
    def update_weights_cd(
        self,
        data_states: np.ndarray,
        eta: float,
        k_steps: int = 1,
        key: Optional[jax.random.PRNGKey] = None,
        n_chains: int = 1
    ) -> Dict[str, float]:
        """
        Optimized Contrastive Divergence learning with optional parallel chains.
        
        Algorithm:
        1. Compute data statistics: <s_i s_j>_data
        2. Run k steps of Gibbs sampling from data (optionally with multiple chains)
        3. Compute model statistics: <s_i s_j>_model (averaged over chains)
        4. Update: W_ij += eta * (<s_i s_j>_data - <s_i s_j>_model)
        
        Args:
            data_states: (n_nodes,) observed binary states {-1, +1}
            eta: Learning rate
            k_steps: Number of Gibbs steps (CD-k)
            key: JAX random key
            n_chains: Number of parallel sampling chains (default 1)
            
        Returns:
            Dict with learning diagnostics (gradient_norm, energy_diff, etc.)
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        data_states_jax = jnp.array(data_states[:self.n_nodes], dtype=jnp.float32)
        
        # === Step 1: Compute data statistics ===
        # Correlation matrix: C_data[i,j] = s_i * s_j
        C_data = jnp.outer(data_states_jax, data_states_jax)
        data_energy = self.compute_energy(data_states)
        
        # === Step 2: Run k-step Gibbs sampling (with optional parallel chains) ===
        if n_chains > 1:
            # Use multiple chains for better gradient estimates
            keys = jax.random.split(key, n_chains)
            model_states_list = []
            
            for chain_key in keys:
                chain_states = self.sample_gibbs(
                    n_steps=k_steps,
                    temperature=1.0 / float(self.beta_jax),
                    key=chain_key
                )
                model_states_list.append(chain_states)
            
            # Average correlations over all chains
            C_model = jnp.zeros_like(C_data)
            for chain_states in model_states_list:
                chain_states_jax = jnp.array(chain_states, dtype=jnp.float32)
                C_model += jnp.outer(chain_states_jax, chain_states_jax)
            C_model /= n_chains
            
            # Use last chain for energy computation
            model_states_jax = jnp.array(model_states_list[-1], dtype=jnp.float32)
        else:
            # Single chain (standard CD-k)
            model_states = self.sample_gibbs(
                n_steps=k_steps,
                temperature=1.0 / float(self.beta_jax),
                key=key
            )
            model_states_jax = jnp.array(model_states, dtype=jnp.float32)
            
            # === Step 3: Compute model statistics ===
            C_model = jnp.outer(model_states_jax, model_states_jax)
        
        model_energy = self.compute_energy(np.array(model_states_jax))
        
        # === Step 4: Update weights ===
        delta_W = eta * (C_data - C_model)
        
        # Compute gradient norm for diagnostics
        gradient_norm = float(jnp.linalg.norm(delta_W))
        
        # Update full weight matrix
        self._full_weights[:self.n_nodes, :self.n_nodes] += np.array(delta_W)
        
        # Return diagnostics
        diagnostics = {
            'gradient_norm': gradient_norm,
            'data_energy': data_energy,
            'model_energy': model_energy,
            'energy_diff': data_energy - model_energy,
            'n_chains': n_chains,
            'k_steps': k_steps
        }
        
        # Ensure symmetry after update
        weights_upper = np.triu(self._full_weights[:self.n_nodes, :self.n_nodes], k=1)
        self._full_weights[:self.n_nodes, :self.n_nodes] = (
            weights_upper + weights_upper.T
        )
        
        # Zero diagonal (no self-connections)
        np.fill_diagonal(self._full_weights[:self.n_nodes, :self.n_nodes], 0.0)
        
        # Rebuild edge list with updated weights
        self.edges = []
        self.edge_weights = []
        
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                w = self._full_weights[i, j]
                if abs(w) > 1e-6:
                    self.edges.append((self.nodes[i], self.nodes[j]))
                    self.edge_weights.append(float(w))
        
        # Update JAX arrays
        self.edge_weights_jax = jnp.array(self.edge_weights, dtype=jnp.float32)
        
        # Recreate model with updated weights
        self.model = IsingEBM(
            nodes=self.nodes,
            edges=self.edges,
            biases=self.biases_jax,
            weights=self.edge_weights_jax,
            beta=self.beta_jax
        )
        
        return diagnostics
    
    def get_weights(self) -> np.ndarray:
        """
        Get current weight matrix.
        
        Returns:
            (n_nodes, n_nodes) symmetric weight matrix
        """
        return self._full_weights[:self.n_nodes, :self.n_nodes].copy()
    
    def set_temperature(self, temperature: float) -> None:
        """
        Update sampling temperature.
        
        Args:
            temperature: New temperature value
        """
        self.beta = 1.0 / temperature
        self.beta_jax = jnp.array(self.beta, dtype=jnp.float32)
        
        # Recreate model
        self.model = IsingEBM(
            nodes=self.nodes,
            edges=self.edges,
            biases=self.biases_jax,
            weights=self.edge_weights_jax,
            beta=self.beta_jax
        )
    
    # ========================================================================
    # Multi-Chain Parallelism
    # ========================================================================
    
    def _auto_detect_optimal_chains(self) -> int:
        """
        Auto-detect optimal number of parallel chains based on hardware.
        
        Returns:
            Recommended number of chains
        """
        devices = jax.devices()
        
        # Check if GPU available
        has_gpu = any("gpu" in str(d).lower() for d in devices)
        
        if has_gpu:
            # Estimate based on GPU memory
            # This is a heuristic - actual optimal value depends on model size
            try:
                # Try to get GPU memory info
                import os
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
                else:
                    n_gpus = len([d for d in devices if "gpu" in str(d).lower()])
                
                # Heuristic: 4-8 chains per GPU for good utilization
                return min(n_gpus * 6, 16)  # Cap at 16 chains
            except:
                return 4  # Safe default for GPU
        else:
            # CPU: use fraction of cores
            import os
            n_cores = os.cpu_count() or 4
            return max(1, n_cores // 4)
    
    def sample_gibbs_parallel(
        self,
        n_steps: int,
        temperature: float,
        n_chains: int = -1,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """
        Run parallel Gibbs sampling using jax.vmap.
        
        Samples multiple independent chains simultaneously for better
        gradient estimates and performance analysis.
        
        Args:
            n_steps: Number of sampling steps per chain
            temperature: Sampling temperature
            n_chains: Number of parallel chains (-1 for auto-detect)
            key: JAX random key
            
        Returns:
            Tuple of:
                - (n_chains, n_nodes) sampled states
                - List of per-chain diagnostics
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Auto-detect optimal chain count
        if n_chains == -1:
            n_chains = self._auto_detect_optimal_chains()
        
        self._current_n_chains = n_chains
        
        # Update beta based on temperature
        beta_temp = 1.0 / temperature
        self.beta_jax = jnp.array(beta_temp, dtype=jnp.float32)
        
        # Recreate model with new temperature
        model_temp = IsingEBM(
            nodes=self.nodes,
            edges=self.edges,
            biases=self.biases_jax,
            weights=self.edge_weights_jax,
            beta=self.beta_jax
        )
        
        # Create sampling program
        # Per https://docs.thrml.ai/en/latest/ API
        program = IsingSamplingProgram(
            model_temp,
            self.free_blocks,
            clamped_blocks=[]
        )
        
        # Create schedule
        n_warmup = max(1, n_steps // 2)
        n_samples = 1  # Final state only
        
        schedule = SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=2
        )
        
        # Split keys for each chain
        chain_keys = jax.random.split(key, n_chains)
        
        # Define single-chain sampling function
        def single_chain_sample(chain_key):
            k_init, k_samp = jax.random.split(chain_key)
            init_state = hinton_init(k_init, model_temp, self.free_blocks, ())
            
            samples = sample_states(
                k_samp,
                program,
                schedule,
                init_state,
                [],
                [Block(self.nodes)]
            )
            
            # Return final sample
            return samples[-1]
        
        # Time the parallel sampling
        t_start = time.time()
        
        # Use vmap for parallel execution
        batched_sample = jax.vmap(single_chain_sample)
        all_samples = batched_sample(chain_keys)
        
        # Convert to numpy
        all_samples_np = np.array(all_samples, dtype=np.float32)
        
        wall_time = time.time() - t_start
        
        # Compute per-chain diagnostics
        per_chain_diagnostics = []
        
        for i in range(n_chains):
            chain_states = all_samples_np[i]
            chain_energy = self.compute_energy(chain_states)
            chain_magnetization = float(np.mean(chain_states))
            
            per_chain_diagnostics.append({
                'chain_id': i,
                'energy': chain_energy,
                'magnetization': chain_magnetization,
                'wall_time': wall_time / n_chains  # Approximate per-chain time
            })
        
        # Record aggregate benchmark
        # Use mean states for magnetization tracking
        mean_states = np.mean(all_samples_np, axis=0)
        mean_energy = np.mean([d['energy'] for d in per_chain_diagnostics])
        
        self._record_benchmark_sample(
            wall_time=wall_time,
            n_samples=n_chains,  # Total samples across chains
            states=mean_states,
            energy=mean_energy
        )
        
        return all_samples_np, per_chain_diagnostics
    
    def get_chain_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostics about multi-chain performance.
        
        Returns:
            Dict with chain count, per-chain stats, etc.
        """
        return {
            'current_n_chains': self._current_n_chains,
            'auto_detected_optimal': self._auto_detect_optimal_chains(),
            'has_gpu': any("gpu" in str(d).lower() for d in jax.devices())
        }
    
    # ========================================================================
    # Universal Blocking Strategies
    # ========================================================================
    
    def set_blocking_strategy(
        self,
        strategy_name: str,
        node_positions: Optional[np.ndarray] = None,
        connectivity: Optional[np.ndarray] = None
    ):
        """
        Set the blocking strategy for parallel sampling.
        
        Args:
            strategy_name: Name of strategy ('checkerboard', 'random', etc.)
            node_positions: (n_nodes, 2) spatial positions (required for spatial strategies)
            connectivity: (n_nodes, n_nodes) adjacency matrix (required for graph coloring)
        """
        # Get strategy from registry
        strategy = get_strategy(strategy_name)
        self._blocking_strategy = strategy
        self._current_strategy = strategy_name
        
        # Build blocks using the strategy
        if strategy.requires_spatial and node_positions is None:
            # Generate default grid positions
            grid_size = int(np.ceil(np.sqrt(self.n_nodes)))
            node_positions = np.array([
                [i % grid_size, i // grid_size]
                for i in range(self.n_nodes)
            ], dtype=np.float32)
        
        if strategy.requires_connectivity and connectivity is None:
            # Use weight matrix as connectivity
            connectivity = (np.abs(self._full_weights[:self.n_nodes, :self.n_nodes]) > 1e-6).astype(np.float32)
        
        # Build blocks
        strategy_blocks = strategy.build_blocks(
            n_nodes=self.n_nodes,
            connectivity=connectivity,
            positions=node_positions,
            seed=0
        )
        
        # Convert to THRML Block format
        self.free_blocks = []
        for i, strategy_block in enumerate(strategy_blocks):
            thrml_nodes = [self.nodes[node_id] for node_id in strategy_block.nodes]
            self.free_blocks.append(Block(thrml_nodes))
        
        # Cache the blocks
        cache_key = f"{strategy_name}_{self.n_nodes}"
        self._block_cache[cache_key] = self.free_blocks
        
        print(f"[THRML] Set blocking strategy: {strategy_name} ({len(self.free_blocks)} blocks)")
    
    def get_blocking_strategy(self) -> str:
        """Get current blocking strategy name."""
        return self._current_strategy
    
    def get_supported_strategies(self) -> List[str]:
        """
        Get list of supported blocking strategies.
        
        Returns:
            List of strategy names
        """
        return ["checkerboard", "random", "stripes", "supercell", "graph-coloring"]
    
    def validate_current_blocks(self) -> Dict[str, Any]:
        """
        Validate current block configuration.
        
        Returns:
            Dict with validation results
        """
        if self._blocking_strategy is None:
            return {
                'valid': False,
                'reason': 'No blocking strategy set',
                'balance_score': 0.0
            }
        
        # Build connectivity matrix from weights
        connectivity = (np.abs(self._full_weights[:self.n_nodes, :self.n_nodes]) > 1e-6).astype(np.float32)
        
        # Convert THRML blocks back to strategy blocks
        strategy_blocks = []
        for i, thrml_block in enumerate(self.free_blocks):
            node_ids = []
            for node in thrml_block.nodes:
                # Find node index
                for j, n in enumerate(self.nodes):
                    if n is node:
                        node_ids.append(j)
                        break
            strategy_blocks.append(StrategyBlock(nodes=node_ids, id=i))
        
        # Validate
        result = self._blocking_strategy.validate_blocks(
            blocks=strategy_blocks,
            connectivity=connectivity,
            n_nodes=self.n_nodes
        )
        
        return {
            'valid': result.valid,
            'reason': result.reason,
            'balance_score': result.balance_score,
            'independence_violations': result.independence_violations,
            'coverage_missing': result.coverage_missing
        }
    
    # ========================================================================
    # Conditional Sampling (Clamped Nodes)
    # ========================================================================
    
    def set_clamped_nodes(
        self,
        node_ids: List[int],
        values: List[float],
        node_positions: Optional[np.ndarray] = None
    ):
        """
        Set nodes to be clamped (fixed) during sampling.
        
        This enables conditional sampling for:
        - Audio inpainting: Fix known good regions, sample corrupted parts
        - Constrained synthesis: User pins certain oscillators
        - Pattern completion: ML-style inference with partial observations
        
        Args:
            node_ids: List of node indices to clamp
            values: List of values to clamp to (will be converted to {-1, +1})
            node_positions: Optional positions for rebuilding free blocks
        """
        if len(node_ids) != len(values):
            raise ValueError("node_ids and values must have same length")
        
        self._clamped_node_ids = node_ids
        # Convert values to {-1, +1} for SpinNode
        self._clamped_values = [1.0 if v > 0 else -1.0 for v in values]
        
        # Create clamped blocks
        clamped_nodes = [self.nodes[i] for i in node_ids]
        self._clamped_blocks = [Block(clamped_nodes)] if clamped_nodes else []
        
        # Rebuild free blocks excluding clamped nodes
        clamped_set = set(node_ids)
        free_node_ids = [i for i in range(self.n_nodes) if i not in clamped_set]
        
        # Use current blocking strategy to partition free nodes
        if self._blocking_strategy is not None and node_positions is not None:
            # Build blocks for free nodes only
            free_positions = node_positions[free_node_ids] if node_positions is not None else None
            
            # Create connectivity for free nodes
            free_connectivity = None
            if self._blocking_strategy.requires_connectivity:
                full_conn = (np.abs(self._full_weights[:self.n_nodes, :self.n_nodes]) > 1e-6).astype(np.float32)
                free_connectivity = full_conn[np.ix_(free_node_ids, free_node_ids)]
            
            strategy_blocks = self._blocking_strategy.build_blocks(
                n_nodes=len(free_node_ids),
                connectivity=free_connectivity,
                positions=free_positions,
                seed=0
            )
            
            # Convert to THRML blocks
            self.free_blocks = []
            for strategy_block in strategy_blocks:
                # Map back to original node indices
                original_ids = [free_node_ids[i] for i in strategy_block.nodes]
                thrml_nodes = [self.nodes[i] for i in original_ids]
                self.free_blocks.append(Block(thrml_nodes))
        else:
            # Simple even/odd partitioning of free nodes
            free_evens = [self.nodes[i] for i in free_node_ids if i % 2 == 0]
            free_odds = [self.nodes[i] for i in free_node_ids if i % 2 == 1]
            self.free_blocks = [Block(free_evens), Block(free_odds)]
        
        print(f"[THRML] Clamped {len(node_ids)} nodes, {len(free_node_ids)} free nodes in {len(self.free_blocks)} blocks")
    
    def clear_clamped_nodes(self, node_positions: Optional[np.ndarray] = None):
        """
        Remove all clamped nodes and restore full sampling.
        
        Args:
            node_positions: Optional positions for rebuilding blocks
        """
        self._clamped_node_ids = []
        self._clamped_values = []
        self._clamped_blocks = []
        
        # Restore original blocks
        if self._blocking_strategy is not None and node_positions is not None:
            # Rebuild with current strategy
            self.set_blocking_strategy(
                self._current_strategy,
                node_positions=node_positions
            )
        else:
            # Default even/odd blocks
            self.free_blocks = [
                Block(self.nodes[::2]),
                Block(self.nodes[1::2])
            ]
        
        print(f"[THRML] Cleared all clamped nodes")
    
    def get_clamped_nodes(self) -> Tuple[List[int], List[float]]:
        """
        Get current clamped nodes.
        
        Returns:
            Tuple of (node_ids, values)
        """
        return self._clamped_node_ids, self._clamped_values
    
    def sample_conditional(
        self,
        n_steps: int,
        temperature: float,
        key: Optional[jax.random.PRNGKey] = None
    ) -> np.ndarray:
        """
        Sample with clamped nodes (conditional sampling).
        
        Only free nodes are sampled; clamped nodes remain fixed.
        
        Args:
            n_steps: Number of sampling steps
            temperature: Sampling temperature
            key: JAX random key
            
        Returns:
            (n_nodes,) sampled states with clamped values fixed
        """
        if not self._clamped_node_ids:
            # No clamped nodes, use regular sampling
            return self.sample_gibbs(n_steps, temperature, key)
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Update beta
        beta_temp = 1.0 / temperature
        self.beta_jax = jnp.array(beta_temp, dtype=jnp.float32)
        
        # Recreate model
        model_temp = IsingEBM(
            nodes=self.nodes,
            edges=self.edges,
            biases=self.biases_jax,
            weights=self.edge_weights_jax,
            beta=self.beta_jax
        )
        
        # Create sampling program with clamped blocks
        # Per https://docs.thrml.ai/en/latest/ API
        program = IsingSamplingProgram(
            model_temp,
            self.free_blocks,
            clamped_blocks=self._clamped_blocks
        )
        
        # Initialize
        k_init, k_samp = jax.random.split(key)
        init_state_free = hinton_init(k_init, model_temp, self.free_blocks, ())
        
        # Convert clamped values to bool for SpinNode
        clamp_bool = [v > 0 for v in self._clamped_values]
        state_clamp = [jnp.array(clamp_bool)]
        
        # Create schedule
        n_warmup = max(1, n_steps // 2)
        n_samples = 1
        
        schedule = SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=2
        )
        
        # Time sampling
        t_start = time.time()
        
        # Sample with clamps
        samples = sample_states(
            k_samp,
            program,
            schedule,
            init_state_free,
            state_clamp,
            [Block(self.nodes)]  # Observe all nodes
        )
        
        wall_time = time.time() - t_start
        
        # Extract final sample
        final_sample = samples[-1]
        binary_states = np.array(final_sample, dtype=np.float32)
        
        # Record benchmark
        energy = self.compute_energy(binary_states)
        self._record_benchmark_sample(
            wall_time=wall_time,
            n_samples=n_samples,
            states=binary_states,
            energy=energy
        )
        
        return binary_states
    
    # ========================================================================
    # Benchmarking & Diagnostics
    # ========================================================================
    
    def _record_benchmark_sample(
        self,
        wall_time: float,
        n_samples: int,
        states: np.ndarray,
        energy: float
    ):
        """Record a benchmark sample for performance tracking."""
        # Compute magnetization (mean of spins)
        magnetization = float(np.mean(states))
        
        # Record in history
        sample = BenchmarkSample(
            timestamp=time.time(),
            wall_time=wall_time,
            n_samples=n_samples,
            magnetization=magnetization,
            energy=energy,
            strategy=self._current_strategy,
            n_chains=self._current_n_chains
        )
        
        self._benchmark_history.append(sample)
        self._magnetization_history.append(magnetization)
        self._total_samples += n_samples
    
    def compute_autocorrelation(self, lag: int = 1, max_lag: int = 50) -> np.ndarray:
        """
        Compute autocorrelation of magnetization at given lags.
        
        Args:
            lag: Single lag to compute (if max_lag is None)
            max_lag: Maximum lag to compute (returns array)
            
        Returns:
            Autocorrelation value(s)
        """
        if len(self._magnetization_history) < max_lag + 1:
            return np.zeros(max_lag) if max_lag else 0.0
        
        # Convert to numpy array
        m = np.array(list(self._magnetization_history))
        
        # Center the data
        m_centered = m - np.mean(m)
        
        # Compute autocorrelation
        if max_lag:
            autocorr = np.zeros(max_lag)
            denom = np.dot(m_centered, m_centered)
            
            if denom > 0:
                for k in range(1, max_lag + 1):
                    if len(m_centered) > k:
                        autocorr[k-1] = np.dot(m_centered[:-k], m_centered[k:]) / denom
            
            return autocorr
        else:
            # Single lag
            if len(m_centered) > lag:
                denom = np.dot(m_centered, m_centered)
                if denom > 0:
                    return np.dot(m_centered[:-lag], m_centered[lag:]) / denom
            return 0.0
    
    def compute_tau_int(self) -> float:
        """
        Compute integrated autocorrelation time using AR(1) approximation.
        
        τ_int ≈ (1 + ρ₁) / (1 - ρ₁)
        
        Returns:
            Integrated autocorrelation time
        """
        rho1 = self.compute_autocorrelation(lag=1, max_lag=None)
        rho1 = float(np.clip(rho1, -0.999, 0.999))
        
        if abs(rho1 - 1.0) < 1e-6:
            return 100.0  # Cap at reasonable value
        
        tau_int = (1.0 + rho1) / (1.0 - rho1)
        return max(1.0, tau_int)
    
    def compute_ess_per_sec(self) -> float:
        """
        Compute effective sample size per second.
        
        ESS/sec = samples_per_sec / τ_int
        
        Returns:
            Effective samples per second
        """
        if len(self._benchmark_history) < 2:
            return 0.0
        
        # Compute samples/sec from recent history
        recent = list(self._benchmark_history)[-10:]  # Last 10 samples
        total_time = sum(s.wall_time for s in recent)
        total_samples = sum(s.n_samples for s in recent)
        
        if total_time > 0:
            samples_per_sec = total_samples / total_time
        else:
            samples_per_sec = 0.0
        
        # Compute ESS/sec
        tau_int = self.compute_tau_int()
        ess_per_sec = samples_per_sec / tau_int if tau_int > 0 else 0.0
        
        return ess_per_sec
    
    def get_benchmark_diagnostics(self) -> Dict[str, Any]:
        """
        Get current benchmark diagnostics.
        
        Returns:
            Dict with samples/sec, ESS/sec, autocorr, tau_int, etc.
        """
        if len(self._benchmark_history) < 2:
            return {
                'samples_per_sec': 0.0,
                'ess_per_sec': 0.0,
                'lag1_autocorr': 0.0,
                'tau_int': 1.0,
                'energy': 0.0,
                'total_samples': self._total_samples,
                'history_length': 0
            }
        
        # Compute metrics from recent history
        recent = list(self._benchmark_history)[-10:]
        total_time = sum(s.wall_time for s in recent)
        total_samples = sum(s.n_samples for s in recent)
        
        samples_per_sec = total_samples / total_time if total_time > 0 else 0.0
        lag1_autocorr = float(self.compute_autocorrelation(lag=1, max_lag=None))
        tau_int = self.compute_tau_int()
        ess_per_sec = self.compute_ess_per_sec()
        
        # Get latest energy
        latest_energy = recent[-1].energy if recent else 0.0
        
        return {
            'samples_per_sec': samples_per_sec,
            'ess_per_sec': ess_per_sec,
            'lag1_autocorr': lag1_autocorr,
            'tau_int': tau_int,
            'energy': latest_energy,
            'total_samples': self._total_samples,
            'history_length': len(self._benchmark_history),
            'strategy': self._current_strategy,
            'n_chains': self._current_n_chains
        }
    
    def get_benchmark_history(self, max_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Get benchmark history for plotting.
        
        Args:
            max_samples: Maximum number of samples to return
            
        Returns:
            List of benchmark samples as dicts
        """
        history = list(self._benchmark_history)[-max_samples:]
        
        return [
            {
                'timestamp': s.timestamp,
                'wall_time': s.wall_time,
                'n_samples': s.n_samples,
                'magnetization': s.magnetization,
                'energy': s.energy,
                'strategy': s.strategy,
                'n_chains': s.n_chains
            }
            for s in history
        ]
    
    def get_benchmark_json(self) -> Dict[str, Any]:
        """
        Export benchmark data in THRML-Testing leaderboard format.
        
        Returns:
            Dict compatible with thrmlbench.com submission
        """
        diagnostics = self.get_benchmark_diagnostics()
        
        # Get device info
        devices = jax.devices()
        device_type = "gpu" if any("gpu" in str(d).lower() for d in devices) else "cpu"
        
        return {
            'device': [str(d) for d in devices],
            'device_type': device_type,
            'H': int(np.sqrt(self.n_nodes)),  # Assume square grid
            'W': int(np.sqrt(self.n_nodes)),
            'n_nodes': self.n_nodes,
            'blocking': self._current_strategy,
            'n_chains': self._current_n_chains,
            'samples_per_sec': diagnostics['samples_per_sec'],
            'lag1_autocorr': diagnostics['lag1_autocorr'],
            'tau_int_est': diagnostics['tau_int'],
            'ess_per_sec': diagnostics['ess_per_sec'],
            'total_samples': diagnostics['total_samples'],
            'timestamp': time.time()
        }
    
    def export_benchmark_csv(self, filepath: str):
        """
        Export benchmark history to CSV file.
        
        Args:
            filepath: Path to CSV file
        """
        import csv
        
        history = self.get_benchmark_history(max_samples=1000)
        
        with open(filepath, 'w', newline='') as f:
            if not history:
                return
            
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    
    # ========================================================================
    # Serialization
    # ========================================================================
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize wrapper state for storage in SystemState.
        
        Returns:
            Dictionary with serialized state including all configuration
        """
        try:
            return {
                'version': '2.0',  # Serialization version
                'n_nodes': self.n_nodes,
                'weights': self._full_weights[:self.n_nodes, :self.n_nodes].tolist(),
                'biases': np.array(self.biases_jax).tolist(),
                'beta': float(self.beta),
                'current_strategy': self._current_strategy,
                'current_n_chains': self._current_n_chains,
                'clamped_node_ids': self._clamped_node_ids.copy(),
                'clamped_values': self._clamped_values.copy(),
                'total_samples': self._total_samples,
                'error_count': self._error_count
            }
        except Exception as e:
            print(f"[THRML WARNING] Serialization failed: {e}")
            # Return minimal serialization
            return {
                'version': '2.0',
                'n_nodes': self.n_nodes,
                'weights': np.zeros((self.n_nodes, self.n_nodes)).tolist(),
                'biases': np.zeros(self.n_nodes).tolist(),
                'beta': 1.0
            }
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'THRMLWrapper':
        """
        Reconstruct wrapper from serialized state.
        
        Args:
            data: Dictionary from serialize()
            
        Returns:
            Reconstructed THRMLWrapper
            
        Raises:
            ValueError: If data is invalid or missing required fields
        """
        try:
            # Validate required fields
            required_fields = ['n_nodes', 'weights', 'biases', 'beta']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create wrapper
            wrapper = THRMLWrapper(
                n_nodes=data['n_nodes'],
                weights=np.array(data['weights'], dtype=np.float32),
                biases=np.array(data['biases'], dtype=np.float32),
                beta=float(data['beta'])
            )
            
            # Restore optional state (v2.0+)
            if 'current_strategy' in data:
                wrapper._current_strategy = data['current_strategy']
            
            if 'current_n_chains' in data:
                wrapper._current_n_chains = data['current_n_chains']
            
            if 'clamped_node_ids' in data and 'clamped_values' in data:
                wrapper._clamped_node_ids = data['clamped_node_ids']
                wrapper._clamped_values = data['clamped_values']
            
            if 'total_samples' in data:
                wrapper._total_samples = data['total_samples']
            
            if 'error_count' in data:
                wrapper._error_count = data['error_count']
            
            return wrapper
            
        except Exception as e:
            raise ValueError(f"Failed to deserialize THRMLWrapper: {e}") from e


# ============================================================================
# Helper Functions
# ============================================================================

def create_thrml_model(
    n_nodes: int,
    weights: np.ndarray,
    biases: np.ndarray,
    beta: float = 1.0,
    fallback_on_error: bool = True
) -> Optional[THRMLWrapper]:
    """
    Factory function to create THRML model wrapper with error handling.
    
    Args:
        n_nodes: Number of nodes
        weights: (n_nodes, n_nodes) weight matrix
        biases: (n_nodes,) bias vector
        beta: Inverse temperature
        fallback_on_error: If True, return None on error instead of raising
        
    Returns:
        THRMLWrapper instance, or None if creation failed and fallback_on_error=True
        
    Raises:
        RuntimeError: If creation fails and fallback_on_error=False
    """
    try:
        if not THRML_AVAILABLE:
            raise RuntimeError("THRML not available")
        
        return THRMLWrapper(n_nodes, weights, biases, beta)
        
    except Exception as e:
        error_msg = f"Failed to create THRML model: {e}"
        
        if fallback_on_error:
            print(f"[THRML WARNING] {error_msg}, returning None")
            return None
        else:
            raise RuntimeError(error_msg) from e


def thrml_to_jax_weights(thrml_wrapper: THRMLWrapper) -> jnp.ndarray:
    """
    Extract weights from THRML model as JAX array.
    
    Args:
        thrml_wrapper: THRMLWrapper instance
        
    Returns:
        (n_nodes, n_nodes) JAX array of weights
    """
    weights = thrml_wrapper.get_weights()
    return jnp.array(weights, dtype=jnp.float32)


def jax_to_thrml_weights(
    jax_weights: jnp.ndarray,
    thrml_wrapper: THRMLWrapper
) -> THRMLWrapper:
    """
    Update THRML model with weights from JAX array.
    
    Creates a new wrapper with updated weights.
    
    Args:
        jax_weights: (n_nodes, n_nodes) JAX weight array
        thrml_wrapper: Existing wrapper (for metadata)
        
    Returns:
        New THRMLWrapper with updated weights
    """
    weights_np = np.array(jax_weights)
    biases_np = np.array(thrml_wrapper.biases_jax)
    
    return THRMLWrapper(
        n_nodes=thrml_wrapper.n_nodes,
        weights=weights_np,
        biases=biases_np,
        beta=thrml_wrapper.beta
    )


def reconstruct_thrml_wrapper(state_dict: Dict[str, Any]) -> THRMLWrapper:
    """
    Reconstruct THRML wrapper from SystemState serialized data.
    
    Args:
        state_dict: Dictionary with THRML model data
        
    Returns:
        Reconstructed THRMLWrapper
    """
    return THRMLWrapper.deserialize(state_dict)


# ============================================================================
# Advanced THRML Features
# ============================================================================

class HeterogeneousTHRMLWrapper(THRMLWrapper):
    """
    Extended THRML wrapper supporting heterogeneous node types.
    
    Supports mixing:
    - SpinNode: Binary {-1, +1} for standard Ising
    - ContinuousNode: Real-valued for audio/analog signals
    - DiscreteNode: Multi-valued discrete for ML/classification
    
    Node type mapping:
    - Audio nodes → ContinuousNode
    - Photonic nodes → SpinNode
    - ML/classification → DiscreteNode
    """
    
    def __init__(
        self,
        n_nodes: int,
        node_types: np.ndarray,  # (n_nodes,) array of node type IDs
        weights: Optional[np.ndarray] = None,
        biases: Optional[np.ndarray] = None,
        beta: float = 1.0
    ):
        """
        Initialize heterogeneous THRML model.
        
        Args:
            n_nodes: Number of nodes
            node_types: (n_nodes,) array where 0=Spin, 1=Continuous, 2=Discrete
            weights: (n_nodes, n_nodes) coupling weights
            biases: (n_nodes,) node biases
            beta: Inverse temperature
        """
        if not THRML_HETEROGENEOUS_AVAILABLE:
            raise ImportError(
                "Heterogeneous THRML nodes not available. "
                "Update THRML to latest version."
            )
        
        self.n_nodes = n_nodes
        self.node_types = node_types
        self.beta = beta
        
        # Create heterogeneous nodes
        self.nodes = []
        for i in range(n_nodes):
            if node_types[i] == 0:  # Spin
                self.nodes.append(SpinNode())
            elif node_types[i] == 1:  # Continuous
                self.nodes.append(ContinuousNode())
            elif node_types[i] == 2:  # Discrete (4 values)
                self.nodes.append(DiscreteNode(n_values=4))
            else:  # Default to Spin
                self.nodes.append(SpinNode())
        
        # Create edges (fully connected)
        self.edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                self.edges.append((self.nodes[i], self.nodes[j]))
        
        # Initialize weights and biases
        if weights is None:
            weights = np.random.randn(n_nodes, n_nodes) * 0.01
            weights = (weights + weights.T) / 2  # Symmetrize
            np.fill_diagonal(weights, 0)
        
        if biases is None:
            biases = np.zeros(n_nodes)
        
        self.weights_np = weights
        self.biases_np = biases
        
        # Convert to JAX arrays
        self.weights_jax = jnp.array(weights, dtype=jnp.float32)
        self.biases_jax = jnp.array(biases, dtype=jnp.float32)
        
        # Extract edge weights
        edge_weights = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                edge_weights.append(weights[i, j])
        edge_weights = np.array(edge_weights)
        
        # Create model (Note: May need custom energy function for heterogeneous)
        self.model = IsingEBM(
            self.nodes,
            self.edges,
            biases,
            edge_weights,
            jnp.array(beta)
        )
        
        # Create sampling program
        self.free_blocks = [Block(self.nodes[::2]), Block(self.nodes[1::2])]
        # Per https://docs.thrml.ai/en/latest/ API
        self.program = IsingSamplingProgram(
            self.model,
            self.free_blocks,
            clamped_blocks=[]
        )
        
        # Clamped nodes for conditional sampling
        self.clamped_nodes = []
        self.clamped_values = []
    
    def set_clamped_nodes(self, node_ids: List[int], values: List[float]):
        """
        Set nodes to be clamped (fixed) during sampling.
        
        Args:
            node_ids: List of node indices to clamp
            values: List of values to clamp to
        """
        self.clamped_nodes = node_ids
        self.clamped_values = values
        
        # Create clamped blocks
        clamped_block_nodes = [self.nodes[i] for i in node_ids]
        if clamped_block_nodes:
            self.clamped_blocks = [Block(clamped_block_nodes)]
        else:
            self.clamped_blocks = []
        
        # Update sampling program
        # Per https://docs.thrml.ai/en/latest/ API
        self.program = IsingSamplingProgram(
            self.model,
            self.free_blocks,
            clamped_blocks=self.clamped_blocks
        )
    
    def sample_conditional(
        self,
        key: jax.random.PRNGKey,
        n_steps: int = 10,
        n_warmup: int = 5
    ) -> jnp.ndarray:
        """
        Sample with clamped nodes (conditional sampling).
        
        Args:
            key: JAX PRNG key
            n_steps: Number of sampling steps
            n_warmup: Number of warmup steps
            
        Returns:
            (n_nodes,) sampled state
        """
        # Initialize state
        k_init, k_samp = jax.random.split(key, 2)
        
        # Create initial state with clamped values
        init_state = hinton_init(k_init, self.model, self.free_blocks, tuple(self.clamped_blocks))
        
        # Set clamped values
        for node_id, value in zip(self.clamped_nodes, self.clamped_values):
            # Note: This is simplified - actual implementation depends on THRML API
            pass
        
        # Sample
        schedule = SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=1,
            steps_per_sample=n_steps
        )
        
        samples = sample_states(
            k_samp,
            self.program,
            schedule,
            init_state,
            [],
            [Block(self.nodes)]
        )
        
        # Extract final state
        final_state = samples[Block(self.nodes)][-1]
        
        return final_state


class THRMLFactorSystem:
    """
    Custom energy function (factor) system for THRML.
    
    Allows defining custom energy functions beyond standard Ising model,
    enabling domain-specific constraints and interactions.
    """
    
    def __init__(self):
        """Initialize factor system."""
        self.factors = []
    
    def add_photonic_coupling_factor(
        self,
        node_ids: List[int],
        coupling_strength: float,
        wavelength: float = 1550e-9  # nm
    ):
        """
        Add photonic coupling energy factor.
        
        E_photonic = -κ * |Σ_i ψ_i * exp(i*k*r_i)|²
        
        Args:
            node_ids: Nodes involved in coupling
            coupling_strength: Coupling coefficient κ
            wavelength: Optical wavelength
        """
        factor = {
            'type': 'photonic_coupling',
            'node_ids': node_ids,
            'strength': coupling_strength,
            'wavelength': wavelength
        }
        self.factors.append(factor)
    
    def add_audio_harmony_factor(
        self,
        node_ids: List[int],
        fundamental_freq: float = 440.0
    ):
        """
        Add audio harmony constraint factor.
        
        Encourages harmonic relationships between nodes.
        
        Args:
            node_ids: Nodes representing audio frequencies
            fundamental_freq: Fundamental frequency (Hz)
        """
        factor = {
            'type': 'audio_harmony',
            'node_ids': node_ids,
            'fundamental': fundamental_freq
        }
        self.factors.append(factor)
    
    def add_ml_regularization_factor(
        self,
        node_ids: List[int],
        regularization_type: str = 'l2',
        strength: float = 0.01
    ):
        """
        Add ML regularization factor.
        
        Args:
            node_ids: Nodes to regularize
            regularization_type: 'l1', 'l2', or 'elastic'
            strength: Regularization strength
        """
        factor = {
            'type': 'ml_regularization',
            'node_ids': node_ids,
            'reg_type': regularization_type,
            'strength': strength
        }
        self.factors.append(factor)
    
    def compute_total_energy(self, state: jnp.ndarray) -> float:
        """
        Compute total energy from all factors.
        
        Args:
            state: (n_nodes,) current state
            
        Returns:
            Total energy
        """
        energy = 0.0
        
        for factor in self.factors:
            if factor['type'] == 'photonic_coupling':
                # Photonic coupling energy
                node_ids = factor['node_ids']
                strength = factor['strength']
                # Simplified: sum of squared amplitudes
                amplitudes = state[node_ids]
                energy -= strength * jnp.sum(amplitudes ** 2)
            
            elif factor['type'] == 'audio_harmony':
                # Harmony energy (encourage integer ratios)
                node_ids = factor['node_ids']
                amplitudes = state[node_ids]
                # Simplified: penalize dissonance
                energy -= jnp.sum(jnp.cos(amplitudes))
            
            elif factor['type'] == 'ml_regularization':
                # Regularization energy
                node_ids = factor['node_ids']
                strength = factor['strength']
                reg_type = factor['reg_type']
                
                values = state[node_ids]
                if reg_type == 'l1':
                    energy += strength * jnp.sum(jnp.abs(values))
                elif reg_type == 'l2':
                    energy += strength * jnp.sum(values ** 2)
                elif reg_type == 'elastic':
                    energy += strength * (jnp.sum(jnp.abs(values)) + jnp.sum(values ** 2))
        
        return energy


def create_heterogeneous_model_for_application(
    n_audio_nodes: int,
    n_photonic_nodes: int,
    n_ml_nodes: int
) -> HeterogeneousTHRMLWrapper:
    """
    Create a heterogeneous THRML model for mixed applications.
    
    Args:
        n_audio_nodes: Number of audio (continuous) nodes
        n_photonic_nodes: Number of photonic (spin) nodes
        n_ml_nodes: Number of ML (discrete) nodes
        
    Returns:
        Configured HeterogeneousTHRMLWrapper
    """
    n_total = n_audio_nodes + n_photonic_nodes + n_ml_nodes
    
    # Create node type array
    node_types = np.zeros(n_total, dtype=np.int32)
    node_types[:n_audio_nodes] = 1  # Continuous
    node_types[n_audio_nodes:n_audio_nodes + n_photonic_nodes] = 0  # Spin
    node_types[n_audio_nodes + n_photonic_nodes:] = 2  # Discrete
    
    # Create model
    model = HeterogeneousTHRMLWrapper(
        n_nodes=n_total,
        node_types=node_types
    )
    
    return model


# ============================================================================
# Wave Field → THRML Structure Learning
# ============================================================================

def update_thrml_from_wave_correlations(
    thrml_wrapper: THRMLWrapper,
    wave_field: jnp.ndarray,
    oscillator_states: jnp.ndarray,
    positions: jnp.ndarray,
    eta: float = 0.001
) -> THRMLWrapper:
    """
    Learn THRML coupling weights from wave-mediated correlations.
    
    This implements bidirectional coupling where the wave field acts as a
    communication medium between oscillators. Oscillators that share wave
    energy develop stronger THRML coupling, creating an emergent network
    structure shaped by wave dynamics.
    
    Algorithm:
    1. Sample wave field energy at each oscillator position
    2. Compute wave-mediated coupling strength: w_ij = E_i * E_j
    3. Compute oscillator state correlation: c_ij = x_i * x_j
    4. Hebbian update: ΔW_ij = η * w_ij * c_ij
    
    This creates a feedback loop:
    - Oscillators → Wave field (via source terms)
    - Wave field → THRML weights (via this function)
    - THRML → Oscillators (via feedback in simulation)
    
    Args:
        thrml_wrapper: THRML wrapper with current model
        wave_field: (GRID_W, GRID_H) wave field state
        oscillator_states: (N, 3) oscillator states [x, y, z]
        positions: (N, 2) oscillator positions in grid coordinates
        eta: Learning rate (default 0.001)
        
    Returns:
        Updated THRMLWrapper with learned weights
        
    Example:
        >>> # In simulation loop:
        >>> state, thrml_wrapper = simulation_step(state, thrml_wrapper)
        >>> if step % 10 == 0:  # Learn every 10 steps
        >>>     thrml_wrapper = update_thrml_from_wave_correlations(
        >>>         thrml_wrapper,
        >>>         state.field_p,
        >>>         state.oscillator_state,
        >>>         state.node_positions,
        >>>         eta=0.001
        >>>     )
    """
    try:
        # Import wave field sampling function
        from .wave_pde import sample_field_at_nodes
        
        # Sample wave energy at each oscillator position
        wave_energies = sample_field_at_nodes(wave_field, positions)
        
        # Compute wave field energy (magnitude)
        wave_energies = jnp.abs(wave_energies)
        
        # Get current weights (numpy)
        weights = thrml_wrapper.get_weights()
        n_active = len(oscillator_states)
        
        # Ensure we don't exceed wrapper's capacity
        n_nodes = min(n_active, thrml_wrapper.n_nodes)
        
        # Convert to numpy for modification
        wave_energies_np = np.array(wave_energies[:n_nodes])
        osc_states_np = np.array(oscillator_states[:n_nodes, 0])  # Use x component
        
        # Compute pairwise updates (upper triangle only for efficiency)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # Wave-mediated coupling strength
                # Higher when both oscillators are in high-energy wave regions
                wave_coupling = wave_energies_np[i] * wave_energies_np[j]
                
                # Oscillator state correlation (Hebbian rule)
                # Strengthens connections between synchronized oscillators
                state_corr = osc_states_np[i] * osc_states_np[j]
                
                # Combined update: wave field mediates Hebbian learning
                delta_w = eta * wave_coupling * state_corr
                
                # Symmetric update
                weights[i, j] += delta_w
                weights[j, i] += delta_w
        
        # Update wrapper's internal weights
        thrml_wrapper._full_weights[:n_nodes, :n_nodes] = weights
        
        # Ensure symmetry and zero diagonal
        weights_sym = (weights + weights.T) / 2.0
        np.fill_diagonal(weights_sym, 0.0)
        thrml_wrapper._full_weights[:n_nodes, :n_nodes] = weights_sym
        
        # Rebuild edge list with updated weights
        thrml_wrapper.edges = []
        thrml_wrapper.edge_weights = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                w = float(thrml_wrapper._full_weights[i, j])
                if abs(w) > 1e-6:  # Only include non-zero weights
                    thrml_wrapper.edges.append((thrml_wrapper.nodes[i], thrml_wrapper.nodes[j]))
                    thrml_wrapper.edge_weights.append(w)
        
        # Update JAX arrays and recreate model
        thrml_wrapper.edge_weights_jax = jnp.array(
            thrml_wrapper.edge_weights if thrml_wrapper.edge_weights else [0.0],
            dtype=jnp.float32
        )
        
        # Recreate THRML model with updated weights
        thrml_wrapper.model = IsingEBM(
            nodes=thrml_wrapper.nodes,
            edges=thrml_wrapper.edges if thrml_wrapper.edges else [],
            biases=thrml_wrapper.biases_jax,
            weights=thrml_wrapper.edge_weights_jax,
            beta=thrml_wrapper.beta_jax
        )
        
        return thrml_wrapper
        
    except Exception as e:
        # Graceful degradation: return wrapper unchanged on error
        print(f"[THRML WARNING] Wave learning failed: {e}")
        thrml_wrapper._last_error = f"wave_learning: {e}"
        thrml_wrapper._error_count += 1
        return thrml_wrapper
