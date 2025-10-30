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
from typing import Tuple, Optional, Dict, Any, List

# THRML imports - full feature set
try:
    from thrml import SpinNode, Block, SamplingSchedule, sample_states
    from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
    # Import observers for advanced sampling diagnostics
    from thrml.observers import (
        EnergyObserver,
        StateObserver,
        CorrelationObserver
    )
    # Import heterogeneous node types
    try:
        from thrml import ContinuousNode, DiscreteNode
        THRML_HETEROGENEOUS_AVAILABLE = True
    except ImportError:
        THRML_HETEROGENEOUS_AVAILABLE = False
    
    THRML_AVAILABLE = True
    THRML_OBSERVERS_AVAILABLE = True
except ImportError as e:
    # Try basic import without observers
    try:
        from thrml import SpinNode, Block, SamplingSchedule, sample_states
        from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
        THRML_AVAILABLE = True
        THRML_OBSERVERS_AVAILABLE = False
        THRML_HETEROGENEOUS_AVAILABLE = False
    except ImportError:
        THRML_AVAILABLE = False
        THRML_OBSERVERS_AVAILABLE = False
        THRML_HETEROGENEOUS_AVAILABLE = False
        raise ImportError(
            "THRML is required for this module. Install with: pip install thrml>=0.1.3"
        )


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
        """
        if not THRML_AVAILABLE:
            raise RuntimeError("THRML not available")
        
        self.n_nodes = n_nodes
        self.beta = beta
        
        # Create THRML spin nodes
        self.nodes = [SpinNode() for _ in range(n_nodes)]
        
        # Build edges from weight matrix
        # Only include non-zero weights, use upper triangle
        self.edges = []
        self.edge_weights = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                w = weights[i, j]
                if abs(w) > 1e-6:
                    self.edges.append((self.nodes[i], self.nodes[j]))
                    self.edge_weights.append(float(w))
        
        # Convert to JAX arrays
        self.edge_weights_jax = jnp.array(self.edge_weights, dtype=jnp.float32)
        self.biases_jax = jnp.array(biases, dtype=jnp.float32)
        self.beta_jax = jnp.array(beta, dtype=jnp.float32)
        
        # Create THRML Ising model
        self.model = IsingEBM(
            nodes=self.nodes,
            edges=self.edges,
            biases=self.biases_jax,
            weights=self.edge_weights_jax,
            beta=self.beta_jax
        )
        
        # Create two-color blocks for efficient Gibbs sampling
        # Even/odd coloring ensures no conflicts within a block
        self.free_blocks = [
            Block(self.nodes[::2]),   # Even indices
            Block(self.nodes[1::2])   # Odd indices
        ]
        
        # Store full weight matrix for convenience
        self._full_weights = weights.copy()
    
    def update_biases(self, gmcs_biases: np.ndarray) -> None:
        """
        Update node biases from GMCS pipeline output.
        
        This allows the GMCS signal processing chain to directly
        influence the EBM sampling distribution.
        
        Args:
            gmcs_biases: (n_nodes,) new bias values
        """
        self.biases_jax = jnp.array(gmcs_biases[:self.n_nodes], dtype=jnp.float32)
        
        # Recreate model with updated biases
        self.model = IsingEBM(
            nodes=self.nodes,
            edges=self.edges,
            biases=self.biases_jax,
            weights=self.edge_weights_jax,
            beta=self.beta_jax
        )
    
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
        
        # Create sampling program with free and clamped blocks
        # Using empty clamped_blocks for fully free sampling
        program = IsingSamplingProgram(
            model=model_temp,
            free_blocks=self.free_blocks,
            clamped_blocks=[]  # No clamped nodes
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
        
        # Extract samples
        # THRML returns samples in model's native format
        if return_all_samples:
            # Return all samples as (n_samples, n_nodes) array
            binary_states = np.array(samples, dtype=np.float32)
        else:
            # Return only final sample
            final_sample = samples[-1]
            binary_states = np.array(final_sample, dtype=np.float32)
        
        return binary_states
    
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
        program = IsingSamplingProgram(
            model=model_temp,
            free_blocks=self.free_blocks,
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
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize wrapper state for storage in SystemState.
        
        Returns:
            Dictionary with serialized state
        """
        return {
            'n_nodes': self.n_nodes,
            'weights': self._full_weights.tolist(),
            'biases': np.array(self.biases_jax).tolist(),
            'beta': float(self.beta)
        }
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'THRMLWrapper':
        """
        Reconstruct wrapper from serialized state.
        
        Args:
            data: Dictionary from serialize()
            
        Returns:
            Reconstructed THRMLWrapper
        """
        return THRMLWrapper(
            n_nodes=data['n_nodes'],
            weights=np.array(data['weights']),
            biases=np.array(data['biases']),
            beta=data['beta']
        )


# ============================================================================
# Helper Functions
# ============================================================================

def create_thrml_model(
    n_nodes: int,
    weights: np.ndarray,
    biases: np.ndarray,
    beta: float = 1.0
) -> THRMLWrapper:
    """
    Factory function to create THRML model wrapper.
    
    Args:
        n_nodes: Number of nodes
        weights: (n_nodes, n_nodes) weight matrix
        biases: (n_nodes,) bias vector
        beta: Inverse temperature
        
    Returns:
        THRMLWrapper instance
    """
    return THRMLWrapper(n_nodes, weights, biases, beta)


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

