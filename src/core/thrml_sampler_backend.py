"""
THRMLSamplerBackend: SamplerBackend implementation for THRML.

This module provides a concrete implementation of the SamplerBackend interface
that wraps THRMLWrapper, enabling THRML to be used as a drop-in sampler backend
in the GMCS platform.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Any, Optional, Tuple, Type

from src.core.sampler_backend import (
    SamplerBackend,
    SamplerBackendRegistry,
    BackendCapabilities,
)
from src.core.thrml_integration import THRMLWrapper


class THRMLSamplerBackend(SamplerBackend):
    """
    THRML-based sampler backend implementation.
    
    Wraps THRMLWrapper to provide the generic SamplerBackend interface,
    enabling all THRML features (blocking strategies, multi-chain, conditional
    sampling, benchmarking) to be accessed through the unified API.
    """
    
    def __init__(
        self,
        n_nodes: int,
        initial_weights: np.ndarray,
        initial_biases: np.ndarray,
        **kwargs
    ):
        """
        Initialize THRML sampler backend.
        
        Args:
            n_nodes: Number of nodes in the model
            initial_weights: (n_nodes, n_nodes) coupling matrix
            initial_biases: (n_nodes,) bias vector
            **kwargs: Additional THRML-specific parameters (e.g., beta)
        """
        super().__init__(
            "thrml",
            n_nodes=n_nodes,
            initial_weights=initial_weights,
            initial_biases=initial_biases,
            **kwargs,
        )
        
        # Extract THRML-specific params
        beta = kwargs.get('beta', 1.0)
        
        # Create THRMLWrapper
        self.thrml_wrapper = THRMLWrapper(
            n_nodes=n_nodes,
            weights=initial_weights,
            biases=initial_biases,
            beta=beta
        )
        
        # Store node positions if provided
        self.node_positions: Optional[np.ndarray] = kwargs.get('node_positions', None)
    
    @classmethod
    def declared_capabilities(cls) -> BackendCapabilities:
        """Static capability metadata used during registry bootstrap."""
        return BackendCapabilities(
            supports_multi_chain=True,
            supports_conditional=True,
            supports_gpu=True,
            supports_distributed=False,
            supports_hot_reload=False,
            max_nodes=0,
            optimal_strategies=["auto", "checkerboard", "random"],
            hardware_type="gpu",
            extra={
                "benchmarking": True,
                "weight_learning": True,
                "autocorrelation_analysis": True,
                "supports_blocking": True,
                "heterogeneous_nodes": False,
            },
        )

    def sample(
        self,
        n_steps: int,
        temperature: float,
        num_chains: int = 1,
        blocking_strategy_name: str = "auto",
        node_positions: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform sampling using THRML.
        
        Args:
            n_steps: Number of Gibbs sampling steps
            temperature: Sampling temperature
            num_chains: Number of parallel chains (1 for single, -1 for auto)
            blocking_strategy_name: Blocking strategy to use
            node_positions: Optional (n_nodes, 2) spatial positions
            **kwargs: Additional THRML parameters
            
        Returns:
            Tuple of (samples, diagnostics)
        """
        # Update node positions if provided
        if node_positions is not None:
            self.node_positions = node_positions
        
        # Set blocking strategy if not "auto" or if changed
        if blocking_strategy_name != "auto":
            if blocking_strategy_name != self.thrml_wrapper.get_blocking_strategy():
                self.thrml_wrapper.set_blocking_strategy(
                    strategy_name=blocking_strategy_name,
                    node_positions=self.node_positions
                )
        
        # Determine if using multi-chain
        use_multi_chain = num_chains != 1
        
        # Generate JAX key
        key_subkey = self._update_rng_key()
        
        if use_multi_chain:
            # Use parallel sampling
            if num_chains == -1:
                # Auto-detect
                num_chains = self.thrml_wrapper._auto_detect_optimal_chains()
            
            samples, per_chain_diag = self.thrml_wrapper.sample_gibbs_parallel(
                n_steps=n_steps,
                temperature=temperature,
                n_chains=num_chains,
                key=key_subkey
            )
            
            # Aggregate diagnostics
            diagnostics = {
                'num_chains': num_chains,
                'per_chain': per_chain_diag,
                'mean_energy': np.mean([d['energy'] for d in per_chain_diag]),
                'mean_magnetization': np.mean([d['magnetization'] for d in per_chain_diag])
            }
            
            # Add benchmark metrics
            bench_diag = self.thrml_wrapper.get_benchmark_diagnostics()
            diagnostics.update(bench_diag)
            
            return samples, diagnostics
        else:
            # Single chain sampling
            # Check if conditional sampling is active
            if self.thrml_wrapper._clamped_node_ids:
                samples = self.thrml_wrapper.sample_conditional(
                    n_steps=n_steps,
                    temperature=temperature,
                    key=key_subkey
                )
            else:
                samples = self.thrml_wrapper.sample_gibbs(
                    n_steps=n_steps,
                    temperature=temperature,
                    key=key_subkey
                )
            
            # Get diagnostics
            energy = self.thrml_wrapper.compute_energy(samples)
            magnetization = float(np.mean(samples))
            
            diagnostics = {
                'num_chains': 1,
                'energy': energy,
                'magnetization': magnetization
            }
            
            # Add benchmark metrics
            bench_diag = self.thrml_wrapper.get_benchmark_diagnostics()
            diagnostics.update(bench_diag)
            
            return samples.reshape(1, -1), diagnostics  # Shape (1, n_nodes) for consistency
    
    def compute_energy(self, states: np.ndarray) -> float:
        """
        Compute energy of given states.
        
        Args:
            states: (n_nodes,) or (num_chains, n_nodes) states
            
        Returns:
            Scalar energy value
        """
        if states.ndim == 2:
            # Multiple chains, compute mean energy
            energies = [self.thrml_wrapper.compute_energy(s) for s in states]
            return float(np.mean(energies))
        else:
            return self.thrml_wrapper.compute_energy(states)
    
    def update_weights(
        self,
        data_states: np.ndarray,
        learning_rate: float,
        k_steps: int = 1,
        num_chains: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Update coupling weights using Contrastive Divergence.
        
        Args:
            data_states: (n_nodes,) or (batch_size, n_nodes) observed data
            learning_rate: Learning rate (eta)
            k_steps: CD-k steps
            num_chains: Number of parallel chains for model phase
            **kwargs: Additional parameters
            
        Returns:
            Dict with learning diagnostics
        """
        # Generate key
        key_subkey = self._update_rng_key()
        
        # If batch, average over batch
        if data_states.ndim == 2:
            # Average data statistics over batch
            batch_size = data_states.shape[0]
            total_diagnostics = {
                'gradient_norm': 0.0,
                'data_energy': 0.0,
                'model_energy': 0.0,
                'energy_diff': 0.0
            }
            
            for i in range(batch_size):
                diag = self.thrml_wrapper.update_weights_cd(
                    data_states=data_states[i],
                    eta=learning_rate / batch_size,  # Scale by batch size
                    k_steps=k_steps,
                    key=jax.random.fold_in(key_subkey, i),
                    n_chains=num_chains
                )
                
                for key in total_diagnostics:
                    total_diagnostics[key] += diag[key]
            
            # Average
            for key in total_diagnostics:
                total_diagnostics[key] /= batch_size
            
            return total_diagnostics
        else:
            # Single sample
            return self.thrml_wrapper.update_weights_cd(
                data_states=data_states,
                eta=learning_rate,
                k_steps=k_steps,
                key=key_subkey,
                n_chains=num_chains
            )
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get current performance diagnostics.
        
        Returns:
            Dict with benchmark metrics, energy, chain info, etc.
        """
        diagnostics = self.thrml_wrapper.get_benchmark_diagnostics()
        
        # Add chain diagnostics
        chain_diag = self.thrml_wrapper.get_chain_diagnostics()
        diagnostics['chain_info'] = chain_diag
        
        # Add blocking strategy info
        diagnostics['blocking_strategy'] = self.thrml_wrapper.get_blocking_strategy()
        
        # Add clamped node info
        clamped_ids, clamped_vals = self.thrml_wrapper.get_clamped_nodes()
        diagnostics['clamped_nodes'] = {
            'count': len(clamped_ids),
            'ids': clamped_ids,
            'values': clamped_vals
        }
        
        return diagnostics
    
    def set_conditional_nodes(self, node_ids: List[int], values: List[Any]):
        """
        Clamp nodes for conditional sampling.
        
        Args:
            node_ids: List of node indices to clamp
            values: List of values to clamp to
        """
        self.thrml_wrapper.set_clamped_nodes(
            node_ids=node_ids,
            values=values,
            node_positions=self.node_positions
        )
        
        # Update internal tracking
        self.clamped_nodes = dict(zip(node_ids, values))
    
    def clear_conditional_nodes(self):
        """Clear all clamped nodes."""
        self.thrml_wrapper.clear_clamped_nodes(node_positions=self.node_positions)
        self.clamped_nodes = {}
    
    def get_capabilities(self) -> BackendCapabilities:
        """Return dynamic THRML capability metadata."""
        base_caps = self.declared_capabilities()
        strategies = self.thrml_wrapper.get_supported_strategies()
        extra = dict(base_caps.extra)
        extra.update(
            {
                "blocking_strategies": strategies,
                "node_positions_supported": self.node_positions is not None,
                "ess_computation": True,
                "hardware": "JAX (CPU/GPU)",
            }
        )

        return BackendCapabilities(
            supports_multi_chain=base_caps.supports_multi_chain,
            supports_conditional=base_caps.supports_conditional,
            supports_gpu=base_caps.supports_gpu,
            supports_distributed=base_caps.supports_distributed,
            supports_hot_reload=base_caps.supports_hot_reload,
            max_nodes=self.n_nodes or base_caps.max_nodes,
            optimal_strategies=strategies or base_caps.optimal_strategies,
            hardware_type=base_caps.hardware_type,
            extra=extra,
        )
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize backend state.
        
        Returns:
            Dict with all state needed to reconstruct backend
        """
        return {
            'backend_type': 'thrml',
            'n_nodes': self.n_nodes,
            'thrml_state': self.thrml_wrapper.serialize(),
            'node_positions': self.node_positions.tolist() if self.node_positions is not None else None,
            'clamped_nodes': self.clamped_nodes
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'THRMLSamplerBackend':
        """
        Deserialize backend from saved state.
        
        Args:
            data: Serialized state dict
            
        Returns:
            Reconstructed THRMLSamplerBackend instance
        """
        thrml_state = data['thrml_state']
        
        # Reconstruct
        backend = cls(
            n_nodes=data['n_nodes'],
            initial_weights=np.array(thrml_state['weights']),
            initial_biases=np.array(thrml_state['biases']),
            beta=thrml_state['beta'],
            node_positions=np.array(data['node_positions']) if data['node_positions'] else None
        )
        
        # Restore clamped nodes if any
        if data.get('clamped_nodes'):
            node_ids = list(data['clamped_nodes'].keys())
            values = list(data['clamped_nodes'].values())
            backend.set_conditional_nodes(node_ids, values)
        
        return backend


# Register THRML backend with static capability metadata
SamplerBackendRegistry.register(
    "thrml",
    THRMLSamplerBackend,
    THRMLSamplerBackend.declared_capabilities(),
    "THRML (JAX) sampler backend",
    "1.0.0",
)

