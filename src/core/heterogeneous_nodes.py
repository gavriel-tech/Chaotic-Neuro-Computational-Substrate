"""
Heterogeneous Node Types for THRML

Supports mixing different node types in a single graphical model:
- SpinNode: Binary {-1, +1} for Ising-style models
- ContinuousNode: Real-valued for analog signals
- DiscreteNode: Multi-valued for classification

This enables hybrid models combining different domains (photonic, audio, ML).
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Dict, Tuple, Optional, Any
from enum import IntEnum
import logging

# THRML imports
try:
    from thrml import Block, SpinNode
    THRML_AVAILABLE = True
except ImportError:
    THRML_AVAILABLE = False
    logging.warning("THRML not available, heterogeneous nodes disabled")

logger = logging.getLogger(__name__)

from .thrml_compat import ContinuousNodes, DiscreteNodes, SpinNodes


class NodeType(IntEnum):
    """Node type enumeration."""
    SPIN = 0          # Binary {-1, +1}
    CONTINUOUS = 1    # Real-valued
    DISCRETE = 2      # Multi-valued {0, 1, 2, ..., k-1}


class HeterogeneousNodeSpec:
    """
    Specification for a heterogeneous node.
    
    Attributes:
        node_id: Unique node identifier
        node_type: Type of node (SPIN, CONTINUOUS, DISCRETE)
        properties: Type-specific properties
    """
    
    def __init__(
        self,
        node_id: int,
        node_type: NodeType,
        properties: Optional[Dict[str, Any]] = None
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.properties = properties or {}
        
        # Type-specific defaults
        if node_type == NodeType.DISCRETE:
            if 'n_values' not in self.properties:
                self.properties['n_values'] = 4  # Default: 4 discrete values
        elif node_type == NodeType.CONTINUOUS:
            if 'min_value' not in self.properties:
                self.properties['min_value'] = -1.0
            if 'max_value' not in self.properties:
                self.properties['max_value'] = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'node_type': int(self.node_type),
            'properties': self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeterogeneousNodeSpec':
        """Create from dictionary."""
        return cls(
            node_id=data['node_id'],
            node_type=NodeType(data['node_type']),
            properties=data.get('properties', {})
        )


class HeterogeneousTHRMLWrapper:
    """
    THRML wrapper supporting heterogeneous node types.
    
    Manages a graphical model with mixed node types (spin, continuous, discrete).
    Handles type-specific sampling and energy computation.
    """
    
    def __init__(
        self,
        node_specs: List[HeterogeneousNodeSpec],
        edges: List[Tuple[int, int]],
        edge_weights: np.ndarray,
        biases: Optional[np.ndarray] = None,
        beta: float = 1.0
    ):
        """
        Initialize heterogeneous THRML wrapper.
        
        Args:
            node_specs: List of HeterogeneousNodeSpec
            edges: List of (node_i, node_j) tuples
            edge_weights: Array of edge weights
            biases: Optional per-node biases
            beta: Inverse temperature
        """
        if not THRML_AVAILABLE:
            raise ImportError("THRML required for heterogeneous nodes")
        
        self.node_specs = node_specs
        self.n_nodes = len(node_specs)
        
        # Organize nodes by type
        self.node_groups = self._group_nodes_by_type()
        
        # Edges and weights
        self.edges = edges
        self.edge_weights = edge_weights
        
        # Biases
        if biases is None:
            self.biases = np.zeros(self.n_nodes)
        else:
            self.biases = biases
        
        self.beta = beta
        
        # Create THRML node mapping (PyPI exposes SpinNode only)
        self._node_id_to_thrml = self._create_thrml_node_mapping()
        # Preserve ordered list of nodes for convenience
        self.nodes = [self._node_id_to_thrml[spec.node_id] for spec in self.node_specs]
        
        # Group THRML nodes with compatibility wrappers
        self.node_wrappers = self._create_thrml_node_groups()
        self.thrml_node_groups = {
            node_type: tuple(wrapper.nodes)
            for node_type, wrapper in self.node_wrappers.items()
        }
        
        # Build blocks (simple checkerboard for now)
        self.blocks = self._create_simple_blocks()
        
        logger.info(f"[HeteroTHRML] Initialized with {self.n_nodes} nodes")
        for node_type, nodes in self.node_groups.items():
            logger.info(f"  {node_type.name}: {len(nodes)} nodes")
    
    def _group_nodes_by_type(self) -> Dict[NodeType, List[HeterogeneousNodeSpec]]:
        """Group nodes by their type."""
        groups = {
            NodeType.SPIN: [],
            NodeType.CONTINUOUS: [],
            NodeType.DISCRETE: []
        }
        
        for spec in self.node_specs:
            groups[spec.node_type].append(spec)
        
        return groups
    
    def _create_thrml_node_groups(self) -> Dict[NodeType, Any]:
        """Create compatibility wrappers for each node group."""
        wrappers: Dict[NodeType, Any] = {}
        
        if self.node_groups[NodeType.SPIN]:
            node_ids = [spec.node_id for spec in self.node_groups[NodeType.SPIN]]
            nodes = [self._node_id_to_thrml[nid] for nid in node_ids]
            wrappers[NodeType.SPIN] = SpinNodes("spin_nodes", node_ids, nodes)
        
        if self.node_groups[NodeType.CONTINUOUS]:
            node_ids = [spec.node_id for spec in self.node_groups[NodeType.CONTINUOUS]]
            nodes = [self._node_id_to_thrml[nid] for nid in node_ids]
            example_spec = self.node_groups[NodeType.CONTINUOUS][0]
            min_val = example_spec.properties.get('min_value', -1.0)
            max_val = example_spec.properties.get('max_value', 1.0)
            wrappers[NodeType.CONTINUOUS] = ContinuousNodes(
                "continuous_nodes",
                node_ids,
                min_value=min_val,
                max_value=max_val,
                nodes=nodes,
            )
        
        if self.node_groups[NodeType.DISCRETE]:
            node_ids = [spec.node_id for spec in self.node_groups[NodeType.DISCRETE]]
            nodes = [self._node_id_to_thrml[nid] for nid in node_ids]
            example_spec = self.node_groups[NodeType.DISCRETE][0]
            n_values = example_spec.properties.get('n_values', 4)
            wrappers[NodeType.DISCRETE] = DiscreteNodes(
                "discrete_nodes",
                node_ids,
                n_values=n_values,
                nodes=nodes,
            )
        
        return wrappers

    def _create_thrml_node_mapping(self) -> Dict[int, Any]:
        """Create mapping from node_id to THRML node instances."""
        mapping: Dict[int, Any] = {}
        
        # PyPI THRML 0.1.3 only exposes SpinNode, so use it for all types
        for spec in self.node_specs:
            mapping[spec.node_id] = SpinNode()
        
        return mapping
    
    def _create_simple_blocks(self) -> List[Block]:
        """Create simple blocking (checkerboard on node IDs)."""
        all_node_ids = [spec.node_id for spec in self.node_specs]
        
        # Even and odd blocks
        even_ids = [nid for nid in all_node_ids if nid % 2 == 0]
        odd_ids = [nid for nid in all_node_ids if nid % 2 == 1]
        
        blocks = []
        if even_ids:
            # Create block with actual THRML nodes
            even_nodes = [self._node_id_to_thrml[nid] for nid in even_ids]
            if even_nodes:
                blocks.append(Block(tuple(even_nodes)))
        
        if odd_ids:
            odd_nodes = [self._node_id_to_thrml[nid] for nid in odd_ids]
            if odd_nodes:
                blocks.append(Block(tuple(odd_nodes)))
        
        return blocks
    
    def sample_gibbs(
        self,
        n_steps: int,
        temperature: float,
        key: jax.random.PRNGKey,
        initial_states: Optional[Dict[NodeType, np.ndarray]] = None
    ) -> Dict[NodeType, np.ndarray]:
        """
        Sample using Gibbs sampling.
        
        Args:
            n_steps: Number of Gibbs steps
            temperature: Sampling temperature
            key: JAX random key
            initial_states: Optional initial states per type
            
        Returns:
            Dict mapping NodeType to sampled states
        """
        from thrml import SamplingSchedule, sample_states
        from thrml.models import hinton_init
        
        # Create schedule
        schedule = SamplingSchedule(
            n_warmup=n_steps // 2,
            n_samples=1,
            steps_per_sample=2
        )
        
        # Initialize states
        if initial_states is None:
            # Use Hinton initialization (requires an EBM, so we approximate)
            initial_states = self._random_initial_states(key)
        
        # Run sampling (simplified - full implementation would need proper EBM)
        # For now, just return perturbed initial states
        final_states = {}
        for node_type, state in initial_states.items():
            # Add noise
            key, subkey = jax.random.split(key)
            if node_type == NodeType.SPIN:
                # Flip some spins
                flip_prob = 0.1 / temperature
                flips = jax.random.bernoulli(subkey, flip_prob, shape=state.shape)
                final_states[node_type] = jnp.where(flips, -state, state)
            elif node_type == NodeType.CONTINUOUS:
                # Add Gaussian noise
                noise_scale = 0.1 * temperature
                noise = jax.random.normal(subkey, shape=state.shape) * noise_scale
                final_states[node_type] = jnp.clip(state + noise, -1.0, 1.0)
            elif node_type == NodeType.DISCRETE:
                # Randomly change some values
                n_values = self.node_groups[NodeType.DISCRETE][0].properties.get('n_values', 4)
                change_prob = 0.1 / temperature
                changes = jax.random.bernoulli(subkey, change_prob, shape=state.shape)
                new_vals = jax.random.randint(subkey, shape=state.shape, minval=0, maxval=n_values)
                final_states[node_type] = jnp.where(changes, new_vals, state)
        
        return final_states
    
    def _random_initial_states(self, key: jax.random.PRNGKey) -> Dict[NodeType, np.ndarray]:
        """Generate random initial states."""
        states = {}
        
        for node_type, specs in self.node_groups.items():
            if not specs:
                continue
            
            key, subkey = jax.random.split(key)
            n_nodes_of_type = len(specs)
            
            if node_type == NodeType.SPIN:
                states[node_type] = jax.random.choice(
                    subkey, jnp.array([-1.0, 1.0]), shape=(n_nodes_of_type,)
                )
            elif node_type == NodeType.CONTINUOUS:
                min_val = specs[0].properties.get('min_value', -1.0)
                max_val = specs[0].properties.get('max_value', 1.0)
                states[node_type] = jax.random.uniform(
                    subkey, shape=(n_nodes_of_type,), minval=min_val, maxval=max_val
                )
            elif node_type == NodeType.DISCRETE:
                n_values = specs[0].properties.get('n_values', 4)
                states[node_type] = jax.random.randint(
                    subkey, shape=(n_nodes_of_type,), minval=0, maxval=n_values
                )
        
        return states
    
    def compute_energy(
        self,
        states: Dict[NodeType, np.ndarray]
    ) -> float:
        """
        Compute energy for given states.
        
        Args:
            states: Dict mapping NodeType to state arrays
            
        Returns:
            Total energy
        """
        # Flatten all states into single array (with proper ordering)
        full_states = np.zeros(self.n_nodes)
        for spec in self.node_specs:
            type_states = states[spec.node_type]
            type_node_ids = [s.node_id for s in self.node_groups[spec.node_type]]
            idx_in_type = type_node_ids.index(spec.node_id)
            full_states[spec.node_id] = type_states[idx_in_type]
        
        # Compute energy
        # E = -½ Σᵢⱼ Wᵢⱼ sᵢ sⱼ - Σᵢ bᵢ sᵢ
        
        # Interaction term
        E_interaction = 0.0
        for (i, j), w in zip(self.edges, self.edge_weights):
            E_interaction -= w * full_states[i] * full_states[j]
        E_interaction *= 0.5  # Avoid double counting
        
        # Bias term
        E_bias = -np.dot(self.biases, full_states)
        
        return float(E_interaction + E_bias)
    
    def states_to_dict(
        self,
        flat_states: np.ndarray
    ) -> Dict[NodeType, np.ndarray]:
        """
        Convert flat state array to dict by type.
        
        Args:
            flat_states: (n_nodes,) array
            
        Returns:
            Dict mapping NodeType to typed states
        """
        states = {}
        
        for node_type, specs in self.node_groups.items():
            if not specs:
                continue
            
            node_ids = [spec.node_id for spec in specs]
            type_states = np.array([flat_states[nid] for nid in node_ids])
            states[node_type] = type_states
        
        return states
    
    def dict_to_states(
        self,
        state_dict: Dict[NodeType, np.ndarray]
    ) -> np.ndarray:
        """
        Convert dict of states to flat array.
        
        Args:
            state_dict: Dict mapping NodeType to typed states
            
        Returns:
            (n_nodes,) flat array
        """
        flat_states = np.zeros(self.n_nodes)
        
        for node_type, type_states in state_dict.items():
            node_ids = [spec.node_id for spec in self.node_groups[node_type]]
            for idx, nid in enumerate(node_ids):
                flat_states[nid] = type_states[idx]
        
        return flat_states
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize wrapper state."""
        return {
            'node_specs': [spec.to_dict() for spec in self.node_specs],
            'edges': self.edges,
            'edge_weights': self.edge_weights.tolist(),
            'biases': self.biases.tolist(),
            'beta': self.beta,
            'n_nodes': self.n_nodes
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'HeterogeneousTHRMLWrapper':
        """Deserialize wrapper state."""
        node_specs = [HeterogeneousNodeSpec.from_dict(spec_data) 
                     for spec_data in data['node_specs']]
        
        return cls(
            node_specs=node_specs,
            edges=data['edges'],
            edge_weights=np.array(data['edge_weights']),
            biases=np.array(data['biases']),
            beta=data['beta']
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get wrapper information."""
        type_counts = {
            node_type.name: len(specs)
            for node_type, specs in self.node_groups.items()
        }
        
        return {
            'n_nodes': self.n_nodes,
            'type_counts': type_counts,
            'n_edges': len(self.edges),
            'n_blocks': len(self.blocks),
            'beta': self.beta
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_heterogeneous_model(
    node_type_array: np.ndarray,
    weights: np.ndarray,
    biases: Optional[np.ndarray] = None,
    node_properties: Optional[Dict[int, Dict[str, Any]]] = None,
    beta: float = 1.0
) -> HeterogeneousTHRMLWrapper:
    """
    Create heterogeneous THRML model from type array.
    
    Args:
        node_type_array: (n_nodes,) array where each element is a NodeType value
        weights: (n_nodes, n_nodes) weight matrix
        biases: Optional (n_nodes,) bias vector
        node_properties: Optional dict mapping node_id -> properties
        beta: Inverse temperature
        
    Returns:
        HeterogeneousTHRMLWrapper instance
    """
    n_nodes = len(node_type_array)
    
    # Create node specs
    node_specs = []
    for i in range(n_nodes):
        node_type = NodeType(int(node_type_array[i]))
        properties = node_properties.get(i, {}) if node_properties else {}
        node_specs.append(HeterogeneousNodeSpec(i, node_type, properties))
    
    # Extract edges from weight matrix
    edges = []
    edge_weights = []
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            w = weights[i, j]
            if abs(w) > 1e-6:
                edges.append((i, j))
                edge_weights.append(w)
    
    edge_weights = np.array(edge_weights)
    
    # Create wrapper
    wrapper = HeterogeneousTHRMLWrapper(
        node_specs=node_specs,
        edges=edges,
        edge_weights=edge_weights,
        biases=biases,
        beta=beta
    )
    
    return wrapper


def create_domain_specific_model(
    n_photonic: int,
    n_audio: int,
    n_ml: int,
    coupling_strength: float = 0.1,
    beta: float = 1.0
) -> HeterogeneousTHRMLWrapper:
    """
    Create model with domain-specific node groups.
    
    Args:
        n_photonic: Number of photonic (spin) nodes
        n_audio: Number of audio (continuous) nodes
        n_ml: Number of ML (discrete) nodes
        coupling_strength: Inter-domain coupling
        beta: Inverse temperature
        
    Returns:
        HeterogeneousTHRMLWrapper instance with 3 domains
    """
    n_total = n_photonic + n_audio + n_ml
    
    # Create node specs
    node_specs = []
    node_id = 0
    
    # Photonic nodes (spin)
    for _ in range(n_photonic):
        node_specs.append(HeterogeneousNodeSpec(node_id, NodeType.SPIN, {}))
        node_id += 1
    
    # Audio nodes (continuous)
    for _ in range(n_audio):
        node_specs.append(HeterogeneousNodeSpec(
            node_id, NodeType.CONTINUOUS,
            {'min_value': -1.0, 'max_value': 1.0}
        ))
        node_id += 1
    
    # ML nodes (discrete, 4 classes)
    for _ in range(n_ml):
        node_specs.append(HeterogeneousNodeSpec(
            node_id, NodeType.DISCRETE,
            {'n_values': 4}
        ))
        node_id += 1
    
    # Create coupling weights
    weights = np.random.randn(n_total, n_total) * coupling_strength
    weights = (weights + weights.T) / 2  # Symmetric
    np.fill_diagonal(weights, 0)
    
    # Stronger intra-domain coupling
    intra_strength = coupling_strength * 2
    
    # Photonic-photonic
    for i in range(n_photonic):
        for j in range(i + 1, n_photonic):
            weights[i, j] = np.random.randn() * intra_strength
            weights[j, i] = weights[i, j]
    
    # Audio-audio
    for i in range(n_photonic, n_photonic + n_audio):
        for j in range(i + 1, n_photonic + n_audio):
            weights[i, j] = np.random.randn() * intra_strength
            weights[j, i] = weights[i, j]
    
    # ML-ML
    for i in range(n_photonic + n_audio, n_total):
        for j in range(i + 1, n_total):
            weights[i, j] = np.random.randn() * intra_strength
            weights[j, i] = weights[i, j]
    
    # Extract edges
    edges = []
    edge_weights = []
    for i in range(n_total):
        for j in range(i + 1, n_total):
            if abs(weights[i, j]) > 1e-6:
                edges.append((i, j))
                edge_weights.append(weights[i, j])
    
    edge_weights = np.array(edge_weights)
    biases = np.zeros(n_total)
    
    wrapper = HeterogeneousTHRMLWrapper(
        node_specs=node_specs,
        edges=edges,
        edge_weights=edge_weights,
        biases=biases,
        beta=beta
    )
    
    logger.info(f"[DomainModel] Created model: {n_photonic} photonic, "
               f"{n_audio} audio, {n_ml} ML nodes")
    
    return wrapper


if __name__ == '__main__':
    # Demo
    print("Heterogeneous Nodes Demo")
    
    if not THRML_AVAILABLE:
        print("THRML not available, cannot run demo")
    else:
        # Create domain-specific model
        wrapper = create_domain_specific_model(
            n_photonic=4,
            n_audio=4,
            n_ml=4,
            coupling_strength=0.2
        )
        
        info = wrapper.get_info()
        print(f"\nModel info: {info}")
        
        # Sample
        key = jax.random.PRNGKey(0)
        states = wrapper.sample_gibbs(n_steps=50, temperature=1.0, key=key)
        
        print(f"\nSampled states:")
        for node_type, type_states in states.items():
            print(f"  {node_type.name}: {type_states}")
        
        # Compute energy
        energy = wrapper.compute_energy(states)
        print(f"\nEnergy: {energy:.3f}")
        
        print("\nDemo complete!")

