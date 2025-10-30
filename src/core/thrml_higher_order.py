"""
Higher-Order THRML Interactions.

Implements 3-way and 4-way interactions for THRML models, enabling
more complex energy landscapes beyond pairwise Ising interactions.

Energy function with higher-order terms:
E = -Σ w_ij s_i s_j - Σ w_ijk s_i s_j s_k - Σ w_ijkl s_i s_j s_k s_l
"""

from typing import List, Tuple, Dict, Any
import jax
import jax.numpy as jnp
import numpy as np


class HigherOrderInteraction:
    """
    Represents a higher-order interaction between multiple nodes.
    
    Attributes:
        node_ids: Tuple of node indices involved
        strength: Interaction strength coefficient
        order: Interaction order (3 for 3-way, 4 for 4-way)
    """
    
    def __init__(self, node_ids: Tuple[int, ...], strength: float):
        """
        Initialize higher-order interaction.
        
        Args:
            node_ids: Tuple of node indices (3 or 4 nodes)
            strength: Interaction strength
        """
        if len(node_ids) not in [3, 4]:
            raise ValueError("Only 3-way and 4-way interactions supported")
        
        self.node_ids = tuple(sorted(node_ids))  # Sort for consistency
        self.strength = strength
        self.order = len(node_ids)
    
    def compute_energy(self, state: jnp.ndarray) -> float:
        """
        Compute energy contribution from this interaction.
        
        Args:
            state: (n_nodes,) current state
            
        Returns:
            Energy contribution
        """
        # Get states of involved nodes
        node_states = state[list(self.node_ids)]
        
        # Compute product of all states
        product = jnp.prod(node_states)
        
        # Energy: -w * Π s_i
        return -self.strength * product
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'node_ids': list(self.node_ids),
            'strength': float(self.strength),
            'order': self.order
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HigherOrderInteraction':
        """Deserialize from dictionary."""
        return cls(
            node_ids=tuple(data['node_ids']),
            strength=data['strength']
        )


class HigherOrderInteractionManager:
    """
    Manages collection of higher-order interactions for THRML model.
    
    Provides methods to add, remove, and compute energies from
    3-way and 4-way interactions.
    """
    
    def __init__(self):
        """Initialize interaction manager."""
        self.interactions: List[HigherOrderInteraction] = []
        self.three_way_count = 0
        self.four_way_count = 0
    
    def add_three_way_interaction(
        self,
        node_i: int,
        node_j: int,
        node_k: int,
        strength: float
    ) -> int:
        """
        Add 3-way interaction.
        
        Energy contribution: -w_ijk * s_i * s_j * s_k
        
        Args:
            node_i: First node index
            node_j: Second node index
            node_k: Third node index
            strength: Interaction strength w_ijk
            
        Returns:
            Interaction ID (index in list)
        """
        interaction = HigherOrderInteraction(
            node_ids=(node_i, node_j, node_k),
            strength=strength
        )
        self.interactions.append(interaction)
        self.three_way_count += 1
        return len(self.interactions) - 1
    
    def add_four_way_interaction(
        self,
        node_i: int,
        node_j: int,
        node_k: int,
        node_l: int,
        strength: float
    ) -> int:
        """
        Add 4-way interaction.
        
        Energy contribution: -w_ijkl * s_i * s_j * s_k * s_l
        
        Args:
            node_i: First node index
            node_j: Second node index
            node_k: Third node index
            node_l: Fourth node index
            strength: Interaction strength w_ijkl
            
        Returns:
            Interaction ID (index in list)
        """
        interaction = HigherOrderInteraction(
            node_ids=(node_i, node_j, node_k, node_l),
            strength=strength
        )
        self.interactions.append(interaction)
        self.four_way_count += 1
        return len(self.interactions) - 1
    
    def remove_interaction(self, interaction_id: int) -> bool:
        """
        Remove interaction by ID.
        
        Args:
            interaction_id: Index in interactions list
            
        Returns:
            True if removed, False if not found
        """
        if 0 <= interaction_id < len(self.interactions):
            removed = self.interactions.pop(interaction_id)
            if removed.order == 3:
                self.three_way_count -= 1
            else:
                self.four_way_count -= 1
            return True
        return False
    
    def compute_total_energy(self, state: jnp.ndarray) -> float:
        """
        Compute total energy from all higher-order interactions.
        
        Args:
            state: (n_nodes,) current state
            
        Returns:
            Total energy contribution
        """
        if not self.interactions:
            return 0.0
        
        total_energy = 0.0
        for interaction in self.interactions:
            total_energy += interaction.compute_energy(state)
        
        return total_energy
    
    @jax.jit
    def compute_total_energy_jit(self, state: jnp.ndarray) -> float:
        """
        JIT-compiled version of energy computation.
        
        Note: This creates a static computation graph based on
        current interactions. If interactions change, recompile.
        
        Args:
            state: (n_nodes,) current state
            
        Returns:
            Total energy contribution
        """
        if not self.interactions:
            return 0.0
        
        # Vectorized computation
        energies = []
        for interaction in self.interactions:
            node_states = state[jnp.array(interaction.node_ids)]
            product = jnp.prod(node_states)
            energy = -interaction.strength * product
            energies.append(energy)
        
        return jnp.sum(jnp.array(energies))
    
    def get_interaction_groups(self) -> Dict[int, List[Tuple[int, ...]]]:
        """
        Get interaction groups organized by order.
        
        Returns:
            Dictionary mapping order (3 or 4) to list of node tuples
        """
        groups = {3: [], 4: []}
        for interaction in self.interactions:
            groups[interaction.order].append(interaction.node_ids)
        return groups
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'interactions': [i.to_dict() for i in self.interactions],
            'three_way_count': self.three_way_count,
            'four_way_count': self.four_way_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HigherOrderInteractionManager':
        """Deserialize from dictionary."""
        manager = cls()
        manager.interactions = [
            HigherOrderInteraction.from_dict(i)
            for i in data['interactions']
        ]
        manager.three_way_count = data['three_way_count']
        manager.four_way_count = data['four_way_count']
        return manager


def create_ring_three_way_interactions(
    n_nodes: int,
    strength: float = 0.1
) -> HigherOrderInteractionManager:
    """
    Create 3-way interactions in a ring topology.
    
    Each triplet of consecutive nodes has a 3-way interaction:
    (0,1,2), (1,2,3), ..., (n-2,n-1,0), (n-1,0,1)
    
    Args:
        n_nodes: Number of nodes
        strength: Interaction strength
        
    Returns:
        Configured HigherOrderInteractionManager
    """
    manager = HigherOrderInteractionManager()
    
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        k = (i + 2) % n_nodes
        manager.add_three_way_interaction(i, j, k, strength)
    
    return manager


def create_fully_connected_four_way(
    n_nodes: int,
    strength: float = 0.05,
    max_interactions: int = 100
) -> HigherOrderInteractionManager:
    """
    Create 4-way interactions between all combinations of 4 nodes.
    
    Warning: Number of interactions grows as C(n,4) = n!/(4!(n-4)!)
    For n=10, this is 210 interactions.
    
    Args:
        n_nodes: Number of nodes
        strength: Interaction strength
        max_interactions: Maximum number of interactions to create
        
    Returns:
        Configured HigherOrderInteractionManager
    """
    manager = HigherOrderInteractionManager()
    
    count = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            for k in range(j + 1, n_nodes):
                for l in range(k + 1, n_nodes):
                    if count >= max_interactions:
                        return manager
                    manager.add_four_way_interaction(i, j, k, l, strength)
                    count += 1
    
    return manager


def create_local_four_way_interactions(
    n_nodes: int,
    strength: float = 0.1
) -> HigherOrderInteractionManager:
    """
    Create local 4-way interactions between consecutive nodes.
    
    Each group of 4 consecutive nodes has a 4-way interaction:
    (0,1,2,3), (1,2,3,4), ..., (n-4,n-3,n-2,n-1)
    
    Args:
        n_nodes: Number of nodes
        strength: Interaction strength
        
    Returns:
        Configured HigherOrderInteractionManager
    """
    manager = HigherOrderInteractionManager()
    
    for i in range(n_nodes - 3):
        manager.add_four_way_interaction(i, i+1, i+2, i+3, strength)
    
    return manager

