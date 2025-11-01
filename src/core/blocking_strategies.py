"""
Universal Blocking Strategy System for GMCS.

Provides abstract blocking strategies that work across all sampler backends
(THRML, photonic, neuromorphic, quantum). Blocking strategies partition nodes
into independent sets for parallel sampling without conflicts.

Key Concepts:
- Blocking = Graph coloring problem
- Independent sets = No edges within blocks
- Enables parallel Gibbs-style sampling
- Works with any backend that supports parallel updates
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict


# ============================================================================
# Data Structures
# ============================================================================

class ValidationResult(NamedTuple):
    """Result of block validation."""
    valid: bool
    reason: str = ""
    balance_score: float = 1.0
    independence_violations: int = 0
    coverage_missing: int = 0


class StrategyInfo(NamedTuple):
    """Metadata about a blocking strategy."""
    name: str
    description: str
    version: str
    requires_spatial: bool
    requires_connectivity: bool
    computational_cost: str
    memory_usage: str
    optimal_topologies: List[str]


class BenchmarkSample(NamedTuple):
    """Single benchmark measurement."""
    timestamp: float
    wall_time: float
    n_samples: int
    magnetization: float
    energy: float
    strategy: str
    n_chains: int


@dataclass
class Block:
    """A block of nodes that can be sampled independently."""
    nodes: List[int]  # Node IDs
    id: int = 0
    
    def __len__(self):
        return len(self.nodes)
    
    def __repr__(self):
        return f"Block(id={self.id}, size={len(self.nodes)})"


# ============================================================================
# Abstract Base Class
# ============================================================================

class BlockingStrategy(ABC):
    """
    Abstract base class for blocking strategies.
    
    A blocking strategy partitions nodes into independent sets (blocks)
    where no two nodes in the same block share an edge. This enables
    parallel sampling without conflicts.
    
    All strategies must implement:
    - build_blocks(): Partition nodes into blocks
    - validate_blocks(): Verify blocks are valid
    - estimate_performance(): Predict speedup
    
    Strategies declare their requirements:
    - requires_spatial: Needs node positions (e.g., checkerboard)
    - requires_connectivity: Needs adjacency matrix (e.g., graph coloring)
    """
    
    def __init__(self):
        """Initialize strategy."""
        self._version = "1.0.0"
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        pass
    
    @property
    def version(self) -> str:
        """Strategy version."""
        return self._version
    
    @property
    @abstractmethod
    def requires_spatial(self) -> bool:
        """Whether strategy needs node positions."""
        pass
    
    @property
    @abstractmethod
    def requires_connectivity(self) -> bool:
        """Whether strategy needs connectivity matrix."""
        pass
    
    @property
    @abstractmethod
    def computational_cost(self) -> str:
        """Big-O complexity (e.g., 'O(n)', 'O(n²)')."""
        pass
    
    @property
    @abstractmethod
    def memory_usage(self) -> str:
        """Memory usage ('low', 'medium', 'high')."""
        pass
    
    @property
    @abstractmethod
    def optimal_topologies(self) -> List[str]:
        """List of optimal topology types."""
        pass
    
    @abstractmethod
    def build_blocks(
        self,
        n_nodes: int,
        connectivity: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        seed: int = 0
    ) -> List[Block]:
        """
        Partition nodes into independent blocks.
        
        Args:
            n_nodes: Number of nodes to partition
            connectivity: (n_nodes, n_nodes) adjacency matrix (optional)
            positions: (n_nodes, 2) spatial positions (optional)
            seed: Random seed for reproducibility
            
        Returns:
            List of Block objects
        """
        pass
    
    def validate_blocks(
        self,
        blocks: List[Block],
        connectivity: Optional[np.ndarray] = None,
        n_nodes: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate that blocks are independent and complete.
        
        Args:
            blocks: List of blocks to validate
            connectivity: Adjacency matrix (optional, for independence check)
            n_nodes: Total number of nodes (optional, for coverage check)
            
        Returns:
            ValidationResult with diagnostics
        """
        if not blocks:
            return ValidationResult(valid=False, reason="No blocks provided")
        
        # Extract all node IDs
        all_assigned = set()
        for block in blocks:
            all_assigned.update(block.nodes)
        
        # Check 1: Coverage (all nodes assigned)
        if n_nodes is not None:
            expected_nodes = set(range(n_nodes))
            missing = expected_nodes - all_assigned
            if missing:
                return ValidationResult(
                    valid=False,
                    reason=f"Incomplete coverage: {len(missing)} nodes missing",
                    coverage_missing=len(missing)
                )
        
        # Check 2: No duplicates
        total_assigned = sum(len(block.nodes) for block in blocks)
        if total_assigned != len(all_assigned):
            return ValidationResult(
                valid=False,
                reason="Duplicate node assignments"
            )
        
        # Check 3: Independence (no edges within blocks)
        violations = 0
        if connectivity is not None:
            for block in blocks:
                node_ids = block.nodes
                for i, node_i in enumerate(node_ids):
                    for node_j in node_ids[i+1:]:
                        if connectivity[node_i, node_j] > 0:
                            violations += 1
            
            if violations > 0:
                return ValidationResult(
                    valid=False,
                    reason=f"Independence violated: {violations} edges within blocks",
                    independence_violations=violations
                )
        
        # Check 4: Balance (similar block sizes)
        sizes = [len(block.nodes) for block in blocks]
        if len(sizes) > 1:
            variance = np.var(sizes)
            mean_size = np.mean(sizes)
            balance_score = 1.0 - (variance / (mean_size ** 2)) if mean_size > 0 else 0.0
        else:
            balance_score = 1.0
        
        return ValidationResult(
            valid=True,
            reason="All checks passed",
            balance_score=max(0.0, min(1.0, balance_score))
        )
    
    def estimate_performance(
        self,
        n_nodes: int,
        connectivity: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Estimate performance metrics for this strategy.
        
        Args:
            n_nodes: Number of nodes
            connectivity: Adjacency matrix (optional)
            
        Returns:
            Dict with estimated speedup, memory_mb, etc.
        """
        # Default estimates (subclasses can override)
        n_blocks = 2  # Most strategies use 2-coloring
        
        return {
            'estimated_speedup': min(n_blocks, 8),  # Cap at 8x
            'estimated_memory_mb': n_nodes * 0.001,  # ~1KB per 1000 nodes
            'n_blocks': n_blocks,
            'parallelism': n_blocks
        }
    
    def get_info(self) -> StrategyInfo:
        """Get strategy metadata."""
        return StrategyInfo(
            name=self.name,
            description=self.description,
            version=self.version,
            requires_spatial=self.requires_spatial,
            requires_connectivity=self.requires_connectivity,
            computational_cost=self.computational_cost,
            memory_usage=self.memory_usage,
            optimal_topologies=self.optimal_topologies
        )


# ============================================================================
# Concrete Strategy Implementations
# ============================================================================

class CheckerboardStrategy(BlockingStrategy):
    """
    2D checkerboard (even/odd) coloring strategy.
    
    Partitions nodes based on (i+j) % 2 where (i,j) are spatial coordinates.
    This is optimal for 2D grids as it respects spatial locality and ensures
    no two adjacent nodes are in the same block.
    
    Performance: 1.4-2.2× better ESS/sec than random for 2D grids.
    
    Optimal for:
    - THRML Ising grids
    - Photonic arrays
    - Memristor crossbars
    - Any 2D spatial lattice
    """
    
    @property
    def name(self) -> str:
        return "checkerboard"
    
    @property
    def description(self) -> str:
        return "2D checkerboard coloring (i+j)%2 - optimal for spatial grids"
    
    @property
    def requires_spatial(self) -> bool:
        return True
    
    @property
    def requires_connectivity(self) -> bool:
        return False
    
    @property
    def computational_cost(self) -> str:
        return "O(n)"
    
    @property
    def memory_usage(self) -> str:
        return "low"
    
    @property
    def optimal_topologies(self) -> List[str]:
        return ["grid_2d", "lattice_2d", "spatial_2d"]
    
    def build_blocks(
        self,
        n_nodes: int,
        connectivity: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        seed: int = 0
    ) -> List[Block]:
        """Build checkerboard blocks from spatial positions."""
        if positions is None:
            raise ValueError("CheckerboardStrategy requires node positions")
        
        if positions.shape[0] != n_nodes:
            raise ValueError(f"Position array size {positions.shape[0]} != n_nodes {n_nodes}")
        
        # Infer grid dimensions
        # Assume positions are in grid coordinates [0, W) x [0, H)
        grid_w = int(np.max(positions[:, 0])) + 1
        grid_h = int(np.max(positions[:, 1])) + 1
        
        # Partition by (i+j) % 2
        evens = []
        odds = []
        
        for node_id in range(n_nodes):
            i = int(positions[node_id, 0])
            j = int(positions[node_id, 1])
            
            if (i + j) % 2 == 0:
                evens.append(node_id)
            else:
                odds.append(node_id)
        
        return [
            Block(nodes=evens, id=0),
            Block(nodes=odds, id=1)
        ]


class RandomStrategy(BlockingStrategy):
    """
    Random shuffle and split strategy.
    
    Randomly shuffles nodes and splits into two halves. This is a universal
    fallback that works with any topology but doesn't respect spatial locality.
    
    Performance: Faster throughput but worse mixing than checkerboard for grids.
    
    Optimal for:
    - General graphs (no spatial structure)
    - Quick exploration
    - Fallback when other strategies fail
    """
    
    @property
    def name(self) -> str:
        return "random"
    
    @property
    def description(self) -> str:
        return "Random shuffle and split - universal fallback"
    
    @property
    def requires_spatial(self) -> bool:
        return False
    
    @property
    def requires_connectivity(self) -> bool:
        return False
    
    @property
    def computational_cost(self) -> str:
        return "O(n log n)"
    
    @property
    def memory_usage(self) -> str:
        return "low"
    
    @property
    def optimal_topologies(self) -> List[str]:
        return ["general", "arbitrary", "fallback"]
    
    def build_blocks(
        self,
        n_nodes: int,
        connectivity: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        seed: int = 0
    ) -> List[Block]:
        """Build random blocks by shuffling and splitting."""
        rng = np.random.default_rng(seed)
        
        # Shuffle node IDs
        node_ids = np.arange(n_nodes)
        rng.shuffle(node_ids)
        
        # Split in half
        half = n_nodes // 2
        
        return [
            Block(nodes=node_ids[:half].tolist(), id=0),
            Block(nodes=node_ids[half:].tolist(), id=1)
        ]


class StripesStrategy(BlockingStrategy):
    """
    Vertical/horizontal stripe coloring strategy.
    
    Partitions nodes by column (or row) index modulo 2. Good for 1D chains
    or when spatial structure is primarily in one dimension.
    
    Optimal for:
    - 1D chains
    - Optical waveguides
    - Columnar structures
    """
    
    @property
    def name(self) -> str:
        return "stripes"
    
    @property
    def description(self) -> str:
        return "Vertical stripe coloring - good for 1D chains"
    
    @property
    def requires_spatial(self) -> bool:
        return True
    
    @property
    def requires_connectivity(self) -> bool:
        return False
    
    @property
    def computational_cost(self) -> str:
        return "O(n)"
    
    @property
    def memory_usage(self) -> str:
        return "low"
    
    @property
    def optimal_topologies(self) -> List[str]:
        return ["chain_1d", "stripe", "columnar"]
    
    def build_blocks(
        self,
        n_nodes: int,
        connectivity: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        seed: int = 0
    ) -> List[Block]:
        """Build stripe blocks from spatial positions."""
        if positions is None:
            raise ValueError("StripesStrategy requires node positions")
        
        # Partition by column (x-coordinate) modulo 2
        evens = []
        odds = []
        
        for node_id in range(n_nodes):
            col = int(positions[node_id, 0])
            
            if col % 2 == 0:
                evens.append(node_id)
            else:
                odds.append(node_id)
        
        return [
            Block(nodes=evens, id=0),
            Block(nodes=odds, id=1)
        ]


class SupercellStrategy(BlockingStrategy):
    """
    2×2 supercell coloring strategy.
    
    Partitions nodes based on ((i//2) + (j//2)) % 2. This creates larger
    blocks than checkerboard, which can be beneficial for certain hardware.
    
    Optimal for:
    - Alternative 2D coloring
    - Larger block sizes needed
    - Testing vs checkerboard
    """
    
    @property
    def name(self) -> str:
        return "supercell"
    
    @property
    def description(self) -> str:
        return "2×2 supercell coloring - alternative 2D pattern"
    
    @property
    def requires_spatial(self) -> bool:
        return True
    
    @property
    def requires_connectivity(self) -> bool:
        return False
    
    @property
    def computational_cost(self) -> str:
        return "O(n)"
    
    @property
    def memory_usage(self) -> str:
        return "low"
    
    @property
    def optimal_topologies(self) -> List[str]:
        return ["grid_2d", "lattice_2d"]
    
    def build_blocks(
        self,
        n_nodes: int,
        connectivity: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        seed: int = 0
    ) -> List[Block]:
        """Build supercell blocks from spatial positions."""
        if positions is None:
            raise ValueError("SupercellStrategy requires node positions")
        
        # Partition by ((i//2) + (j//2)) % 2
        evens = []
        odds = []
        
        for node_id in range(n_nodes):
            i = int(positions[node_id, 0])
            j = int(positions[node_id, 1])
            
            cell = ((i // 2) + (j // 2)) % 2
            
            if cell == 0:
                evens.append(node_id)
            else:
                odds.append(node_id)
        
        return [
            Block(nodes=evens, id=0),
            Block(nodes=odds, id=1)
        ]


class GraphColoringStrategy(BlockingStrategy):
    """
    General graph 2-coloring strategy using greedy algorithm.
    
    Works with arbitrary connectivity graphs. Uses greedy coloring to partition
    nodes into two independent sets. Slower than spatial strategies but universal.
    
    Optimal for:
    - Arbitrary connectivity
    - General graphs
    - When spatial structure unavailable
    """
    
    @property
    def name(self) -> str:
        return "graph-coloring"
    
    @property
    def description(self) -> str:
        return "Greedy graph 2-coloring - works with arbitrary connectivity"
    
    @property
    def requires_spatial(self) -> bool:
        return False
    
    @property
    def requires_connectivity(self) -> bool:
        return True
    
    @property
    def computational_cost(self) -> str:
        return "O(n²)"
    
    @property
    def memory_usage(self) -> str:
        return "medium"
    
    @property
    def optimal_topologies(self) -> List[str]:
        return ["general", "arbitrary", "graph"]
    
    def build_blocks(
        self,
        n_nodes: int,
        connectivity: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        seed: int = 0
    ) -> List[Block]:
        """Build blocks using greedy graph coloring."""
        if connectivity is None:
            raise ValueError("GraphColoringStrategy requires connectivity matrix")
        
        # Greedy 2-coloring
        colors = np.full(n_nodes, -1, dtype=np.int32)
        
        for node in range(n_nodes):
            # Find neighbors' colors
            neighbor_colors = set()
            for neighbor in range(n_nodes):
                if connectivity[node, neighbor] > 0 and colors[neighbor] != -1:
                    neighbor_colors.add(colors[neighbor])
            
            # Assign first available color (0 or 1)
            if 0 not in neighbor_colors:
                colors[node] = 0
            elif 1 not in neighbor_colors:
                colors[node] = 1
            else:
                # If both colors used by neighbors, graph not 2-colorable
                # Fall back to random assignment
                rng = np.random.default_rng(seed + node)
                colors[node] = rng.integers(0, 2)
        
        # Partition by color
        color_0 = np.where(colors == 0)[0].tolist()
        color_1 = np.where(colors == 1)[0].tolist()
        
        return [
            Block(nodes=color_0, id=0),
            Block(nodes=color_1, id=1)
        ]


# ============================================================================
# Strategy Registry
# ============================================================================

class StrategyRegistry:
    """
    Global registry for blocking strategies.
    
    Manages built-in and plugin strategies. Provides discovery,
    registration, and retrieval.
    """
    
    def __init__(self):
        """Initialize registry with built-in strategies."""
        self._strategies: Dict[str, BlockingStrategy] = {}
        self._register_builtin_strategies()
    
    def _register_builtin_strategies(self):
        """Register all built-in strategies."""
        self.register("checkerboard", CheckerboardStrategy())
        self.register("random", RandomStrategy())
        self.register("stripes", StripesStrategy())
        self.register("supercell", SupercellStrategy())
        self.register("graph-coloring", GraphColoringStrategy())
    
    def register(self, name: str, strategy: BlockingStrategy):
        """
        Register a strategy.
        
        Args:
            name: Strategy name (must be unique)
            strategy: Strategy instance
        """
        if name in self._strategies:
            print(f"Warning: Overwriting existing strategy '{name}'")
        
        self._strategies[name] = strategy
    
    def get(self, name: str) -> BlockingStrategy:
        """
        Get strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy instance or None if not found
        """
        return self._strategies.get(name, None)
    
    def list_strategies(self, filter_by_topology: Optional[str] = None) -> List[StrategyInfo]:
        """
        List all registered strategies.
        
        Args:
            filter_by_topology: Optional topology filter (e.g., "grid_2d")
            
        Returns:
            List of StrategyInfo objects
        """
        infos = []
        
        for strategy in self._strategies.values():
            info = strategy.get_info()
            
            # Apply filter if specified
            if filter_by_topology is not None:
                if filter_by_topology not in info.optimal_topologies:
                    continue
            
            infos.append(info)
        
        return infos
    
    def load_plugin_strategy(self, path: str):
        """
        Load a custom strategy plugin.
        
        Args:
            path: Path to plugin module
            
        Note: Plugin must define a class inheriting from BlockingStrategy
        and a `register_strategy()` function.
        """
        # TODO: Implement plugin loading in Phase 1.3
        raise NotImplementedError("Plugin loading not yet implemented")


# Global registry instance
_global_registry = StrategyRegistry()


def register_strategy(name: str, strategy: BlockingStrategy):
    """Register a strategy in the global registry."""
    _global_registry.register(name, strategy)


def get_strategy(name: str) -> BlockingStrategy:
    """Get a strategy from the global registry."""
    return _global_registry.get(name)


def list_strategies(filter_by_topology: Optional[str] = None) -> List[StrategyInfo]:
    """List all registered strategies."""
    return _global_registry.list_strategies(filter_by_topology)


# ============================================================================
# Auto-register all built-in strategies
# ============================================================================

# Register checkerboard strategy
_global_registry.register("checkerboard", CheckerboardStrategy())

# Register random strategy
_global_registry.register("random", RandomStrategy())

# Register stripes strategy
_global_registry.register("stripes", StripesStrategy())

# Register supercell strategy
_global_registry.register("supercell", SupercellStrategy())

# Register graph coloring strategy
_global_registry.register("graph-coloring", GraphColoringStrategy())

print("[Blocking] Registered 5 blocking strategies: checkerboard, random, stripes, supercell, graph-coloring")
