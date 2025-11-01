"""
Storage Nodes for GMCS Node Graph.

Provides data storage and retrieval capabilities:
- TopKStorage: Keep best K solutions
- ReplayBufferNode: Experience replay for RL
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import deque
import heapq


# ============================================================================
# Top-K Storage Node
# ============================================================================

@dataclass
class TopKStorageConfig:
    """Configuration for Top-K storage."""
    top_k: int = 10
    format: str = "json"  # "json", "sdf", "source_code", "dict"
    metric_key: str = "score"  # Which key to use for ranking
    mode: str = "max"  # "max" or "min"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")


class TopKStorage:
    """
    Store and rank the top-K solutions/candidates.
    
    Maintains a heap of best solutions based on a metric.
    Used for architecture search, molecular design, etc.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'topk_storage')
        else:
            self.node_id = node_id_or_config
            # Map 'k' parameter to 'top_k' for config
            if 'k' in kwargs and 'top_k' not in kwargs:
                kwargs['top_k'] = kwargs.pop('k')
            config = kwargs
        
        self.config = TopKStorageConfig(**config)
        self.storage = []  # Min-heap for top-K (negative scores for max mode)
        self.all_candidates = []  # Keep all for Pareto front
        self.count = 0
        
        print(f"[TopKStorage] Initialized, K={self.config.top_k}, mode={self.config.mode}")
    
    def process(self, candidates: Any = None, **inputs) -> Dict[str, Any]:
        """
        Add candidate(s) to storage.
        
        Args:
            candidates: Single candidate or list of candidates
            **inputs: Can include individual candidate fields
            
        Returns:
            Dictionary with storage status and best candidates
        """
        # Handle various input formats
        if candidates is not None:
            if isinstance(candidates, list):
                for cand in candidates:
                    self._add_candidate(cand)
            else:
                self._add_candidate(candidates)
        elif inputs:
            # Treat inputs as a single candidate
            self._add_candidate(inputs)
        
        # Get top-K
        top_k = self._get_top_k()
        
        return {
            'top_k': top_k,
            'best': top_k[0] if top_k else None,
            'count': self.count,
            'pareto_front': self._compute_pareto_front() if len(self.all_candidates) > 0 else []
        }
    
    def _add_candidate(self, candidate: Any):
        """Add a candidate to storage."""
        self.count += 1
        
        # Extract metric
        if isinstance(candidate, dict):
            metric = candidate.get(self.config.metric_key, 0.0)
        else:
            metric = float(candidate)
            candidate = {'value': candidate, self.config.metric_key: metric}
        
        # Convert to float
        try:
            metric = float(metric)
        except:
            print(f"[TopKStorage WARNING] Could not convert metric to float: {metric}")
            metric = 0.0
        
        # Add to all candidates (for Pareto front)
        candidate_copy = candidate.copy() if isinstance(candidate, dict) else {'value': candidate}
        candidate_copy['_id'] = self.count
        self.all_candidates.append(candidate_copy)
        
        # Maintain top-K heap
        # For max mode: keep K largest, so min-heap root = smallest of K largest (threshold)
        # For min mode: keep K smallest, so we negate to use min-heap for largest negated
        if self.config.mode == "max":
            score = metric  # No negation for max mode
        else:
            score = -metric  # Negate for min mode
        
        if len(self.storage) < self.config.top_k:
            heapq.heappush(self.storage, (score, self.count, candidate))
        else:
            # Only add if better than worst in heap
            # For max mode: if new score > root (worst of K largest), replace
            # For min mode: if new negated score < root (best of K most negative), replace
            if (self.config.mode == "max" and score > self.storage[0][0]) or \
               (self.config.mode == "min" and score < self.storage[0][0]):
                heapq.heapreplace(self.storage, (score, self.count, candidate))
    
    def _get_top_k(self) -> List[Any]:
        """Get top-K candidates in sorted order (best first)."""
        # The heap stores (score, count, candidate) tuples
        # For max mode: sort descending (best first = largest scores first)
        # For min mode: sort ascending (best first = smallest negated = largest original)
        if self.config.mode == "max":
            sorted_storage = sorted(self.storage, key=lambda x: x[0], reverse=True)
        else:
            sorted_storage = sorted(self.storage, key=lambda x: x[0])
        return [item[2] for item in sorted_storage]
    
    def _compute_pareto_front(self) -> List[Any]:
        """
        Compute Pareto front for multi-objective optimization.
        
        Currently simplified - returns top candidates.
        Full implementation would check domination relationships.
        """
        # Simplified: just return top-K
        return self._get_top_k()[:min(20, len(self.storage))]
    
    def add_item(self, score: float, data: Any):
        """
        Add an item with a given score to the storage.
        
        Args:
            score: Score/metric value for ranking
            data: Associated data (can be dict or any type)
        """
        # Create candidate from score and data
        if isinstance(data, dict):
            candidate = {self.config.metric_key: score, **data}
        else:
            candidate = {self.config.metric_key: score, 'data': data}
        
        # Use process method to add (which calls _add_candidate internally)
        self.process(candidate)
    
    def get_top_k(self) -> List[Any]:
        """
        Get top-K items (public method for tests).
        
        Returns:
            List of top-K candidates sorted by score
        """
        return self._get_top_k()


# ============================================================================
# Experience Replay Buffer Node
# ============================================================================

@dataclass
class ReplayBufferConfig:
    """Configuration for experience replay buffer."""
    capacity: int = 100000
    batch_size: int = 64
    prioritization: str = "uniform"  # "uniform", "diversity", "priority"


class ReplayBufferNode:
    """
    Experience replay buffer for reinforcement learning.
    
    Stores (state, action, reward, next_state, done) transitions
    and provides batched sampling for training.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'replay_buffer')
        else:
            self.node_id = node_id_or_config
            # Map 'buffer_size' to 'capacity' if needed
            if 'buffer_size' in kwargs and 'capacity' not in kwargs:
                kwargs['capacity'] = kwargs.pop('buffer_size')
            config = kwargs
        
        self.config = ReplayBufferConfig(**config)
        self.buffer = deque(maxlen=self.config.capacity)
        self.priorities = deque(maxlen=self.config.capacity)
        
        print(f"[ReplayBuffer] Initialized, capacity={self.config.capacity}, prioritization={self.config.prioritization}")
    
    def process(self, add: Optional[Dict[str, Any]] = None, sample: bool = False, **inputs) -> Dict[str, Any]:
        """
        Add experience or sample batch.
        
        Args:
            add: Experience dict with keys: state, action, reward, next_state, done
            sample: Whether to sample a batch
            **inputs: Can include individual transition fields
            
        Returns:
            Dictionary with batch (if sampling) or status
        """
        # Add experience
        if add is not None:
            self._add_experience(add)
        elif inputs and not sample:
            # Treat inputs as experience
            self._add_experience(inputs)
        
        # Sample batch
        if sample:
            batch = self._sample_batch()
            return {
                'batch': batch,
                'buffer_size': len(self.buffer)
            }
        else:
            return {
                'buffer_size': len(self.buffer),
                'is_ready': len(self.buffer) >= self.config.batch_size
            }
    
    def _add_experience(self, experience: Dict[str, Any]):
        """Add an experience to the buffer."""
        # Validate experience
        required_keys = ['state', 'action', 'reward']
        if not all(k in experience for k in required_keys):
            print(f"[ReplayBuffer WARNING] Incomplete experience: {experience.keys()}")
            return
        
        # Add to buffer
        self.buffer.append(experience)
        
        # Compute priority (simplified)
        if self.config.prioritization == "priority":
            # Use TD error or reward magnitude as priority
            priority = abs(experience.get('reward', 0.0))
        elif self.config.prioritization == "diversity":
            # Use distance to existing states (simplified)
            priority = 1.0  # Placeholder
        else:
            priority = 1.0
        
        self.priorities.append(priority)
    
    def _sample_batch(self) -> Dict[str, np.ndarray]:
        """
        Sample a batch of experiences.
        
        Returns:
            Dictionary with arrays of states, actions, rewards, next_states, dones
        """
        if len(self.buffer) < self.config.batch_size:
            return {}
        
        # Sample indices
        if self.config.prioritization == "uniform":
            indices = np.random.choice(len(self.buffer), size=self.config.batch_size, replace=False)
        else:
            # Prioritized sampling
            priorities = np.array(list(self.priorities))
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), size=self.config.batch_size, replace=False, p=probs)
        
        # Collect batch
        batch = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        for idx in indices:
            exp = self.buffer[idx]
            batch['states'].append(exp['state'])
            batch['actions'].append(exp['action'])
            batch['rewards'].append(exp['reward'])
            batch['next_states'].append(exp.get('next_state', exp['state']))
            batch['dones'].append(exp.get('done', False))
        
        # Convert to arrays
        batch['states'] = np.array(batch['states'])
        batch['actions'] = np.array(batch['actions'])
        batch['rewards'] = np.array(batch['rewards'])
        batch['next_states'] = np.array(batch['next_states'])
        batch['dones'] = np.array(batch['dones'])
        
        return batch
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities.clear()


# ============================================================================
# Node Factory Registration Helper
# ============================================================================

def create_storage_node(node_name: str, config: Dict[str, Any]) -> Any:
    """
    Factory function to create storage nodes.
    
    Args:
        node_name: Name of the storage node type
        config: Node configuration
        
    Returns:
        Storage node instance
    """
    if 'Replay' in node_name or 'Buffer' in node_name:
        return ReplayBufferNode(config)
    else:
        # Default to Top-K storage
        return TopKStorage(config)


# ============================================================================
# Alias for Test Compatibility
# ============================================================================

# Tests expect this exact class name
TopKStorageNode = TopKStorage
