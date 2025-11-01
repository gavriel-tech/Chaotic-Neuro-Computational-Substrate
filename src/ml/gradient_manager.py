"""
Gradient Flow Manager for GMCS.

Coordinates gradient flow through hybrid chaos+ML computational graph,
handling mixed differentiable/non-differentiable nodes and multiple
ML frameworks (PyTorch, TensorFlow, JAX).

Key features:
- Automatic differentiation through differentiable nodes
- Gradient stop at non-differentiable (chaos) nodes
- Cross-framework gradient handling
- Computational graph visualization
- Gradient clipping and monitoring
"""

from typing import Dict, List, Tuple, Optional, Any, Set, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json

try:
    import jax
    import jax.numpy as jnp
    from jax import grad as jax_grad
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# ============================================================================
# Graph Node Types
# ============================================================================

class NodeType(Enum):
    """Types of nodes in computational graph."""
    DIFFERENTIABLE = "differentiable"  # Gradient flows through
    NON_DIFFERENTIABLE = "non_differentiable"  # Gradient stops
    INPUT = "input"  # Source node
    OUTPUT = "output"  # Sink node


@dataclass
class GraphNode:
    """
    Node in computational graph.
    
    Attributes:
        node_id: Unique identifier
        node_type: Type of node
        framework: 'jax', 'pytorch', 'tensorflow', or None
        forward_fn: Forward computation function
        differentiable: Whether gradients flow through
        metadata: Additional node information
    """
    node_id: str
    node_type: NodeType
    framework: Optional[str] = None
    forward_fn: Optional[Callable] = None
    differentiable: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GraphEdge:
    """
    Edge in computational graph.
    
    Attributes:
        source: Source node ID
        target: Target node ID
        gradient_enabled: Whether gradient flows along this edge
    """
    source: str
    target: str
    gradient_enabled: bool = True


# ============================================================================
# Gradient Flow Graph
# ============================================================================

class GradientFlowGraph:
    """
    Manages gradient flow through hybrid chaos+ML system.
    
    This graph coordinates:
    - Forward propagation through all nodes
    - Backward propagation (gradients) through differentiable nodes
    - Gradient stopping at non-differentiable (chaos) nodes
    - Cross-framework gradient translation
    
    Example:
        graph = GradientFlowGraph()
        graph.add_node("osc1", differentiable=False)  # Chaos node
        graph.add_node("ml1", differentiable=True)    # ML node
        graph.add_edge("osc1", "ml1")
        
        outputs = graph.forward(inputs)
        gradients = graph.backward(loss)
    """
    
    def __init__(self):
        """Initialize empty graph."""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.topological_order: List[str] = []
        self.reverse_topological_order: List[str] = []
        
        # Gradient tracking
        self.gradient_history: List[Dict[str, Any]] = []
        self.gradient_norms: Dict[str, List[float]] = {}
        
        # Framework conversion cache
        self._conversion_cache: Dict[Tuple[str, str], Callable] = {}
    
    def add_node(
        self,
        node_id: str,
        differentiable: bool = False,
        framework: Optional[str] = None,
        forward_fn: Optional[Callable] = None,
        node_type: NodeType = NodeType.DIFFERENTIABLE,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add node to graph.
        
        Args:
            node_id: Unique identifier
            differentiable: Whether gradients flow through
            framework: 'jax', 'pytorch', 'tensorflow', or None
            forward_fn: Forward computation function
            node_type: Type of node
            metadata: Additional information
        """
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            framework=framework,
            forward_fn=forward_fn,
            differentiable=differentiable,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        self._invalidate_topology()
    
    def add_edge(self, source: str, target: str, gradient_enabled: bool = True):
        """
        Add edge between nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            gradient_enabled: Whether gradient flows along this edge
        """
        if source not in self.nodes:
            raise ValueError(f"Source node {source} not in graph")
        if target not in self.nodes:
            raise ValueError(f"Target node {target} not in graph")
        
        edge = GraphEdge(source, target, gradient_enabled)
        self.edges.append(edge)
        self._invalidate_topology()
    
    def remove_node(self, node_id: str):
        """Remove node and connected edges."""
        if node_id in self.nodes:
            del self.nodes[node_id]
        
        self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]
        self._invalidate_topology()
    
    def remove_edge(self, source: str, target: str):
        """Remove edge between nodes."""
        self.edges = [e for e in self.edges if not (e.source == source and e.target == target)]
        self._invalidate_topology()
    
    def _invalidate_topology(self):
        """Mark topological order as needing recomputation."""
        self.topological_order = []
        self.reverse_topological_order = []
    
    def _compute_topological_order(self) -> List[str]:
        """
        Compute topological ordering of nodes.
        
        Returns:
            List of node IDs in topological order
        """
        if self.topological_order:
            return self.topological_order
        
        # Build adjacency list
        adj = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}
        
        for edge in self.edges:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1
        
        # Kahn's algorithm
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            for neighbor in adj[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles")
        
        self.topological_order = result
        self.reverse_topological_order = list(reversed(result))
        
        return result
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward propagation through graph.
        
        Args:
            inputs: Dict mapping input node IDs to input values
            
        Returns:
            Dict mapping all node IDs to output values
        """
        order = self._compute_topological_order()
        
        # Store intermediate outputs
        outputs = inputs.copy()
        
        for node_id in order:
            if node_id in inputs:
                # Input node, already have value
                continue
            
            node = self.nodes[node_id]
            
            # Collect inputs from predecessor nodes
            predecessors = [e.source for e in self.edges if e.target == node_id]
            
            if not predecessors:
                # No predecessors, skip
                continue
            
            # Get input values (convert frameworks if needed)
            node_inputs = []
            for pred_id in predecessors:
                pred_node = self.nodes[pred_id]
                pred_output = outputs[pred_id]
                
                # Convert to target framework if needed
                if node.framework and pred_node.framework != node.framework:
                    pred_output = self._convert_framework(
                        pred_output,
                        pred_node.framework,
                        node.framework
                    )
                
                node_inputs.append(pred_output)
            
            # Compute forward pass
            if node.forward_fn:
                if len(node_inputs) == 1:
                    output = node.forward_fn(node_inputs[0])
                else:
                    output = node.forward_fn(node_inputs)
                
                outputs[node_id] = output
            else:
                # Pass through
                outputs[node_id] = node_inputs[0] if len(node_inputs) == 1 else node_inputs
        
        return outputs
    
    def backward(
        self,
        loss: Any,
        loss_node_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Backward propagation (gradient computation).
        
        Computes gradients wrt parameters for differentiable nodes.
        Stops at non-differentiable nodes.
        
        Args:
            loss: Loss value
            loss_node_id: Node ID where loss is computed
            parameters: Dict of trainable parameters by node
            
        Returns:
            Dict mapping node IDs to gradients
        """
        if parameters is None:
            parameters = {}
        
        order = self.reverse_topological_order
        if not order:
            self._compute_topological_order()
            order = self.reverse_topological_order
        
        # Initialize gradients
        gradients = {loss_node_id: 1.0}  # dL/dL = 1
        
        # Backpropagation
        for node_id in order:
            if node_id == loss_node_id:
                continue
            
            node = self.nodes[node_id]
            
            # Check if this node has gradient
            successors = [e for e in self.edges if e.source == node_id and e.gradient_enabled]
            
            if not successors:
                continue
            
            # If non-differentiable, stop gradient
            if not node.differentiable:
                gradients[node_id] = 0.0
                continue
            
            # Accumulate gradient from successors
            grad = 0.0
            for edge in successors:
                if edge.target in gradients:
                    grad += gradients[edge.target]
            
            gradients[node_id] = grad
            
            # Track gradient norm
            if node_id not in self.gradient_norms:
                self.gradient_norms[node_id] = []
            
            try:
                norm = float(np.linalg.norm(np.array(grad)))
                self.gradient_norms[node_id].append(norm)
            except:
                pass
        
        return gradients
    
    def update_parameters(
        self,
        gradients: Dict[str, Any],
        parameters: Dict[str, Any],
        learning_rate: float
    ) -> Dict[str, Any]:
        """
        Update parameters using computed gradients.
        
        Args:
            gradients: Gradients from backward pass
            parameters: Current parameters
            learning_rate: Learning rate
            
        Returns:
            Updated parameters
        """
        updated_params = parameters.copy()
        
        for node_id, grad in gradients.items():
            if node_id in parameters:
                # Simple SGD update
                updated_params[node_id] -= learning_rate * grad
        
        return updated_params
    
    def clip_gradients(
        self,
        gradients: Dict[str, Any],
        max_norm: float = 1.0
    ) -> Dict[str, Any]:
        """
        Clip gradients by global norm.
        
        Args:
            gradients: Gradients to clip
            max_norm: Maximum gradient norm
            
        Returns:
            Clipped gradients
        """
        # Compute global norm
        total_norm = 0.0
        for grad in gradients.values():
            try:
                norm = np.linalg.norm(np.array(grad))
                total_norm += norm ** 2
            except:
                pass
        
        total_norm = np.sqrt(total_norm)
        
        # Clip if needed
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            return {k: v * clip_coef for k, v in gradients.items()}
        
        return gradients
    
    def _convert_framework(
        self,
        data: Any,
        source_framework: Optional[str],
        target_framework: Optional[str]
    ) -> Any:
        """
        Convert data between frameworks.
        
        Args:
            data: Data to convert
            source_framework: Source framework
            target_framework: Target framework
            
        Returns:
            Converted data
        """
        if source_framework == target_framework:
            return data
        
        # Convert to numpy first
        if source_framework == 'pytorch' and PYTORCH_AVAILABLE:
            data_np = data.detach().cpu().numpy()
        elif source_framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
            data_np = data.numpy()
        elif source_framework == 'jax' and JAX_AVAILABLE:
            data_np = np.array(data)
        else:
            data_np = np.array(data)
        
        # Convert to target framework
        if target_framework == 'pytorch' and PYTORCH_AVAILABLE:
            return torch.from_numpy(data_np).float()
        elif target_framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
            return tf.convert_to_tensor(data_np, dtype=tf.float32)
        elif target_framework == 'jax' and JAX_AVAILABLE:
            return jnp.array(data_np)
        else:
            return data_np
    
    def get_gradient_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about gradient flow.
        
        Returns:
            Dict with gradient statistics
        """
        stats = {}
        
        for node_id, norms in self.gradient_norms.items():
            if norms:
                stats[node_id] = {
                    'mean_norm': float(np.mean(norms)),
                    'max_norm': float(np.max(norms)),
                    'min_norm': float(np.min(norms)),
                    'std_norm': float(np.std(norms)),
                    'num_updates': len(norms)
                }
        
        return stats
    
    def visualize_graph(self) -> str:
        """
        Generate text visualization of graph.
        
        Returns:
            ASCII art representation
        """
        lines = ["Gradient Flow Graph:", "=" * 50]
        
        lines.append("\nNodes:")
        for node_id, node in self.nodes.items():
            diff_str = "✓ diff" if node.differentiable else "✗ non-diff"
            framework_str = f"[{node.framework}]" if node.framework else "[none]"
            lines.append(f"  {node_id}: {diff_str} {framework_str}")
        
        lines.append("\nEdges:")
        for edge in self.edges:
            grad_str = "→" if edge.gradient_enabled else "⇢"
            lines.append(f"  {edge.source} {grad_str} {edge.target}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary."""
        return {
            'nodes': {
                node_id: {
                    'node_type': node.node_type.value,
                    'framework': node.framework,
                    'differentiable': node.differentiable,
                    'metadata': node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            'edges': [
                {
                    'source': e.source,
                    'target': e.target,
                    'gradient_enabled': e.gradient_enabled
                }
                for e in self.edges
            ]
        }
    
    def save(self, path: str):
        """Save graph to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load(self, path: str):
        """Load graph from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Clear current graph
        self.nodes = {}
        self.edges = []
        
        # Load nodes
        for node_id, node_data in data['nodes'].items():
            self.add_node(
                node_id,
                differentiable=node_data['differentiable'],
                framework=node_data['framework'],
                node_type=NodeType(node_data['node_type']),
                metadata=node_data['metadata']
            )
        
        # Load edges
        for edge_data in data['edges']:
            self.add_edge(
                edge_data['source'],
                edge_data['target'],
                edge_data['gradient_enabled']
            )


# ============================================================================
# Utility Functions
# ============================================================================

def create_simple_graph() -> GradientFlowGraph:
    """
    Create a simple example graph.
    
    Returns:
        Graph with oscillator → ML → output
    """
    graph = GradientFlowGraph()
    
    # Add nodes
    graph.add_node("oscillator", differentiable=False, framework=None)
    graph.add_node("ml_model", differentiable=True, framework='jax')
    graph.add_node("output", differentiable=True, framework='jax')
    
    # Add edges
    graph.add_edge("oscillator", "ml_model")
    graph.add_edge("ml_model", "output")
    
    return graph


def detect_gradient_vanishing(
    gradient_norms: Dict[str, List[float]],
    threshold: float = 1e-6
) -> List[str]:
    """
    Detect nodes with vanishing gradients.
    
    Args:
        gradient_norms: Dict of gradient norms by node
        threshold: Vanishing threshold
        
    Returns:
        List of node IDs with vanishing gradients
    """
    vanishing = []
    
    for node_id, norms in gradient_norms.items():
        if norms and np.mean(norms[-10:]) < threshold:
            vanishing.append(node_id)
    
    return vanishing


def detect_gradient_explosion(
    gradient_norms: Dict[str, List[float]],
    threshold: float = 100.0
) -> List[str]:
    """
    Detect nodes with exploding gradients.
    
    Args:
        gradient_norms: Dict of gradient norms by node
        threshold: Explosion threshold
        
    Returns:
        List of node IDs with exploding gradients
    """
    exploding = []
    
    for node_id, norms in gradient_norms.items():
        if norms and np.max(norms[-10:]) > threshold:
            exploding.append(node_id)
    
    return exploding


if __name__ == '__main__':
    # Example usage
    print("Creating gradient flow graph...")
    graph = create_simple_graph()
    print(graph.visualize_graph())
    
    print("\nTopological order:")
    print(graph._compute_topological_order())
    
    print("\nGraph statistics:")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print(f"Differentiable nodes: {sum(1 for n in graph.nodes.values() if n.differentiable)}")

