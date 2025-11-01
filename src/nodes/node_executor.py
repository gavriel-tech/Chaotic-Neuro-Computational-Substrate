"""
Node Graph Executor for GMCS.

Executes node graphs by routing data between nodes according to connections.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict, deque

from src.performance import ProfilerContext, get_global_profiler, get_global_tracker


class NodeGraph:
    """
    Represents and executes a node graph.
    
    Handles topological sorting, data routing, and execution order.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Any] = {}
        self.connections: List[Dict[str, str]] = []
        self.execution_order: List[str] = []
        self.node_outputs: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    def add_node(self, node_id: str, node_instance: Any):
        """Add a node to the graph."""
        self.nodes[node_id] = node_instance
    
    def add_connection(self, from_node: str, from_output: str, to_node: str, to_input: str):
        """
        Add a connection between nodes.
        
        Args:
            from_node: Source node ID
            from_output: Output name from source node
            to_node: Destination node ID
            to_input: Input name for destination node
        """
        self.connections.append({
            'from_node': from_node,
            'from_output': from_output,
            'to_node': to_node,
            'to_input': to_input
        })
    
    def build_execution_order(self):
        """
        Build topological execution order using Kahn's algorithm.
        
        Ensures nodes are executed in correct dependency order.
        """
        # Build adjacency list and in-degree count
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Initialize all nodes with zero in-degree
        for node_id in self.nodes:
            in_degree[node_id] = 0
        
        # Build graph
        for conn in self.connections:
            from_node = conn['from_node']
            to_node = conn['to_node']
            graph[from_node].append(to_node)
            in_degree[to_node] += 1
        
        # Find nodes with no dependencies
        queue = deque([node_id for node_id in self.nodes if in_degree[node_id] == 0])
        order = []
        
        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            
            # Reduce in-degree for dependent nodes
            for dependent in graph[node_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(order) != len(self.nodes):
            print("[WARNING] Cycle detected in node graph - some nodes may not execute")
        
        self.execution_order = order
    
    def execute_step(self) -> Dict[str, Dict[str, Any]]:
        """
        Execute one step of the node graph.
        
        Returns:
            Dictionary mapping node IDs to their outputs
        """
        if not self.execution_order:
            self.build_execution_order()
        
        # Profile entire step execution
        with ProfilerContext("node_graph_step"):
            # Execute nodes in topological order
            for node_id in self.execution_order:
                node = self.nodes[node_id]
                
                # Gather inputs for this node
                inputs = self._gather_inputs(node_id)
                
                # Execute node with profiling
                try:
                    node_type = type(node).__name__ if hasattr(node, '__class__') else 'unknown'
                    
                    with ProfilerContext(f"node_{node_type}", metadata={"node_id": node_id}):
                        if hasattr(node, 'process'):
                            outputs = node.process(**inputs)
                        elif isinstance(node, dict):
                            # Stub node - return empty
                            outputs = {}
                        else:
                            outputs = {}
                    
                    # Store outputs
                    self.node_outputs[node_id] = outputs
                    
                    # Record metrics
                    tracker = get_global_tracker()
                    profiler = get_global_profiler()
                    # Get the last entry for this node
                    entries = profiler.get_entries(name_filter=f"node_{node_type}")
                    if entries:
                        last_entry = entries[-1]
                        if last_entry.duration is not None:
                            tracker.record(
                                name=f"node_{node_type}",
                                duration=last_entry.duration,
                                memory_delta=last_entry.memory_delta,
                                gpu_memory_delta=last_entry.gpu_memory_delta,
                                gpu_utilization=last_entry.gpu_utilization
                            )
                    
                except Exception as e:
                    print(f"[ERROR] Node {node_id} execution failed: {e}")
                    self.node_outputs[node_id] = {}
        
        return dict(self.node_outputs)
    
    def _gather_inputs(self, node_id: str) -> Dict[str, Any]:
        """
        Gather inputs for a node from connected outputs.
        
        Args:
            node_id: ID of node to gather inputs for
            
        Returns:
            Dictionary of input name -> value
        """
        inputs = {}
        
        for conn in self.connections:
            if conn['to_node'] == node_id:
                from_node = conn['from_node']
                from_output = conn['from_output']
                to_input = conn['to_input']
                
                # Get output from source node
                if from_node in self.node_outputs:
                    node_output = self.node_outputs[from_node]
                    if from_output in node_output:
                        inputs[to_input] = node_output[from_output]
        
        return inputs
    
    def get_node_output(self, node_id: str, output_name: str) -> Optional[Any]:
        """Get a specific output from a node."""
        if node_id in self.node_outputs:
            return self.node_outputs[node_id].get(output_name)
        return None
    
    def reset(self):
        """Reset execution state."""
        self.node_outputs.clear()


class PresetExecutor:
    """
    High-level executor for loading and running preset node graphs.
    """
    
    def __init__(self):
        self.graph = NodeGraph()
        self.preset_config = None
    
    def load_preset(self, preset: Dict[str, Any]):
        """
        Load a preset configuration into the executor.
        
        Args:
            preset: Preset dictionary with 'nodes' and 'connections'
        """
        from src.nodes.node_factory import create_node_from_preset
        
        self.preset_config = preset
        self.graph = NodeGraph()
        
        # Create all nodes
        for node_def in preset['nodes']:
            node_id = node_def['id']
            node_instance = create_node_from_preset(node_def)
            self.graph.add_node(node_id, node_instance)
        
        # Create all connections
        for conn_def in preset['connections']:
            # Parse connection string "node_id.output_name"
            from_parts = conn_def['from'].split('.')
            to_parts = conn_def['to'].split('.')
            
            if len(from_parts) == 2 and len(to_parts) == 2:
                self.graph.add_connection(
                    from_node=from_parts[0],
                    from_output=from_parts[1],
                    to_node=to_parts[0],
                    to_input=to_parts[1]
                )
        
        # Build execution order
        self.graph.build_execution_order()
        
        print(f"[PRESET] Loaded '{preset['name']}' with {len(preset['nodes'])} nodes")
        print(f"[PRESET] Execution order: {' -> '.join(self.graph.execution_order[:5])}...")
    
    def run_step(self) -> Dict[str, Dict[str, Any]]:
        """Run one step of the loaded preset."""
        return self.graph.execute_step()
    
    def run(self, num_steps: int = 100) -> List[Dict[str, Dict[str, Any]]]:
        """
        Run the preset for multiple steps.
        
        Args:
            num_steps: Number of steps to execute
            
        Returns:
            List of output dictionaries (one per step)
        """
        results = []
        for i in range(num_steps):
            outputs = self.run_step()
            results.append(outputs)
            
            if i % 10 == 0:
                print(f"[PRESET] Step {i}/{num_steps}")
        
        return results
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for visualizing the current graph state.
        
        Returns:
            Dictionary with visualization-ready data
        """
        return {
            'node_outputs': dict(self.graph.node_outputs),
            'num_nodes': len(self.graph.nodes),
            'num_connections': len(self.graph.connections),
            'execution_order': self.graph.execution_order
        }

