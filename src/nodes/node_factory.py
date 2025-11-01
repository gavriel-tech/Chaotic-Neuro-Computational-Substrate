"""
Node Factory for GMCS Node Graph System.

Central registry and factory for instantiating all node types from JSON configurations.
"""

from typing import Dict, Any, Optional
import importlib


# Import all node modules
from src.ml.ml_nodes import MLNode
from src.analysis.analysis_nodes import create_analysis_node
from src.control.control_nodes import create_control_node
from src.generator.generator_nodes import create_generator_node
from src.processor.processor_nodes import create_processor_node
from src.nodes.input_nodes import create_input_node
from src.nodes.output_nodes import create_output_node
from src.nodes.storage_nodes import create_storage_node
from src.nodes.rl_env_nodes import create_rl_env_node
from src.nodes.simulation_bridge import create_simulation_bridge_node


class NodeFactory:
    """
    Factory for creating node instances from type strings and configs.
    
    Centralizes node creation logic and maintains a registry of available node types.
    """
    
    # Node type registry
    NODE_TYPES = {
        # ML nodes
        'ml': {
            'MLP Predictor': 'ml',
            'CNN Classifier': 'ml',
            'Transformer Encoder': 'ml',
            'Diffusion Generator': 'ml',
            'GAN Generator': 'ml',
            'RL Controller': 'ml',
            'Autoencoder': 'ml',
        },
        
        # Analysis nodes
        'analysis': {
            'FFT Analyzer': 'analysis',
            'Pattern Recognizer': 'analysis',
            'Lyapunov Calculator': 'analysis',
            'Attractor Analyzer': 'analysis',
        },
        
        # Control nodes
        'control': {
            'Parameter Optimizer': 'control',
            'Chaos Controller': 'control',
            'PID Controller': 'control',
        },
        
        # Generator nodes
        'generator': {
            'Noise Generator': 'generator',
            'Pattern Generator': 'generator',
            'Sequence Generator': 'generator',
        },
        
        # Processor nodes
        'processor': {
            'Pitch Quantizer': 'processor',
            'Sprite Formatter': 'processor',
            'Color Mapper': 'processor',
            'Molecule Builder': 'processor',
            'Circuit Builder': 'processor',
            'Data Encoder': 'processor',
            'Validator': 'processor',
            'Formatter': 'processor',
            'Physics Simulator': 'processor',
            'Mixer': 'processor',
        },
        
        # Core GMCS nodes (existing)
        'oscillator': {},
        'thrml': {},
        'wave_pde': {},
        'visualizer': {},
    }
    
    @classmethod
    def create_node(cls, node_type: str, node_name: str, config: Dict[str, Any]) -> Any:
        """
        Create a node instance from type, name, and config.
        
        Args:
            node_type: Category of node ('ml', 'analysis', 'control', etc.)
            node_name: Specific node name ('MLP Predictor', 'FFT Analyzer', etc.)
            config: Node configuration dictionary
            
        Returns:
            Node instance
            
        Raises:
            ValueError: If node type/name is unknown
        """
        try:
            if node_type == 'ml':
                return cls._create_ml_node(node_name, config)
            elif node_type == 'analysis':
                return create_analysis_node(node_name, config)
            elif node_type == 'control':
                return create_control_node(node_name, config)
            elif node_type == 'generator':
                return create_generator_node(node_name, config)
            elif node_type == 'processor':
                return create_processor_node(node_name, config)
            elif node_type == 'oscillator':
                return create_simulation_bridge_node('oscillator', config)
            elif node_type == 'thrml':
                return create_simulation_bridge_node('thrml', config)
            elif node_type == 'wave_pde':
                return create_simulation_bridge_node('wave_pde', config)
            elif node_type == 'visualizer':
                return cls._create_visualizer_node(config)
            elif node_type == 'input':
                return create_input_node(node_name, config)
            elif node_type == 'output':
                return create_output_node(node_name, config)
            elif node_type == 'storage':
                return create_storage_node(node_name, config)
            elif node_type == 'rl_env':
                return create_rl_env_node(node_name, config)
            else:
                raise ValueError(f"Unknown node type: {node_type}")
                
        except Exception as e:
            print(f"Error creating node {node_name} of type {node_type}: {e}")
            # Return a stub node that does nothing
            return StubNode(node_type, node_name, config)
    
    @classmethod
    def _create_ml_node(cls, node_name: str, config: Dict[str, Any]) -> Any:
        """Create ML node."""
        return StubNode('ml', node_name, config)
    
    @classmethod
    def _create_visualizer_node(cls, config: Dict[str, Any]) -> Any:
        """Create visualizer node (stub for now - visualization happens client-side)."""
        return {
            'type': 'visualizer',
            'config': config
        }
    
    @classmethod
    def list_available_nodes(cls) -> Dict[str, list]:
        """
        Get list of all available node types.
        
        Returns:
            Dictionary mapping categories to node name lists
        """
        available = {}
        for category, nodes in cls.NODE_TYPES.items():
            if nodes:  # Skip empty categories
                available[category] = list(nodes.keys())
        return available
    
    @classmethod
    def get_node_info(cls, node_type: str, node_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific node type.
        
        Args:
            node_type: Node category
            node_name: Specific node name
            
        Returns:
            Node information dictionary or None if not found
        """
        if node_type in cls.NODE_TYPES:
            if node_name in cls.NODE_TYPES[node_type]:
                return {
                    'type': node_type,
                    'name': node_name,
                    'category': node_type,
                    'available': True
                }
        return None


class StubNode:
    """Stub node for unimplemented types."""
    is_stub: bool = True
    
    def __init__(self, node_type: str, node_name: str, config: Dict[str, Any]):
        self.type = node_type
        self.name = node_name
        self.config = config
    
    def process(self, **inputs) -> Dict[str, Any]:
        """Stub process that returns zero outputs."""
        print(f"[STUB] {self.name} ({self.type}) - not fully implemented")
        return {}


# Convenience function
def create_node_from_preset(node_def: Dict[str, Any]) -> Any:
    """
    Create a node from a preset node definition.
    
    Args:
        node_def: Node definition from preset JSON with keys:
            - id: Node ID
            - type: Node type
            - name: Node name
            - config: Node configuration
            
    Returns:
        Node instance
    """
    return NodeFactory.create_node(
        node_type=node_def['type'],
        node_name=node_def['name'],
        config=node_def['config']
    )

