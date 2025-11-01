"""
Node system for GMCS.

Provides node factory and executor for running node graph presets.
"""

from .node_factory import NodeFactory, create_node_from_preset
from .node_executor import NodeGraph, PresetExecutor

__all__ = [
    'NodeFactory',
    'create_node_from_preset',
    'NodeGraph',
    'PresetExecutor'
]

