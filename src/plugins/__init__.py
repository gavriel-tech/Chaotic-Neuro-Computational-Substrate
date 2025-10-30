"""
GMCS Plugin System.

Provides extensible plugin architecture for custom algorithms,
processors, analyzers, and visualizers.
"""

from src.plugins.plugin_base import (
    GMCSPlugin,
    PluginMetadata,
    AlgorithmPlugin,
    ProcessorPlugin,
    AnalyzerPlugin,
    VisualizerPlugin,
    StatefulPlugin,
    CompositePlugin,
    PluginPipeline,
    PluginParallel
)

from src.plugins.plugin_registry import (
    PluginRegistry,
    get_global_registry,
    register_plugin,
    get_plugin,
    list_plugins,
    execute_plugin
)

__all__ = [
    # Base classes
    'GMCSPlugin',
    'PluginMetadata',
    'AlgorithmPlugin',
    'ProcessorPlugin',
    'AnalyzerPlugin',
    'VisualizerPlugin',
    'StatefulPlugin',
    'CompositePlugin',
    'PluginPipeline',
    'PluginParallel',
    # Registry
    'PluginRegistry',
    'get_global_registry',
    'register_plugin',
    'get_plugin',
    'list_plugins',
    'execute_plugin',
]

