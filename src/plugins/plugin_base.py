"""
Plugin System Base Classes for GMCS.

Provides the foundation for creating custom algorithms, processors,
and extensions to the GMCS system.
"""

from typing import Dict, Any, Optional, List, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    author: str
    description: str
    category: str  # 'algorithm', 'processor', 'visualizer', 'analyzer'
    tags: List[str]
    parameters: List[Dict[str, Any]]  # List of parameter definitions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "parameters": self.parameters
        }


class GMCSPlugin(ABC):
    """
    Base class for all GMCS plugins.
    
    Plugins extend GMCS functionality by providing custom algorithms,
    processors, or analysis tools.
    """
    
    def __init__(self):
        """Initialize plugin."""
        self.metadata = self.get_metadata()
        self.state = {}  # Plugin-specific state
        self.enabled = True
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata.
        
        Returns:
            PluginMetadata instance
        """
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """
        Initialize plugin with configuration.
        
        Args:
            config: Configuration dictionary
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """
        Main processing function.
        
        Args:
            input_data: Input data (type depends on plugin category)
            **kwargs: Additional parameters
            
        Returns:
            Processed output
        """
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate parameters against metadata.
        
        Args:
            params: Parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        for param_def in self.metadata.parameters:
            param_name = param_def["name"]
            
            # Check required parameters
            if param_def.get("required", False) and param_name not in params:
                return False
            
            # Check types
            if param_name in params:
                expected_type = param_def.get("type")
                if expected_type and not isinstance(params[param_name], eval(expected_type)):
                    return False
            
            # Check ranges
            if param_name in params and "range" in param_def:
                value = params[param_name]
                min_val, max_val = param_def["range"]
                if not (min_val <= value <= max_val):
                    return False
        
        return True
    
    def get_state(self) -> Dict[str, Any]:
        """Get plugin state."""
        return self.state
    
    def set_state(self, state: Dict[str, Any]):
        """Set plugin state."""
        self.state = state
    
    def reset(self):
        """Reset plugin to initial state."""
        self.state = {}
    
    def enable(self):
        """Enable plugin."""
        self.enabled = True
    
    def disable(self):
        """Disable plugin."""
        self.enabled = False


class AlgorithmPlugin(GMCSPlugin):
    """
    Base class for custom GMCS algorithm plugins.
    
    Algorithm plugins provide new signal processing algorithms
    that can be used in the GMCS pipeline.
    """
    
    @abstractmethod
    def compute(
        self,
        input_signal: jnp.ndarray,
        params: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        """
        Compute algorithm output.
        
        Args:
            input_signal: Input signal value(s)
            params: Algorithm parameters
            **kwargs: Additional arguments
            
        Returns:
            Processed signal
        """
        pass
    
    def process(self, input_data: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Process wrapper for compute."""
        params = kwargs.get("params", jnp.zeros(8))
        return self.compute(input_data, params, **kwargs)
    
    def get_jax_function(self) -> Callable:
        """
        Get JAX-jittable version of the algorithm.
        
        Returns:
            JAX function
        """
        @jax.jit
        def jax_compute(h: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
            return self.compute(h, params)
        
        return jax_compute


class ProcessorPlugin(GMCSPlugin):
    """
    Base class for processor plugins.
    
    Processor plugins operate on system state or data streams,
    providing analysis, transformation, or side effects.
    """
    
    @abstractmethod
    def process_state(
        self,
        system_state: Any,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Process system state.
        
        Args:
            system_state: Current system state
            **kwargs: Additional arguments
            
        Returns:
            (modified_state, metadata) tuple
        """
        pass
    
    def process(self, input_data: Any, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Process wrapper."""
        return self.process_state(input_data, **kwargs)


class AnalyzerPlugin(GMCSPlugin):
    """
    Base class for analyzer plugins.
    
    Analyzer plugins extract information, compute metrics,
    or detect patterns in GMCS data.
    """
    
    @abstractmethod
    def analyze(
        self,
        data: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze data.
        
        Args:
            data: Data to analyze
            **kwargs: Additional arguments
            
        Returns:
            Analysis results dictionary
        """
        pass
    
    def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Process wrapper."""
        return self.analyze(input_data, **kwargs)


class VisualizerPlugin(GMCSPlugin):
    """
    Base class for visualizer plugins.
    
    Visualizer plugins generate visual representations
    or export data for external visualization.
    """
    
    @abstractmethod
    def render(
        self,
        data: Any,
        **kwargs
    ) -> Any:
        """
        Render visualization.
        
        Args:
            data: Data to visualize
            **kwargs: Rendering options
            
        Returns:
            Rendered output (format depends on implementation)
        """
        pass
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process wrapper."""
        return self.render(input_data, **kwargs)


class StatefulPlugin(GMCSPlugin):
    """
    Base class for stateful plugins.
    
    Stateful plugins maintain internal state across invocations,
    enabling temporal analysis, filtering, or accumulation.
    """
    
    def __init__(self):
        """Initialize stateful plugin."""
        super().__init__()
        self.history = []
        self.max_history_length = 100
    
    def add_to_history(self, data: Any):
        """
        Add data to history buffer.
        
        Args:
            data: Data to store
        """
        self.history.append(data)
        
        # Limit history size
        if len(self.history) > self.max_history_length:
            self.history.pop(0)
    
    def get_history(self, n: Optional[int] = None) -> List[Any]:
        """
        Get history.
        
        Args:
            n: Number of recent items (None = all)
            
        Returns:
            List of historical data
        """
        if n is None:
            return self.history
        return self.history[-n:]
    
    def clear_history(self):
        """Clear history buffer."""
        self.history = []
    
    @abstractmethod
    def process_with_history(
        self,
        current_data: Any,
        history: List[Any],
        **kwargs
    ) -> Any:
        """
        Process data with access to history.
        
        Args:
            current_data: Current data
            history: Historical data
            **kwargs: Additional arguments
            
        Returns:
            Processed output
        """
        pass
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """Process wrapper that manages history."""
        # Add to history
        self.add_to_history(input_data)
        
        # Process with history
        return self.process_with_history(
            input_data,
            self.history,
            **kwargs
        )


class CompositePlugin(GMCSPlugin):
    """
    Base class for composite plugins.
    
    Composite plugins combine multiple plugins into a pipeline
    or parallel processing structure.
    """
    
    def __init__(self):
        """Initialize composite plugin."""
        super().__init__()
        self.plugins: List[GMCSPlugin] = []
    
    def add_plugin(self, plugin: GMCSPlugin):
        """
        Add plugin to composite.
        
        Args:
            plugin: Plugin to add
        """
        self.plugins.append(plugin)
    
    def remove_plugin(self, plugin_name: str) -> bool:
        """
        Remove plugin by name.
        
        Args:
            plugin_name: Name of plugin to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, plugin in enumerate(self.plugins):
            if plugin.metadata.name == plugin_name:
                self.plugins.pop(i)
                return True
        return False
    
    def get_plugin(self, plugin_name: str) -> Optional[GMCSPlugin]:
        """
        Get plugin by name.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        for plugin in self.plugins:
            if plugin.metadata.name == plugin_name:
                return plugin
        return None
    
    @abstractmethod
    def process(self, input_data: Any, **kwargs) -> Any:
        """
        Process data through composite plugin.
        
        Args:
            input_data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Processed output
        """
        pass


class PluginPipeline(CompositePlugin):
    """
    Sequential pipeline of plugins.
    
    Data flows through plugins in order, with each plugin's
    output becoming the next plugin's input.
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Get metadata."""
        return PluginMetadata(
            name="PluginPipeline",
            version="1.0.0",
            author="GMCS",
            description="Sequential plugin pipeline",
            category="composite",
            tags=["pipeline", "sequential"],
            parameters=[]
        )
    
    def initialize(self, config: Dict[str, Any]):
        """Initialize pipeline."""
        for plugin in self.plugins:
            plugin.initialize(config.get(plugin.metadata.name, {}))
    
    def process(self, input_data: Any, **kwargs) -> Any:
        """
        Process data through pipeline.
        
        Args:
            input_data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Final output after all plugins
        """
        data = input_data
        
        for plugin in self.plugins:
            if plugin.enabled:
                data = plugin.process(data, **kwargs)
        
        return data


class PluginParallel(CompositePlugin):
    """
    Parallel execution of plugins.
    
    All plugins process the same input independently,
    and outputs are collected.
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Get metadata."""
        return PluginMetadata(
            name="PluginParallel",
            version="1.0.0",
            author="GMCS",
            description="Parallel plugin execution",
            category="composite",
            tags=["parallel", "concurrent"],
            parameters=[]
        )
    
    def initialize(self, config: Dict[str, Any]):
        """Initialize parallel processor."""
        for plugin in self.plugins:
            plugin.initialize(config.get(plugin.metadata.name, {}))
    
    def process(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Process data through all plugins in parallel.
        
        Args:
            input_data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Dictionary mapping plugin names to outputs
        """
        outputs = {}
        
        for plugin in self.plugins:
            if plugin.enabled:
                try:
                    outputs[plugin.metadata.name] = plugin.process(input_data, **kwargs)
                except Exception as e:
                    print(f"Error in plugin {plugin.metadata.name}: {e}")
                    outputs[plugin.metadata.name] = None
        
        return outputs

