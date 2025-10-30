"""
Plugin Registry and Management System.

Provides discovery, loading, and lifecycle management for GMCS plugins.
"""

from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import importlib
import importlib.util
import json
import sys

from src.plugins.plugin_base import GMCSPlugin, PluginMetadata


class PluginRegistry:
    """
    Central registry for GMCS plugins.
    
    Manages plugin discovery, loading, registration, and lifecycle.
    """
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """
        Initialize plugin registry.
        
        Args:
            plugin_dirs: List of directories to search for plugins
        """
        self.plugins: Dict[str, GMCSPlugin] = {}
        self.plugin_classes: Dict[str, Type[GMCSPlugin]] = {}
        self.plugin_dirs = plugin_dirs or ["src/plugins/custom", "plugins"]
        
        # Ensure plugin directories exist
        for plugin_dir in self.plugin_dirs:
            Path(plugin_dir).mkdir(parents=True, exist_ok=True)
    
    def register_plugin(
        self,
        plugin_class: Type[GMCSPlugin],
        plugin_id: Optional[str] = None
    ) -> str:
        """
        Register a plugin class.
        
        Args:
            plugin_class: Plugin class (not instance)
            plugin_id: Optional custom plugin ID
            
        Returns:
            Plugin ID
        """
        # Create instance to get metadata
        instance = plugin_class()
        metadata = instance.get_metadata()
        
        # Generate ID if not provided
        if plugin_id is None:
            plugin_id = f"{metadata.category}_{metadata.name}_{metadata.version}"
        
        # Store class and instance
        self.plugin_classes[plugin_id] = plugin_class
        self.plugins[plugin_id] = instance
        
        return plugin_id
    
    def register_plugin_instance(
        self,
        plugin: GMCSPlugin,
        plugin_id: Optional[str] = None
    ) -> str:
        """
        Register a plugin instance.
        
        Args:
            plugin: Plugin instance
            plugin_id: Optional custom plugin ID
            
        Returns:
            Plugin ID
        """
        metadata = plugin.get_metadata()
        
        if plugin_id is None:
            plugin_id = f"{metadata.category}_{metadata.name}_{metadata.version}"
        
        self.plugins[plugin_id] = plugin
        self.plugin_classes[plugin_id] = type(plugin)
        
        return plugin_id
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            True if unregistered, False if not found
        """
        if plugin_id in self.plugins:
            del self.plugins[plugin_id]
        if plugin_id in self.plugin_classes:
            del self.plugin_classes[plugin_id]
        return True
    
    def get_plugin(self, plugin_id: str) -> Optional[GMCSPlugin]:
        """
        Get plugin instance by ID.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Plugin instance or None
        """
        return self.plugins.get(plugin_id)
    
    def list_plugins(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List registered plugins.
        
        Args:
            category: Filter by category
            tags: Filter by tags (any match)
            
        Returns:
            List of plugin metadata dictionaries
        """
        plugins = []
        
        for plugin_id, plugin in self.plugins.items():
            metadata = plugin.get_metadata()
            
            # Apply filters
            if category and metadata.category != category:
                continue
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            plugins.append({
                "plugin_id": plugin_id,
                "metadata": metadata.to_dict(),
                "enabled": plugin.enabled
            })
        
        return plugins
    
    def discover_plugins(self):
        """
        Discover and load plugins from plugin directories.
        
        Searches for Python files with GMCSPlugin subclasses.
        """
        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir)
            
            if not plugin_path.exists():
                continue
            
            # Find all Python files
            for py_file in plugin_path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                try:
                    self._load_plugin_from_file(py_file)
                except Exception as e:
                    print(f"Error loading plugin from {py_file}: {e}")
    
    def _load_plugin_from_file(self, filepath: Path):
        """
        Load plugin from Python file.
        
        Args:
            filepath: Path to Python file
        """
        # Create module name
        module_name = f"gmcs_plugin_{filepath.stem}"
        
        # Load module
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            return
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Find GMCSPlugin subclasses
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            # Check if it's a class and subclass of GMCSPlugin
            if (isinstance(attr, type) and
                issubclass(attr, GMCSPlugin) and
                attr is not GMCSPlugin):
                
                try:
                    self.register_plugin(attr)
                    print(f"Loaded plugin: {attr_name} from {filepath.name}")
                except Exception as e:
                    print(f"Error registering plugin {attr_name}: {e}")
    
    def initialize_plugin(self, plugin_id: str, config: Dict[str, Any]):
        """
        Initialize a plugin with configuration.
        
        Args:
            plugin_id: Plugin identifier
            config: Configuration dictionary
        """
        plugin = self.get_plugin(plugin_id)
        if plugin:
            plugin.initialize(config)
    
    def enable_plugin(self, plugin_id: str):
        """Enable a plugin."""
        plugin = self.get_plugin(plugin_id)
        if plugin:
            plugin.enable()
    
    def disable_plugin(self, plugin_id: str):
        """Disable a plugin."""
        plugin = self.get_plugin(plugin_id)
        if plugin:
            plugin.disable()
    
    def execute_plugin(
        self,
        plugin_id: str,
        input_data: Any,
        **kwargs
    ) -> Any:
        """
        Execute a plugin.
        
        Args:
            plugin_id: Plugin identifier
            input_data: Input data
            **kwargs: Additional arguments
            
        Returns:
            Plugin output
        """
        plugin = self.get_plugin(plugin_id)
        if plugin is None:
            raise ValueError(f"Plugin {plugin_id} not found")
        
        if not plugin.enabled:
            raise ValueError(f"Plugin {plugin_id} is disabled")
        
        return plugin.process(input_data, **kwargs)
    
    def get_plugin_metadata(self, plugin_id: str) -> Optional[PluginMetadata]:
        """
        Get plugin metadata.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            PluginMetadata or None
        """
        plugin = self.get_plugin(plugin_id)
        if plugin:
            return plugin.get_metadata()
        return None
    
    def export_registry(self, output_path: str):
        """
        Export registry to JSON file.
        
        Args:
            output_path: Output file path
        """
        registry_data = {
            "plugins": [
                {
                    "plugin_id": plugin_id,
                    "metadata": plugin.get_metadata().to_dict(),
                    "enabled": plugin.enabled
                }
                for plugin_id, plugin in self.plugins.items()
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats = {
            "total_plugins": len(self.plugins),
            "enabled_plugins": sum(1 for p in self.plugins.values() if p.enabled),
            "disabled_plugins": sum(1 for p in self.plugins.values() if not p.enabled),
            "by_category": {},
            "by_author": {}
        }
        
        for plugin in self.plugins.values():
            metadata = plugin.get_metadata()
            
            # Count by category
            category = metadata.category
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            # Count by author
            author = metadata.author
            stats["by_author"][author] = stats["by_author"].get(author, 0) + 1
        
        return stats


# Global registry instance
_global_registry: Optional[PluginRegistry] = None


def get_global_registry() -> PluginRegistry:
    """Get or create global plugin registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
        # Auto-discover plugins
        _global_registry.discover_plugins()
    return _global_registry


def register_plugin(*args, **kwargs) -> str:
    """Convenience function to register plugin in global registry."""
    return get_global_registry().register_plugin(*args, **kwargs)


def get_plugin(plugin_id: str) -> Optional[GMCSPlugin]:
    """Convenience function to get plugin from global registry."""
    return get_global_registry().get_plugin(plugin_id)


def list_plugins(*args, **kwargs) -> List[Dict[str, Any]]:
    """Convenience function to list plugins in global registry."""
    return get_global_registry().list_plugins(*args, **kwargs)


def execute_plugin(plugin_id: str, input_data: Any, **kwargs) -> Any:
    """Convenience function to execute plugin from global registry."""
    return get_global_registry().execute_plugin(plugin_id, input_data, **kwargs)

