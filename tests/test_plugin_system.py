"""
Tests for Plugin System.

Tests plugin base classes, registry, and example plugins.
"""

import pytest
import jax.numpy as jnp

from src.plugins.plugin_base import (
    GMCSPlugin,
    PluginMetadata,
    AlgorithmPlugin,
    ProcessorPlugin,
    AnalyzerPlugin,
    StatefulPlugin,
    PluginPipeline,
    PluginParallel
)
from src.plugins.plugin_registry import PluginRegistry, get_global_registry


class SimpleTestPlugin(AlgorithmPlugin):
    """Simple test plugin."""
    
    def get_metadata(self):
        return PluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            author="Test",
            description="Test plugin",
            category="algorithm",
            tags=["test"],
            parameters=[
                {"name": "gain", "type": "float", "range": [0.0, 2.0], "default": 1.0}
            ]
        )
    
    def initialize(self, config):
        self.state = {"initialized": True}
    
    def compute(self, input_signal, params, **kwargs):
        gain = params[0] if len(params) > 0 else 1.0
        return input_signal * gain


class TestPluginBase:
    """Test plugin base classes."""
    
    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="Test",
            version="1.0.0",
            author="Author",
            description="Description",
            category="algorithm",
            tags=["test"],
            parameters=[]
        )
        
        assert metadata.name == "Test"
        assert metadata.version == "1.0.0"
        assert metadata.category == "algorithm"
    
    def test_algorithm_plugin_creation(self):
        """Test creating algorithm plugin."""
        plugin = SimpleTestPlugin()
        
        assert plugin is not None
        assert plugin.enabled
        
        metadata = plugin.get_metadata()
        assert metadata.name == "TestPlugin"
    
    def test_algorithm_plugin_compute(self):
        """Test algorithm plugin computation."""
        plugin = SimpleTestPlugin()
        plugin.initialize({})
        
        # Test computation
        input_signal = jnp.array(2.0)
        params = jnp.array([1.5])
        
        output = plugin.compute(input_signal, params)
        
        assert jnp.allclose(output, 3.0)
    
    def test_plugin_enable_disable(self):
        """Test enabling/disabling plugins."""
        plugin = SimpleTestPlugin()
        
        assert plugin.enabled
        
        plugin.disable()
        assert not plugin.enabled
        
        plugin.enable()
        assert plugin.enabled
    
    def test_plugin_state_management(self):
        """Test plugin state management."""
        plugin = SimpleTestPlugin()
        
        # Set state
        plugin.set_state({"test": "value"})
        state = plugin.get_state()
        
        assert state["test"] == "value"
        
        # Reset
        plugin.reset()
        state = plugin.get_state()
        assert "test" not in state


class TestStatefulPlugin:
    """Test stateful plugin."""
    
    def test_stateful_plugin_history(self):
        """Test stateful plugin history management."""
        
        class TestStateful(StatefulPlugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="TestStateful",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    category="analyzer",
                    tags=[],
                    parameters=[]
                )
            
            def initialize(self, config):
                pass
            
            def process_with_history(self, current_data, history, **kwargs):
                return len(history)
        
        plugin = TestStateful()
        
        # Add data
        plugin.add_to_history(1)
        plugin.add_to_history(2)
        plugin.add_to_history(3)
        
        history = plugin.get_history()
        assert len(history) == 3
        
        # Clear
        plugin.clear_history()
        history = plugin.get_history()
        assert len(history) == 0


class TestPluginComposite:
    """Test composite plugins."""
    
    def test_plugin_pipeline(self):
        """Test plugin pipeline."""
        pipeline = PluginPipeline()
        pipeline.initialize({})
        
        # Add plugins
        plugin1 = SimpleTestPlugin()
        plugin1.initialize({})
        plugin2 = SimpleTestPlugin()
        plugin2.initialize({})
        
        pipeline.add_plugin(plugin1)
        pipeline.add_plugin(plugin2)
        
        # Process through pipeline
        input_data = jnp.array(2.0)
        params = jnp.array([1.5])
        
        output = pipeline.process(input_data, params=params)
        
        # Should be 2.0 * 1.5 * 1.5 = 4.5
        assert jnp.allclose(output, 4.5)
    
    def test_plugin_parallel(self):
        """Test parallel plugin execution."""
        parallel = PluginParallel()
        parallel.initialize({})
        
        # Add plugins
        plugin1 = SimpleTestPlugin()
        plugin1.initialize({})
        plugin2 = SimpleTestPlugin()
        plugin2.initialize({})
        
        parallel.add_plugin(plugin1)
        parallel.add_plugin(plugin2)
        
        # Process in parallel
        input_data = jnp.array(2.0)
        params = jnp.array([1.5])
        
        outputs = parallel.process(input_data, params=params)
        
        assert isinstance(outputs, dict)
        assert len(outputs) == 2


class TestPluginRegistry:
    """Test plugin registry."""
    
    def test_registry_creation(self):
        """Test creating registry."""
        registry = PluginRegistry()
        
        assert registry is not None
        assert len(registry.plugins) == 0
    
    def test_register_plugin(self):
        """Test registering plugin."""
        registry = PluginRegistry()
        
        plugin_id = registry.register_plugin(SimpleTestPlugin)
        
        assert plugin_id is not None
        assert len(registry.plugins) == 1
    
    def test_get_plugin(self):
        """Test getting plugin."""
        registry = PluginRegistry()
        
        plugin_id = registry.register_plugin(SimpleTestPlugin)
        plugin = registry.get_plugin(plugin_id)
        
        assert plugin is not None
        assert plugin.get_metadata().name == "TestPlugin"
    
    def test_list_plugins(self):
        """Test listing plugins."""
        registry = PluginRegistry()
        
        registry.register_plugin(SimpleTestPlugin)
        
        plugins = registry.list_plugins()
        
        assert len(plugins) == 1
        assert plugins[0]["metadata"]["name"] == "TestPlugin"
    
    def test_filter_by_category(self):
        """Test filtering plugins by category."""
        registry = PluginRegistry()
        
        registry.register_plugin(SimpleTestPlugin)
        
        # Filter by category
        plugins = registry.list_plugins(category="algorithm")
        assert len(plugins) == 1
        
        plugins = registry.list_plugins(category="processor")
        assert len(plugins) == 0
    
    def test_enable_disable_plugin(self):
        """Test enabling/disabling plugin."""
        registry = PluginRegistry()
        
        plugin_id = registry.register_plugin(SimpleTestPlugin)
        
        # Disable
        registry.disable_plugin(plugin_id)
        plugin = registry.get_plugin(plugin_id)
        assert not plugin.enabled
        
        # Enable
        registry.enable_plugin(plugin_id)
        plugin = registry.get_plugin(plugin_id)
        assert plugin.enabled
    
    def test_execute_plugin(self):
        """Test executing plugin."""
        registry = PluginRegistry()
        
        plugin_id = registry.register_plugin(SimpleTestPlugin)
        registry.initialize_plugin(plugin_id, {})
        
        # Execute
        input_data = jnp.array(2.0)
        output = registry.execute_plugin(plugin_id, input_data, params=jnp.array([1.5]))
        
        assert jnp.allclose(output, 3.0)
    
    def test_global_registry(self):
        """Test global registry singleton."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        
        assert registry1 is registry2
    
    def test_registry_statistics(self):
        """Test registry statistics."""
        registry = PluginRegistry()
        
        registry.register_plugin(SimpleTestPlugin)
        
        stats = registry.get_statistics()
        
        assert stats['total_plugins'] == 1
        assert stats['enabled_plugins'] == 1
        assert 'by_category' in stats


def test_plugin_module_imports():
    """Test that plugin modules can be imported."""
    from src.plugins import (
        GMCSPlugin,
        PluginMetadata,
        AlgorithmPlugin,
        PluginRegistry,
        get_global_registry
    )
    
    assert GMCSPlugin is not None
    assert PluginMetadata is not None
    assert AlgorithmPlugin is not None
    assert PluginRegistry is not None
    assert get_global_registry is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

