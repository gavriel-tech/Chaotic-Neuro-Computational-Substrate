"""
Plugin System API Endpoints.

Provides REST API for plugin management and execution.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.plugins import get_global_registry, execute_plugin


# Create router
router = APIRouter(prefix="/plugins", tags=["plugins"])


class ExecutePluginRequest(BaseModel):
    """Request to execute a plugin."""
    input_data: Any = Field(..., description="Input data for plugin")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RegisterPluginRequest(BaseModel):
    """Request to register a plugin (metadata only)."""
    name: str
    version: str
    author: str
    description: str
    category: str
    tags: List[str] = Field(default_factory=list)
    parameters: List[Dict[str, Any]] = Field(default_factory=list)


@router.get("/list")
async def list_all_plugins(
    category: Optional[str] = None,
    tags: Optional[str] = None
):
    """
    List all registered plugins.
    
    Query params:
        - category: Filter by category
        - tags: Comma-separated list of tags
    """
    registry = get_global_registry()
    
    tag_list = tags.split(",") if tags else None
    plugins = registry.list_plugins(category=category, tags=tag_list)
    
    return {
        "plugins": plugins,
        "total": len(plugins)
    }


@router.get("/{plugin_id}")
async def get_plugin_info(plugin_id: str):
    """Get detailed information about a plugin."""
    registry = get_global_registry()
    
    plugin = registry.get_plugin(plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    metadata = plugin.get_metadata()
    
    return {
        "plugin_id": plugin_id,
        "metadata": metadata.to_dict(),
        "enabled": plugin.enabled,
        "state": plugin.get_state()
    }


@router.post("/{plugin_id}/execute")
async def execute_plugin_endpoint(plugin_id: str, request: ExecutePluginRequest):
    """
    Execute a plugin.
    
    Args:
        plugin_id: Plugin identifier
        request: Execution request with input data
    """
    try:
        result = execute_plugin(
            plugin_id,
            request.input_data,
            **request.parameters
        )
        
        return {
            "status": "success",
            "result": result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{plugin_id}/enable")
async def enable_plugin(plugin_id: str):
    """Enable a plugin."""
    registry = get_global_registry()
    
    plugin = registry.get_plugin(plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    registry.enable_plugin(plugin_id)
    
    return {
        "status": "success",
        "plugin_id": plugin_id,
        "enabled": True
    }


@router.post("/{plugin_id}/disable")
async def disable_plugin(plugin_id: str):
    """Disable a plugin."""
    registry = get_global_registry()
    
    plugin = registry.get_plugin(plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    registry.disable_plugin(plugin_id)
    
    return {
        "status": "success",
        "plugin_id": plugin_id,
        "enabled": False
    }


@router.post("/{plugin_id}/reset")
async def reset_plugin(plugin_id: str):
    """Reset plugin state."""
    registry = get_global_registry()
    
    plugin = registry.get_plugin(plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    plugin.reset()
    
    return {
        "status": "success",
        "plugin_id": plugin_id,
        "message": "Plugin state reset"
    }


@router.delete("/{plugin_id}")
async def unregister_plugin(plugin_id: str):
    """Unregister a plugin."""
    registry = get_global_registry()
    
    success = registry.unregister_plugin(plugin_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    return {
        "status": "success",
        "message": f"Plugin {plugin_id} unregistered"
    }


@router.get("/categories/list")
async def list_categories():
    """List available plugin categories."""
    return {
        "categories": [
            {
                "name": "algorithm",
                "description": "Signal processing algorithms for GMCS pipeline"
            },
            {
                "name": "processor",
                "description": "System state processors and transformers"
            },
            {
                "name": "analyzer",
                "description": "Analysis and pattern detection tools"
            },
            {
                "name": "visualizer",
                "description": "Visualization and rendering plugins"
            },
            {
                "name": "composite",
                "description": "Composite plugins (pipelines, parallel)"
            }
        ]
    }


@router.get("/statistics")
async def get_plugin_statistics():
    """Get plugin system statistics."""
    registry = get_global_registry()
    stats = registry.get_statistics()
    
    return stats


@router.post("/discover")
async def discover_plugins():
    """
    Discover and load plugins from plugin directories.
    
    Scans configured plugin directories for new plugins.
    """
    registry = get_global_registry()
    
    initial_count = len(registry.plugins)
    registry.discover_plugins()
    final_count = len(registry.plugins)
    
    return {
        "status": "success",
        "plugins_discovered": final_count - initial_count,
        "total_plugins": final_count
    }


@router.get("/examples")
async def get_plugin_examples():
    """Get example plugin implementations."""
    return {
        "examples": [
            {
                "name": "Waveshaper",
                "category": "algorithm",
                "description": "Custom waveshaping algorithm with multiple modes",
                "file": "src/plugins/examples/waveshaper_plugin.py"
            },
            {
                "name": "PatternDetector",
                "category": "analyzer",
                "description": "Detects patterns in oscillator dynamics",
                "file": "src/plugins/examples/pattern_detector_plugin.py"
            },
            {
                "name": "FrequencyAnalyzer",
                "category": "analyzer",
                "description": "FFT-based frequency analysis",
                "file": "src/plugins/examples/pattern_detector_plugin.py"
            }
        ]
    }


@router.get("/documentation")
async def get_plugin_documentation():
    """Get plugin development documentation."""
    return {
        "overview": "GMCS Plugin System allows extending functionality with custom algorithms and processors",
        "base_classes": [
            {
                "class": "AlgorithmPlugin",
                "description": "For custom GMCS signal processing algorithms",
                "methods": ["compute", "get_jax_function"]
            },
            {
                "class": "ProcessorPlugin",
                "description": "For system state processors",
                "methods": ["process_state"]
            },
            {
                "class": "AnalyzerPlugin",
                "description": "For analysis and pattern detection",
                "methods": ["analyze"]
            },
            {
                "class": "StatefulPlugin",
                "description": "For plugins that maintain history",
                "methods": ["process_with_history", "add_to_history"]
            }
        ],
        "quick_start": {
            "1": "Import base class: from src.plugins.plugin_base import AlgorithmPlugin",
            "2": "Create class: class MyPlugin(AlgorithmPlugin)",
            "3": "Implement get_metadata() and compute() methods",
            "4": "Save to src/plugins/custom/my_plugin.py",
            "5": "Call /plugins/discover to load"
        },
        "example_code": """
from src.plugins.plugin_base import AlgorithmPlugin, PluginMetadata
import jax.numpy as jnp

class MyPlugin(AlgorithmPlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="MyPlugin",
            version="1.0.0",
            author="Your Name",
            description="My custom algorithm",
            category="algorithm",
            tags=["custom"],
            parameters=[
                {"name": "gain", "type": "float", "range": [0.0, 2.0], "default": 1.0}
            ]
        )
    
    def initialize(self, config):
        self.state = {"initialized": True}
    
    def compute(self, input_signal, params, **kwargs):
        gain = params[0]
        return input_signal * gain
"""
    }


@router.post("/{plugin_id}/initialize")
async def initialize_plugin(plugin_id: str, config: Dict[str, Any]):
    """
    Initialize a plugin with configuration.
    
    Args:
        plugin_id: Plugin identifier
        config: Configuration dictionary
    """
    registry = get_global_registry()
    
    plugin = registry.get_plugin(plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    try:
        registry.initialize_plugin(plugin_id, config)
        return {
            "status": "success",
            "plugin_id": plugin_id,
            "message": "Plugin initialized"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

