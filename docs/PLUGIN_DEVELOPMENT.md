# Plugin Development Guide

Learn how to create custom plugins for GMCS.

## Quick Start

```python
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
                {
                    "name": "gain",
                    "type": "float",
                    "range": [0.0, 2.0],
                    "default": 1.0,
                    "required": True
                }
            ]
        )
    
    def initialize(self, config):
        self.state = {"initialized": True}
    
    def compute(self, input_signal, params, **kwargs):
        gain = params[0]
        return input_signal * gain
```

Save as `src/plugins/custom/my_plugin.py` and call `/plugins/discover`.

## Plugin Types

### 1. AlgorithmPlugin
For GMCS signal processing algorithms.

**Methods**:
- `compute(input_signal, params, **kwargs)`: Main computation
- `get_jax_function()`: Returns JAX-jitted version

### 2. ProcessorPlugin
For system state processors.

**Methods**:
- `process_state(system_state, **kwargs)`: Process state

### 3. AnalyzerPlugin
For analysis and pattern detection.

**Methods**:
- `analyze(data, **kwargs)`: Analyze data

### 4. StatefulPlugin
For plugins that maintain history.

**Methods**:
- `process_with_history(current_data, history, **kwargs)`
- `add_to_history(data)`
- `get_history(n=None)`

### 5. VisualizerPlugin
For visualization and rendering.

**Methods**:
- `render(data, **kwargs)`: Generate visualization

## Complete Examples

### Example 1: Waveshaper
```python
class WaveshaperPlugin(AlgorithmPlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="Waveshaper",
            version="1.0.0",
            author="GMCS Team",
            description="Custom waveshaping",
            category="algorithm",
            tags=["waveshaping", "distortion"],
            parameters=[
                {"name": "drive", "type": "float", "range": [0.0, 10.0], "default": 1.0},
                {"name": "mode", "type": "int", "range": [0, 3], "default": 0}
            ]
        )
    
    def initialize(self, config):
        self.state = {"initialized": True}
    
    def compute(self, input_signal, params, **kwargs):
        drive = params[0]
        mode = jnp.int32(params[1])
        
        driven = input_signal * drive
        
        # Different modes
        soft = jnp.tanh(driven)
        hard = jnp.clip(driven, -1.0, 1.0)
        fold = jnp.where(jnp.abs(driven) > 1.0, 2.0 - jnp.abs(driven), driven)
        wrap = (driven + 1.0) % 2.0 - 1.0
        
        return jnp.where(mode == 0, soft,
               jnp.where(mode == 1, hard,
               jnp.where(mode == 2, fold, wrap)))
```

### Example 2: Pattern Detector
```python
class PatternDetectorPlugin(StatefulPlugin):
    def __init__(self):
        super().__init__()
        self.max_history_length = 200
    
    def get_metadata(self):
        return PluginMetadata(
            name="PatternDetector",
            version="1.0.0",
            author="GMCS Team",
            description="Detects patterns in dynamics",
            category="analyzer",
            tags=["pattern", "detection"],
            parameters=[
                {"name": "window_size", "type": "int", "range": [10, 100], "default": 50}
            ]
        )
    
    def initialize(self, config):
        self.state = {"initialized": True}
    
    def process_with_history(self, current_data, history, **kwargs):
        window_size = kwargs.get("window_size", 50)
        
        if len(history) < window_size:
            return {"status": "insufficient_data"}
        
        recent = history[-window_size:]
        states_array = jnp.array(recent)
        
        # Detect periodicity
        flat = states_array.reshape(len(states_array), -1)
        mean = jnp.mean(flat, axis=0)
        centered = flat - mean
        
        # Autocorrelation
        autocorr = []
        for lag in range(1, min(50, len(flat) // 2)):
            corr = jnp.mean(jnp.sum(centered[:-lag] * centered[lag:], axis=1))
            autocorr.append(float(corr))
        
        return {
            "status": "success",
            "autocorrelation": autocorr,
            "max_corr": float(jnp.max(jnp.array(autocorr)))
        }
```

## Plugin Metadata

### Required Fields
- `name`: Plugin name
- `version`: Semantic version (e.g., "1.0.0")
- `author`: Author name
- `description`: Brief description
- `category`: One of: algorithm, processor, analyzer, visualizer, composite
- `tags`: List of tags for categorization
- `parameters`: List of parameter definitions

### Parameter Definition
```python
{
    "name": "param_name",
    "type": "float",  # or "int", "bool", "string"
    "description": "Parameter description",
    "range": [min_val, max_val],
    "default": default_value,
    "required": True  # or False
}
```

## JAX Optimization

### Use JAX Functions
```python
import jax
import jax.numpy as jnp

def compute(self, input_signal, params, **kwargs):
    # Use JAX operations for GPU acceleration
    return jnp.tanh(input_signal * params[0])
```

### JIT Compilation
```python
def get_jax_function(self):
    @jax.jit
    def jax_compute(h, params):
        return self.compute(h, params)
    return jax_compute
```

## State Management

### Stateless Plugin
```python
def initialize(self, config):
    self.state = {"initialized": True}
```

### Stateful Plugin
```python
def __init__(self):
    super().__init__()
    self.history = []
    self.max_history_length = 100

def process_with_history(self, current_data, history, **kwargs):
    # Access history
    recent = history[-50:]
    # Process...
    return result
```

## Testing Plugins

```python
import pytest
from my_plugin import MyPlugin

def test_plugin_creation():
    plugin = MyPlugin()
    assert plugin is not None

def test_plugin_compute():
    plugin = MyPlugin()
    plugin.initialize({})
    
    input_signal = jnp.array(2.0)
    params = jnp.array([1.5])
    output = plugin.compute(input_signal, params)
    
    assert jnp.allclose(output, 3.0)
```

## Composite Plugins

### Pipeline
```python
from src.plugins.plugin_base import PluginPipeline

pipeline = PluginPipeline()
pipeline.add_plugin(Plugin1())
pipeline.add_plugin(Plugin2())
pipeline.add_plugin(Plugin3())

output = pipeline.process(input_data)
```

### Parallel
```python
from src.plugins.plugin_base import PluginParallel

parallel = PluginParallel()
parallel.add_plugin(Plugin1())
parallel.add_plugin(Plugin2())

outputs = parallel.process(input_data)  # Returns dict
```

## API Integration

### Register Plugin
```bash
# Automatic discovery
POST /plugins/discover

# Manual registration (metadata only)
POST /plugins/register
```

### Execute Plugin
```bash
POST /plugins/{plugin_id}/execute
{
  "input_data": [...],
  "parameters": {"gain": 1.5}
}
```

## Best Practices

1. **Use JAX**: Always use JAX operations for GPU acceleration
2. **Validate Parameters**: Check parameter ranges in `initialize()`
3. **Document**: Add docstrings to all methods
4. **Test**: Write unit tests for your plugin
5. **Error Handling**: Use try/except for robustness
6. **Performance**: Profile with `jax.profiler`
7. **Versioning**: Use semantic versioning

## Publishing Plugins

1. Create plugin file in `src/plugins/custom/`
2. Add tests in `tests/test_my_plugin.py`
3. Document in plugin docstring
4. Call `/plugins/discover` to register
5. Share via GitHub/PyPI

## Troubleshooting

### Plugin Not Found
- Check file location: `src/plugins/custom/`
- Ensure class inherits from plugin base
- Call `/plugins/discover`

### JAX Errors
- Use `jnp` instead of `np`
- Avoid Python loops, use `jax.lax.scan`
- Check array shapes

### Performance Issues
- Use `@jax.jit` decorator
- Avoid Python callbacks in compute
- Profile with `jax.profiler.trace()`

## See Also

- [Algorithm Reference](ALGORITHM_REFERENCE.md)
- [API Reference](API_REFERENCE.md)
- [Example Plugins](../src/plugins/examples/)

