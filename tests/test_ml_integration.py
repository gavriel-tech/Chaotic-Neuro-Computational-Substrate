"""
Tests for ML Integration Module.

Tests PyTorch, TensorFlow, and HuggingFace integration.
"""

import pytest
import jax.numpy as jnp
import numpy as np

# Import with availability checks
try:
    from src.ml.pytorch_integration import PyTorchModelWrapper, PYTORCH_AVAILABLE
    from src.ml.tensorflow_integration import TensorFlowModelWrapper, TENSORFLOW_AVAILABLE
    from src.ml.huggingface_integration import HuggingFaceModelWrapper, HUGGINGFACE_AVAILABLE
    from src.ml.model_registry import ModelRegistry, get_global_registry
except ImportError as e:
    pytest.skip(f"ML modules not available: {e}", allow_module_level=True)


class TestPyTorchIntegration:
    """Test PyTorch integration."""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_wrapper_creation(self):
        """Test creating PyTorch wrapper."""
        wrapper = PyTorchModelWrapper()
        assert wrapper is not None
        assert wrapper.device in ['cuda', 'cpu']
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_jax_torch_conversion(self):
        """Test JAX <-> PyTorch conversion."""
        wrapper = PyTorchModelWrapper()
        
        # JAX -> PyTorch
        jax_array = jnp.array([1.0, 2.0, 3.0])
        torch_tensor = wrapper.jax_to_torch(jax_array)
        assert torch_tensor.shape == (3,)
        
        # PyTorch -> JAX
        jax_back = wrapper.torch_to_jax(torch_tensor)
        assert jnp.allclose(jax_array, jax_back)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_simple_model_forward(self):
        """Test forward pass through simple model."""
        import torch.nn as nn
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )
        
        wrapper = PyTorchModelWrapper(model)
        
        # Forward pass
        input_data = jnp.ones(10)
        output = wrapper.forward(input_data)
        
        assert output.shape == (3,)
        assert isinstance(output, jnp.ndarray)


class TestTensorFlowIntegration:
    """Test TensorFlow integration."""
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_tensorflow_wrapper_creation(self):
        """Test creating TensorFlow wrapper."""
        wrapper = TensorFlowModelWrapper()
        assert wrapper is not None
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_jax_tf_conversion(self):
        """Test JAX <-> TensorFlow conversion."""
        wrapper = TensorFlowModelWrapper()
        
        # JAX -> TensorFlow
        jax_array = jnp.array([1.0, 2.0, 3.0])
        tf_tensor = wrapper.jax_to_tf(jax_array)
        assert tf_tensor.shape == (3,)
        
        # TensorFlow -> JAX
        jax_back = wrapper.tf_to_jax(tf_tensor)
        assert jnp.allclose(jax_array, jax_back)
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_keras_model_forward(self):
        """Test forward pass through Keras model."""
        from tensorflow import keras
        
        # Create simple model
        model = keras.Sequential([
            keras.layers.Dense(5, activation='relu', input_shape=(10,)),
            keras.layers.Dense(3)
        ])
        
        wrapper = TensorFlowModelWrapper(model)
        
        # Forward pass
        input_data = jnp.ones(10)
        output = wrapper.forward(input_data)
        
        assert output.shape == (3,)
        assert isinstance(output, jnp.ndarray)


class TestModelRegistry:
    """Test model registry."""
    
    def test_registry_creation(self):
        """Test creating registry."""
        registry = ModelRegistry()
        assert registry is not None
        assert len(registry.plugins) == 0
    
    def test_global_registry(self):
        """Test global registry singleton."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        assert registry1 is registry2
    
    def test_registry_statistics(self):
        """Test registry statistics."""
        registry = ModelRegistry()
        stats = registry.get_statistics()
        
        assert 'total_models' in stats
        assert 'by_framework' in stats
        assert 'by_type' in stats


class TestMLIntegrationAPI:
    """Test ML integration API endpoints."""
    
    def test_model_registry_operations(self):
        """Test basic registry operations."""
        registry = ModelRegistry()
        
        # Initially empty
        models = registry.list_models()
        assert len(models) == 0
        
        # Get statistics
        stats = registry.get_statistics()
        assert stats['total_models'] == 0


def test_ml_module_imports():
    """Test that ML modules can be imported."""
    from src.ml import (
        ModelRegistry,
        get_global_registry,
        PYTORCH_AVAILABLE,
        TENSORFLOW_AVAILABLE,
        HUGGINGFACE_AVAILABLE
    )
    
    assert ModelRegistry is not None
    assert get_global_registry is not None
    assert isinstance(PYTORCH_AVAILABLE, bool)
    assert isinstance(TENSORFLOW_AVAILABLE, bool)
    assert isinstance(HUGGINGFACE_AVAILABLE, bool)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

