"""
ML Model Registry for GMCS.

Centralized registry for managing multiple ML models, their metadata,
and integration with the GMCS simulation.
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import json

import jax.numpy as jnp

from src.ml.pytorch_integration import PyTorchModelWrapper, PYTORCH_AVAILABLE
from src.ml.tensorflow_integration import TensorFlowModelWrapper, TENSORFLOW_AVAILABLE
from src.ml.huggingface_integration import HuggingFaceModelWrapper, HUGGINGFACE_AVAILABLE


class ModelMetadata:
    """Metadata for a registered model."""
    
    def __init__(
        self,
        model_id: str,
        model_type: str,
        name: str,
        description: str = "",
        framework: str = "pytorch",
        input_shape: Optional[List[int]] = None,
        output_shape: Optional[List[int]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize model metadata.
        
        Args:
            model_id: Unique model identifier
            model_type: Type of model (feedback, predictor, encoder, etc.)
            name: Human-readable name
            description: Model description
            framework: ML framework (pytorch, tensorflow, huggingface)
            input_shape: Expected input shape
            output_shape: Expected output shape
            tags: List of tags for categorization
        """
        self.model_id = model_id
        self.model_type = model_type
        self.name = name
        self.description = description
        self.framework = framework
        self.input_shape = input_shape or []
        self.output_shape = output_shape or []
        self.tags = tags or []
        self.created_at = datetime.now().isoformat()
        self.last_used = None
        self.usage_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "name": self.name,
            "description": self.description,
            "framework": self.framework,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "tags": self.tags,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "usage_count": self.usage_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        metadata = cls(
            model_id=data["model_id"],
            model_type=data["model_type"],
            name=data["name"],
            description=data.get("description", ""),
            framework=data.get("framework", "pytorch"),
            input_shape=data.get("input_shape"),
            output_shape=data.get("output_shape"),
            tags=data.get("tags")
        )
        metadata.created_at = data.get("created_at", metadata.created_at)
        metadata.last_used = data.get("last_used")
        metadata.usage_count = data.get("usage_count", 0)
        return metadata


class ModelRegistry:
    """
    Central registry for ML models in GMCS.
    
    Manages model lifecycle, metadata, and provides unified interface
    for different ML frameworks.
    """
    
    def __init__(self, registry_dir: str = "model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory to store registry data
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk."""
        metadata_file = self.registry_dir / "registry.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                for model_id, meta_dict in data.items():
                    self.metadata[model_id] = ModelMetadata.from_dict(meta_dict)
    
    def _save_registry(self):
        """Save registry to disk."""
        metadata_file = self.registry_dir / "registry.json"
        data = {
            model_id: meta.to_dict()
            for model_id, meta in self.metadata.items()
        }
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        model: Any,
        model_type: str,
        name: str,
        framework: str = "pytorch",
        description: str = "",
        tags: Optional[List[str]] = None,
        model_id: Optional[str] = None
    ) -> str:
        """
        Register a new model.
        
        Args:
            model: Model instance (PyTorch, TensorFlow, or HuggingFace)
            model_type: Type of model
            name: Model name
            framework: Framework type
            description: Description
            tags: Tags
            model_id: Optional custom model ID
            
        Returns:
            Model ID
        """
        # Generate ID if not provided
        if model_id is None:
            model_id = f"{framework}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create wrapper based on framework
        if framework == "pytorch" and PYTORCH_AVAILABLE:
            wrapper = PyTorchModelWrapper(model)
        elif framework == "tensorflow" and TENSORFLOW_AVAILABLE:
            wrapper = TensorFlowModelWrapper(model)
        elif framework == "huggingface" and HUGGINGFACE_AVAILABLE:
            wrapper = model  # Already wrapped
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        # Get model info
        info = wrapper.get_model_info()
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=model_type,
            name=name,
            description=description,
            framework=framework,
            input_shape=info.get("input_shape"),
            output_shape=info.get("output_shape"),
            tags=tags
        )
        
        # Store
        self.models[model_id] = wrapper
        self.metadata[model_id] = metadata
        
        # Save registry
        self._save_registry()
        
        return model_id
    
    def get_model(self, model_id: str) -> Any:
        """
        Get model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model wrapper
        """
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found in registry")
        
        # Update usage stats
        metadata = self.metadata[model_id]
        metadata.last_used = datetime.now().isoformat()
        metadata.usage_count += 1
        self._save_registry()
        
        return self.models[model_id]
    
    def get_metadata(self, model_id: str) -> ModelMetadata:
        """Get model metadata."""
        if model_id not in self.metadata:
            raise KeyError(f"Model {model_id} not found in registry")
        return self.metadata[model_id]
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        framework: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List registered models with optional filtering.
        
        Args:
            model_type: Filter by model type
            framework: Filter by framework
            tags: Filter by tags (any match)
            
        Returns:
            List of model metadata dictionaries
        """
        models = []
        
        for model_id, metadata in self.metadata.items():
            # Apply filters
            if model_type and metadata.model_type != model_type:
                continue
            if framework and metadata.framework != framework:
                continue
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            models.append(metadata.to_dict())
        
        # Sort by last used (most recent first)
        models.sort(
            key=lambda x: x.get("last_used") or "",
            reverse=True
        )
        
        return models
    
    def remove_model(self, model_id: str):
        """
        Remove model from registry.
        
        Args:
            model_id: Model identifier
        """
        if model_id in self.models:
            del self.models[model_id]
        if model_id in self.metadata:
            del self.metadata[model_id]
        
        self._save_registry()
    
    def forward(self, model_id: str, input_data: jnp.ndarray) -> jnp.ndarray:
        """
        Run forward pass through a model.
        
        Args:
            model_id: Model identifier
            input_data: Input data
            
        Returns:
            Model output
        """
        model = self.get_model(model_id)
        return model.forward(input_data)
    
    def batch_forward(
        self,
        model_ids: List[str],
        input_data: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """
        Run forward pass through multiple models.
        
        Args:
            model_ids: List of model identifiers
            input_data: Input data
            
        Returns:
            Dictionary mapping model IDs to outputs
        """
        outputs = {}
        for model_id in model_ids:
            try:
                outputs[model_id] = self.forward(model_id, input_data)
            except Exception as e:
                print(f"Error running model {model_id}: {e}")
                outputs[model_id] = None
        
        return outputs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        stats = {
            "total_models": len(self.models),
            "by_framework": {},
            "by_type": {},
            "most_used": None,
            "recently_added": []
        }
        
        # Count by framework and type
        for metadata in self.metadata.values():
            framework = metadata.framework
            model_type = metadata.model_type
            
            stats["by_framework"][framework] = stats["by_framework"].get(framework, 0) + 1
            stats["by_type"][model_type] = stats["by_type"].get(model_type, 0) + 1
        
        # Most used
        if self.metadata:
            most_used = max(
                self.metadata.values(),
                key=lambda m: m.usage_count
            )
            stats["most_used"] = {
                "model_id": most_used.model_id,
                "name": most_used.name,
                "usage_count": most_used.usage_count
            }
        
        # Recently added
        recent = sorted(
            self.metadata.values(),
            key=lambda m: m.created_at,
            reverse=True
        )[:5]
        stats["recently_added"] = [
            {"model_id": m.model_id, "name": m.name, "created_at": m.created_at}
            for m in recent
        ]
        
        return stats
    
    def export_model_info(self, model_id: str, output_path: str):
        """
        Export model information to JSON file.
        
        Args:
            model_id: Model identifier
            output_path: Output file path
        """
        metadata = self.get_metadata(model_id)
        model = self.get_model(model_id)
        
        info = {
            "metadata": metadata.to_dict(),
            "model_info": model.get_model_info()
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_global_registry() -> ModelRegistry:
    """Get or create global model registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


def register_model(*args, **kwargs) -> str:
    """Convenience function to register model in global registry."""
    return get_global_registry().register_model(*args, **kwargs)


def get_model(model_id: str) -> Any:
    """Convenience function to get model from global registry."""
    return get_global_registry().get_model(model_id)


def list_models(*args, **kwargs) -> List[Dict[str, Any]]:
    """Convenience function to list models in global registry."""
    return get_global_registry().list_models(*args, **kwargs)

