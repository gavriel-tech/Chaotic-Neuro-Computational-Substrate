"""
ML model persistence and versioning.
"""

import os
import json
from typing import Any, Optional, Dict
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ModelMetadata:
    """Metadata for saved models."""
    model_name: str
    model_type: str
    framework: str
    version: str
    training_steps: int
    training_time: float
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ModelSaver:
    """
    Save and load ML model checkpoints.
    
    Handles:
    - Model weights
    - Optimizer state
    - Training history
    - Model versioning
    """
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def save_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        optimizer_state: Optional[Any] = None
    ) -> str:
        """
        Save model with metadata.
        
        Args:
            model: Model object (PyTorch, TF, etc.)
            metadata: Model metadata
            optimizer_state: Optional optimizer state
            
        Returns:
            Path to saved model directory
        """
        # Create model directory
        model_dir = os.path.join(
            self.models_dir,
            f"{metadata.model_name}_{metadata.version}"
        )
        os.makedirs(model_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Save model weights (framework-specific)
        weights_path = os.path.join(model_dir, 'weights.pt')
        
        try:
            import torch
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), weights_path)
        except ImportError:
            pass
        
        # Save optimizer state if provided
        if optimizer_state is not None:
            optimizer_path = os.path.join(model_dir, 'optimizer.pt')
            try:
                import torch
                torch.save(optimizer_state, optimizer_path)
            except ImportError:
                pass
        
        print(f"[ModelSaver] Saved model: {model_dir}")
        return model_dir
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load model and metadata.
        
        Args:
            model_name: Model name
            version: Specific version (latest if None)
            
        Returns:
            Dict with weights_path, metadata, optimizer_path
        """
        if version:
            model_dir = os.path.join(self.models_dir, f"{model_name}_{version}")
        else:
            # Find latest version
            model_dir = self._find_latest_version(model_name)
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model not found: {model_name}")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Paths to weights and optimizer
        weights_path = os.path.join(model_dir, 'weights.pt')
        optimizer_path = os.path.join(model_dir, 'optimizer.pt')
        
        return {
            'model_dir': model_dir,
            'weights_path': weights_path if os.path.exists(weights_path) else None,
            'optimizer_path': optimizer_path if os.path.exists(optimizer_path) else None,
            'metadata': metadata
        }
    
    def list_models(self) -> list:
        """List all saved models."""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        for dirname in os.listdir(self.models_dir):
            model_dir = os.path.join(self.models_dir, dirname)
            metadata_path = os.path.join(model_dir, 'metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                models.append(metadata)
        
        return models
    
    def _find_latest_version(self, model_name: str) -> str:
        """Find latest version of a model."""
        matching_dirs = []
        
        for dirname in os.listdir(self.models_dir):
            if dirname.startswith(f"{model_name}_"):
                model_dir = os.path.join(self.models_dir, dirname)
                metadata_path = os.path.join(model_dir, 'metadata.json')
                
                if os.path.exists(metadata_path):
                    matching_dirs.append((model_dir, os.path.getmtime(metadata_path)))
        
        if not matching_dirs:
            raise FileNotFoundError(f"No versions found for model: {model_name}")
        
        # Sort by modification time (most recent first)
        matching_dirs.sort(key=lambda x: x[1], reverse=True)
        return matching_dirs[0][0]


_global_model_saver: Optional[ModelSaver] = None


def get_model_saver(models_dir: str = 'models') -> ModelSaver:
    """Get or create global model saver."""
    global _global_model_saver
    
    if _global_model_saver is None:
        _global_model_saver = ModelSaver(models_dir)
    
    return _global_model_saver

