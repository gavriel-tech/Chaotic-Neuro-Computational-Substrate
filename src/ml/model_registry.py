"""
Model Registry for GMCS.

Central registry for pre-trained models, providing easy access to
models from HuggingFace Hub, local storage, and custom definitions.

Key features:
- Model catalog with metadata
- HuggingFace Hub integration
- Local model management
- Model versioning
- Automatic downloading
- Model search and filtering

Use cases:
- Browse available models
- Load pre-trained models
- Register custom models
- Version management
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import os

try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


# ============================================================================
# Model Metadata
# ============================================================================

@dataclass
class ModelMetadata:
    """
    Metadata for a registered model.
    """
    model_id: str
    name: str
    model_type: str  # 'transformer', 'diffusion', 'gan', 'supervised', 'rl'
    framework: str  # 'pytorch', 'tensorflow', 'jax'
    task: str  # 'classification', 'generation', 'embedding', etc.
    
    # Model details
    architecture: str  # 'bert', 'gpt2', 'resnet', etc.
    parameters: int  # Number of parameters
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    
    # Source
    source: str = 'huggingface'  # 'huggingface', 'local', 'custom'
    hub_id: Optional[str] = None  # HuggingFace Hub ID
    local_path: Optional[str] = None
    
    # Training
    trained_on: Optional[str] = None  # Dataset name
    training_details: Optional[Dict[str, Any]] = None
    
    # Performance
    metrics: Optional[Dict[str, float]] = None
    
    # Metadata
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    version: str = '1.0.0'
    author: Optional[str] = None
    license: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(**data)


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """
    Central registry for ML models.
    
    Manages catalog of available models, provides search and loading functionality.
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry file (default: ./models/registry.json)
        """
        if registry_path is None:
            registry_path = os.path.join(os.getcwd(), 'models', 'registry.json')
        
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load registry
        self.models: Dict[str, ModelMetadata] = {}
        self._load_registry()
        
        # Initialize with default models
        if not self.models:
            self._initialize_default_models()
    
    def _load_registry(self):
        """Load registry from file."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                self.models = {
                    model_id: ModelMetadata.from_dict(metadata)
                    for model_id, metadata in data.items()
                }
    
    def _save_registry(self):
        """Save registry to file."""
        data = {
            model_id: metadata.to_dict()
            for model_id, metadata in self.models.items()
        }
        
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _initialize_default_models(self):
        """Initialize with default models."""
        
        # BERT models
        self.register_model(ModelMetadata(
            model_id='bert-base',
            name='BERT Base',
            model_type='transformer',
            framework='pytorch',
            task='embedding',
            architecture='bert',
            parameters=110_000_000,
            source='huggingface',
            hub_id='bert-base-uncased',
            description='Base BERT model for text embeddings',
            tags=['nlp', 'embedding', 'bert'],
            license='apache-2.0'
        ))
        
        self.register_model(ModelMetadata(
            model_id='bert-large',
            name='BERT Large',
            model_type='transformer',
            framework='pytorch',
            task='embedding',
            architecture='bert',
            parameters=340_000_000,
            source='huggingface',
            hub_id='bert-large-uncased',
            description='Large BERT model for text embeddings',
            tags=['nlp', 'embedding', 'bert'],
            license='apache-2.0'
        ))
        
        # GPT models
        self.register_model(ModelMetadata(
            model_id='gpt2',
            name='GPT-2',
            model_type='transformer',
            framework='pytorch',
            task='generation',
            architecture='gpt2',
            parameters=117_000_000,
            source='huggingface',
            hub_id='gpt2',
            description='GPT-2 for text generation',
            tags=['nlp', 'generation', 'gpt'],
            license='mit'
        ))
        
        self.register_model(ModelMetadata(
            model_id='gpt2-medium',
            name='GPT-2 Medium',
            model_type='transformer',
            framework='pytorch',
            task='generation',
            architecture='gpt2',
            parameters=345_000_000,
            source='huggingface',
            hub_id='gpt2-medium',
            description='Medium GPT-2 for text generation',
            tags=['nlp', 'generation', 'gpt'],
            license='mit'
        ))
        
        # T5 models
        self.register_model(ModelMetadata(
            model_id='t5-small',
            name='T5 Small',
            model_type='transformer',
            framework='pytorch',
            task='seq2seq',
            architecture='t5',
            parameters=60_000_000,
            source='huggingface',
            hub_id='t5-small',
            description='Small T5 model for sequence-to-sequence',
            tags=['nlp', 'seq2seq', 't5'],
            license='apache-2.0'
        ))
        
        # DistilBERT (fast)
        self.register_model(ModelMetadata(
            model_id='distilbert',
            name='DistilBERT',
            model_type='transformer',
            framework='pytorch',
            task='embedding',
            architecture='distilbert',
            parameters=66_000_000,
            source='huggingface',
            hub_id='distilbert-base-uncased',
            description='Distilled BERT for fast embeddings',
            tags=['nlp', 'embedding', 'fast'],
            license='apache-2.0'
        ))
        
        # Sentence transformers
        self.register_model(ModelMetadata(
            model_id='sentence-transformer',
            name='Sentence Transformer',
            model_type='transformer',
            framework='pytorch',
            task='embedding',
            architecture='sentence-bert',
            parameters=110_000_000,
            source='huggingface',
            hub_id='sentence-transformers/all-MiniLM-L6-v2',
            description='Sentence embeddings optimized for similarity',
            tags=['nlp', 'embedding', 'similarity'],
            license='apache-2.0'
        ))
        
        self._save_registry()
    
    def register_model(self, metadata: ModelMetadata):
        """
        Register a model.
        
        Args:
            metadata: Model metadata
        """
        self.models[metadata.model_id] = metadata
        self._save_registry()
    
    def unregister_model(self, model_id: str):
        """
        Unregister a model.
        
        Args:
            model_id: Model ID to remove
        """
        if model_id in self.models:
            del self.models[model_id]
            self._save_registry()
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model metadata or None
        """
        return self.models.get(model_id)
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        framework: Optional[str] = None,
        task: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelMetadata]:
        """
        List models with optional filtering.
        
        Args:
            model_type: Filter by model type
            framework: Filter by framework
            task: Filter by task
            tags: Filter by tags (any match)
            
        Returns:
            List of matching models
        """
        results = []
        
        for metadata in self.models.values():
            # Apply filters
            if model_type and metadata.model_type != model_type:
                continue
            if framework and metadata.framework != framework:
                continue
            if task and metadata.task != task:
                continue
            if tags and not any(tag in (metadata.tags or []) for tag in tags):
                continue
            
            results.append(metadata)
        
        return results
    
    def search_models(self, query: str) -> List[ModelMetadata]:
        """
        Search models by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching models
        """
        query_lower = query.lower()
        results = []
        
        for metadata in self.models.values():
            # Search in name
            if query_lower in metadata.name.lower():
                results.append(metadata)
                continue
            
            # Search in description
            if metadata.description and query_lower in metadata.description.lower():
                results.append(metadata)
                continue
            
            # Search in tags
            if metadata.tags and any(query_lower in tag.lower() for tag in metadata.tags):
                results.append(metadata)
                continue
        
        return results
    
    def load_model(
        self,
        model_id: str,
        device: str = 'auto',
        **kwargs
    ) -> Any:
        """
        Load a model.
        
        Args:
            model_id: Model ID
            device: Device to load on
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model
        """
        metadata = self.get_model_metadata(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Load based on source
        if metadata.source == 'huggingface':
            return self._load_from_huggingface(metadata, device, **kwargs)
        elif metadata.source == 'local':
            return self._load_from_local(metadata, device, **kwargs)
        else:
            raise ValueError(f"Unknown source: {metadata.source}")
    
    def _load_from_huggingface(
        self,
        metadata: ModelMetadata,
        device: str = 'auto',
        **kwargs
    ) -> Any:
        """Load model from HuggingFace Hub."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers")
        
        if metadata.hub_id is None:
            raise ValueError(f"Model {metadata.model_id} has no hub_id")
        
        # Determine device
        if device == 'auto':
            device = 'cuda' if PYTORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
        # Load model based on task
        if metadata.task == 'generation':
            model = AutoModelForCausalLM.from_pretrained(metadata.hub_id, **kwargs)
        else:
            model = AutoModel.from_pretrained(metadata.hub_id, **kwargs)
        
        if PYTORCH_AVAILABLE:
            model = model.to(device)
        
        return model
    
    def _load_from_local(
        self,
        metadata: ModelMetadata,
        device: str = 'auto',
        **kwargs
    ) -> Any:
        """Load model from local path."""
        if metadata.local_path is None:
            raise ValueError(f"Model {metadata.model_id} has no local_path")
        
        path = Path(metadata.local_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load based on framework
        if metadata.framework == 'pytorch' and PYTORCH_AVAILABLE:
            model = torch.load(path, map_location=device)
            return model
        else:
            raise NotImplementedError(f"Loading {metadata.framework} models not yet implemented")
    
    def download_model(self, model_id: str, save_path: Optional[str] = None):
        """
        Download model to local storage.
        
        Args:
            model_id: Model ID
            save_path: Path to save (default: ./models/{model_id})
        """
        metadata = self.get_model_metadata(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")
        
        if save_path is None:
            save_path = os.path.join('models', model_id)
        
        # Create directory
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Download from source
        if metadata.source == 'huggingface':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers required")
            
            # Download model and tokenizer
            model = AutoModel.from_pretrained(metadata.hub_id)
            tokenizer = AutoTokenizer.from_pretrained(metadata.hub_id)
            
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            print(f"Downloaded {model_id} to {save_path}")
        else:
            raise ValueError(f"Cannot download from source: {metadata.source}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Statistics dict
        """
        total_models = len(self.models)
        
        # Count by type
        by_type = {}
        by_framework = {}
        by_task = {}
        
        for metadata in self.models.values():
            by_type[metadata.model_type] = by_type.get(metadata.model_type, 0) + 1
            by_framework[metadata.framework] = by_framework.get(metadata.framework, 0) + 1
            by_task[metadata.task] = by_task.get(metadata.task, 0) + 1
        
        return {
            'total_models': total_models,
            'by_type': by_type,
            'by_framework': by_framework,
            'by_task': by_task
        }


# ============================================================================
# Global Registry Instance
# ============================================================================

_global_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """
    Get global model registry instance.
    
    Returns:
        ModelRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
    return _global_registry


# Alias for backwards compatibility with existing code
get_global_registry = get_registry


# ============================================================================
# Convenience Functions
# ============================================================================

def list_models(**kwargs) -> List[ModelMetadata]:
    """List models from global registry."""
    return get_registry().list_models(**kwargs)


def search_models(query: str) -> List[ModelMetadata]:
    """Search models in global registry."""
    return get_registry().search_models(query)


def load_model(model_id: str, **kwargs) -> Any:
    """Load model from global registry."""
    return get_registry().load_model(model_id, **kwargs)


def register_model(metadata: ModelMetadata):
    """Register model in global registry."""
    get_registry().register_model(metadata)


def get_model(model_id: str) -> Optional[ModelMetadata]:
    """Get model metadata from global registry."""
    return get_registry().get_model_metadata(model_id)


if __name__ == '__main__':
    # Example usage
    print("Model Registry Demo\n" + "=" * 50)
    
    # Get registry
    registry = get_registry()
    
    # List all models
    print("\n1. All models:")
    for model in registry.list_models():
        print(f"   - {model.model_id}: {model.name} ({model.parameters:,} params)")
    
    # Search models
    print("\n2. Search for 'bert':")
    results = registry.search_models('bert')
    for model in results:
        print(f"   - {model.model_id}: {model.name}")
    
    # Filter by type
    print("\n3. Transformer models:")
    transformers = registry.list_models(model_type='transformer')
    print(f"   Found {len(transformers)} transformer models")
    
    # Get statistics
    print("\n4. Registry statistics:")
    stats = registry.get_statistics()
    print(f"   Total models: {stats['total_models']}")
    print(f"   By type: {stats['by_type']}")
    
    # Load a model (if transformers available)
    if TRANSFORMERS_AVAILABLE:
        print("\n5. Loading DistilBERT...")
        try:
            model = registry.load_model('distilbert')
            print(f"   ✓ Loaded successfully!")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print("\n✓ Registry demo complete!")
