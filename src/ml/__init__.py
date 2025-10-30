"""
Machine Learning Integration Module for GMCS.

Provides integration with PyTorch, TensorFlow, and HuggingFace models.
"""

from src.ml.model_registry import (
    ModelRegistry,
    ModelMetadata,
    get_global_registry,
    register_model,
    get_model,
    list_models
)

# Try to import framework integrations
try:
    from src.ml.pytorch_integration import (
        PyTorchModelWrapper,
        GMCSPyTorchTrainer,
        create_gmcs_feedback_model,
        create_oscillator_predictor,
        PYTORCH_AVAILABLE
    )
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from src.ml.tensorflow_integration import (
        TensorFlowModelWrapper,
        GMCSTensorFlowTrainer,
        create_gmcs_feedback_model_tf,
        create_oscillator_predictor_tf,
        TENSORFLOW_AVAILABLE
    )
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from src.ml.huggingface_integration import (
        HuggingFaceModelWrapper,
        GMCSTextInterface,
        GMCSPatternRecognizer,
        HUGGINGFACE_AVAILABLE
    )
except ImportError:
    HUGGINGFACE_AVAILABLE = False


__all__ = [
    # Registry
    'ModelRegistry',
    'ModelMetadata',
    'get_global_registry',
    'register_model',
    'get_model',
    'list_models',
    # Availability flags
    'PYTORCH_AVAILABLE',
    'TENSORFLOW_AVAILABLE',
    'HUGGINGFACE_AVAILABLE',
]

# Add framework-specific exports if available
if PYTORCH_AVAILABLE:
    __all__.extend([
        'PyTorchModelWrapper',
        'GMCSPyTorchTrainer',
        'create_gmcs_feedback_model',
        'create_oscillator_predictor',
    ])

if TENSORFLOW_AVAILABLE:
    __all__.extend([
        'TensorFlowModelWrapper',
        'GMCSTensorFlowTrainer',
        'create_gmcs_feedback_model_tf',
        'create_oscillator_predictor_tf',
    ])

if HUGGINGFACE_AVAILABLE:
    __all__.extend([
        'HuggingFaceModelWrapper',
        'GMCSTextInterface',
        'GMCSPatternRecognizer',
    ])

