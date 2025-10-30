"""
ML Integration API Endpoints.

Provides REST API for ML model management and integration.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import jax.numpy as jnp

from src.ml.model_registry import get_global_registry


# Create router
router = APIRouter(prefix="/ml", tags=["machine-learning"])


class RegisterModelRequest(BaseModel):
    """Request to register a model."""
    model_type: str = Field(..., description="Model type (feedback, predictor, encoder)")
    name: str = Field(..., min_length=1)
    framework: str = Field(..., pattern="^(pytorch|tensorflow|huggingface)$")
    description: Optional[str] = ""
    tags: Optional[List[str]] = []
    model_path: Optional[str] = None


class ModelForwardRequest(BaseModel):
    """Request for model forward pass."""
    input_data: List[float] = Field(..., description="Input data as flat list")


class ModelTrainRequest(BaseModel):
    """Request for model training."""
    input_data: List[List[float]] = Field(..., description="Batch of input data")
    target_data: List[List[float]] = Field(..., description="Batch of target data")
    n_epochs: int = Field(10, ge=1, le=1000)
    learning_rate: float = Field(1e-3, gt=0, lt=1)


@router.get("/models/list")
async def list_ml_models(
    model_type: Optional[str] = None,
    framework: Optional[str] = None
):
    """
    List all registered ML models.
    
    Query params:
        - model_type: Filter by model type
        - framework: Filter by framework
    """
    registry = get_global_registry()
    models = registry.list_models(model_type=model_type, framework=framework)
    
    return {
        "models": models,
        "total": len(models)
    }


@router.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get detailed information about a model."""
    registry = get_global_registry()
    
    try:
        metadata = registry.get_metadata(model_id)
        model = registry.get_model(model_id)
        model_info = model.get_model_info()
        
        return {
            "metadata": metadata.to_dict(),
            "model_info": model_info
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/register")
async def register_ml_model(request: RegisterModelRequest):
    """
    Register a new ML model.
    
    Note: This endpoint registers metadata. The actual model should be
    loaded separately via the appropriate framework integration.
    """
    registry = get_global_registry()
    
    try:
        # For now, we'll create a placeholder
        # In production, you'd load the actual model from model_path
        model_id = f"{request.framework}_{request.model_type}_{request.name}"
        
        # Store metadata
        from src.ml.model_registry import ModelMetadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=request.model_type,
            name=request.name,
            description=request.description,
            framework=request.framework,
            tags=request.tags
        )
        
        registry.metadata[model_id] = metadata
        registry._save_registry()
        
        return {
            "status": "success",
            "model_id": model_id,
            "message": f"Model '{request.name}' registered"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/models/{model_id}/forward")
async def model_forward(model_id: str, request: ModelForwardRequest):
    """
    Run forward pass through a model.
    
    Args:
        model_id: Model identifier
        request: Input data
    """
    registry = get_global_registry()
    
    try:
        # Convert input to JAX array
        input_array = jnp.array(request.input_data)
        
        # Forward pass
        output = registry.forward(model_id, input_array)
        
        return {
            "status": "success",
            "output": output.tolist(),
            "output_shape": list(output.shape)
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_id}")
async def remove_ml_model(model_id: str):
    """Remove a model from the registry."""
    registry = get_global_registry()
    
    try:
        registry.remove_model(model_id)
        return {
            "status": "success",
            "message": f"Model {model_id} removed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/statistics")
async def get_registry_statistics():
    """Get ML model registry statistics."""
    registry = get_global_registry()
    stats = registry.get_statistics()
    
    return stats


@router.post("/models/batch-forward")
async def batch_forward(model_ids: List[str], input_data: List[float]):
    """
    Run forward pass through multiple models.
    
    Args:
        model_ids: List of model identifiers
        input_data: Input data
    """
    registry = get_global_registry()
    
    try:
        input_array = jnp.array(input_data)
        outputs = registry.batch_forward(model_ids, input_array)
        
        # Convert outputs to lists
        result = {}
        for model_id, output in outputs.items():
            if output is not None:
                result[model_id] = output.tolist()
            else:
                result[model_id] = None
        
        return {
            "status": "success",
            "outputs": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frameworks/available")
async def get_available_frameworks():
    """Get list of available ML frameworks."""
    from src.ml.pytorch_integration import PYTORCH_AVAILABLE
    from src.ml.tensorflow_integration import TENSORFLOW_AVAILABLE
    from src.ml.huggingface_integration import HUGGINGFACE_AVAILABLE
    
    return {
        "frameworks": {
            "pytorch": {
                "available": PYTORCH_AVAILABLE,
                "description": "PyTorch deep learning framework"
            },
            "tensorflow": {
                "available": TENSORFLOW_AVAILABLE,
                "description": "TensorFlow/Keras deep learning framework"
            },
            "huggingface": {
                "available": HUGGINGFACE_AVAILABLE,
                "description": "HuggingFace Transformers library"
            }
        }
    }


@router.get("/model-types")
async def get_model_types():
    """Get available model types and their descriptions."""
    return {
        "model_types": [
            {
                "type": "feedback",
                "description": "Models that provide feedback to GMCS parameters",
                "use_cases": ["Parameter optimization", "Adaptive control"]
            },
            {
                "type": "predictor",
                "description": "Models that predict future oscillator states",
                "use_cases": ["State prediction", "Anomaly detection"]
            },
            {
                "type": "encoder",
                "description": "Models that encode GMCS states into latent space",
                "use_cases": ["Dimensionality reduction", "Feature extraction"]
            },
            {
                "type": "generator",
                "description": "Models that generate GMCS configurations",
                "use_cases": ["Configuration synthesis", "Creative exploration"]
            },
            {
                "type": "classifier",
                "description": "Models that classify GMCS states or patterns",
                "use_cases": ["Pattern recognition", "State categorization"]
            }
        ]
    }


@router.post("/models/{model_id}/extract-features")
async def extract_features(
    model_id: str,
    input_data: List[float],
    layer_name: Optional[str] = None
):
    """
    Extract features from intermediate layer.
    
    Args:
        model_id: Model identifier
        input_data: Input data
        layer_name: Name of layer to extract from
    """
    registry = get_global_registry()
    
    try:
        model = registry.get_model(model_id)
        input_array = jnp.array(input_data)
        
        if layer_name and hasattr(model, 'extract_features'):
            features = model.extract_features(input_array, layer_name)
        else:
            # Just run forward pass
            features = model.forward(input_array)
        
        return {
            "status": "success",
            "features": features.tolist(),
            "feature_shape": list(features.shape)
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/integration/examples")
async def get_integration_examples():
    """Get examples of ML integration patterns."""
    return {
        "examples": [
            {
                "name": "Feedback Control",
                "description": "Use ML model to adjust GMCS parameters based on system state",
                "flow": "GMCS State → Model → Parameter Updates → GMCS"
            },
            {
                "name": "State Prediction",
                "description": "Predict future oscillator states for anticipatory control",
                "flow": "Historical States → Model → Predicted State → Preemptive Adjustment"
            },
            {
                "name": "Feature Extraction",
                "description": "Extract high-level features for external processing",
                "flow": "GMCS State → Model → Features → External System"
            },
            {
                "name": "Pattern Recognition",
                "description": "Classify current system behavior",
                "flow": "GMCS State → Model → Pattern Label → Conditional Logic"
            },
            {
                "name": "Data Augmentation",
                "description": "Use GMCS as data augmentation for training other models",
                "flow": "Training Data → GMCS → Augmented Data → Target Model"
            }
        ]
    }

