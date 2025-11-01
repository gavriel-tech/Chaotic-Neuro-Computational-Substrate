"""
ML Model API Endpoints for GMCS.

REST API for ML model management, loading, inference, and training control.

Endpoints:
- GET /api/ml/models - List available models
- POST /api/ml/models/load - Load a model
- POST /api/ml/models/{model_id}/inference - Run inference
- GET /api/ml/models/{model_id}/status - Get model status
- DELETE /api/ml/models/{model_id} - Unload model
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import numpy as np

# Try to import ML modules
try:
    from ..ml.model_registry import get_registry, ModelMetadata
    from ..ml.ml_nodes import MLNode
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

router = APIRouter(prefix="/api/ml", tags=["ml"])

# In-memory model cache
loaded_models: Dict[str, Any] = {}

# ============================================================================
# Request/Response Models
# ============================================================================

class ModelLoadRequest(BaseModel):
    model_id: str
    device: str = "auto"
    config: Optional[Dict[str, Any]] = None


class ModelInferenceRequest(BaseModel):
    input_data: List[List[float]]  # 2D array as list of lists
    batch_size: Optional[int] = 1


class ModelSearchRequest(BaseModel):
    query: Optional[str] = None
    model_type: Optional[str] = None
    framework: Optional[str] = None
    task: Optional[str] = None


class ModelResponse(BaseModel):
    model_id: str
    name: str
    model_type: str
    framework: str
    task: str
    parameters: int
    status: str
    loaded: bool


class InferenceResponse(BaseModel):
    predictions: List[List[float]]
    model_id: str
    inference_time_ms: float


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/models")
async def list_models(
    model_type: Optional[str] = None,
    framework: Optional[str] = None,
    task: Optional[str] = None
):
    """
    List available models with optional filtering.
    
    Query parameters:
    - model_type: Filter by model type (transformer, diffusion, etc.)
    - framework: Filter by framework (pytorch, jax, tensorflow)
    - task: Filter by task (embedding, generation, etc.)
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML modules not available")
    
    try:
        registry = get_registry()
        models = registry.list_models(
            model_type=model_type,
            framework=framework,
            task=task
        )
        
        return {
            "models": [
                {
                    **metadata.to_dict(),
                    "loaded": metadata.model_id in loaded_models
                }
                for metadata in models
            ],
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/search")
async def search_models(query: str):
    """
    Search models by name, description, or tags.
    
    Query parameters:
    - query: Search query string
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML modules not available")
    
    try:
        registry = get_registry()
        models = registry.search_models(query)
        
        return {
            "models": [
                {
                    **metadata.to_dict(),
                    "loaded": metadata.model_id in loaded_models
                }
                for metadata in models
            ],
            "count": len(models),
            "query": query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/load")
async def load_model(request: ModelLoadRequest, background_tasks: BackgroundTasks):
    """
    Load a model into memory.
    
    Request body:
    - model_id: Model identifier
    - device: Device to load on ("cpu", "cuda", "auto")
    - config: Optional model configuration
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML modules not available")
    
    if request.model_id in loaded_models:
        return {
            "status": "already_loaded",
            "model_id": request.model_id,
            "message": f"Model {request.model_id} is already loaded"
        }
    
    try:
        registry = get_registry()
        
        # Load model
        model = registry.load_model(
            request.model_id,
            device=request.device,
            **(request.config or {})
        )
        
        # Cache loaded model
        loaded_models[request.model_id] = {
            "model": model,
            "device": request.device,
            "metadata": registry.get_model_metadata(request.model_id)
        }
        
        return {
            "status": "loaded",
            "model_id": request.model_id,
            "device": request.device,
            "message": f"Model {request.model_id} loaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.get("/models/{model_id}/status")
async def get_model_status(model_id: str):
    """
    Get status of a specific model.
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML modules not available")
    
    registry = get_registry()
    metadata = registry.get_model_metadata(model_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    is_loaded = model_id in loaded_models
    
    return {
        "model_id": model_id,
        "loaded": is_loaded,
        "metadata": metadata.to_dict(),
        "device": loaded_models[model_id]["device"] if is_loaded else None
    }


@router.post("/models/{model_id}/inference")
async def run_inference(model_id: str, request: ModelInferenceRequest):
    """
    Run inference on a loaded model.
    
    Request body:
    - input_data: Input data as 2D array (list of lists)
    - batch_size: Optional batch size for processing
    """
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML modules not available")
    
    if model_id not in loaded_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id} not loaded. Load it first with POST /api/ml/models/load"
        )
    
    try:
        import time
        start_time = time.time()
        
        # Get model
        model_data = loaded_models[model_id]
        model = model_data["model"]
        
        # Convert input to numpy
        input_array = np.array(request.input_data)
        
        # Run inference
        # Note: This assumes model has a forward() or __call__ method
        if hasattr(model, 'forward'):
            predictions = model.forward(input_array)
        elif callable(model):
            predictions = model(input_array)
        else:
            raise ValueError("Model does not have a callable inference method")
        
        # Convert predictions to list
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
        predictions = np.array(predictions).tolist()
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return {
            "predictions": predictions,
            "model_id": model_id,
            "inference_time_ms": inference_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.delete("/models/{model_id}")
async def unload_model(model_id: str):
    """
    Unload a model from memory.
    """
    if model_id not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not loaded")
    
    try:
        del loaded_models[model_id]
        
        return {
            "status": "unloaded",
            "model_id": model_id,
            "message": f"Model {model_id} unloaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@router.get("/models/loaded")
async def list_loaded_models():
    """
    List currently loaded models.
    """
    return {
        "loaded_models": [
            {
                "model_id": model_id,
                "device": data["device"],
                "metadata": data["metadata"].to_dict() if data["metadata"] else None
            }
            for model_id, data in loaded_models.items()
        ],
        "count": len(loaded_models)
    }


@router.get("/health")
async def health_check():
    """
    Health check for ML API.
    """
    return {
        "status": "healthy",
        "ml_available": ML_AVAILABLE,
        "loaded_models_count": len(loaded_models)
    }
