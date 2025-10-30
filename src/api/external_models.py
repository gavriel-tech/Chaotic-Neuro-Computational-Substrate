"""
External Model Connection API.

Provides endpoints for registering and managing external ML models
and GPU connections.
"""

from typing import List, Dict, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import jax
import jax.numpy as jnp


# Create router
router = APIRouter(prefix="/external", tags=["external"])


class ExternalModel:
    """External ML model registration."""
    
    def __init__(self, model_id: str, model_type: str, metadata: Dict):
        """
        Initialize external model.
        
        Args:
            model_id: Unique model ID
            model_type: Model type (pytorch, tensorflow, huggingface)
            metadata: Model metadata
        """
        self.model_id = model_id
        self.model_type = model_type
        self.name = metadata.get("name", "Unnamed Model")
        self.description = metadata.get("description", "")
        self.input_shape = metadata.get("input_shape", [])
        self.output_shape = metadata.get("output_shape", [])
        self.registered_at = datetime.now().isoformat()
        self.last_output = None
        self.is_connected = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "name": self.name,
            "description": self.description,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "registered_at": self.registered_at,
            "is_connected": self.is_connected
        }


class RegisterModelRequest(BaseModel):
    """Request to register external model."""
    model_type: str = Field(..., pattern="^(pytorch|tensorflow|huggingface)$")
    name: str = Field(..., min_length=1)
    description: Optional[str] = ""
    input_shape: List[int] = Field(default_factory=list)
    output_shape: List[int] = Field(default_factory=list)


class UpdateModelOutputRequest(BaseModel):
    """Request to update model output."""
    output: List[float] = Field(..., description="Model output values")


# Global model registry
model_registry: Dict[str, ExternalModel] = {}


@router.post("/models/register")
async def register_model(request: RegisterModelRequest):
    """
    Register an external ML model.
    
    Once registered, the model can send outputs to the modulation matrix.
    """
    # Generate unique ID
    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create model
    metadata = {
        "name": request.name,
        "description": request.description,
        "input_shape": request.input_shape,
        "output_shape": request.output_shape
    }
    
    model = ExternalModel(model_id, request.model_type, metadata)
    model_registry[model_id] = model
    
    return {
        "status": "success",
        "model_id": model_id,
        "model": model.to_dict()
    }


@router.get("/models/list")
async def list_models():
    """List all registered external models."""
    models = [model.to_dict() for model in model_registry.values()]
    return {
        "models": models,
        "total": len(models)
    }


@router.post("/models/{model_id}/update")
async def update_model_output(model_id: str, request: UpdateModelOutputRequest):
    """
    Update output from an external model.
    
    This output can be used as a modulation source.
    """
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = model_registry[model_id]
    model.last_output = jnp.array(request.output)
    
    return {
        "status": "success",
        "model_id": model_id,
        "output_shape": len(request.output)
    }


@router.delete("/models/{model_id}")
async def disconnect_model(model_id: str):
    """Disconnect and remove an external model."""
    if model_id not in model_registry:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del model_registry[model_id]
    
    return {
        "status": "success",
        "message": f"Model {model_id} disconnected"
    }


@router.get("/gpus/list")
async def list_gpus():
    """
    List available GPUs/accelerators.
    
    Returns information about JAX-visible devices.
    """
    from src.core.multi_gpu import get_device_info
    
    device_info = get_device_info()
    
    return {
        "devices": device_info["devices"],
        "n_devices": device_info["n_devices"],
        "default_backend": device_info["default_backend"]
    }


@router.post("/gpus/{device_id}/connect")
async def connect_gpu(device_id: int):
    """
    Connect to a specific GPU device.
    
    Note: This is informational - JAX manages device allocation automatically.
    """
    from src.core.multi_gpu import detect_devices
    
    devices = detect_devices()
    
    if device_id < 0 or device_id >= len(devices):
        raise HTTPException(status_code=404, detail="Device not found")
    
    device = devices[device_id]
    
    return {
        "status": "success",
        "device_id": device_id,
        "device_info": {
            "platform": device.platform,
            "device_kind": device.device_kind
        },
        "message": f"Connected to device {device_id}"
    }


@router.get("/gpus/benchmark")
async def benchmark_gpus(n_devices: Optional[int] = None, n_steps: int = 100):
    """
    Benchmark multi-GPU performance.
    
    Args:
        n_devices: Number of devices to benchmark (None = all)
        n_steps: Number of simulation steps
    """
    from src.core.multi_gpu import benchmark_multi_gpu
    
    try:
        results = benchmark_multi_gpu(
            n_devices=n_devices or jax.device_count(),
            n_steps=n_steps
        )
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

