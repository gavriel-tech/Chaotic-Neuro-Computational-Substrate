"""
Training Control API Endpoints for GMCS.

REST API for controlling ML model training, monitoring progress,
and managing training sessions.

Endpoints:
- POST /api/training/start - Start training
- POST /api/training/stop - Stop training
- GET /api/training/status - Get training status
- GET /api/training/metrics - Get training metrics
- POST /api/training/pause - Pause training
- POST /api/training/resume - Resume training
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import json

router = APIRouter(prefix="/api/training", tags=["training"])

# Training state storage
training_sessions: Dict[str, Dict[str, Any]] = {}
active_websockets: List[WebSocket] = []

# ============================================================================
# Request/Response Models
# ============================================================================

class TrainingStartRequest(BaseModel):
    session_id: str
    model_id: str
    dataset_id: Optional[str] = None
    config: Dict[str, Any] = {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss_function": "mse"
    }


class TrainingStatusResponse(BaseModel):
    session_id: str
    status: str  # "running", "paused", "stopped", "completed"
    current_epoch: int
    total_epochs: int
    current_loss: Optional[float]
    metrics: Dict[str, Any]


class TrainingMetricsResponse(BaseModel):
    session_id: str
    history: List[Dict[str, Any]]


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/start")
async def start_training(request: TrainingStartRequest):
    """
    Start a new training session.
    
    Request body:
    - session_id: Unique session identifier
    - model_id: Model to train
    - dataset_id: Dataset to use (optional)
    - config: Training configuration (epochs, batch_size, lr, etc.)
    """
    if request.session_id in training_sessions:
        existing = training_sessions[request.session_id]
        if existing["status"] == "running":
            raise HTTPException(
                status_code=400,
                detail=f"Training session {request.session_id} is already running"
            )
    
    # Initialize training session
    training_sessions[request.session_id] = {
        "model_id": request.model_id,
        "dataset_id": request.dataset_id,
        "config": request.config,
        "status": "running",
        "current_epoch": 0,
        "total_epochs": request.config.get("epochs", 100),
        "metrics": {
            "loss": [],
            "val_loss": [],
            "accuracy": []
        },
        "start_time": asyncio.get_event_loop().time()
    }
    
    # Start training in background (simulated)
    asyncio.create_task(simulate_training(request.session_id))
    
    return {
        "status": "started",
        "session_id": request.session_id,
        "message": f"Training session {request.session_id} started"
    }


@router.post("/stop")
async def stop_training(session_id: str):
    """
    Stop a training session.
    
    Query parameters:
    - session_id: Session to stop
    """
    if session_id not in training_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Training session {session_id} not found"
        )
    
    session = training_sessions[session_id]
    session["status"] = "stopped"
    
    # Broadcast to WebSocket clients
    await broadcast_training_update(session_id, session)
    
    return {
        "status": "stopped",
        "session_id": session_id,
        "message": f"Training session {session_id} stopped"
    }


@router.post("/pause")
async def pause_training(session_id: str):
    """
    Pause a training session.
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = training_sessions[session_id]
    if session["status"] != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Session {session_id} is not running"
        )
    
    session["status"] = "paused"
    
    return {
        "status": "paused",
        "session_id": session_id
    }


@router.post("/resume")
async def resume_training(session_id: str):
    """
    Resume a paused training session.
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = training_sessions[session_id]
    if session["status"] != "paused":
        raise HTTPException(
            status_code=400,
            detail=f"Session {session_id} is not paused"
        )
    
    session["status"] = "running"
    asyncio.create_task(simulate_training(session_id))
    
    return {
        "status": "resumed",
        "session_id": session_id
    }


@router.get("/status/{session_id}")
async def get_training_status(session_id: str):
    """
    Get status of a training session.
    """
    if session_id not in training_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Training session {session_id} not found"
        )
    
    session = training_sessions[session_id]
    
    return {
        "session_id": session_id,
        "status": session["status"],
        "current_epoch": session["current_epoch"],
        "total_epochs": session["total_epochs"],
        "current_loss": session["metrics"]["loss"][-1] if session["metrics"]["loss"] else None,
        "config": session["config"]
    }


@router.get("/metrics/{session_id}")
async def get_training_metrics(session_id: str):
    """
    Get training metrics history.
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = training_sessions[session_id]
    
    # Format metrics as time series
    history = []
    for i in range(len(session["metrics"]["loss"])):
        history.append({
            "epoch": i + 1,
            "loss": session["metrics"]["loss"][i],
            "val_loss": session["metrics"]["val_loss"][i] if i < len(session["metrics"]["val_loss"]) else None,
            "accuracy": session["metrics"]["accuracy"][i] if i < len(session["metrics"]["accuracy"]) else None
        })
    
    return {
        "session_id": session_id,
        "history": history,
        "count": len(history)
    }


@router.get("/sessions")
async def list_training_sessions():
    """
    List all training sessions.
    """
    return {
        "sessions": [
            {
                "session_id": session_id,
                "model_id": data["model_id"],
                "status": data["status"],
                "current_epoch": data["current_epoch"],
                "total_epochs": data["total_epochs"]
            }
            for session_id, data in training_sessions.items()
        ],
        "count": len(training_sessions)
    }


@router.delete("/sessions/{session_id}")
async def delete_training_session(session_id: str):
    """
    Delete a training session.
    """
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = training_sessions[session_id]
    if session["status"] == "running":
        session["status"] = "stopped"
        await asyncio.sleep(0.1)  # Give time to stop
    
    del training_sessions[session_id]
    
    return {
        "status": "deleted",
        "session_id": session_id
    }


# ============================================================================
# WebSocket for Real-Time Updates
# ============================================================================

@router.websocket("/ws/{session_id}")
async def training_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time training updates.
    
    Sends training metrics as they're generated.
    """
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        # Send initial state
        if session_id in training_sessions:
            await websocket.send_json({
                "type": "initial_state",
                "data": training_sessions[session_id]
            })
        
        # Keep connection alive and receive client messages
        while True:
            data = await websocket.receive_text()
            # Echo back or handle client commands
            await websocket.send_text(f"Received: {data}")
            
    except WebSocketDisconnect:
        active_websockets.remove(websocket)


# ============================================================================
# Helper Functions
# ============================================================================

async def simulate_training(session_id: str):
    """
    Simulate training progress (for demonstration).
    In production, this would integrate with actual training loops.
    """
    import random
    
    if session_id not in training_sessions:
        return
    
    session = training_sessions[session_id]
    
    while session["status"] == "running" and session["current_epoch"] < session["total_epochs"]:
        await asyncio.sleep(0.5)  # Simulate epoch time
        
        if session["status"] != "running":
            break
        
        # Update metrics
        epoch = session["current_epoch"] + 1
        loss = 1.0 / epoch + random.random() * 0.1
        val_loss = 1.2 / epoch + random.random() * 0.1
        accuracy = min(0.95, epoch * 0.01)
        
        session["current_epoch"] = epoch
        session["metrics"]["loss"].append(loss)
        session["metrics"]["val_loss"].append(val_loss)
        session["metrics"]["accuracy"].append(accuracy)
        
        # Broadcast update
        await broadcast_training_update(session_id, session)
    
    if session["status"] == "running":
        session["status"] = "completed"
        await broadcast_training_update(session_id, session)


async def broadcast_training_update(session_id: str, session_data: Dict[str, Any]):
    """
    Broadcast training update to all connected WebSocket clients.
    """
    message = {
        "type": "training_update",
        "session_id": session_id,
        "data": {
            "status": session_data["status"],
            "current_epoch": session_data["current_epoch"],
            "total_epochs": session_data["total_epochs"],
            "latest_metrics": {
                "loss": session_data["metrics"]["loss"][-1] if session_data["metrics"]["loss"] else None,
                "val_loss": session_data["metrics"]["val_loss"][-1] if session_data["metrics"]["val_loss"] else None,
                "accuracy": session_data["metrics"]["accuracy"][-1] if session_data["metrics"]["accuracy"] else None
            }
        }
    }
    
    # Send to all connected clients
    dead_sockets = []
    for websocket in active_websockets:
        try:
            await websocket.send_json(message)
        except:
            dead_sockets.append(websocket)
    
    # Remove dead sockets
    for socket in dead_sockets:
        active_websockets.remove(socket)

