"""
API Endpoints for Preset Management.

Provides REST API for browsing, loading, and saving GMCS application presets.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path

router = APIRouter(prefix="/api/presets", tags=["presets"])

# ============================================================================
# Global State
# ============================================================================

# Active preset executors (preset_id -> executor)
_active_executors = {}

# ============================================================================
# Configuration
# ============================================================================

# Preset directories
PRESET_DIR = Path(__file__).parent.parent.parent / "frontend" / "presets"
USER_PRESET_DIR = Path.home() / ".gmcs" / "user_presets"

# Ensure user preset directory exists
USER_PRESET_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Models
# ============================================================================

class PresetNode(BaseModel):
    id: str
    type: str
    name: str
    position: Dict[str, float]
    config: Dict[str, Any]


class PresetConnection(BaseModel):
    from_: Optional[str] = Field(default=None, alias='from')
    to: str
    description: Optional[str] = None

    model_config = {'populate_by_name': True}


class PresetControl(BaseModel):
    name: str
    label: str
    node: str
    field: str
    type: str
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[Any] = None
    default: Optional[Any] = None


class Preset(BaseModel):
    id: str
    name: str
    version: str
    description: str
    category: str
    tags: List[str]
    author: str
    created: str
    nodes: List[PresetNode]
    connections: List[PresetConnection]
    controls: Optional[Dict[str, Any]] = None
    requirements: Optional[Dict[str, Any]] = None
    documentation: Optional[Dict[str, Any]] = None
    initialState: Optional[Dict[str, Any]] = None


class PresetSummary(BaseModel):
    """Lightweight preset summary for browsing."""
    id: str
    name: str
    description: str
    category: str
    tags: List[str]
    node_count: int
    connection_count: int


# ============================================================================
# Helper Functions
# ============================================================================

def load_preset_file(file_path: Path) -> Optional[Preset]:
    """Load a preset from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Preset(**data)
    except Exception as e:
        print(f"Error loading preset from {file_path}: {e}")
        return None


def get_all_preset_files() -> List[Path]:
    """Get all preset JSON files from both system and user directories."""
    files = []
    
    # System presets
    if PRESET_DIR.exists():
        files.extend(PRESET_DIR.glob("*.json"))
    
    # User presets
    if USER_PRESET_DIR.exists():
        files.extend(USER_PRESET_DIR.glob("*.json"))
    
    return files


def preset_to_summary(preset: Preset) -> PresetSummary:
    """Convert full preset to lightweight summary."""
    return PresetSummary(
        id=preset.id,
        name=preset.name,
        description=preset.description,
        category=preset.category,
        tags=preset.tags,
        node_count=len(preset.nodes),
        connection_count=len(preset.connections)
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/")
async def list_presets(
    category: Optional[str] = None,
    tag: Optional[str] = None
) -> Dict[str, List[PresetSummary]]:
    """
    List all available presets.
    
    Query params:
        category: Filter by category (AI/ML, Creative, Scientific)
        tag: Filter by tag
    
    Returns:
        Dictionary with "presets" key containing list of preset summaries
    """
    preset_files = get_all_preset_files()
    presets = []
    
    for file_path in preset_files:
        preset = load_preset_file(file_path)
        if preset is None:
            continue
        
        # Apply filters
        if category and preset.category != category:
            continue
        if tag and tag not in preset.tags:
            continue
        
        presets.append(preset_to_summary(preset))
    
    # Sort by name
    presets.sort(key=lambda p: p.name)
    
    return {"presets": presets}


@router.get("/{preset_id}")
async def get_preset(preset_id: str) -> Preset:
    """
    Get full preset details by ID.
    
    Args:
        preset_id: Preset identifier
    
    Returns:
        Complete preset configuration
    
    Raises:
        HTTPException: If preset not found
    """
    # Search in both directories
    for preset_dir in [PRESET_DIR, USER_PRESET_DIR]:
        file_path = preset_dir / f"{preset_id}.json"
        if file_path.exists():
            preset = load_preset_file(file_path)
            if preset:
                return preset
    
    raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")


@router.post("/{preset_id}/load")
async def load_preset(preset_id: str, preset: Optional[Preset] = None) -> Dict[str, Any]:
    """
    Load a preset into the system.
    
    This endpoint:
    1. Validates the preset
    2. Creates all nodes
    3. Establishes all connections
    4. Applies initial state
    5. Returns the loaded node graph
    
    Args:
        preset_id: Preset identifier
        preset: Optional full preset (if not provided, loads from file)
    
    Returns:
        Status and loaded node graph info
    """
    # Load preset if not provided
    if preset is None:
        preset = await get_preset(preset_id)
    
    # Validate preset
    if not preset.nodes:
        raise HTTPException(status_code=400, detail="Preset has no nodes")
    
    # Load preset into executor
    try:
        from src.nodes.node_executor import PresetExecutor
        
        executor = PresetExecutor()
        executor.load_preset(preset.dict())
        
        # Store executor in global registry (simplified - would use proper state management)
        _active_executors[preset.id] = executor
        
        return {
            "status": "success",
            "message": f"Preset '{preset.name}' loaded successfully",
            "nodes_created": len(preset.nodes),
            "connections_made": len(preset.connections),
            "preset_id": preset.id,
            "executor_ready": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load preset: {str(e)}"
        )


@router.post("/")
async def save_preset(preset: Preset) -> Dict[str, str]:
    """
    Save a new preset.
    
    Saves to user preset directory (~/.gmcs/user_presets/).
    
    Args:
        preset: Complete preset configuration
    
    Returns:
        Status message and saved preset ID
    
    Raises:
        HTTPException: If preset ID already exists or save fails
    """
    # Check if preset ID already exists
    file_path = USER_PRESET_DIR / f"{preset.id}.json"
    if file_path.exists():
        raise HTTPException(
            status_code=400, 
            detail=f"Preset with ID '{preset.id}' already exists"
        )
    
    # Save preset
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(preset.dict(), f, indent=2)
        
        return {
            "status": "success",
            "message": f"Preset '{preset.name}' saved successfully",
            "preset_id": preset.id,
            "file_path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save preset: {str(e)}"
        )


@router.put("/{preset_id}")
async def update_preset(preset_id: str, preset: Preset) -> Dict[str, str]:
    """
    Update an existing preset.
    
    Only user presets can be updated (not system presets).
    
    Args:
        preset_id: Preset identifier
        preset: Updated preset configuration
    
    Returns:
        Status message
    
    Raises:
        HTTPException: If preset not found or is a system preset
    """
    # Check if it's a user preset
    file_path = USER_PRESET_DIR / f"{preset_id}.json"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"User preset '{preset_id}' not found. System presets cannot be modified."
        )
    
    # Save updated preset
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(preset.dict(), f, indent=2)
        
        return {
            "status": "success",
            "message": f"Preset '{preset.name}' updated successfully",
            "preset_id": preset.id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update preset: {str(e)}"
        )


@router.delete("/{preset_id}")
async def delete_preset(preset_id: str) -> Dict[str, str]:
    """
    Delete a user preset.
    
    System presets cannot be deleted.
    
    Args:
        preset_id: Preset identifier
    
    Returns:
        Status message
    
    Raises:
        HTTPException: If preset not found or is a system preset
    """
    # Check if it's a user preset
    file_path = USER_PRESET_DIR / f"{preset_id}.json"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"User preset '{preset_id}' not found. System presets cannot be deleted."
        )
    
    # Delete preset
    try:
        os.remove(file_path)
        return {
            "status": "success",
            "message": f"Preset '{preset_id}' deleted successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete preset: {str(e)}"
        )


@router.post("/{preset_id}/export")
async def export_preset(preset_id: str) -> Preset:
    """
    Export a preset (returns full JSON for download).
    
    Args:
        preset_id: Preset identifier
    
    Returns:
        Complete preset configuration
    """
    return await get_preset(preset_id)


@router.post("/import")
async def import_preset(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Import a preset from uploaded JSON file.
    
    Args:
        file: Uploaded JSON file
    
    Returns:
        Status and imported preset ID
    
    Raises:
        HTTPException: If file is invalid or import fails
    """
    try:
        # Read file content
        content = await file.read()
        data = json.loads(content)
        
        # Validate and create preset
        preset = Preset(**data)
        
        # Save to user presets
        return await save_preset(preset)
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to import preset: {str(e)}"
        )


@router.get("/categories/list")
async def list_categories() -> Dict[str, List[str]]:
    """
    Get list of all preset categories.
    
    Returns:
        Dictionary with "categories" key containing list of category names
    """
    preset_files = get_all_preset_files()
    categories = set()
    
    for file_path in preset_files:
        preset = load_preset_file(file_path)
        if preset:
            categories.add(preset.category)
    
    return {"categories": sorted(list(categories))}


@router.get("/tags/list")
async def list_tags() -> Dict[str, List[str]]:
    """
    Get list of all preset tags.
    
    Returns:
        Dictionary with "tags" key containing list of unique tags
    """
    preset_files = get_all_preset_files()
    tags = set()
    
    for file_path in preset_files:
        preset = load_preset_file(file_path)
        if preset:
            tags.update(preset.tags)
    
    return {"tags": sorted(list(tags))}

