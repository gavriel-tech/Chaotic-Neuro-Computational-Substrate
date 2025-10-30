"""
Node Configuration Management API.

Provides endpoints for saving, loading, and managing node configurations.
"""

from typing import List, Dict, Optional
from datetime import datetime
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.routes import NodeConfigurationRequest, NodeConfigurationResponse


# Create router
router = APIRouter(prefix="/configs", tags=["configurations"])

# Configuration storage directory
CONFIG_DIR = Path("saved_configs")
CONFIG_DIR.mkdir(exist_ok=True)


class NodeConfiguration:
    """Node configuration storage and management."""
    
    def __init__(self, config_id: str, data: Dict):
        """
        Initialize configuration.
        
        Args:
            config_id: Unique configuration ID
            data: Configuration data
        """
        self.config_id = config_id
        self.name = data.get("name", "Unnamed")
        self.description = data.get("description", "")
        self.nodes = data.get("nodes", [])
        self.modulation_routes = data.get("modulation_routes", [])
        self.created_at = data.get("created_at", datetime.now().isoformat())
        self.node_count = len(self.nodes)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "config_id": self.config_id,
            "name": self.name,
            "description": self.description,
            "nodes": self.nodes,
            "modulation_routes": self.modulation_routes,
            "created_at": self.created_at,
            "node_count": self.node_count
        }
    
    def save(self):
        """Save configuration to disk."""
        filepath = CONFIG_DIR / f"{self.config_id}.json"
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, config_id: str) -> 'NodeConfiguration':
        """Load configuration from disk."""
        filepath = CONFIG_DIR / f"{config_id}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration {config_id} not found")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(config_id, data)
    
    @classmethod
    def list_all(cls) -> List[Dict]:
        """List all saved configurations."""
        configs = []
        for filepath in CONFIG_DIR.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    configs.append({
                        "config_id": filepath.stem,
                        "name": data.get("name", "Unnamed"),
                        "description": data.get("description", ""),
                        "node_count": len(data.get("nodes", [])),
                        "created_at": data.get("created_at", "")
                    })
            except Exception as e:
                print(f"Error loading config {filepath}: {e}")
                continue
        
        # Sort by creation date (newest first)
        configs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return configs
    
    @classmethod
    def delete(cls, config_id: str) -> bool:
        """Delete configuration from disk."""
        filepath = CONFIG_DIR / f"{config_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False


@router.post("/save")
async def save_configuration(request: NodeConfigurationRequest):
    """
    Save current node configuration.
    
    Saves node positions, algorithms, parameters, and modulation routes.
    """
    # Generate unique ID
    config_id = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create configuration
    config_data = {
        "name": request.name,
        "description": request.description,
        "nodes": request.nodes,
        "modulation_routes": request.modulation_routes,
        "created_at": datetime.now().isoformat()
    }
    
    config = NodeConfiguration(config_id, config_data)
    config.save()
    
    return NodeConfigurationResponse(
        config_id=config_id,
        name=config.name,
        description=config.description,
        node_count=config.node_count,
        created_at=config.created_at
    )


@router.get("/list")
async def list_configurations():
    """List all saved configurations."""
    configs = NodeConfiguration.list_all()
    return {
        "configurations": configs,
        "total": len(configs)
    }


@router.get("/{config_id}")
async def get_configuration(config_id: str):
    """Get configuration details."""
    try:
        config = NodeConfiguration.load(config_id)
        return config.to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Configuration not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{config_id}/load")
async def load_configuration(config_id: str):
    """
    Load a saved configuration.
    
    Returns the configuration data to be applied by the client.
    """
    try:
        config = NodeConfiguration.load(config_id)
        return {
            "status": "success",
            "config": config.to_dict(),
            "message": f"Configuration '{config.name}' loaded"
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Configuration not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{config_id}")
async def delete_configuration(config_id: str):
    """Delete a saved configuration."""
    success = NodeConfiguration.delete(config_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    return {
        "status": "success",
        "message": f"Configuration {config_id} deleted"
    }


@router.post("/export")
async def export_configuration(config_id: str):
    """Export configuration as JSON file."""
    try:
        config = NodeConfiguration.load(config_id)
        return {
            "status": "success",
            "data": config.to_dict(),
            "filename": f"{config.name.replace(' ', '_')}.json"
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Configuration not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import")
async def import_configuration(data: Dict):
    """Import configuration from JSON data."""
    try:
        # Generate new ID
        config_id = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}_imported"
        
        # Ensure required fields
        if "name" not in data:
            data["name"] = "Imported Configuration"
        if "created_at" not in data:
            data["created_at"] = datetime.now().isoformat()
        
        config = NodeConfiguration(config_id, data)
        config.save()
        
        return NodeConfigurationResponse(
            config_id=config_id,
            name=config.name,
            description=config.description,
            node_count=config.node_count,
            created_at=config.created_at
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

