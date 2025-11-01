"""
API key management for GMCS.

Provides API key generation, validation, and management.
"""

from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
import secrets
import hashlib
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
import json
from pathlib import Path


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKey(BaseModel):
    """API key model."""
    key_id: str
    key_hash: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    permissions: List[str] = []
    metadata: Dict[str, Any] = {}
    is_active: bool = True
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIKeyManager:
    """
    Manages API keys for GMCS.
    
    Stores API keys securely with hashing.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize API key manager.
        
        Args:
            storage_path: Path to store API keys (default: data/api_keys.json)
        """
        self.storage_path = storage_path or Path("data/api_keys.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.keys: Dict[str, APIKey] = {}
        self._load_keys()
    
    def _hash_key(self, key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _load_keys(self):
        """Load API keys from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    for key_data in data:
                        api_key = APIKey(**key_data)
                        self.keys[api_key.key_id] = api_key
            except Exception as e:
                print(f"Warning: Could not load API keys: {e}")
    
    def _save_keys(self):
        """Save API keys to storage."""
        try:
            data = [key.dict() for key in self.keys.values()]
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error: Could not save API keys: {e}")
    
    def generate_key(
        self,
        name: str,
        permissions: List[str] = None,
        expires_in_days: Optional[int] = None,
        metadata: Dict[str, Any] = None
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.
        
        Args:
            name: Name/description for the key
            permissions: List of permissions
            expires_in_days: Optional expiration in days
            metadata: Optional metadata
            
        Returns:
            Tuple of (raw_key, APIKey object)
        """
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        key_hash = self._hash_key(raw_key)
        key_id = secrets.token_urlsafe(16)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            permissions=permissions or [],
            metadata=metadata or {}
        )
        
        # Store key
        self.keys[key_id] = api_key
        self._save_keys()
        
        return raw_key, api_key
    
    def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key.
        
        Args:
            raw_key: Raw API key to validate
            
        Returns:
            APIKey object if valid, None otherwise
        """
        key_hash = self._hash_key(raw_key)
        
        # Find key by hash
        for api_key in self.keys.values():
            if api_key.key_hash == key_hash:
                # Check if active
                if not api_key.is_active:
                    return None
                
                # Check expiration
                if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                    return None
                
                # Update last used
                api_key.last_used = datetime.utcnow()
                self._save_keys()
                
                return api_key
        
        return None
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: Key ID to revoke
            
        Returns:
            True if revoked, False if not found
        """
        if key_id in self.keys:
            self.keys[key_id].is_active = False
            self._save_keys()
            return True
        return False
    
    def delete_key(self, key_id: str) -> bool:
        """
        Delete an API key permanently.
        
        Args:
            key_id: Key ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if key_id in self.keys:
            del self.keys[key_id]
            self._save_keys()
            return True
        return False
    
    def list_keys(self) -> List[APIKey]:
        """List all API keys (without raw keys)."""
        return list(self.keys.values())
    
    def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get an API key by ID."""
        return self.keys.get(key_id)


# Global API key manager
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def generate_api_key(
    name: str,
    permissions: List[str] = None,
    expires_in_days: Optional[int] = None
) -> tuple[str, APIKey]:
    """Generate a new API key."""
    manager = get_api_key_manager()
    return manager.generate_key(name, permissions, expires_in_days)


async def validate_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> APIKey:
    """
    Validate API key from header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        Validated APIKey object
        
    Raises:
        HTTPException: If key is invalid or missing
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    manager = get_api_key_manager()
    validated_key = manager.validate_key(api_key)
    
    if not validated_key:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    
    return validated_key

