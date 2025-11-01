"""
Permission and role management for GMCS.

Defines roles, permissions, and access control.
"""

from enum import Enum
from typing import List, Set
from dataclasses import dataclass


class Permission(Enum):
    """System permissions."""
    # Preset permissions
    PRESET_READ = "preset:read"
    PRESET_WRITE = "preset:write"
    PRESET_DELETE = "preset:delete"
    PRESET_SHARE = "preset:share"
    
    # Node permissions
    NODE_CREATE = "node:create"
    NODE_UPDATE = "node:update"
    NODE_DELETE = "node:delete"
    NODE_EXECUTE = "node:execute"
    
    # System permissions
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"
    
    # API key permissions
    API_KEY_CREATE = "api_key:create"
    API_KEY_REVOKE = "api_key:revoke"
    API_KEY_LIST = "api_key:list"
    
    # User management
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"
    
    # Monitoring
    METRICS_READ = "metrics:read"
    HEALTH_READ = "health:read"
    
    # ML permissions
    MODEL_LOAD = "model:load"
    MODEL_TRAIN = "model:train"
    MODEL_EXPORT = "model:export"


class Role(Enum):
    """Predefined user roles with permission sets."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API = "api"


# Role to permissions mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        # All permissions
        Permission.PRESET_READ,
        Permission.PRESET_WRITE,
        Permission.PRESET_DELETE,
        Permission.PRESET_SHARE,
        Permission.NODE_CREATE,
        Permission.NODE_UPDATE,
        Permission.NODE_DELETE,
        Permission.NODE_EXECUTE,
        Permission.SYSTEM_READ,
        Permission.SYSTEM_WRITE,
        Permission.SYSTEM_ADMIN,
        Permission.API_KEY_CREATE,
        Permission.API_KEY_REVOKE,
        Permission.API_KEY_LIST,
        Permission.USER_READ,
        Permission.USER_WRITE,
        Permission.USER_DELETE,
        Permission.METRICS_READ,
        Permission.HEALTH_READ,
        Permission.MODEL_LOAD,
        Permission.MODEL_TRAIN,
        Permission.MODEL_EXPORT,
    },
    Role.USER: {
        # Standard user permissions
        Permission.PRESET_READ,
        Permission.PRESET_WRITE,
        Permission.PRESET_DELETE,
        Permission.PRESET_SHARE,
        Permission.NODE_CREATE,
        Permission.NODE_UPDATE,
        Permission.NODE_DELETE,
        Permission.NODE_EXECUTE,
        Permission.SYSTEM_READ,
        Permission.METRICS_READ,
        Permission.HEALTH_READ,
        Permission.MODEL_LOAD,
        Permission.MODEL_TRAIN,
    },
    Role.VIEWER: {
        # Read-only permissions
        Permission.PRESET_READ,
        Permission.SYSTEM_READ,
        Permission.METRICS_READ,
        Permission.HEALTH_READ,
    },
    Role.API: {
        # API access permissions
        Permission.PRESET_READ,
        Permission.NODE_EXECUTE,
        Permission.SYSTEM_READ,
        Permission.METRICS_READ,
        Permission.HEALTH_READ,
    }
}


def get_role_permissions(role: Role) -> Set[Permission]:
    """Get all permissions for a role."""
    return ROLE_PERMISSIONS.get(role, set())


def check_permission(user_roles: List[str], required_permission: Permission) -> bool:
    """
    Check if user has required permission based on their roles.
    
    Args:
        user_roles: List of role names the user has
        required_permission: Permission to check
        
    Returns:
        True if user has permission, False otherwise
    """
    for role_name in user_roles:
        try:
            role = Role(role_name)
            permissions = get_role_permissions(role)
            if required_permission in permissions:
                return True
        except ValueError:
            continue
    
    return False


def require_permission(permission: Permission):
    """
    Decorator to require specific permission for an endpoint.
    
    Usage:
        @router.get("/admin")
        @require_permission(Permission.SYSTEM_ADMIN)
        async def admin_endpoint(current_user: TokenData = Depends(get_current_user)):
            ...
    """
    from fastapi import HTTPException, Depends
    from .jwt_auth import get_current_user, TokenData
    
    def decorator(func):
        async def wrapper(*args, current_user: TokenData = Depends(get_current_user), **kwargs):
            if not check_permission(current_user.roles, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied. Required: {permission.value}"
                )
            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator

