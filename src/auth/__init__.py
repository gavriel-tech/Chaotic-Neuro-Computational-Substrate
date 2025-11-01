"""
Authentication and security for GMCS.

Provides JWT authentication, API key management, and rate limiting.
"""

from .jwt_auth import JWTAuth, create_access_token, verify_token, get_current_user, TokenData, Token
from .api_keys import APIKeyManager, APIKey, generate_api_key, get_api_key_manager, validate_api_key
from .rate_limiting import RateLimiter, rate_limit
from .permissions import Permission, Role, check_permission

__all__ = [
    "JWTAuth",
    "create_access_token",
    "verify_token",
    "get_current_user",
    "TokenData",
    "Token",
    "APIKeyManager",
    "APIKey",
    "generate_api_key",
    "get_api_key_manager",
    "validate_api_key",
    "RateLimiter",
    "rate_limit",
    "Permission",
    "Role",
    "check_permission",
]

