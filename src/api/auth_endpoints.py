"""
Authentication API endpoints for GMCS.

Provides login, registration, token management, and API key operations.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import timedelta
import hashlib
import secrets

from src.auth import (
    create_access_token,
    get_current_user,
    TokenData,
    Token,
    get_api_key_manager,
    APIKey,
    validate_api_key,
    rate_limit
)
from src.auth.permissions import Permission, Role, check_permission


router = APIRouter(prefix="/auth", tags=["authentication"])


# ============================================
# Request/Response Models
# ============================================

class UserRegister(BaseModel):
    """User registration request."""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """User login request."""
    username: str
    password: str


class TokenRefresh(BaseModel):
    """Token refresh request."""
    refresh_token: str


class APIKeyCreate(BaseModel):
    """API key creation request."""
    name: str
    permissions: List[str] = []
    expires_in_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    """API key creation response."""
    key_id: str
    api_key: str
    name: str
    created_at: str
    expires_at: Optional[str]
    permissions: List[str]


class UserInfo(BaseModel):
    """User information response."""
    user_id: str
    username: str
    email: Optional[str]
    roles: List[str]
    permissions: List[str]


# ============================================
# In-Memory User Store (Replace with Database)
# ============================================

# TEMPORARY: In production, use a real database
_users_db = {
    "admin": {
        "user_id": "1",
        "username": "admin",
        "email": "admin@gmcs.local",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "roles": ["admin"],
        "full_name": "Administrator"
    },
    "demo": {
        "user_id": "2",
        "username": "demo",
        "email": "demo@gmcs.local",
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "roles": ["user"],
        "full_name": "Demo User"
    }
}


def hash_password(password: str) -> str:
    """Hash a password."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(plain_password) == hashed_password


def get_user(username: str) -> Optional[dict]:
    """Get user from database."""
    return _users_db.get(username)


# ============================================
# Authentication Endpoints
# ============================================

@router.post("/register", response_model=Token, status_code=201)
async def register(user_data: UserRegister, request: Request):
    """
    Register a new user.
    
    Creates a new user account and returns access tokens.
    """
    # Check if username exists
    if user_data.username in _users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create user
    user_id = str(len(_users_db) + 1)
    _users_db[user_data.username] = {
        "user_id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "password_hash": hash_password(user_data.password),
        "roles": ["user"],  # Default role
        "full_name": user_data.full_name
    }
    
    # Create access token
    token = create_access_token(
        user_id=user_id,
        username=user_data.username,
        email=user_data.email,
        roles=["user"],
        permissions=[p.value for p in Role.__members__.get("USER", Role.USER) if hasattr(Role, "USER")]
    )
    
    return token


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, request: Request):
    """
    Login with username and password.
    
    Returns access and refresh tokens.
    """
    # Get user
    user = get_user(credentials.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Verify password
    if not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create access token
    token = create_access_token(
        user_id=user["user_id"],
        username=user["username"],
        email=user.get("email"),
        roles=user.get("roles", ["user"])
    )
    
    return token


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_data: TokenRefresh):
    """
    Refresh access token using refresh token.
    
    Returns a new access token.
    """
    from src.auth.jwt_auth import verify_token
    
    try:
        # Verify refresh token
        payload = verify_token(refresh_data.refresh_token)
        
        # Check token type
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        # Create new access token
        token = create_access_token(
            user_id=payload["sub"],
            username=payload["username"],
            email=payload.get("email"),
            roles=payload.get("roles", [])
        )
        
        return token
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid refresh token: {str(e)}")


@router.get("/me", response_model=UserInfo)
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    Requires valid JWT token.
    """
    return UserInfo(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        roles=current_user.roles,
        permissions=current_user.permissions
    )


@router.post("/logout")
async def logout(current_user: TokenData = Depends(get_current_user)):
    """
    Logout current user.
    
    Note: With JWT, logout is typically handled client-side by discarding the token.
    Server-side logout requires token blacklisting (not implemented here).
    """
    return {"message": "Logged out successfully", "username": current_user.username}


# ============================================
# API Key Management
# ============================================

@router.post("/api-keys", response_model=APIKeyResponse, status_code=201)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Create a new API key.
    
    Requires authentication and API_KEY_CREATE permission.
    """
    # Check permission
    if not check_permission(current_user.roles, Permission.API_KEY_CREATE):
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Generate API key
    manager = get_api_key_manager()
    raw_key, api_key = manager.generate_key(
        name=key_data.name,
        permissions=key_data.permissions,
        expires_in_days=key_data.expires_in_days,
        metadata={"created_by": current_user.username}
    )
    
    return APIKeyResponse(
        key_id=api_key.key_id,
        api_key=raw_key,  # Only returned once!
        name=api_key.name,
        created_at=api_key.created_at.isoformat(),
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
        permissions=api_key.permissions
    )


@router.get("/api-keys", response_model=List[APIKey])
async def list_api_keys(current_user: TokenData = Depends(get_current_user)):
    """
    List all API keys for current user.
    
    Requires authentication.
    """
    manager = get_api_key_manager()
    return manager.list_keys()


@router.get("/api-keys/{key_id}", response_model=APIKey)
async def get_api_key(
    key_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Get details of a specific API key.
    
    Requires authentication.
    """
    manager = get_api_key_manager()
    api_key = manager.get_key(key_id)
    
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return api_key


@router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(
    key_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Revoke an API key.
    
    Requires authentication and API_KEY_REVOKE permission.
    """
    # Check permission
    if not check_permission(current_user.roles, Permission.API_KEY_REVOKE):
        raise HTTPException(status_code=403, detail="Permission denied")
    
    manager = get_api_key_manager()
    success = manager.revoke_key(key_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return None


@router.delete("/api-keys/{key_id}/permanent", status_code=204)
async def delete_api_key(
    key_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Permanently delete an API key.
    
    Requires authentication and admin role.
    """
    # Check if admin
    if "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    manager = get_api_key_manager()
    success = manager.delete_key(key_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return None


# ============================================
# Test Endpoints (Development Only)
# ============================================

@router.get("/test/protected")
async def test_protected_endpoint(current_user: TokenData = Depends(get_current_user)):
    """Test endpoint that requires JWT authentication."""
    return {
        "message": "Access granted",
        "user": current_user.username,
        "roles": current_user.roles
    }


@router.get("/test/api-key")
async def test_api_key_endpoint(api_key: APIKey = Depends(validate_api_key)):
    """Test endpoint that requires API key authentication."""
    return {
        "message": "API key valid",
        "key_name": api_key.name,
        "permissions": api_key.permissions
    }


@router.get("/test/public")
async def test_public_endpoint():
    """Test endpoint with no authentication required."""
    return {"message": "Public access"}


# ============================================
# Password Management (Future)
# ============================================

class PasswordChange(BaseModel):
    """Password change request."""
    old_password: str
    new_password: str


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Change user password.
    
    Requires authentication.
    """
    # Get user
    user = get_user(current_user.username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Verify old password
    if not verify_password(password_data.old_password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid current password")
    
    # Update password
    user["password_hash"] = hash_password(password_data.new_password)
    
    return {"message": "Password updated successfully"}


class PasswordReset(BaseModel):
    """Password reset request."""
    email: EmailStr


@router.post("/reset-password")
async def reset_password_request(reset_data: PasswordReset):
    """
    Request password reset.
    
    Sends reset link to email (not implemented - would need email service).
    """
    # TODO: Implement email sending
    return {
        "message": "If the email exists, a reset link has been sent",
        "note": "Email sending not yet implemented"
    }

