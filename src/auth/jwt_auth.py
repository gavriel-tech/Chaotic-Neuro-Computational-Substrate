"""
JWT authentication for GMCS.

Handles JWT token creation, validation, and user authentication.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
import secrets


# Security configuration
SECRET_KEY = os.getenv("GMCS_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


class TokenData(BaseModel):
    """Token payload data."""
    user_id: str
    username: str
    email: Optional[str] = None
    roles: list = []
    permissions: list = []


class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int


class JWTAuth:
    """JWT authentication handler."""
    
    def __init__(
        self,
        secret_key: str = SECRET_KEY,
        algorithm: str = ALGORITHM,
        access_token_expire_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES
    ):
        """
        Initialize JWT auth.
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm (default: HS256)
            access_token_expire_minutes: Access token expiration in minutes
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.security = HTTPBearer()
    
    def create_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT token.
        
        Args:
            data: Data to encode in token
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a refresh token with longer expiration."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
    ) -> TokenData:
        """
        Get current authenticated user from token.
        
        Args:
            credentials: HTTP authorization credentials
            
        Returns:
            TokenData with user information
        """
        token = credentials.credentials
        payload = self.verify_token(token)
        
        # Verify token type
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        return TokenData(
            user_id=payload.get("sub"),
            username=payload.get("username"),
            email=payload.get("email"),
            roles=payload.get("roles", []),
            permissions=payload.get("permissions", [])
        )


# Global JWT auth instance
_jwt_auth = JWTAuth()


def create_access_token(
    user_id: str,
    username: str,
    email: Optional[str] = None,
    roles: list = None,
    permissions: list = None,
    expires_delta: Optional[timedelta] = None
) -> Token:
    """
    Create an access token for a user.
    
    Args:
        user_id: User ID
        username: Username
        email: User email
        roles: User roles
        permissions: User permissions
        expires_delta: Optional custom expiration
        
    Returns:
        Token response with access and refresh tokens
    """
    data = {
        "sub": user_id,
        "username": username,
        "email": email,
        "roles": roles or [],
        "permissions": permissions or []
    }
    
    access_token = _jwt_auth.create_token(data, expires_delta)
    refresh_token = _jwt_auth.create_refresh_token(data)
    
    expires_in = expires_delta.total_seconds() if expires_delta else ACCESS_TOKEN_EXPIRE_MINUTES * 60
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(expires_in)
    )


def verify_token(token: str) -> Dict[str, Any]:
    """Verify a JWT token."""
    return _jwt_auth.verify_token(token)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
) -> TokenData:
    """Get current authenticated user."""
    return await _jwt_auth.get_current_user(credentials)


def require_auth(func):
    """Decorator to require authentication."""
    async def wrapper(*args, current_user: TokenData = Depends(get_current_user), **kwargs):
        return await func(*args, current_user=current_user, **kwargs)
    return wrapper

