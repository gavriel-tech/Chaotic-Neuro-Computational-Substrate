"""
Rate limiting for GMCS API.

Prevents abuse by limiting request rates per user/API key.
"""

from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
import time
from collections import defaultdict
from dataclasses import dataclass, field
from fastapi import HTTPException, Request
import asyncio


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting."""
    capacity: int
    tokens: float
    last_update: float = field(default_factory=time.time)
    refill_rate: float = 1.0  # tokens per second
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if bucket is empty
        """
        # Refill tokens based on time elapsed
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + (elapsed * self.refill_rate)
        )
        self.last_update = now
        
        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until tokens will be available."""
        if self.tokens >= tokens:
            return 0.0
        
        needed = tokens - self.tokens
        return needed / self.refill_rate


class RateLimiter:
    """
    Rate limiter with token bucket algorithm.
    
    Supports per-user, per-IP, and per-API-key rate limiting.
    """
    
    def __init__(
        self,
        default_capacity: int = 100,
        default_refill_rate: float = 10.0,
        cleanup_interval: int = 3600
    ):
        """
        Initialize rate limiter.
        
        Args:
            default_capacity: Default bucket capacity
            default_refill_rate: Default refill rate (tokens per second)
            cleanup_interval: Cleanup interval for old buckets (seconds)
        """
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate
        self.cleanup_interval = cleanup_interval
        
        self._buckets: Dict[str, RateLimitBucket] = {}
        self._last_cleanup = time.time()
    
    def _cleanup(self):
        """Remove old unused buckets."""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return
        
        # Remove buckets not used in last hour
        cutoff = now - 3600
        to_remove = [
            key for key, bucket in self._buckets.items()
            if bucket.last_update < cutoff
        ]
        
        for key in to_remove:
            del self._buckets[key]
        
        self._last_cleanup = now
    
    def _get_bucket(
        self,
        key: str,
        capacity: Optional[int] = None,
        refill_rate: Optional[float] = None
    ) -> RateLimitBucket:
        """Get or create a rate limit bucket."""
        if key not in self._buckets:
            self._buckets[key] = RateLimitBucket(
                capacity=capacity or self.default_capacity,
                tokens=capacity or self.default_capacity,
                refill_rate=refill_rate or self.default_refill_rate
            )
        
        return self._buckets[key]
    
    def check_rate_limit(
        self,
        key: str,
        tokens: int = 1,
        capacity: Optional[int] = None,
        refill_rate: Optional[float] = None
    ) -> tuple[bool, Optional[float]]:
        """
        Check if request is within rate limit.
        
        Args:
            key: Unique key for rate limiting (user ID, IP, etc.)
            tokens: Number of tokens to consume
            capacity: Custom capacity for this key
            refill_rate: Custom refill rate for this key
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        self._cleanup()
        
        bucket = self._get_bucket(key, capacity, refill_rate)
        allowed = bucket.consume(tokens)
        
        if not allowed:
            retry_after = bucket.time_until_available(tokens)
            return False, retry_after
        
        return True, None
    
    def reset(self, key: str):
        """Reset rate limit for a key."""
        if key in self._buckets:
            del self._buckets[key]


# Global rate limiter
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def rate_limit(
    requests_per_minute: int = 60,
    key_func: Optional[Callable[[Request], str]] = None
):
    """
    Decorator for rate limiting endpoints.
    
    Args:
        requests_per_minute: Maximum requests per minute
        key_func: Optional function to extract rate limit key from request
    """
    def decorator(func):
        async def wrapper(*args, request: Request, **kwargs):
            limiter = get_rate_limiter()
            
            # Get rate limit key
            if key_func:
                key = key_func(request)
            else:
                # Default: use IP address
                key = request.client.host if request.client else "unknown"
            
            # Calculate tokens and rate
            capacity = requests_per_minute
            refill_rate = requests_per_minute / 60.0
            
            # Check rate limit
            allowed, retry_after = limiter.check_rate_limit(
                key=key,
                tokens=1,
                capacity=capacity,
                refill_rate=refill_rate
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Retry after {retry_after:.1f} seconds",
                    headers={"Retry-After": str(int(retry_after) + 1)}
                )
            
            return await func(*args, request=request, **kwargs)
        
        return wrapper
    
    return decorator


async def check_rate_limit_middleware(request: Request, call_next):
    """
    Middleware for rate limiting all requests.
    
    Can be added to FastAPI app:
        app.middleware("http")(check_rate_limit_middleware)
    """
    limiter = get_rate_limiter()
    
    # Use IP address as key
    key = request.client.host if request.client else "unknown"
    
    # Global rate limit: 1000 requests per minute per IP
    allowed, retry_after = limiter.check_rate_limit(
        key=f"global_{key}",
        capacity=1000,
        refill_rate=1000/60.0
    )
    
    if not allowed:
        return HTTPException(
            status_code=429,
            detail=f"Global rate limit exceeded. Retry after {retry_after:.1f} seconds",
            headers={"Retry-After": str(int(retry_after) + 1)}
        )
    
    response = await call_next(request)
    return response

