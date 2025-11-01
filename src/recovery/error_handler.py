"""
Error handling with automatic retry and circuit breaker.
"""

import time
import functools
from typing import Optional, Callable, Any, List, Type
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Retry Configuration
# ============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    exceptions: List[Type[Exception]] = None
    
    def __post_init__(self):
        if self.exceptions is None:
            self.exceptions = [Exception]


# ============================================================================
# Circuit Breaker
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, block requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.
    
    Prevents cascading failures by blocking requests after threshold.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before trying half-open
            success_threshold: Successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerOpen("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.last_failure_time = time.time()
        self.failure_count += 1
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def reset(self):
        """Reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# ============================================================================
# Error Handler
# ============================================================================

class ErrorHandler:
    """
    Comprehensive error handling with retry and circuit breaker.
    """
    
    def __init__(self):
        self.circuit_breakers = {}
        self.error_stats = {
            'total_errors': 0,
            'retry_successes': 0,
            'retry_failures': 0,
            'circuit_breaker_trips': 0
        }
    
    def with_retry(self, config: Optional[RetryConfig] = None):
        """
        Decorator for automatic retry with exponential backoff.
        
        Usage:
            @error_handler.with_retry(RetryConfig(max_attempts=3))
            def my_function():
                ...
        """
        if config is None:
            config = RetryConfig()
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._retry_with_backoff(func, config, *args, **kwargs)
            return wrapper
        return decorator
    
    def _retry_with_backoff(
        self,
        func: Callable,
        config: RetryConfig,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry and exponential backoff."""
        last_exception = None
        delay = config.initial_delay
        
        for attempt in range(config.max_attempts):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.error_stats['retry_successes'] += 1
                
                return result
                
            except tuple(config.exceptions) as e:
                last_exception = e
                self.error_stats['total_errors'] += 1
                
                if attempt < config.max_attempts - 1:
                    print(f"[ErrorHandler] Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * config.backoff_factor, config.max_delay)
                else:
                    self.error_stats['retry_failures'] += 1
        
        raise last_exception
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker for a component."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(**kwargs)
        return self.circuit_breakers[name]
    
    def with_circuit_breaker(self, name: str, **breaker_kwargs):
        """
        Decorator for circuit breaker pattern.
        
        Usage:
            @error_handler.with_circuit_breaker('my_service')
            def my_function():
                ...
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                breaker = self.get_circuit_breaker(name, **breaker_kwargs)
                try:
                    return breaker.call(func, *args, **kwargs)
                except CircuitBreakerOpen:
                    self.error_stats['circuit_breaker_trips'] += 1
                    raise
            return wrapper
        return decorator
    
    def get_stats(self) -> dict:
        """Get error handling statistics."""
        stats = self.error_stats.copy()
        stats['active_circuit_breakers'] = len(self.circuit_breakers)
        stats['open_circuits'] = sum(
            1 for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.OPEN
        )
        return stats


# ============================================================================
# Global Error Handler
# ============================================================================

_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get or create global error handler."""
    global _global_error_handler
    
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    
    return _global_error_handler

