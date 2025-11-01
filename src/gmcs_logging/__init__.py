"""
Enhanced logging system for GMCS.

Provides structured logging with JSON format, rotating file handlers,
and context injection.
"""

from .structured_logger import (
    get_logger,
    configure_logging,
    GMCSLogger,
    LogContext
)
from .log_aggregator import (
    LogAggregator,
    get_aggregator
)

__all__ = [
    'get_logger',
    'configure_logging',
    'GMCSLogger',
    'LogContext',
    'LogAggregator',
    'get_aggregator'
]

