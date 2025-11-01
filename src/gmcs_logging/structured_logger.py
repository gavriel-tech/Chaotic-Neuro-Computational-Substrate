"""
Structured JSON logging for GMCS.

Provides:
- JSON structured logging
- Rotating file handlers (size + time based)
- Context injection (request ID, user ID, node ID)
- Per-module log levels
- Performance metrics logging
"""

import logging
import logging.handlers
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager
import threading


# ============================================================================
# JSON Formatter
# ============================================================================

class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context if available
        if self.include_context and hasattr(record, 'context'):
            log_data['context'] = record.context
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info', 'context']:
                log_data[key] = value
        
        return json.dumps(log_data)


# ============================================================================
# Log Context
# ============================================================================

class LogContext:
    """Thread-local context for logging."""
    
    _context = threading.local()
    
    @classmethod
    def get(cls) -> Dict[str, Any]:
        """Get current context."""
        if not hasattr(cls._context, 'data'):
            cls._context.data = {}
        return cls._context.data.copy()
    
    @classmethod
    def set(cls, **kwargs):
        """Set context fields."""
        if not hasattr(cls._context, 'data'):
            cls._context.data = {}
        cls._context.data.update(kwargs)
    
    @classmethod
    def clear(cls):
        """Clear context."""
        if hasattr(cls._context, 'data'):
            cls._context.data = {}
    
    @classmethod
    @contextmanager
    def bind(cls, **kwargs):
        """Temporarily bind context."""
        old_context = cls.get()
        cls.set(**kwargs)
        try:
            yield
        finally:
            cls._context.data = old_context


# ============================================================================
# Context Injection Filter
# ============================================================================

class ContextFilter(logging.Filter):
    """Inject context into log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record."""
        context = LogContext.get()
        if context:
            record.context = context
        return True


# ============================================================================
# GMCS Logger
# ============================================================================

class GMCSLogger(logging.LoggerAdapter):
    """
    Enhanced logger with context support and performance tracking.
    """
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Process log message with context."""
        # Add context from LogContext
        context = LogContext.get()
        if context:
            kwargs['extra'] = kwargs.get('extra', {})
            kwargs['extra'].update(context)
        
        return msg, kwargs
    
    def log_metric(self, metric_name: str, value: float, **tags):
        """
        Log performance metric.
        
        Args:
            metric_name: Metric name
            value: Metric value
            **tags: Additional tags
        """
        self.info(
            f"METRIC: {metric_name}={value}",
            extra={'metric_name': metric_name, 'metric_value': value, **tags}
        )
    
    def log_timing(self, operation: str, duration: float, **tags):
        """
        Log operation timing.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            **tags: Additional tags
        """
        self.info(
            f"TIMING: {operation} took {duration:.4f}s",
            extra={'operation': operation, 'duration': duration, **tags}
        )
    
    @contextmanager
    def timer(self, operation: str, **tags):
        """
        Time an operation.
        
        Usage:
            with logger.timer('data_processing'):
                # ... code ...
        """
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.log_timing(operation, duration, **tags)


# ============================================================================
# Logger Configuration
# ============================================================================

_loggers = {}
_configured = False


def configure_logging(
    log_dir: str = 'logs',
    level: str = 'INFO',
    console_output: bool = True,
    json_format: bool = True,
    rotate_size: int = 10 * 1024 * 1024,  # 10 MB
    rotate_count: int = 5,
    per_module_levels: Optional[Dict[str, str]] = None
):
    """
    Configure logging for GMCS.
    
    Args:
        log_dir: Directory for log files
        level: Default log level
        console_output: Whether to log to console
        json_format: Whether to use JSON format
        rotate_size: Max size per log file (bytes)
        rotate_count: Number of backup files
        per_module_levels: Module-specific log levels
    """
    global _configured
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger('gmcs')
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Context filter
    context_filter = ContextFilter()
    
    # File handler with rotation
    log_file = os.path.join(log_dir, 'gmcs.log')
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=rotate_size,
        backupCount=rotate_count
    )
    file_handler.setLevel(logging.DEBUG)
    
    if json_format:
        file_handler.setFormatter(JSONFormatter())
    else:
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    file_handler.addFilter(context_filter)
    root_logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    error_file = os.path.join(log_dir, 'gmcs_errors.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=rotate_size,
        backupCount=rotate_count
    )
    error_handler.setLevel(logging.ERROR)
    
    if json_format:
        error_handler.setFormatter(JSONFormatter())
    else:
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    error_handler.addFilter(context_filter)
    root_logger.addHandler(error_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Use simple format for console
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        )
        
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)
    
    # Set per-module levels
    if per_module_levels:
        for module, module_level in per_module_levels.items():
            logging.getLogger(f'gmcs.{module}').setLevel(
                getattr(logging, module_level.upper())
            )
    
    _configured = True
    
    root_logger.info("Logging configured", extra={
        'log_dir': log_dir,
        'level': level,
        'json_format': json_format
    })


def get_logger(name: str) -> GMCSLogger:
    """
    Get or create a logger for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        GMCSLogger instance
    """
    global _configured, _loggers
    
    # Auto-configure if not configured
    if not _configured:
        configure_logging()
    
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]
    
    # Create new logger
    logger = logging.getLogger(f'gmcs.{name}')
    gmcs_logger = GMCSLogger(logger)
    _loggers[name] = gmcs_logger
    
    return gmcs_logger


# ============================================================================
# Convenience Functions
# ============================================================================

def set_log_level(level: str, module: Optional[str] = None):
    """
    Set log level.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        module: Optional module name (None = root logger)
    """
    level_int = getattr(logging, level.upper())
    
    if module:
        logging.getLogger(f'gmcs.{module}').setLevel(level_int)
    else:
        logging.getLogger('gmcs').setLevel(level_int)


def get_log_stats(log_dir: str = 'logs') -> Dict[str, Any]:
    """
    Get logging statistics.
    
    Args:
        log_dir: Log directory
        
    Returns:
        Dict with log file stats
    """
    stats = {}
    
    if not os.path.exists(log_dir):
        return stats
    
    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            filepath = os.path.join(log_dir, filename)
            stats[filename] = {
                'size': os.path.getsize(filepath),
                'modified': datetime.fromtimestamp(
                    os.path.getmtime(filepath)
                ).isoformat()
            }
    
    return stats


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    # Configure logging
    configure_logging(level='DEBUG', json_format=True)
    
    # Get logger
    logger = get_logger('example')
    
    # Basic logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # With context
    LogContext.set(request_id='123', user_id='user1')
    logger.info("Message with context")
    
    # With temporary context
    with LogContext.bind(node_id='node_42'):
        logger.info("Message with node context")
    
    # Timing
    with logger.timer('processing'):
        time.sleep(0.1)
    
    # Metrics
    logger.log_metric('processing_rate', 1000.0, unit='items/sec')
    
    # Clear context
    LogContext.clear()

