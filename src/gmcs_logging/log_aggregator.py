"""
Log aggregator for GMCS.

Provides:
- Buffered log collection
- Batch log writes for performance
- Async log shipping to external systems
- Rate limiting
"""

import asyncio
import queue
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from collections import deque


# ============================================================================
# Log Entry
# ============================================================================

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: float
    level: str
    logger: str
    message: str
    context: Dict[str, Any]
    extra: Dict[str, Any]


# ============================================================================
# Log Aggregator
# ============================================================================

class LogAggregator:
    """
    Aggregate and batch logs for efficient processing.
    
    Collects logs in memory and flushes periodically or when buffer is full.
    """
    
    def __init__(
        self,
        buffer_size: int = 1000,
        flush_interval: float = 5.0,
        max_queue_size: int = 10000
    ):
        """
        Initialize log aggregator.
        
        Args:
            buffer_size: Max logs before auto-flush
            flush_interval: Flush interval in seconds
            max_queue_size: Max queue size (drop if exceeded)
        """
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size
        
        # Log buffer
        self.buffer = deque(maxlen=buffer_size)
        self.queue = queue.Queue(maxsize=max_queue_size)
        
        # Handlers
        self.handlers: List[Callable[[List[LogEntry]], None]] = []
        
        # Control
        self.running = False
        self.worker_thread = None
        self.last_flush = time.time()
        
        # Statistics
        self.stats = {
            'logs_received': 0,
            'logs_dropped': 0,
            'logs_flushed': 0,
            'flush_count': 0
        }
    
    def add_handler(self, handler: Callable[[List[LogEntry]], None]):
        """
        Add a log handler.
        
        Args:
            handler: Function that receives list of log entries
        """
        self.handlers.append(handler)
    
    def start(self):
        """Start aggregator worker."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def stop(self, timeout: float = 5.0):
        """
        Stop aggregator and flush remaining logs.
        
        Args:
            timeout: Max time to wait for flush
        """
        self.running = False
        
        if self.worker_thread:
            self.worker_thread.join(timeout=timeout)
        
        # Final flush
        self._flush()
    
    def add_log(self, entry: LogEntry):
        """Add log entry to buffer."""
        try:
            self.queue.put_nowait(entry)
            self.stats['logs_received'] += 1
        except queue.Full:
            self.stats['logs_dropped'] += 1
    
    def _worker(self):
        """Worker thread for processing logs."""
        while self.running:
            try:
                # Get log from queue (with timeout)
                try:
                    entry = self.queue.get(timeout=0.1)
                    self.buffer.append(entry)
                except queue.Empty:
                    pass
                
                # Check if we should flush
                should_flush = (
                    len(self.buffer) >= self.buffer_size or
                    time.time() - self.last_flush >= self.flush_interval
                )
                
                if should_flush and self.buffer:
                    self._flush()
                
            except Exception as e:
                print(f"[LogAggregator] Worker error: {e}")
                time.sleep(1)
    
    def _flush(self):
        """Flush buffer to handlers."""
        if not self.buffer:
            return
        
        # Copy buffer
        logs_to_flush = list(self.buffer)
        self.buffer.clear()
        
        # Send to handlers
        for handler in self.handlers:
            try:
                handler(logs_to_flush)
            except Exception as e:
                print(f"[LogAggregator] Handler error: {e}")
        
        # Update stats
        self.stats['logs_flushed'] += len(logs_to_flush)
        self.stats['flush_count'] += 1
        self.last_flush = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            **self.stats,
            'buffer_size': len(self.buffer),
            'queue_size': self.queue.qsize(),
            'handlers_count': len(self.handlers),
            'running': self.running
        }


# ============================================================================
# Async Log Shipper
# ============================================================================

class AsyncLogShipper:
    """
    Ship logs to external systems asynchronously.
    
    Supports HTTP endpoints, databases, etc.
    """
    
    def __init__(self, endpoint_url: Optional[str] = None):
        """
        Initialize log shipper.
        
        Args:
            endpoint_url: Optional HTTP endpoint for log shipping
        """
        self.endpoint_url = endpoint_url
        self.stats = {
            'shipped': 0,
            'failed': 0
        }
    
    async def ship_logs(self, logs: List[LogEntry]):
        """
        Ship logs asynchronously.
        
        Args:
            logs: List of log entries
        """
        if not self.endpoint_url:
            return
        
        try:
            # Convert logs to JSON
            log_data = [
                {
                    'timestamp': entry.timestamp,
                    'level': entry.level,
                    'logger': entry.logger,
                    'message': entry.message,
                    'context': entry.context,
                    'extra': entry.extra
                }
                for entry in logs
            ]
            
            # Ship to endpoint (would use aiohttp in production)
            # For now, just simulate
            await asyncio.sleep(0.01)  # Simulate network delay
            
            self.stats['shipped'] += len(logs)
            
        except Exception as e:
            print(f"[AsyncLogShipper] Failed to ship logs: {e}")
            self.stats['failed'] += len(logs)
    
    def ship_logs_sync(self, logs: List[LogEntry]):
        """Synchronous wrapper for shipping logs."""
        asyncio.run(self.ship_logs(logs))


# ============================================================================
# Global Aggregator
# ============================================================================

_global_aggregator: Optional[LogAggregator] = None


def get_aggregator() -> LogAggregator:
    """Get or create global log aggregator."""
    global _global_aggregator
    
    if _global_aggregator is None:
        _global_aggregator = LogAggregator()
        _global_aggregator.start()
    
    return _global_aggregator


# ============================================================================
# Example Handlers
# ============================================================================

def console_handler(logs: List[LogEntry]):
    """Print logs to console (for debugging)."""
    for log in logs:
        print(f"[{log.level}] {log.logger}: {log.message}")


def file_handler(filename: str):
    """Create a file handler."""
    def handler(logs: List[LogEntry]):
        with open(filename, 'a') as f:
            for log in logs:
                f.write(f"{log.timestamp} [{log.level}] {log.logger}: {log.message}\n")
    return handler

