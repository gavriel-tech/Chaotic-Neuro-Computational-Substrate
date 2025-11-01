"""
Checkpoint manager for system state snapshots.
"""

import os
import gzip
import pickle
import time
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    checkpoint_dir: str = 'checkpoints'
    interval: float = 300.0  # 5 minutes
    max_checkpoints: int = 10
    compress: bool = True
    incremental: bool = False


class CheckpointManager:
    """
    Manage system state checkpoints.
    
    Features:
    - Periodic snapshots
    - Compression
    - Automatic cleanup
    - Resume from checkpoint
    """
    
    def __init__(self, config: Optional[CheckpointConfig] = None):
        self.config = config or CheckpointConfig()
        self.last_checkpoint_time = 0
        self.checkpoint_count = 0
        
        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def should_checkpoint(self) -> bool:
        """Check if it's time to checkpoint."""
        return time.time() - self.last_checkpoint_time >= self.config.interval
    
    def save_checkpoint(
        self,
        state: Any,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Save checkpoint.
        
        Args:
            state: System state to save
            metadata: Optional metadata
            name: Optional checkpoint name (auto-generated if None)
            
        Returns:
            Path to saved checkpoint
        """
        if name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"checkpoint_{timestamp}_{self.checkpoint_count}"
        
        filename = f"{name}.pkl"
        if self.config.compress:
            filename += ".gz"
        
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'state': state,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'checkpoint_id': self.checkpoint_count
        }
        
        # Save
        if self.config.compress:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        
        self.last_checkpoint_time = time.time()
        self.checkpoint_count += 1
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"[CheckpointManager] Saved checkpoint: {filepath}")
        return filepath
    
    def load_checkpoint(self, name: str) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            name: Checkpoint name or path
            
        Returns:
            Checkpoint data
        """
        # Handle full path or just name
        if os.path.exists(name):
            filepath = name
        else:
            # Try with and without compression extension
            filepath = os.path.join(self.config.checkpoint_dir, name)
            if not os.path.exists(filepath):
                filepath += ".pkl"
            if not os.path.exists(filepath):
                filepath += ".gz"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {name}")
        
        # Load
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        
        print(f"[CheckpointManager] Loaded checkpoint: {filepath}")
        return data
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)
        latest = checkpoints[0]
        
        return self.load_checkpoint(latest['path'])
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        if not os.path.exists(self.config.checkpoint_dir):
            return []
        
        checkpoints = []
        for filename in os.listdir(self.config.checkpoint_dir):
            if filename.startswith('checkpoint_'):
                filepath = os.path.join(self.config.checkpoint_dir, filename)
                checkpoints.append({
                    'name': filename,
                    'path': filepath,
                    'size': os.path.getsize(filepath),
                    'modified': os.path.getmtime(filepath)
                })
        
        return checkpoints
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= self.config.max_checkpoints:
            return
        
        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda x: x['modified'])
        
        # Remove oldest
        to_remove = len(checkpoints) - self.config.max_checkpoints
        for checkpoint in checkpoints[:to_remove]:
            os.remove(checkpoint['path'])
            print(f"[CheckpointManager] Removed old checkpoint: {checkpoint['name']}")
    
    def delete_checkpoint(self, name: str):
        """Delete specific checkpoint."""
        filepath = os.path.join(self.config.checkpoint_dir, name)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"[CheckpointManager] Deleted checkpoint: {name}")


_global_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(config: Optional[CheckpointConfig] = None) -> CheckpointManager:
    """Get or create global checkpoint manager."""
    global _global_checkpoint_manager
    
    if _global_checkpoint_manager is None:
        _global_checkpoint_manager = CheckpointManager(config)
    
    return _global_checkpoint_manager

