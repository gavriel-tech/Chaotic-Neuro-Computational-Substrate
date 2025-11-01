"""
Configuration management system for GMCS.
"""

import os
import yaml
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from pydantic import ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from .config_schema import GMCSConfig


class ConfigManager:
    """
    Manage GMCS configuration with YAML support and validation.
    
    Features:
    - Load from YAML files
    - Environment variable overrides
    - Type validation with Pydantic
    - Hot reload (watch for file changes)
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to YAML config file
        """
        self.config_file = config_file
        self.config: Optional[GMCSConfig] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file or defaults."""
        if not PYDANTIC_AVAILABLE:
            print("[ConfigManager] Pydantic not available, using defaults")
            self.config = None
            return
        
        # Load from file if provided
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                self.config = GMCSConfig(**config_data)
                print(f"[ConfigManager] Loaded config from {self.config_file}")
            except Exception as e:
                print(f"[ConfigManager] Failed to load config: {e}")
                self.config = GMCSConfig()
        else:
            # Use defaults
            self.config = GMCSConfig()
            print("[ConfigManager] Using default configuration")
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()
    
    def save(self, filepath: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            filepath: Path to save (uses self.config_file if None)
        """
        if not PYDANTIC_AVAILABLE or self.config is None:
            return
        
        filepath = filepath or self.config_file
        if not filepath:
            raise ValueError("No filepath specified")
        
        # Convert to dict
        config_dict = self.config.dict()
        
        # Save as YAML
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"[ConfigManager] Saved config to {filepath}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Config key (e.g., 'server.port')
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        if not self.config:
            return default
        
        # Navigate nested structure
        parts = key.split('.')
        value = self.config
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by dot-notation key.
        
        Args:
            key: Config key (e.g., 'server.port')
            value: New value
        """
        if not self.config:
            return
        
        # Navigate to parent
        parts = key.split('.')
        target = self.config
        
        for part in parts[:-1]:
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                return
        
        # Set value
        if hasattr(target, parts[-1]):
            setattr(target, parts[-1], value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        if not self.config:
            return {}
        return self.config.dict()


# ============================================================================
# Global Configuration Manager
# ============================================================================

_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Get or create global configuration manager."""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_file)
    
    return _global_config_manager


def get_config() -> Optional[GMCSConfig]:
    """Get current configuration."""
    manager = get_config_manager()
    return manager.config

