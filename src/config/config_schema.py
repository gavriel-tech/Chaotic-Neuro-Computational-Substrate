"""
Type-safe configuration schemas using Pydantic.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator


# ============================================================================
# Server Configuration
# ============================================================================

class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default='0.0.0.0', description='Server host')
    port: int = Field(default=8000, description='Server port')
    reload: bool = Field(default=False, description='Auto-reload on code changes')
    workers: int = Field(default=1, description='Number of worker processes')
    log_level: str = Field(default='INFO', description='Logging level')


# ============================================================================
# Database Configuration
# ============================================================================

class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(default='sqlite:///gmcs.db', description='Database URL')
    echo: bool = Field(default=False, description='Echo SQL statements')
    pool_size: int = Field(default=5, description='Connection pool size')
    max_overflow: int = Field(default=10, description='Max overflow connections')


# ============================================================================
# Logging Configuration
# ============================================================================

class LoggingConfig(BaseModel):
    """Logging configuration."""
    log_dir: str = Field(default='logs', description='Log directory')
    level: str = Field(default='INFO', description='Log level')
    json_format: bool = Field(default=True, description='Use JSON format')
    console_output: bool = Field(default=True, description='Output to console')
    rotate_size: int = Field(default=10485760, description='Max log file size (bytes)')
    rotate_count: int = Field(default=5, description='Number of backup files')
    per_module_levels: Optional[Dict[str, str]] = Field(default=None)


# ============================================================================
# Checkpoint Configuration
# ============================================================================

class CheckpointConfig(BaseModel):
    """Checkpoint configuration."""
    checkpoint_dir: str = Field(default='checkpoints', description='Checkpoint directory')
    interval: float = Field(default=300.0, description='Checkpoint interval (seconds)')
    max_checkpoints: int = Field(default=10, description='Maximum checkpoints to keep')
    compress: bool = Field(default=True, description='Compress checkpoints')
    

# ============================================================================
# ML Configuration
# ============================================================================

class MLConfig(BaseModel):
    """ML system configuration."""
    models_dir: str = Field(default='models', description='Model storage directory')
    default_framework: str = Field(default='pytorch', description='Default ML framework')
    enable_gpu: bool = Field(default=True, description='Enable GPU acceleration')
    batch_size: int = Field(default=32, description='Default batch size')
    learning_rate: float = Field(default=1e-3, description='Default learning rate')


# ============================================================================
# Performance Configuration
# ============================================================================

class PerformanceConfig(BaseModel):
    """Performance and optimization configuration."""
    enable_jit: bool = Field(default=True, description='Enable JIT compilation')
    enable_multithreading: bool = Field(default=True, description='Enable multithreading')
    max_threads: Optional[int] = Field(default=None, description='Max threads (None = auto)')
    cache_size: int = Field(default=1000, description='Cache size for computations')


# ============================================================================
# Security Configuration
# ============================================================================

class SecurityConfig(BaseModel):
    """Security configuration."""
    enable_auth: bool = Field(default=False, description='Enable authentication')
    secret_key: Optional[str] = Field(default=None, description='Secret key for JWT')
    cors_origins: List[str] = Field(default=['*'], description='CORS allowed origins')
    rate_limit: Optional[int] = Field(default=None, description='Rate limit (requests/min)')


# ============================================================================
# Main Configuration
# ============================================================================

class GMCSConfig(BaseModel):
    """Main GMCS configuration."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    model_config = ConfigDict(validate_assignment=True, extra='allow')

    @model_validator(mode='before')
    def env_override(cls, data: Any):
        """Allow environment variable overrides for top-level fields."""
        if not isinstance(data, dict):
            data = {} if data is None else dict(data)
        else:
            data = dict(data)

        import os

        for field_name in cls.model_fields:
            env_name = f"GMCS_{field_name.upper()}"
            env_value = os.getenv(env_name)
            if env_value is None:
                continue

            current_value = data.get(field_name)
            if current_value is None:
                field_info = cls.model_fields[field_name]
                if field_info.default is not None:
                    current_value = field_info.default
                elif field_info.default_factory is not None:  # type: ignore[attr-defined]
                    current_value = field_info.default_factory()
            data[field_name] = cls._coerce_env_value(env_value, current_value)

        return data

    @staticmethod
    def _coerce_env_value(env_value: str, current_value: Any) -> Any:
        """Coerce environment value to the type of the current field value."""
        if isinstance(current_value, bool):
            return env_value.lower() in ('true', '1', 'yes')
        if isinstance(current_value, int):
            try:
                return int(env_value)
            except ValueError:
                return current_value
        if isinstance(current_value, float):
            try:
                return float(env_value)
            except ValueError:
                return current_value
        return env_value

