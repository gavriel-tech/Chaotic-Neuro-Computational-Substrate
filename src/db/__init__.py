"""
Database layer for GMCS.

Provides SQLAlchemy models and database connection management.
"""

from .models import (
    SessionModel,
    PresetModel,
    ModelCheckpointModel,
    UserModel,
    MetricsModel,
    Base
)
from .database import (
    get_engine,
    get_session,
    init_database,
    DatabaseManager
)

__all__ = [
    'SessionModel',
    'PresetModel',
    'ModelCheckpointModel',
    'UserModel',
    'MetricsModel',
    'Base',
    'get_engine',
    'get_session',
    'init_database',
    'DatabaseManager'
]

