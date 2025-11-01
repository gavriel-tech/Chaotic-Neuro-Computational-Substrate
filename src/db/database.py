"""
Database connection management for GMCS.

Provides database initialization, session management, and utilities.
"""

import os
from contextlib import contextmanager
from typing import Optional, Generator

try:
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import StaticPool
    
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("[Database] SQLAlchemy not available")

from .models import Base, SQLALCHEMY_AVAILABLE as MODELS_AVAILABLE


# Global engine and session factory
_engine = None
_SessionFactory = None


# ============================================================================
# Engine Management
# ============================================================================

def get_database_url() -> str:
    """
    Get database URL from environment or default.
    
    Returns:
        Database connection URL
    """
    return os.getenv('GMCS_DB_URL', 'sqlite:///gmcs.db')


def get_engine(echo: bool = False):
    """
    Get or create database engine.
    
    Args:
        echo: Whether to log SQL statements
        
    Returns:
        SQLAlchemy engine
    """
    global _engine
    
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy not available")
    
    if _engine is None:
        db_url = get_database_url()
        
        # Special handling for SQLite
        if db_url.startswith('sqlite'):
            _engine = create_engine(
                db_url,
                echo=echo,
                connect_args={'check_same_thread': False},
                poolclass=StaticPool
            )
            
            # Enable foreign keys for SQLite
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            _engine = create_engine(db_url, echo=echo)
    
    return _engine


def init_database(drop_existing: bool = False, echo: bool = False):
    """
    Initialize database schema.
    
    Args:
        drop_existing: Whether to drop existing tables
        echo: Whether to log SQL statements
    """
    if not SQLALCHEMY_AVAILABLE or not MODELS_AVAILABLE:
        print("[Database] SQLAlchemy/models not available, skipping initialization")
        return
    
    engine = get_engine(echo=echo)
    
    if drop_existing:
        Base.metadata.drop_all(engine)
        print("[Database] Dropped existing tables")
    
    Base.metadata.create_all(engine)
    print(f"[Database] Initialized database at {get_database_url()}")


# ============================================================================
# Session Management
# ============================================================================

def get_session_factory():
    """Get or create session factory."""
    global _SessionFactory
    
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy not available")
    
    if _SessionFactory is None:
        engine = get_engine()
        _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    
    return _SessionFactory


def get_session() -> Optional[Session]:
    """
    Get a new database session.
    
    Returns:
        SQLAlchemy session or None if not available
    """
    if not SQLALCHEMY_AVAILABLE:
        return None
    
    factory = get_session_factory()
    return factory()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional scope for database operations.
    
    Usage:
        with session_scope() as session:
            session.add(model)
            # Automatic commit on success, rollback on error
    """
    if not SQLALCHEMY_AVAILABLE:
        yield None
        return
    
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


# ============================================================================
# Database Manager
# ============================================================================

class DatabaseManager:
    """
    High-level database management interface.
    
    Provides convenient methods for common database operations.
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            db_url: Optional database URL (uses env var if None)
        """
        if db_url:
            os.environ['GMCS_DB_URL'] = db_url
        
        self.engine = None
        self.initialized = False
    
    def initialize(self, drop_existing: bool = False, echo: bool = False):
        """Initialize database."""
        if not SQLALCHEMY_AVAILABLE:
            print("[DatabaseManager] SQLAlchemy not available")
            return False
        
        try:
            init_database(drop_existing=drop_existing, echo=echo)
            self.engine = get_engine()
            self.initialized = True
            return True
        except Exception as e:
            print(f"[DatabaseManager] Initialization failed: {e}")
            return False
    
    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a session context."""
        with session_scope() as session:
            yield session
    
    def close(self):
        """Close database connections."""
        global _engine, _SessionFactory
        
        if _engine:
            _engine.dispose()
            _engine = None
        
        _SessionFactory = None
        self.initialized = False
    
    # ========================================================================
    # Query Helpers
    # ========================================================================
    
    def get_by_id(self, model_class, record_id: int):
        """Get record by ID."""
        with self.session() as session:
            if session is None:
                return None
            return session.query(model_class).filter_by(id=record_id).first()
    
    def get_by_field(self, model_class, field_name: str, value):
        """Get record by field value."""
        with self.session() as session:
            if session is None:
                return None
            return session.query(model_class).filter_by(**{field_name: value}).first()
    
    def list_all(self, model_class, limit: Optional[int] = None, offset: int = 0):
        """List all records of a model."""
        with self.session() as session:
            if session is None:
                return []
            
            query = session.query(model_class).offset(offset)
            if limit:
                query = query.limit(limit)
            
            return query.all()
    
    def count(self, model_class) -> int:
        """Count records."""
        with self.session() as session:
            if session is None:
                return 0
            return session.query(model_class).count()
    
    def delete_by_id(self, model_class, record_id: int) -> bool:
        """Delete record by ID."""
        with self.session() as session:
            if session is None:
                return False
            
            record = session.query(model_class).filter_by(id=record_id).first()
            if record:
                session.delete(record)
                return True
            return False
    
    # ========================================================================
    # Maintenance
    # ========================================================================
    
    def vacuum(self):
        """Vacuum database (SQLite only)."""
        if not self.engine:
            return False
        
        if 'sqlite' in str(self.engine.url):
            try:
                with self.engine.connect() as conn:
                    conn.execute("VACUUM")
                return True
            except Exception as e:
                print(f"[DatabaseManager] Vacuum failed: {e}")
                return False
        
        return False
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        if not SQLALCHEMY_AVAILABLE or not self.initialized:
            return {}
        
        from .models import (
            SessionModel, PresetModel, ModelCheckpointModel,
            UserModel, MetricsModel
        )
        
        stats = {
            'sessions': self.count(SessionModel),
            'presets': self.count(PresetModel),
            'checkpoints': self.count(ModelCheckpointModel),
            'users': self.count(UserModel),
            'metrics': self.count(MetricsModel),
            'database_url': str(self.engine.url) if self.engine else None
        }
        
        return stats

