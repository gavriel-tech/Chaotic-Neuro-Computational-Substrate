"""
SQLAlchemy models for GMCS database.

Provides persistent storage for:
- Sessions: Saved simulation states
- Presets: User-created presets
- Model Checkpoints: Trained ML model weights
- Users: User accounts (if auth enabled)
- Metrics: Performance monitoring data
"""

from datetime import datetime
from typing import Optional, Dict, Any
import json

try:
    from sqlalchemy import (
        Column, Integer, String, Float, Boolean, DateTime, 
        Text, ForeignKey, Index, JSON, LargeBinary
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship
    
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
    
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("[Database] SQLAlchemy not available - database features disabled")
    
    # Stub base class
    class Base:
        pass


if SQLALCHEMY_AVAILABLE:
    
    # ============================================================================
    # Session Model
    # ============================================================================
    
    class SessionModel(Base):
        """
        Saved simulation sessions.
        
        Stores complete system state snapshots for later resumption.
        """
        __tablename__ = 'sessions'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(String(64), unique=True, nullable=False, index=True)
        name = Column(String(255), nullable=False)
        description = Column(Text, nullable=True)
        
        # State data
        state_data = Column(LargeBinary, nullable=False)  # Compressed msgpack
        state_metadata = Column(JSON, nullable=True)  # Human-readable summary
        
        # Configuration
        preset_id = Column(String(64), nullable=True)
        config_json = Column(Text, nullable=True)
        
        # Statistics
        num_nodes = Column(Integer, default=0)
        simulation_time = Column(Float, default=0.0)
        total_iterations = Column(Integer, default=0)
        
        # Timestamps
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        last_accessed = Column(DateTime, default=datetime.utcnow)
        
        # Ownership
        user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
        
        # Relationships
        user = relationship("UserModel", back_populates="sessions")
        
        # Indexes
        __table_args__ = (
            Index('idx_session_user', 'user_id'),
            Index('idx_session_created', 'created_at'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary."""
            return {
                'id': self.id,
                'session_id': self.session_id,
                'name': self.name,
                'description': self.description,
                'preset_id': self.preset_id,
                'num_nodes': self.num_nodes,
                'simulation_time': self.simulation_time,
                'total_iterations': self.total_iterations,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None,
                'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
                'user_id': self.user_id
            }
    
    
    # ============================================================================
    # Preset Model
    # ============================================================================
    
    class PresetModel(Base):
        """
        User-created presets.
        
        Stores custom preset configurations.
        """
        __tablename__ = 'presets'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        preset_id = Column(String(64), unique=True, nullable=False, index=True)
        name = Column(String(255), nullable=False)
        description = Column(Text, nullable=True)
        category = Column(String(100), nullable=True, index=True)
        
        # Preset data
        preset_json = Column(Text, nullable=False)  # Full preset JSON
        
        # Metadata
        tags = Column(JSON, nullable=True)  # List of tags
        author = Column(String(255), nullable=True)
        version = Column(String(20), default='1.0')
        
        # Statistics
        num_nodes = Column(Integer, default=0)
        num_connections = Column(Integer, default=0)
        usage_count = Column(Integer, default=0)
        
        # Flags
        is_public = Column(Boolean, default=False)
        is_featured = Column(Boolean, default=False)
        is_validated = Column(Boolean, default=False)
        
        # Timestamps
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # Ownership
        user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
        
        # Relationships
        user = relationship("UserModel", back_populates="presets")
        
        # Indexes
        __table_args__ = (
            Index('idx_preset_category', 'category'),
            Index('idx_preset_public', 'is_public'),
            Index('idx_preset_user', 'user_id'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary."""
            return {
                'id': self.id,
                'preset_id': self.preset_id,
                'name': self.name,
                'description': self.description,
                'category': self.category,
                'tags': self.tags,
                'author': self.author,
                'version': self.version,
                'num_nodes': self.num_nodes,
                'num_connections': self.num_connections,
                'usage_count': self.usage_count,
                'is_public': self.is_public,
                'is_featured': self.is_featured,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None,
                'user_id': self.user_id
            }
    
    
    # ============================================================================
    # Model Checkpoint Model
    # ============================================================================
    
    class ModelCheckpointModel(Base):
        """
        ML model checkpoints.
        
        Stores trained model weights and training state.
        """
        __tablename__ = 'model_checkpoints'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        checkpoint_id = Column(String(64), unique=True, nullable=False, index=True)
        model_name = Column(String(255), nullable=False)
        model_type = Column(String(100), nullable=False, index=True)
        
        # Model data
        weights_data = Column(LargeBinary, nullable=False)  # PyTorch/TF weights
        optimizer_state = Column(LargeBinary, nullable=True)  # Optimizer state
        metadata = Column(JSON, nullable=True)  # Architecture, hyperparams
        
        # Training info
        training_steps = Column(Integer, default=0)
        training_time = Column(Float, default=0.0)  # seconds
        best_loss = Column(Float, nullable=True)
        best_metric = Column(Float, nullable=True)
        
        # Metadata
        framework = Column(String(50), nullable=True)  # pytorch, tensorflow, jax
        version = Column(String(20), default='1.0')
        notes = Column(Text, nullable=True)
        
        # Timestamps
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # Ownership
        user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
        session_id = Column(Integer, ForeignKey('sessions.id'), nullable=True)
        
        # Relationships
        user = relationship("UserModel", back_populates="checkpoints")
        
        # Indexes
        __table_args__ = (
            Index('idx_checkpoint_model_type', 'model_type'),
            Index('idx_checkpoint_user', 'user_id'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary."""
            return {
                'id': self.id,
                'checkpoint_id': self.checkpoint_id,
                'model_name': self.model_name,
                'model_type': self.model_type,
                'training_steps': self.training_steps,
                'training_time': self.training_time,
                'best_loss': self.best_loss,
                'best_metric': self.best_metric,
                'framework': self.framework,
                'version': self.version,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None,
                'user_id': self.user_id
            }
    
    
    # ============================================================================
    # User Model
    # ============================================================================
    
    class UserModel(Base):
        """
        User accounts.
        
        Stores user information and authentication data.
        """
        __tablename__ = 'users'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        username = Column(String(100), unique=True, nullable=False, index=True)
        email = Column(String(255), unique=True, nullable=True, index=True)
        
        # Authentication (if enabled)
        password_hash = Column(String(255), nullable=True)
        api_key = Column(String(255), unique=True, nullable=True, index=True)
        
        # Metadata
        full_name = Column(String(255), nullable=True)
        organization = Column(String(255), nullable=True)
        role = Column(String(50), default='user')  # user, admin, researcher
        
        # Flags
        is_active = Column(Boolean, default=True)
        is_verified = Column(Boolean, default=False)
        
        # Statistics
        total_sessions = Column(Integer, default=0)
        total_presets = Column(Integer, default=0)
        
        # Timestamps
        created_at = Column(DateTime, default=datetime.utcnow)
        last_login = Column(DateTime, nullable=True)
        
        # Relationships
        sessions = relationship("SessionModel", back_populates="user", cascade="all, delete-orphan")
        presets = relationship("PresetModel", back_populates="user", cascade="all, delete-orphan")
        checkpoints = relationship("ModelCheckpointModel", back_populates="user", cascade="all, delete-orphan")
        metrics = relationship("MetricsModel", back_populates="user", cascade="all, delete-orphan")
        
        def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
            """Convert to dictionary."""
            data = {
                'id': self.id,
                'username': self.username,
                'email': self.email if include_sensitive else None,
                'full_name': self.full_name,
                'organization': self.organization,
                'role': self.role,
                'is_active': self.is_active,
                'is_verified': self.is_verified,
                'total_sessions': self.total_sessions,
                'total_presets': self.total_presets,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'last_login': self.last_login.isoformat() if self.last_login else None
            }
            
            if include_sensitive:
                data['api_key'] = self.api_key
            
            return data
    
    
    # ============================================================================
    # Metrics Model
    # ============================================================================
    
    class MetricsModel(Base):
        """
        Performance and monitoring metrics.
        
        Stores time-series data for monitoring and analysis.
        """
        __tablename__ = 'metrics'
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        
        # Identification
        metric_type = Column(String(100), nullable=False, index=True)
        metric_name = Column(String(255), nullable=False)
        
        # Values
        value_float = Column(Float, nullable=True)
        value_int = Column(Integer, nullable=True)
        value_str = Column(String(500), nullable=True)
        value_json = Column(JSON, nullable=True)
        
        # Context
        session_id = Column(String(64), nullable=True, index=True)
        node_id = Column(String(100), nullable=True)
        component = Column(String(100), nullable=True, index=True)
        
        # Metadata
        tags = Column(JSON, nullable=True)
        
        # Timestamp
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        
        # Ownership
        user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
        
        # Relationships
        user = relationship("UserModel", back_populates="metrics")
        
        # Indexes
        __table_args__ = (
            Index('idx_metrics_type_time', 'metric_type', 'timestamp'),
            Index('idx_metrics_component', 'component', 'timestamp'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary."""
            return {
                'id': self.id,
                'metric_type': self.metric_type,
                'metric_name': self.metric_name,
                'value_float': self.value_float,
                'value_int': self.value_int,
                'value_str': self.value_str,
                'value_json': self.value_json,
                'session_id': self.session_id,
                'node_id': self.node_id,
                'component': self.component,
                'tags': self.tags,
                'timestamp': self.timestamp.isoformat() if self.timestamp else None,
                'user_id': self.user_id
            }

else:
    # Stub models when SQLAlchemy not available
    class SessionModel:
        pass
    
    class PresetModel:
        pass
    
    class ModelCheckpointModel:
        pass
    
    class UserModel:
        pass
    
    class MetricsModel:
        pass

