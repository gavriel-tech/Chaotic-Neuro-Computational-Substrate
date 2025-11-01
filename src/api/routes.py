"""
REST API routes for node management and control.

Defines Pydantic models and endpoint handlers for adding, updating, and removing nodes.
"""

from typing import List, Dict, Optional
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    ValidationInfo,
    field_validator,
)
import jax.numpy as jnp

from src.core.state import N_MAX, GRID_W, GRID_H, MAX_CHAIN_LEN
from src.core.gmcs_pipeline import (
    ALGO_NOP, ALGO_LIMITER, ALGO_COMPRESSOR, ALGO_EXPANDER,
    ALGO_THRESHOLD, ALGO_PHASEMOD, ALGO_FOLD,
    # Audio/Signal algorithms
    ALGO_RESONATOR, ALGO_HILBERT, ALGO_RECTIFIER, ALGO_QUANTIZER,
    ALGO_SLEW_LIMITER, ALGO_CROSS_MOD, ALGO_BIPOLAR_FOLD,
    # Photonic algorithms
    ALGO_OPTICAL_KERR, ALGO_ELECTRO_OPTIC, ALGO_OPTICAL_SWITCH,
    ALGO_FOUR_WAVE_MIXING, ALGO_RAMAN_AMPLIFIER, ALGO_SATURATION, ALGO_OPTICAL_GAIN,
    get_algorithm_name
)


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class AddNodeRequest(BaseModel):
    """Request to add a new node to simulation."""
    position: List[float] = Field(..., min_items=2, max_items=2, description="[x, y] position in grid coordinates")
    config: Optional[Dict[str, float]] = Field(default_factory=dict, description="GMCS parameter overrides")
    chain: Optional[List[int]] = Field(default_factory=list, description="Algorithm chain IDs")
    initial_perturbation: float = Field(default=0.1, ge=-1.0, le=1.0, description="Initial oscillator x value")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "position": [128.0, 128.0],
                "config": {
                    "A_max": 0.8,
                    "R_comp": 2.0,
                    "T_comp": 0.5
                },
                "chain": [1, 2],  # Limiter + Compressor
                "initial_perturbation": 0.1
            }
        }
    )
    
    @field_validator('position')
    def validate_position(cls, v):
        """Validate position is within grid bounds."""
        x, y = v
        if not (0 <= x <= GRID_W):
            raise ValueError(f"x position {x} must be in range [0, {GRID_W}]")
        if not (0 <= y <= GRID_H):
            raise ValueError(f"y position {y} must be in range [0, {GRID_H}]")
        return v
    
    @field_validator('chain')
    def validate_chain(cls, v):
        """Validate algorithm chain."""
        if len(v) > MAX_CHAIN_LEN:
            raise ValueError(f"Chain length {len(v)} exceeds maximum {MAX_CHAIN_LEN}")
        
        for algo_id in v:
            if algo_id not in VALID_ALGORITHM_IDS:
                raise ValueError(f"Invalid algorithm ID: {algo_id}")
        
        return v
    

class UpdateNodeRequest(BaseModel):
    """Request to update existing nodes."""
    node_ids: List[int] = Field(..., min_items=1, description="List of node IDs to update")
    config_updates: Optional[Dict[str, float]] = Field(default_factory=dict, description="Parameter updates")
    chain_update: Optional[List[int]] = Field(default=None, description="New algorithm chain")
    position_update: Optional[List[float]] = Field(default=None, description="New [x, y] position")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "node_ids": [0, 1, 2],
                "config_updates": {
                    "A_max": 1.0,
                    "Phi": 0.5
                },
                "chain_update": [1, 2, 6]
            }
        }
    )
    
    @field_validator('node_ids')
    def validate_node_ids(cls, v):
        """Validate node IDs are in valid range."""
        for node_id in v:
            if not (0 <= node_id < N_MAX):
                raise ValueError(f"Node ID {node_id} must be in range [0, {N_MAX})")
        return v
    
    @field_validator('chain_update')
    def validate_chain(cls, v):
        """Validate algorithm chain if provided."""
        if v is None:
            return v
        
        if len(v) > MAX_CHAIN_LEN:
            raise ValueError(f"Chain length {len(v)} exceeds maximum {MAX_CHAIN_LEN}")
        
        for algo_id in v:
            if algo_id not in VALID_ALGORITHM_IDS:
                raise ValueError(f"Invalid algorithm ID: {algo_id}")
        
        return v
    
    @field_validator('position_update')
    def validate_position(cls, v):
        """Validate position if provided."""
        if v is None:
            return v
        
        if len(v) != 2:
            raise ValueError("Position must be [x, y]")
        
        x, y = v
        if not (0 <= x <= GRID_W):
            raise ValueError(f"x position {x} must be in range [0, {GRID_W}]")
        if not (0 <= y <= GRID_H):
            raise ValueError(f"y position {y} must be in range [0, {GRID_H}]")
        return v


class NodeResponse(BaseModel):
    """Response with node information."""
    status: str
    node_id: Optional[int] = None
    message: Optional[str] = None
    data: Optional[Dict] = None


class SystemStatusResponse(BaseModel):
    """Response with system status information."""
    active_nodes: int
    total_capacity: int
    simulation_time: float
    dt: float
    grid_size: List[int]


class NodeInfoResponse(BaseModel):
    """Response with detailed node information."""
    node_id: int
    active: bool
    position: List[float]
    oscillator_state: List[float]
    config: Dict[str, float]
    chain: List[int]


class NodeListItem(BaseModel):
    """Summary information for an active node."""
    node_id: int
    active: bool
    position: List[float]
    oscillator_state: List[float]
    config: Dict[str, float]
    chain: List[int]


class NodeListResponse(BaseModel):
    """Response containing all active nodes."""
    nodes: List[NodeListItem]


# ============================================================================
# Default Parameter Values
# ============================================================================

DEFAULT_GMCS_PARAMS = {
    'A_max': 0.8,
    'R_comp': 2.0,
    'T_comp': 0.5,
    'R_exp': 2.0,
    'T_exp': 0.2,
    'Phi': 0.0,
    'omega': 1.0,
    'gamma': 1.0,
    'beta': 1.0,
}


VALID_ALGORITHM_IDS = {
    # Basic (0-6)
    ALGO_NOP, ALGO_LIMITER, ALGO_COMPRESSOR, ALGO_EXPANDER,
    ALGO_THRESHOLD, ALGO_PHASEMOD, ALGO_FOLD,
    # Audio/Signal (7-13)
    ALGO_RESONATOR, ALGO_HILBERT, ALGO_RECTIFIER, ALGO_QUANTIZER,
    ALGO_SLEW_LIMITER, ALGO_CROSS_MOD, ALGO_BIPOLAR_FOLD,
    # Photonic (14-20)
    ALGO_OPTICAL_KERR, ALGO_ELECTRO_OPTIC, ALGO_OPTICAL_SWITCH,
    ALGO_FOUR_WAVE_MIXING, ALGO_RAMAN_AMPLIFIER, ALGO_SATURATION, ALGO_OPTICAL_GAIN,
}


def merge_config_with_defaults(config: Dict[str, float]) -> Dict[str, float]:
    """Merge user config with defaults."""
    merged = DEFAULT_GMCS_PARAMS.copy()
    merged.update(config)
    return merged


def pad_chain_to_max_length(chain: List[int]) -> jnp.ndarray:
    """Pad algorithm chain to MAX_CHAIN_LEN with NOPs."""
    padded = chain + [ALGO_NOP] * (MAX_CHAIN_LEN - len(chain))
    return jnp.array(padded[:MAX_CHAIN_LEN], dtype=jnp.int32)


def validate_parameter_ranges(config: Dict[str, float]) -> Dict[str, float]:
    """
    Validate and clamp parameter values to safe ranges.
    
    Args:
        config: Parameter dictionary
        
    Returns:
        Validated and clamped parameters
    """
    ranges = {
        # Basic parameters (0-8)
        'A_max': (0.1, 5.0),
        'R_comp': (1.0, 10.0),
        'T_comp': (0.0, 2.0),
        'R_exp': (1.0, 10.0),
        'T_exp': (0.0, 2.0),
        'Phi': (0.0, 1.0),
        'omega': (0.0, 10.0),
        'gamma': (0.1, 5.0),
        'beta': (0.1, 5.0),
        # Extended parameters (9-15)
        'f0': (20.0, 20000.0),  # Audio frequency range
        'Q': (0.5, 100.0),  # Resonator Q factor
        'levels': (2.0, 256.0),  # Quantizer levels
        'rate_limit': (0.01, 10.0),  # Slew rate
        'n2': (1e-22, 1e-18),  # Kerr coefficient (m²/W)
        'V': (0.0, 10.0),  # Electro-optic voltage
        'V_pi': (1.0, 10.0),  # Half-wave voltage
    }
    
    validated = {}
    for key, value in config.items():
        if key in ranges:
            min_val, max_val = ranges[key]
            validated[key] = max(min_val, min(max_val, value))
        else:
            validated[key] = value
    
    return validated


# ============================================================================
# Extended API Models for New Features
# ============================================================================

class AlgorithmInfo(BaseModel):
    """Information about a GMCS algorithm."""
    id: int
    name: str
    category: str  # 'basic', 'audio', 'photonic'
    description: str
    parameters: List[str]


class ModulationRouteRequest(BaseModel):
    """Request to add a modulation route."""
    source_type: int = Field(..., ge=0, le=9, description="Source type ID")
    source_node_id: int = Field(..., ge=-1, description="Source node ID (-1 for global)")
    target_type: int = Field(..., ge=0, le=20, description="Target type ID")
    target_node_id: int = Field(..., ge=-1, description="Target node ID (-1 for global)")
    strength: float = Field(..., description="Modulation strength")
    mode: str = Field(..., pattern="^(add|multiply|replace|conditional)$", description="Modulation mode")
    condition_node_id: int = Field(default=-1, ge=-1, description="Condition node ID")
    condition_value: float = Field(default=0.0, description="Condition threshold")


class ModulationRouteResponse(BaseModel):
    """Response with modulation route info."""
    route_id: int
    route: Dict
    status: str


class NodeConfigurationRequest(BaseModel):
    """Request to save/load node configuration."""
    name: str = Field(..., description="Configuration name")
    description: Optional[str] = Field(default="", description="Configuration description")
    nodes: List[Dict] = Field(..., description="Node configurations")
    modulation_routes: Optional[List[Dict]] = Field(default_factory=list, description="Modulation routes")


class NodeConfigurationResponse(BaseModel):
    """Response with saved configuration."""
    config_id: str
    name: str
    description: str
    node_count: int
    created_at: str


class THRMLHeterogeneousRequest(BaseModel):
    """Request to configure heterogeneous THRML model."""
    node_types: List[int] = Field(..., description="Node type IDs (0=Spin, 1=Continuous, 2=Discrete)")
    
    @field_validator('node_types')
    def validate_node_types(cls, v):
        """Validate node types."""
        for node_type in v:
            if node_type not in [0, 1, 2]:
                raise ValueError(f"Invalid node type: {node_type}. Must be 0, 1, or 2.")
        return v


class THRMLClampNodesRequest(BaseModel):
    """Request to clamp THRML nodes."""
    node_ids: List[int] = Field(..., description="Node IDs to clamp")
    values: List[float] = Field(..., description="Values to clamp to")
    
    @field_validator('values')
    def validate_values_length(cls, v, info: ValidationInfo):
        """Validate values length matches node_ids."""
        node_ids = info.data.get('node_ids', []) if info.data else []
        if len(v) != len(node_ids):
            raise ValueError("Length of values must match length of node_ids")
        return v


class THRMLFactorRequest(BaseModel):
    """Request to add custom THRML factor."""
    factor_type: str = Field(..., pattern="^(photonic_coupling|audio_harmony|ml_regularization)$")
    node_ids: List[int] = Field(..., description="Nodes involved in factor")
    strength: float = Field(default=1.0, description="Factor strength")
    parameters: Optional[Dict[str, float]] = Field(default_factory=dict, description="Factor-specific parameters")

