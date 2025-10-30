"""
Default configuration values and parameter mappings for GMCS.

This module provides default values for all system parameters and utility
functions for mapping external inputs (like audio) to simulation parameters.
"""

from typing import Dict, Callable
import numpy as np


# ============================================
# System Defaults
# ============================================

DEFAULT_N_NODES = 64
DEFAULT_GRID_SIZE = 256
DEFAULT_DT = 0.01
DEFAULT_WAVE_SPEED = 1.0

# Chua oscillator parameters
CHUA_ALPHA = 10.0
CHUA_BETA = 14.87
CHUA_M0 = -1.143
CHUA_M1 = -0.714

# Wave PDE parameters
WAVE_DAMPING = 0.1
WAVE_SOURCE_ALPHA = 0.05

# GMCS parameters
GMCS_A_MAX_DEFAULT = 0.8
GMCS_R_COMP_DEFAULT = 2.0
GMCS_T_COMP_DEFAULT = 0.5
GMCS_GAMMA_DEFAULT = 1.0
GMCS_BETA_DEFAULT = 1.0
GMCS_PHI_DEFAULT = 0.0
GMCS_OMEGA_DEFAULT = 1.0

# EBM parameters
EBM_LEARNING_RATE = 0.01
EBM_FEEDBACK_GAIN = 0.05
EBM_UPDATE_INTERVAL = 10

# WebSocket parameters
WS_PING_INTERVAL = 20.0
WS_PING_TIMEOUT = 20.0
WS_MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# ============================================
# Audio Mapping Functions
# ============================================

def map_pitch_to_c(pitch_hz: float, fmin: float = 80.0, fmax: float = 800.0) -> float:
    """
    Map pitch frequency to wave speed.
    
    Args:
        pitch_hz: Input pitch in Hz
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Wave speed in range [0.5, 2.0]
    """
    if pitch_hz <= 0 or np.isnan(pitch_hz):
        return 1.0  # Default
    
    # Logarithmic mapping for musical perception
    log_pitch = np.log(np.clip(pitch_hz, fmin, fmax))
    log_min = np.log(fmin)
    log_max = np.log(fmax)
    
    normalized = (log_pitch - log_min) / (log_max - log_min)
    return 0.5 + normalized * 1.5  # Map to [0.5, 2.0]


def map_rms_to_k(rms: float, rms_max: float = 0.3) -> float:
    """
    Map RMS amplitude to source strength.
    
    Args:
        rms: Root mean square amplitude
        rms_max: Maximum expected RMS
        
    Returns:
        Source strength in range [0.0, 10.0]
    """
    if rms <= 0 or np.isnan(rms):
        return 1.0  # Default
    
    normalized = np.clip(rms / rms_max, 0.0, 1.0)
    return normalized * 10.0


def map_pitch_to_color_index(pitch_hz: float, fmin: float = 80.0, fmax: float = 800.0) -> int:
    """
    Map pitch to color scheme index (0-3).
    
    Args:
        pitch_hz: Input pitch in Hz
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Color scheme index (0=cyan, 1=magenta, 2=chromatic, 3=static)
    """
    if pitch_hz <= 0 or np.isnan(pitch_hz):
        return 0  # Default to cyan
    
    log_pitch = np.log(np.clip(pitch_hz, fmin, fmax))
    log_min = np.log(fmin)
    log_max = np.log(fmax)
    
    normalized = (log_pitch - log_min) / (log_max - log_min)
    return int(normalized * 4) % 4


def map_audio_features(pitch_hz: float | None, rms: float | None) -> Dict[str, float]:
    """
    Map audio features to simulation parameters.
    
    Args:
        pitch_hz: Pitch frequency in Hz (None if not detected)
        rms: RMS amplitude (None if not available)
        
    Returns:
        Dictionary of parameter updates
    """
    updates = {}
    
    if pitch_hz is not None and pitch_hz > 0:
        updates['c_val'] = map_pitch_to_c(pitch_hz)
    
    if rms is not None and rms > 0:
        # Map RMS to multiple parameters for richer response
        k_strength = map_rms_to_k(rms)
        updates['k_strengths_gain'] = k_strength
        
        # Also modulate GMCS parameters
        updates['gmcs_Phi_gain'] = rms * 2.0  # Phase modulation depth
        updates['gmcs_omega_gain'] = 1.0 + rms  # Phase modulation frequency
    
    return updates


# ============================================
# Parameter Ranges & Validation
# ============================================

PARAMETER_RANGES = {
    # Time parameters
    'dt': (0.001, 0.1),
    't': (0.0, float('inf')),
    
    # Wave parameters
    'c_val': (0.1, 5.0),
    'k_strengths': (0.0, 20.0),
    
    # GMCS parameters
    'gmcs_A_max': (0.1, 2.0),
    'gmcs_R_comp': (1.0, 10.0),
    'gmcs_T_comp': (0.0, 1.0),
    'gmcs_R_exp': (1.0, 10.0),
    'gmcs_T_exp': (0.0, 1.0),
    'gmcs_Phi': (0.0, 2.0),
    'gmcs_omega': (0.1, 10.0),
    'gmcs_gamma': (0.1, 5.0),
    'gmcs_beta': (0.1, 5.0),
    
    # EBM parameters
    'ebm_eta': (0.001, 0.1),
    'ebm_feedback_gain': (0.0, 0.2),
    
    # Node parameters
    'node_positions': (0.0, 256.0),  # Grid coordinates
}


def validate_parameter(name: str, value: float) -> float:
    """
    Validate and clip parameter to acceptable range.
    
    Args:
        name: Parameter name
        value: Proposed value
        
    Returns:
        Clipped value within valid range
        
    Raises:
        ValueError: If parameter name is unknown
    """
    if name not in PARAMETER_RANGES:
        raise ValueError(f"Unknown parameter: {name}")
    
    min_val, max_val = PARAMETER_RANGES[name]
    return np.clip(value, min_val, max_val)


# ============================================
# Preset Parameter Sets
# ============================================

AUDIO_REACTIVE_PARAMS = {
    'dt': 0.01,
    'c_val': 1.0,
    'gmcs_Phi_base': 0.5,
    'gmcs_omega_base': 2.0,
    'ebm_feedback_gain': 0.05,
    'ebm_update_interval': 10,
}

STABLE_DYNAMICS_PARAMS = {
    'dt': 0.005,
    'c_val': 0.8,
    'gmcs_A_max': 0.6,
    'gmcs_R_comp': 3.0,
    'ebm_learning_rate': 0.005,
    'ebm_feedback_gain': 0.02,
}

CHAOTIC_EXPLORATION_PARAMS = {
    'dt': 0.01,
    'c_val': 1.5,
    'gmcs_gamma': 2.0,
    'gmcs_beta': 1.5,
    'ebm_learning_rate': 0.02,
    'ebm_feedback_gain': 0.1,
}

PHOTONIC_SIMULATION_PARAMS = {
    'dt': 0.0001,  # Femtosecond timescale
    'c_val': 2.99792458,  # Speed of light / refractive index
    'gmcs_A_max': 1.0,  # Optical intensity limits
    'ebm_learning_rate': 0.001,
    'ebm_feedback_gain': 0.0,  # Disable for pure optical simulation
}


# ============================================
# Domain-Specific Mappings
# ============================================

DOMAIN_CONTROL_MAPPINGS: Dict[str, Dict[str, Callable]] = {
    'audio': {
        'pitch': map_pitch_to_c,
        'amplitude': map_rms_to_k,
    },
    'crypto': {
        # Password entropy → chaos initial conditions
        'entropy': lambda x: np.clip(x / 256.0, 0.0, 1.0),
    },
    'generative': {
        # Seed → deterministic noise
        'seed': lambda x: (x % 1000) / 1000.0,
        # Roughness → chaos intensity
        'roughness': lambda x: np.clip(x, 0.0, 2.0),
    },
    'anomaly': {
        # Input variance → EBM sensitivity
        'variance': lambda x: np.clip(x * 10.0, 0.0, 1.0),
    },
}


def get_domain_mapping(domain: str, control_name: str) -> Callable | None:
    """
    Get control mapping function for a specific domain.
    
    Args:
        domain: Domain name (e.g., 'audio', 'crypto')
        control_name: Control parameter name
        
    Returns:
        Mapping function, or None if not found
    """
    return DOMAIN_CONTROL_MAPPINGS.get(domain, {}).get(control_name)


# ============================================
# CFL Stability Check
# ============================================

def check_cfl_stability(c: float, dt: float, dx: float = 1.0) -> bool:
    """
    Check CFL (Courant-Friedrichs-Lewy) stability condition for wave equation.
    
    Condition: c * dt / dx <= 1/sqrt(2) for 2D wave equation
    
    Args:
        c: Wave speed
        dt: Time step
        dx: Spatial step
        
    Returns:
        True if stable, False otherwise
    """
    cfl_number = c * dt / dx
    cfl_limit = 1.0 / np.sqrt(2)
    return cfl_number <= cfl_limit


def compute_stable_dt(c: float, dx: float = 1.0, safety_factor: float = 0.9) -> float:
    """
    Compute maximum stable time step for given wave speed.
    
    Args:
        c: Wave speed
        dx: Spatial step
        safety_factor: Safety margin (< 1.0)
        
    Returns:
        Maximum stable dt
    """
    cfl_limit = 1.0 / np.sqrt(2)
    dt_max = (cfl_limit * dx) / c
    return dt_max * safety_factor
