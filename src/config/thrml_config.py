"""
THRML Performance Mode Configuration for GMCS.

This module defines performance presets that control THRML sampling
and learning parameters, allowing users to balance speed vs. accuracy
based on their application needs.

Performance Modes:
- SPEED: Minimal sampling, fast updates, suitable for real-time visualization
- ACCURACY: More sampling, better gradients, suitable for production applications
- RESEARCH: Extensive sampling, research-quality results, suitable for experiments
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class PerformanceMode(Enum):
    """THRML performance mode enumeration."""
    SPEED = "speed"
    ACCURACY = "accuracy"
    RESEARCH = "research"


@dataclass
class THRMLPerformanceConfig:
    """
    Configuration for THRML sampling and learning parameters.
    
    Attributes:
        mode: Performance mode identifier
        gibbs_steps: Number of Gibbs sampling steps per simulation update
        temperature: Sampling temperature (higher = more stochastic)
        learning_rate: Weight update learning rate (eta)
        cd_k_steps: Number of Gibbs steps for CD-k learning
        weight_update_freq: Update weights every N simulation steps
        use_jit: Whether to use JIT compilation (True for production)
        batch_size: Batch size for sampling (currently 1)
        n_chains: Number of parallel sampling chains for CD learning
        use_observers: Enable THRML observers for diagnostics
        track_energy: Track energy trajectory during sampling
        track_correlations: Track spin correlations
        description: Human-readable description of the mode
    """
    mode: PerformanceMode
    
    # Sampling parameters
    gibbs_steps: int
    temperature: float
    
    # Learning parameters
    learning_rate: float
    cd_k_steps: int
    weight_update_freq: int
    n_chains: int  # Parallel chains for better gradient estimates
    
    # Optimization flags
    use_jit: bool
    batch_size: int
    
    # Diagnostic features
    use_observers: bool
    track_energy: bool
    track_correlations: bool
    
    # Metadata
    description: str


# ============================================================================
# Performance Presets
# ============================================================================

PERFORMANCE_PRESETS: Dict[PerformanceMode, THRMLPerformanceConfig] = {
    PerformanceMode.SPEED: THRMLPerformanceConfig(
        mode=PerformanceMode.SPEED,
        gibbs_steps=5,           # Minimal sampling for real-time
        temperature=1.0,         # Standard temperature
        learning_rate=0.01,      # Moderate learning rate
        cd_k_steps=1,            # CD-1 (fastest)
        weight_update_freq=20,   # Update less often
        n_chains=1,              # Single chain for speed
        use_jit=True,            # Enable JIT for speed
        batch_size=1,            # Single sample
        use_observers=False,     # No observers for speed
        track_energy=False,      # No tracking
        track_correlations=False,
        description=(
            "Speed mode: Optimized for real-time visualization and interactive "
            "applications. Uses minimal Gibbs sampling (5 steps) and CD-1 learning "
            "with single chain for fast updates. Best for live audio-reactive "
            "performances and rapid prototyping."
        )
    ),
    
    PerformanceMode.ACCURACY: THRMLPerformanceConfig(
        mode=PerformanceMode.ACCURACY,
        gibbs_steps=50,          # More sampling for accuracy
        temperature=0.5,         # Lower temp = more deterministic
        learning_rate=0.001,     # Lower learning rate for stability
        cd_k_steps=5,            # CD-5 (better gradient estimates)
        weight_update_freq=5,    # Update more often
        n_chains=3,              # Multiple chains for better gradients
        use_jit=True,            # Keep JIT enabled
        batch_size=1,            # Single sample
        use_observers=True,      # Enable observers
        track_energy=True,       # Track energy for monitoring
        track_correlations=False, # Skip correlations for speed
        description=(
            "Accuracy mode: Balanced performance for production applications. "
            "Uses 50 Gibbs steps and CD-5 learning with 3 parallel chains for "
            "better gradient estimates. Lower temperature (0.5) produces more "
            "deterministic behavior. Energy tracking enabled. Suitable for "
            "photonic processor simulation and ML model integration."
        )
    ),
    
    PerformanceMode.RESEARCH: THRMLPerformanceConfig(
        mode=PerformanceMode.RESEARCH,
        gibbs_steps=100,         # Extensive sampling
        temperature=1.0,         # Standard temperature
        learning_rate=0.005,     # Moderate learning rate
        cd_k_steps=10,           # CD-10 (research quality)
        weight_update_freq=10,   # Balanced update frequency
        n_chains=5,              # Many chains for best gradients
        use_jit=False,           # Disable JIT for inspection
        batch_size=1,            # Single sample
        use_observers=True,      # Full diagnostics
        track_energy=True,       # Track everything
        track_correlations=True, # Full correlation tracking
        description=(
            "Research mode: High-quality sampling for scientific experiments and "
            "benchmarking. Uses 100 Gibbs steps and CD-10 learning with 5 parallel "
            "chains for accurate gradient estimation. Full diagnostic observers "
            "enabled including energy and correlation tracking. JIT disabled to "
            "allow for intermediate state inspection. Best for publications and "
            "detailed analysis."
        )
    )
}


def get_performance_config(mode: str) -> THRMLPerformanceConfig:
    """
    Get performance configuration by mode name.
    
    Args:
        mode: Mode name ('speed', 'accuracy', or 'research')
        
    Returns:
        THRMLPerformanceConfig for the specified mode
        
    Raises:
        ValueError: If mode is not recognized
    """
    try:
        mode_enum = PerformanceMode(mode.lower())
        return PERFORMANCE_PRESETS[mode_enum]
    except (ValueError, KeyError):
        raise ValueError(
            f"Invalid performance mode: {mode}. "
            f"Must be one of: {[m.value for m in PerformanceMode]}"
        )


def list_performance_modes() -> Dict[str, str]:
    """
    List all available performance modes with descriptions.
    
    Returns:
        Dictionary mapping mode names to descriptions
    """
    return {
        mode.value: config.description
        for mode, config in PERFORMANCE_PRESETS.items()
    }


def validate_performance_mode(mode: str) -> bool:
    """
    Check if a performance mode name is valid.
    
    Args:
        mode: Mode name to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        PerformanceMode(mode.lower())
        return True
    except ValueError:
        return False


# ============================================================================
# Helper Functions for Dynamic Configuration
# ============================================================================

def create_custom_config(
    base_mode: str = 'speed',
    **overrides
) -> THRMLPerformanceConfig:
    """
    Create a custom configuration based on a preset with overrides.
    
    Args:
        base_mode: Base performance mode to start from
        **overrides: Keyword arguments to override specific parameters
        
    Returns:
        New THRMLPerformanceConfig with overrides applied
        
    Example:
        >>> config = create_custom_config('speed', gibbs_steps=10, temperature=0.8)
        >>> print(config.gibbs_steps)  # 10
        >>> print(config.cd_k_steps)   # 1 (from speed preset)
    """
    base_config = get_performance_config(base_mode)
    
    # Create a dictionary from the base config
    config_dict = {
        'mode': base_config.mode,
        'gibbs_steps': base_config.gibbs_steps,
        'temperature': base_config.temperature,
        'learning_rate': base_config.learning_rate,
        'cd_k_steps': base_config.cd_k_steps,
        'weight_update_freq': base_config.weight_update_freq,
        'n_chains': base_config.n_chains,
        'use_jit': base_config.use_jit,
        'batch_size': base_config.batch_size,
        'use_observers': base_config.use_observers,
        'track_energy': base_config.track_energy,
        'track_correlations': base_config.track_correlations,
        'description': base_config.description + ' (custom)'
    }
    
    # Apply overrides
    config_dict.update(overrides)
    
    return THRMLPerformanceConfig(**config_dict)


def interpolate_configs(
    mode1: str,
    mode2: str,
    alpha: float = 0.5
) -> THRMLPerformanceConfig:
    """
    Interpolate between two performance configurations.
    
    Useful for creating intermediate configurations between presets.
    
    Args:
        mode1: First mode name
        mode2: Second mode name
        alpha: Interpolation factor (0.0 = mode1, 1.0 = mode2)
        
    Returns:
        Interpolated THRMLPerformanceConfig
        
    Example:
        >>> # Create a config halfway between speed and accuracy
        >>> config = interpolate_configs('speed', 'accuracy', alpha=0.5)
    """
    config1 = get_performance_config(mode1)
    config2 = get_performance_config(mode2)
    
    # Clamp alpha to [0, 1]
    alpha = max(0.0, min(1.0, alpha))
    
    # Interpolate numeric parameters
    return THRMLPerformanceConfig(
        mode=config1.mode,  # Use first mode's enum
        gibbs_steps=int(
            config1.gibbs_steps * (1 - alpha) + config2.gibbs_steps * alpha
        ),
        temperature=config1.temperature * (1 - alpha) + config2.temperature * alpha,
        learning_rate=config1.learning_rate * (1 - alpha) + config2.learning_rate * alpha,
        cd_k_steps=int(
            config1.cd_k_steps * (1 - alpha) + config2.cd_k_steps * alpha
        ),
        weight_update_freq=int(
            config1.weight_update_freq * (1 - alpha) + config2.weight_update_freq * alpha
        ),
        n_chains=int(
            config1.n_chains * (1 - alpha) + config2.n_chains * alpha
        ),
        use_jit=config1.use_jit if alpha < 0.5 else config2.use_jit,
        batch_size=config1.batch_size,
        use_observers=config1.use_observers if alpha < 0.5 else config2.use_observers,
        track_energy=config1.track_energy if alpha < 0.5 else config2.track_energy,
        track_correlations=config1.track_correlations if alpha < 0.5 else config2.track_correlations,
        description=f"Interpolated between {mode1} and {mode2} (alpha={alpha:.2f})"
    )


# ============================================================================
# Application-Specific Recommendations
# ============================================================================

# Recommended modes for different GMCS applications
APPLICATION_RECOMMENDATIONS = {
    # Generative AI
    'terrain_generation': PerformanceMode.SPEED,
    'texture_synthesis': PerformanceMode.SPEED,
    'latent_space_exploration': PerformanceMode.ACCURACY,
    
    # Machine Learning
    'data_augmentation': PerformanceMode.SPEED,
    'reinforcement_learning': PerformanceMode.ACCURACY,
    'feature_extraction': PerformanceMode.ACCURACY,
    
    # Cryptography
    'stream_cipher': PerformanceMode.ACCURACY,
    'key_generation': PerformanceMode.RESEARCH,
    'puf_simulation': PerformanceMode.RESEARCH,
    
    # Scientific Computing
    'photonic_processor': PerformanceMode.ACCURACY,
    'emergence_study': PerformanceMode.RESEARCH,
    'anomaly_detection': PerformanceMode.ACCURACY,
    
    # Audio/Visual
    'live_performance': PerformanceMode.SPEED,
    'music_production': PerformanceMode.ACCURACY,
    'visualization': PerformanceMode.SPEED,
}


def get_recommended_mode(application: str) -> PerformanceMode:
    """
    Get recommended performance mode for a specific application.
    
    Args:
        application: Application name (e.g., 'terrain_generation')
        
    Returns:
        Recommended PerformanceMode
        
    Raises:
        ValueError: If application is not recognized
    """
    if application not in APPLICATION_RECOMMENDATIONS:
        # Default to SPEED for unknown applications
        return PerformanceMode.SPEED
    
    return APPLICATION_RECOMMENDATIONS[application]

