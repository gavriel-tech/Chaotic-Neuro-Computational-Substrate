"""
Energy-Based Model (EBM) learning using THRML for block Gibbs sampling.

Implements Contrastive Divergence learning for coupling oscillators through
learned weights using THRML's probabilistic graphical model framework.

THRML provides:
- Blocked Gibbs sampling for discrete PGMs
- Efficient GPU-accelerated sampling
- Support for heterogeneous graphical models
- Energy-based model utilities

This module now uses THRML as the primary EBM implementation.
"""

from collections import deque
from typing import Tuple, Optional, Deque, List

import jax
import jax.numpy as jnp
import numpy as np

# THRML integration (required)
from src.core.thrml_integration import (
    THRMLWrapper,
    create_thrml_model,
    reconstruct_thrml_wrapper
)


THRML_SAMPLE_HISTORY_LENGTH = 512

# Global buffers for THRML visualizers
_last_thrml_sample: Optional[np.ndarray] = None
_thrml_sample_history: Deque[np.ndarray] = deque(maxlen=THRML_SAMPLE_HISTORY_LENGTH)
_last_thrml_feedback: Optional[np.ndarray] = None
_last_thrml_feedback_norm: float = 0.0


def _record_thrml_sample(sample: np.ndarray) -> None:
    """Store latest THRML sample for visualizers and history."""
    global _last_thrml_sample

    if sample is None:
        return

    # Ensure copy to avoid accidental mutation from callers
    _last_thrml_sample = np.array(sample, dtype=np.float32).copy()
    _thrml_sample_history.append(_last_thrml_sample.copy())


def get_last_thrml_sample() -> Optional[np.ndarray]:
    """Return last recorded THRML binary state sample."""
    if _last_thrml_sample is None:
        return None
    return _last_thrml_sample.copy()


def update_thrml_sample(sample: np.ndarray) -> None:
    """Externally record a THRML sample (used by diagnostics or endpoints)."""
    _record_thrml_sample(sample)


def get_thrml_sample_history(max_samples: Optional[int] = None) -> List[List[float]]:
    """Return history of THRML samples (most recent first)."""
    if not _thrml_sample_history:
        return []

    history = list(_thrml_sample_history)
    if max_samples is not None:
        history = history[-max_samples:]

    return [sample.astype(np.float32).tolist() for sample in history]


def get_last_thrml_feedback_norm() -> float:
    return float(_last_thrml_feedback_norm)


def get_last_thrml_feedback() -> Optional[np.ndarray]:
    if _last_thrml_feedback is None:
        return None
    return _last_thrml_feedback.copy()


@jax.jit
def binary_state_from_x(x_vec: jnp.ndarray, threshold: float = 0.0) -> jnp.ndarray:
    """
    Convert continuous oscillator x values to binary states {-1, +1}.
    
    Uses sign function: s_i = sgn(x_i - threshold)
    
    Args:
        x_vec: (N,) array of x values
        threshold: threshold value (default 0.0)
        
    Returns:
        (N,) array of binary states {-1, +1}
    """
    return jnp.where(x_vec > threshold, 1.0, -1.0)


def ebm_update_with_thrml(
    thrml_wrapper: THRMLWrapper,
    oscillator_states: jnp.ndarray,
    mask: jnp.ndarray,
    eta: float,
    k_steps: int,
    key: jax.random.PRNGKey,
    x_threshold: float = 0.0,
    n_chains: int = 1
) -> Tuple[THRMLWrapper, dict]:
    """
    Update THRML model using optimized CD-k learning with parallel chains.
    
    This replaces the custom JAX CD-1 implementation with THRML's
    efficient block Gibbs sampling, now with support for parallel chains
    for better gradient estimates.
    
    Algorithm:
    1. Convert oscillator x-values to binary states
    2. Apply mask to get active node states
    3. Call THRML wrapper's update_weights_cd() with n_chains
    4. Return updated wrapper and diagnostics
    
    Args:
        thrml_wrapper: THRMLWrapper instance
        oscillator_states: (N_MAX, 3) oscillator [x, y, z] states
        mask: (N_MAX,) active node mask
        eta: Learning rate
        k_steps: Number of Gibbs steps for CD-k
        key: JAX random key
        x_threshold: Threshold for binary conversion
        n_chains: Number of parallel sampling chains (default 1)
        
    Returns:
        Tuple of (updated THRMLWrapper, diagnostics dict)
    """
    # Extract x values (first dimension of oscillator states)
    x_values = oscillator_states[:, 0]
    
    # Convert to binary states
    s_data = binary_state_from_x(x_values, x_threshold)
    
    # Apply mask to data states
    s_data_masked = s_data * mask
    
    # Convert to numpy for THRML
    s_data_np = np.array(s_data_masked)
    
    # Update THRML weights using CD-k with parallel chains
    diagnostics = thrml_wrapper.update_weights_cd(
        data_states=s_data_np,
        eta=eta,
        k_steps=k_steps,
        key=key,
        n_chains=n_chains  # Use parallel chains for better gradient estimates
    )
    
    return thrml_wrapper, diagnostics


def compute_thrml_feedback(
    thrml_wrapper: THRMLWrapper,
    gmcs_biases: jnp.ndarray,
    temperature: float,
    gibbs_steps: int,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """
    Sample THRML and compute feedback for oscillators.
    
    This is the primary feedback mechanism from the EBM to the oscillators.
    
    Algorithm:
    1. Update THRML biases from GMCS pipeline
    2. Sample binary states via THRML block Gibbs
    3. Compute feedback: W @ binary_states
    4. Return as JAX array
    
    Args:
        thrml_wrapper: THRMLWrapper instance
        gmcs_biases: (N_MAX,) biases from GMCS pipeline
        temperature: Sampling temperature
        gibbs_steps: Number of Gibbs steps
        key: JAX random key
        
    Returns:
        (N_MAX,) array of feedback terms
    """
    # Update THRML biases from GMCS
    gmcs_biases_np = np.array(gmcs_biases)
    thrml_wrapper.update_biases(gmcs_biases_np)
    
    # Sample binary states using THRML
    binary_states = thrml_wrapper.sample_gibbs(
        n_steps=gibbs_steps,
        temperature=temperature,
        key=key
    )
    
    # Record sample for visualizers
    _record_thrml_sample(binary_states)

    # Get weight matrix
    weights = thrml_wrapper.get_weights()
    
    # Compute feedback: W @ s
    feedback = np.dot(weights, binary_states)
    
    global _last_thrml_feedback, _last_thrml_feedback_norm
    _last_thrml_feedback = np.array(feedback, dtype=np.float32).copy()
    _last_thrml_feedback_norm = float(np.linalg.norm(_last_thrml_feedback))

    # Log feedback stats for diagnostics
    feedback_norm = _last_thrml_feedback_norm
    feedback_max = np.max(np.abs(feedback))
    if feedback_norm > 0:
        print(f"[THRML] Feedback computed: norm={feedback_norm:.4f}, max={feedback_max:.4f}, n_nodes={thrml_wrapper.n_nodes}")
    
    # Convert to JAX array and pad to N_MAX
    n_max = gmcs_biases.shape[0]
    feedback_jax = jnp.zeros(n_max, dtype=jnp.float32)
    feedback_jax = feedback_jax.at[:len(feedback)].set(jnp.array(feedback, dtype=jnp.float32))
    
    return feedback_jax


@jax.jit
def compute_ebm_bias(
    ebm_weights: jnp.ndarray,
    oscillator_states: jnp.ndarray,
    mask: jnp.ndarray,
    beta: float,
    x_threshold: float = 0.0
) -> jnp.ndarray:
    """
    Compute EBM feedback bias for oscillator dynamics (legacy function).
    
    Note: This is kept for backward compatibility. New code should use
    compute_thrml_feedback() instead.
    
    Bias term: β * Σ_j W_ij * s_j
    
    This is added to the oscillator dynamics to create feedback from
    learned coupling weights.
    
    Args:
        ebm_weights: (N_MAX, N_MAX) weight matrix
        oscillator_states: (N_MAX, 3) oscillator states
        mask: (N_MAX,) active node mask
        beta: feedback gain (typically 0.0-0.1)
        x_threshold: threshold for binary conversion
        
    Returns:
        (N_MAX,) array of bias terms
    """
    # Extract x values
    x_values = oscillator_states[:, 0]
    
    # Compute binary states
    s_vec = binary_state_from_x(x_values, x_threshold)
    
    # Apply mask
    s_vec = s_vec * mask
    
    # Compute bias: β * W @ s
    bias = beta * jnp.dot(ebm_weights, s_vec)
    
    # Apply mask to bias
    bias = bias * mask
    
    return bias


def compute_ebm_energy_thrml(
    thrml_wrapper: THRMLWrapper,
    oscillator_states: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    x_threshold: float = 0.0
) -> float:
    """
    Compute EBM energy using THRML wrapper.
    
    Energy: E = -1/2 * s^T @ W @ s - b^T @ s
    
    Lower energy indicates more probable states.
    
    Args:
        thrml_wrapper: THRMLWrapper instance
        oscillator_states: (N_MAX, 3) oscillator states (optional, will sample if None)
        mask: (N_MAX,) active node mask (optional)
        x_threshold: Threshold for binary conversion
        
    Returns:
        Scalar energy value
    """
    if oscillator_states is not None:
        # Extract x values and compute binary states
        x_values = oscillator_states[:, 0]
        s_vec = binary_state_from_x(x_values, x_threshold)
        
        if mask is not None:
            s_vec = s_vec * mask
        
        # Convert to numpy
        s_vec_np = np.array(s_vec[:thrml_wrapper.n_nodes])
        
        # Compute energy using THRML
        energy = thrml_wrapper.compute_energy(s_vec_np)
    else:
        # Sample current state and compute energy
        energy = thrml_wrapper.compute_energy()
    
    return float(energy)


@jax.jit
def compute_ebm_energy(
    ebm_weights: jnp.ndarray,
    oscillator_states: jnp.ndarray,
    mask: jnp.ndarray,
    x_threshold: float = 0.0
) -> jnp.ndarray:
    """
    Compute EBM energy for current state (legacy function).
    
    Note: This is kept for backward compatibility. New code should use
    compute_ebm_energy_thrml() instead.
    
    Energy: E = -1/2 * Σ_i Σ_j W_ij * s_i * s_j
    
    Lower energy indicates more probable states.
    
    Args:
        ebm_weights: (N_MAX, N_MAX) weight matrix
        oscillator_states: (N_MAX, 3) oscillator states
        mask: (N_MAX,) active node mask
        x_threshold: threshold for binary conversion
        
    Returns:
        Scalar energy value
    """
    # Extract x values and compute binary states
    x_values = oscillator_states[:, 0]
    s_vec = binary_state_from_x(x_values, x_threshold)
    s_vec = s_vec * mask
    
    # Energy: E = -1/2 * s^T @ W @ s
    energy = -0.5 * jnp.dot(s_vec, jnp.dot(ebm_weights, s_vec))
    
    return energy


def normalize_weights(
    ebm_weights: jnp.ndarray,
    max_weight: float = 1.0
) -> jnp.ndarray:
    """
    Normalize EBM weights to prevent unbounded growth.
    
    Args:
        ebm_weights: (N_MAX, N_MAX) weight matrix
        max_weight: maximum allowed weight magnitude
        
    Returns:
        Normalized weight matrix
    """
    # Clip weights to range
    clipped = jnp.clip(ebm_weights, -max_weight, max_weight)
    
    return clipped


# ============================================================================
# Utility Functions
# ============================================================================


@jax.jit
def compute_weight_statistics(ebm_weights: jnp.ndarray, mask: jnp.ndarray) -> dict:
    """
    Compute statistics about weight matrix.
    
    Args:
        ebm_weights: (N_MAX, N_MAX) weight matrix
        mask: (N_MAX,) active node mask
        
    Returns:
        Dictionary with statistics
    """
    # Get active portion of weights
    n_active = int(jnp.sum(mask))
    
    if n_active == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'max': 0.0,
            'min': 0.0,
        }
    
    # Extract active submatrix
    mask_2d = jnp.outer(mask, mask)
    active_weights = ebm_weights * mask_2d
    
    # Compute statistics (excluding zeros from inactive nodes)
    nonzero_mask = mask_2d > 0.5
    active_values = active_weights[nonzero_mask]
    
    return {
        'mean': float(jnp.mean(active_values)),
        'std': float(jnp.std(active_values)),
        'max': float(jnp.max(active_values)),
        'min': float(jnp.min(active_values)),
    }
