"""
Loss Functions for GMCS ML Training.

Comprehensive loss functions for training ML models on chaotic data,
including chaos-specific losses, standard losses, and combined objectives.

Key features:
- Trajectory matching losses
- Attractor-based losses
- Lyapunov/chaos-aware losses
- Energy landscape losses (THRML integration)
- Standard ML losses
- Multi-objective combinations

Use cases:
- Train to match target dynamics
- Optimize for chaos properties
- Align with THRML energy
- Standard supervised/unsupervised learning
"""

from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# ============================================================================
# Chaos-Specific Losses
# ============================================================================

def trajectory_mse_loss(
    predicted: Union[np.ndarray, jnp.ndarray, torch.Tensor],
    target: Union[np.ndarray, jnp.ndarray, torch.Tensor],
    framework: str = 'numpy'
) -> float:
    """
    Mean squared error between trajectories.
    
    Args:
        predicted: Predicted trajectory
        target: Target trajectory
        framework: 'numpy', 'jax', 'pytorch', or 'tensorflow'
        
    Returns:
        MSE loss
    """
    if framework == 'jax':
        return float(jnp.mean((predicted - target) ** 2))
    elif framework == 'pytorch':
        return float(F.mse_loss(predicted, target).item())
    elif framework == 'tensorflow':
        return float(tf.reduce_mean((predicted - target) ** 2).numpy())
    else:
        return float(np.mean((predicted - target) ** 2))


def trajectory_mae_loss(
    predicted: Union[np.ndarray, jnp.ndarray],
    target: Union[np.ndarray, jnp.ndarray]
) -> float:
    """
    Mean absolute error between trajectories.
    
    Args:
        predicted: Predicted trajectory
        target: Target trajectory
        
    Returns:
        MAE loss
    """
    if JAX_AVAILABLE and isinstance(predicted, jnp.ndarray):
        return float(jnp.mean(jnp.abs(predicted - target)))
    return float(np.mean(np.abs(predicted - target)))


def attractor_distance_loss(
    predicted_traj: np.ndarray,
    target_traj: np.ndarray,
    metric: str = 'frobenius'
) -> float:
    """
    Loss based on distance between attractors in phase space.
    
    Compares statistical properties of attractors rather than
    point-wise trajectory matching.
    
    Args:
        predicted_traj: (n_steps, dim) predicted trajectory
        target_traj: (n_steps, dim) target trajectory
        metric: 'frobenius', 'hausdorff', or 'correlation'
        
    Returns:
        Attractor distance loss
    """
    if metric == 'frobenius':
        # Compare distance matrices (pairwise distances within attractors)
        pred_dists = np.linalg.norm(
            predicted_traj[:, None] - predicted_traj[None, :], axis=-1
        )
        target_dists = np.linalg.norm(
            target_traj[:, None] - target_traj[None, :], axis=-1
        )
        return float(np.linalg.norm(pred_dists - target_dists, ord='fro'))
    
    elif metric == 'hausdorff':
        # Approximate Hausdorff distance
        # d(A, B) = max(max_a min_b ||a-b||, max_b min_a ||b-a||)
        dists_pred_to_target = np.linalg.norm(
            predicted_traj[:, None] - target_traj[None, :], axis=-1
        )
        dists_target_to_pred = dists_pred_to_target.T
        
        forward = np.max(np.min(dists_pred_to_target, axis=1))
        backward = np.max(np.min(dists_target_to_pred, axis=1))
        
        return float(max(forward, backward))
    
    elif metric == 'correlation':
        # Compare correlation matrices
        pred_corr = np.corrcoef(predicted_traj.T)
        target_corr = np.corrcoef(target_traj.T)
        return float(np.mean((pred_corr - target_corr) ** 2))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def lyapunov_loss(
    trajectory: np.ndarray,
    target_lyapunov: float = 2.0,
    dt: float = 0.01
) -> float:
    """
    Loss to match target Lyapunov exponent.
    
    Encourages/discourages chaos by matching Lyapunov exponent.
    - Positive target → encourage chaos
    - Negative target → discourage chaos (stabilize)
    - Zero target → critical regime
    
    Args:
        trajectory: (n_steps, dim) trajectory
        target_lyapunov: Target Lyapunov exponent
        dt: Time step
        
    Returns:
        Lyapunov matching loss
    """
    # Simple approximation: rate of divergence
    diffs = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=-1)
    
    # Avoid log(0)
    distances = np.clip(distances, 1e-8, None)
    
    # Estimate exponential growth rate
    log_distances = np.log(distances)
    estimated_lyapunov = np.mean(np.diff(log_distances)) / dt
    
    return float((estimated_lyapunov - target_lyapunov) ** 2)


def chaos_entropy_loss(
    trajectory: np.ndarray,
    bins: int = 20
) -> float:
    """
    Loss based on trajectory entropy (diversity measure).
    
    Higher entropy → more chaotic/diverse
    Lower entropy → more regular/predictable
    
    Args:
        trajectory: (n_steps, dim) trajectory
        bins: Number of bins for histogram
        
    Returns:
        Negative entropy (minimize for more chaos)
    """
    # Compute histogram for each dimension
    total_entropy = 0.0
    
    for dim in range(trajectory.shape[1]):
        hist, _ = np.histogram(trajectory[:, dim], bins=bins, density=True)
        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]
        # Shannon entropy
        entropy = -np.sum(hist * np.log(hist))
        total_entropy += entropy
    
    # Return negative (minimize for more chaos)
    return -float(total_entropy)


def periodicity_loss(
    trajectory: np.ndarray,
    max_period: int = 100
) -> float:
    """
    Loss that penalizes periodic behavior.
    
    Lower loss → more periodic
    Higher loss → less periodic (more chaotic)
    
    Args:
        trajectory: (n_steps, dim) trajectory
        max_period: Maximum period to search
        
    Returns:
        Periodicity loss
    """
    # Autocorrelation to detect periodicity
    n_steps = len(trajectory)
    autocorr = []
    
    for lag in range(1, min(max_period, n_steps // 2)):
        # Compute autocorrelation for this lag
        corr = 0.0
        for dim in range(trajectory.shape[1]):
            signal = trajectory[:, dim]
            mean = np.mean(signal)
            var = np.var(signal)
            
            if var > 0:
                c = np.mean((signal[:-lag] - mean) * (signal[lag:] - mean)) / var
                corr += c
        
        autocorr.append(corr / trajectory.shape[1])
    
    # Find maximum autocorrelation (excluding lag=0)
    max_autocorr = np.max(np.abs(autocorr))
    
    # Return as loss (high autocorr = periodic = high loss if we want chaos)
    return float(max_autocorr)


# ============================================================================
# Energy-Based Losses (THRML Integration)
# ============================================================================

def thrml_energy_loss(
    states: np.ndarray,
    energy_fn: Callable[[np.ndarray], float],
    target_energy: Optional[float] = None
) -> float:
    """
    Loss based on THRML energy function.
    
    Can optimize to either:
    - Minimize energy (target_energy=None)
    - Match target energy (target_energy specified)
    
    Args:
        states: States to evaluate
        energy_fn: Function that computes energy
        target_energy: Target energy value (None to minimize)
        
    Returns:
        Energy-based loss
    """
    energies = np.array([energy_fn(state) for state in states])
    mean_energy = np.mean(energies)
    
    if target_energy is None:
        # Minimize energy
        return float(mean_energy)
    else:
        # Match target energy
        return float((mean_energy - target_energy) ** 2)


def thrml_alignment_loss(
    oscillator_states: np.ndarray,
    ebm_states: np.ndarray,
    temperature: float = 1.0
) -> float:
    """
    Loss to align oscillator dynamics with EBM energy landscape.
    
    Encourages oscillators to follow low-energy paths in THRML model.
    
    Args:
        oscillator_states: Oscillator trajectory
        ebm_states: Corresponding EBM states
        temperature: Temperature parameter
        
    Returns:
        Alignment loss
    """
    # Simple distance-based alignment
    distances = np.linalg.norm(oscillator_states - ebm_states, axis=-1)
    
    # Weighted by temperature
    loss = np.mean(distances) / temperature
    
    return float(loss)


# ============================================================================
# Standard ML Losses
# ============================================================================

def mse_loss(predicted, target, framework='numpy'):
    """Mean squared error (wrapper)."""
    return trajectory_mse_loss(predicted, target, framework)


def mae_loss(predicted, target):
    """Mean absolute error (wrapper)."""
    return trajectory_mae_loss(predicted, target)


def cross_entropy_loss(
    logits: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    framework: str = 'numpy'
) -> float:
    """
    Cross-entropy loss for classification.
    
    Args:
        logits: Predicted logits
        targets: Target class indices
        framework: Framework to use
        
    Returns:
        Cross-entropy loss
    """
    if framework == 'pytorch' and PYTORCH_AVAILABLE:
        return float(F.cross_entropy(logits, targets).item())
    elif framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return float(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits
        ).numpy().mean())
    else:
        # Numpy implementation
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Cross-entropy
        n = len(targets)
        log_probs = -np.log(probs[range(n), targets] + 1e-8)
        return float(np.mean(log_probs))


def binary_cross_entropy_loss(
    predicted: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    framework: str = 'numpy'
) -> float:
    """
    Binary cross-entropy loss.
    
    Args:
        predicted: Predicted probabilities
        target: Target labels (0 or 1)
        framework: Framework to use
        
    Returns:
        BCE loss
    """
    if framework == 'pytorch' and PYTORCH_AVAILABLE:
        return float(F.binary_cross_entropy(predicted, target).item())
    elif framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return float(tf.keras.losses.binary_crossentropy(target, predicted).numpy().mean())
    else:
        # Numpy implementation
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Clip to avoid log(0)
        predicted = np.clip(predicted, 1e-8, 1 - 1e-8)
        
        loss = -(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))
        return float(np.mean(loss))


def kl_divergence_loss(
    predicted_dist: np.ndarray,
    target_dist: np.ndarray
) -> float:
    """
    KL divergence between distributions.
    
    Args:
        predicted_dist: Predicted distribution
        target_dist: Target distribution
        
    Returns:
        KL divergence
    """
    # Clip to avoid log(0)
    predicted_dist = np.clip(predicted_dist, 1e-8, 1.0)
    target_dist = np.clip(target_dist, 1e-8, 1.0)
    
    kl = np.sum(target_dist * np.log(target_dist / predicted_dist))
    return float(kl)


def huber_loss(
    predicted: np.ndarray,
    target: np.ndarray,
    delta: float = 1.0
) -> float:
    """
    Huber loss (robust to outliers).
    
    Args:
        predicted: Predicted values
        target: Target values
        delta: Threshold for switching between L1 and L2
        
    Returns:
        Huber loss
    """
    error = predicted - target
    abs_error = np.abs(error)
    
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    
    loss = 0.5 * quadratic ** 2 + delta * linear
    return float(np.mean(loss))


# ============================================================================
# Regularization Losses
# ============================================================================

def l1_regularization(weights: np.ndarray, lambda_reg: float = 0.01) -> float:
    """
    L1 (Lasso) regularization.
    
    Args:
        weights: Model weights
        lambda_reg: Regularization strength
        
    Returns:
        L1 penalty
    """
    return float(lambda_reg * np.sum(np.abs(weights)))


def l2_regularization(weights: np.ndarray, lambda_reg: float = 0.01) -> float:
    """
    L2 (Ridge) regularization.
    
    Args:
        weights: Model weights
        lambda_reg: Regularization strength
        
    Returns:
        L2 penalty
    """
    return float(lambda_reg * np.sum(weights ** 2))


def elastic_net_regularization(
    weights: np.ndarray,
    l1_ratio: float = 0.5,
    lambda_reg: float = 0.01
) -> float:
    """
    Elastic net (L1 + L2) regularization.
    
    Args:
        weights: Model weights
        l1_ratio: Ratio of L1 to L2 (0=pure L2, 1=pure L1)
        lambda_reg: Regularization strength
        
    Returns:
        Elastic net penalty
    """
    l1 = l1_ratio * np.sum(np.abs(weights))
    l2 = (1 - l1_ratio) * np.sum(weights ** 2)
    return float(lambda_reg * (l1 + l2))


# ============================================================================
# Combined/Multi-Objective Losses
# ============================================================================

class CombinedLoss:
    """
    Combine multiple loss functions with weights.
    """
    
    def __init__(self, losses: Dict[str, Callable], weights: Dict[str, float]):
        """
        Initialize combined loss.
        
        Args:
            losses: Dict mapping loss names to loss functions
            weights: Dict mapping loss names to weights
        """
        self.losses = losses
        self.weights = weights
    
    def __call__(self, *args, **kwargs) -> Tuple[float, Dict[str, float]]:
        """
        Compute combined loss.
        
        Returns:
            (total_loss, individual_losses)
        """
        individual_losses = {}
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(*args, **kwargs)
            individual_losses[name] = loss_value
            total_loss += self.weights[name] * loss_value
        
        return total_loss, individual_losses


def create_hybrid_chaos_ml_loss(
    trajectory_weight: float = 1.0,
    lyapunov_weight: float = 0.1,
    energy_weight: float = 0.5
) -> CombinedLoss:
    """
    Create hybrid loss for chaos+ML training.
    
    Combines:
    - Trajectory matching (accuracy)
    - Lyapunov matching (chaos properties)
    - Energy alignment (THRML integration)
    
    Args:
        trajectory_weight: Weight for trajectory loss
        lyapunov_weight: Weight for Lyapunov loss
        energy_weight: Weight for energy loss
        
    Returns:
        CombinedLoss instance
    """
    losses = {
        'trajectory': trajectory_mse_loss,
        'lyapunov': lyapunov_loss,
        'energy': lambda states, energy_fn: thrml_energy_loss(states, energy_fn)
    }
    
    weights = {
        'trajectory': trajectory_weight,
        'lyapunov': lyapunov_weight,
        'energy': energy_weight
    }
    
    return CombinedLoss(losses, weights)


# ============================================================================
# Loss Utilities
# ============================================================================

def compute_loss_gradient(
    loss_fn: Callable,
    params: Dict[str, jnp.ndarray],
    *args,
    **kwargs
) -> Dict[str, jnp.ndarray]:
    """
    Compute gradient of loss wrt parameters (JAX).
    
    Args:
        loss_fn: Loss function
        params: Parameters dict
        *args, **kwargs: Arguments to loss function
        
    Returns:
        Gradients dict
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX required for automatic differentiation")
    
    def loss_wrapper(params_dict):
        return loss_fn(*args, params=params_dict, **kwargs)
    
    return jax.grad(loss_wrapper)(params)


def loss_landscape_2d(
    loss_fn: Callable,
    param1_range: Tuple[float, float],
    param2_range: Tuple[float, float],
    resolution: int = 50,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D loss landscape for visualization.
    
    Args:
        loss_fn: Loss function
        param1_range: (min, max) for parameter 1
        param2_range: (min, max) for parameter 2
        resolution: Grid resolution
        **kwargs: Arguments to loss function
        
    Returns:
        (param1_grid, param2_grid, loss_grid)
    """
    param1_vals = np.linspace(param1_range[0], param1_range[1], resolution)
    param2_vals = np.linspace(param2_range[0], param2_range[1], resolution)
    
    param1_grid, param2_grid = np.meshgrid(param1_vals, param2_vals)
    loss_grid = np.zeros_like(param1_grid)
    
    for i in range(resolution):
        for j in range(resolution):
            loss_grid[i, j] = loss_fn(param1_grid[i, j], param2_grid[i, j], **kwargs)
    
    return param1_grid, param2_grid, loss_grid


# ============================================================================
# Loss Registry
# ============================================================================

LOSS_REGISTRY = {
    # Chaos-specific
    'trajectory_mse': trajectory_mse_loss,
    'trajectory_mae': trajectory_mae_loss,
    'attractor_distance': attractor_distance_loss,
    'lyapunov': lyapunov_loss,
    'chaos_entropy': chaos_entropy_loss,
    'periodicity': periodicity_loss,
    
    # Energy-based
    'thrml_energy': thrml_energy_loss,
    'thrml_alignment': thrml_alignment_loss,
    
    # Standard ML
    'mse': mse_loss,
    'mae': mae_loss,
    'cross_entropy': cross_entropy_loss,
    'binary_cross_entropy': binary_cross_entropy_loss,
    'kl_divergence': kl_divergence_loss,
    'huber': huber_loss,
    
    # Regularization
    'l1_reg': l1_regularization,
    'l2_reg': l2_regularization,
    'elastic_net': elastic_net_regularization,
}


def get_loss_fn(loss_name: str) -> Callable:
    """
    Get loss function by name.
    
    Args:
        loss_name: Name of loss function
        
    Returns:
        Loss function
    """
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(LOSS_REGISTRY.keys())}")
    
    return LOSS_REGISTRY[loss_name]


if __name__ == '__main__':
    # Example usage
    print("Testing loss functions...\n")
    
    # Generate sample data
    predicted = np.random.randn(100, 3)
    target = np.random.randn(100, 3)
    
    # Test trajectory losses
    mse = trajectory_mse_loss(predicted, target)
    mae = trajectory_mae_loss(predicted, target)
    print(f"Trajectory MSE: {mse:.6f}")
    print(f"Trajectory MAE: {mae:.6f}")
    
    # Test attractor loss
    attractor_loss = attractor_distance_loss(predicted, target, metric='frobenius')
    print(f"Attractor distance: {attractor_loss:.6f}")
    
    # Test Lyapunov loss
    lyap_loss = lyapunov_loss(predicted, target_lyapunov=2.0)
    print(f"Lyapunov loss: {lyap_loss:.6f}")
    
    # Test combined loss
    print("\nTesting combined loss...")
    combined = create_hybrid_chaos_ml_loss()
    total, individual = combined(predicted, target)
    print(f"Total loss: {total:.6f}")
    print(f"Individual losses: {individual}")
    
    print("\n✓ All loss functions working!")

