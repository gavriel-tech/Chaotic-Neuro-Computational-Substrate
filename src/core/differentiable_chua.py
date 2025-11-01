"""
Differentiable Chua Oscillators for GMCS.

JAX-based implementation of Chua oscillators with full gradient support,
enabling gradient-based optimization of oscillator parameters and dynamics.

Key features:
- Full JAX differentiability
- Gradient flow through dynamics
- Parameter optimization via backprop
- Compatible with existing non-differentiable oscillators

Use cases:
- Learn oscillator parameters to match target dynamics
- Gradient-based chaos control
- Optimize coupling strengths
- Train controllers for stabilization
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Tuple, Optional, Callable, Dict, Any
import numpy as np


# ============================================================================
# Differentiable Chua Dynamics
# ============================================================================

@jit
def chua_nonlinearity(x: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
    """
    Chua diode nonlinearity (differentiable).
    
    f(x) = b*x + 0.5*(a - b)*(|x + 1| - |x - 1|)
    
    Args:
        x: State variable
        a, b: Nonlinearity parameters
        
    Returns:
        f(x)
    """
    return b * x + 0.5 * (a - b) * (jnp.abs(x + 1) - jnp.abs(x - 1))


@jit
def chua_dynamics(
    state: jnp.ndarray,
    params: Dict[str, float],
    forcing: float = 0.0
) -> jnp.ndarray:
    """
    Chua oscillator dynamics (fully differentiable).
    
    dx/dt = alpha*(y - x - f(x)) + forcing
    dy/dt = x - y + z
    dz/dt = -beta*y
    
    Args:
        state: [x, y, z] state vector
        params: {'alpha', 'beta', 'a', 'b'} parameters
        forcing: External forcing term
        
    Returns:
        [dx/dt, dy/dt, dz/dt] derivatives
    """
    x, y, z = state[0], state[1], state[2]
    
    alpha = params['alpha']
    beta = params['beta']
    a = params['a']
    b = params['b']
    
    fx = chua_nonlinearity(x, a, b)
    
    dx = alpha * (y - x - fx) + forcing
    dy = x - y + z
    dz = -beta * y
    
    return jnp.array([dx, dy, dz])


@jit
def chua_step_euler(
    state: jnp.ndarray,
    params: Dict[str, float],
    forcing: float,
    dt: float
) -> jnp.ndarray:
    """
    Single Euler integration step (differentiable).
    
    Args:
        state: Current [x, y, z]
        params: Oscillator parameters
        forcing: External forcing
        dt: Time step
        
    Returns:
        New state [x, y, z]
    """
    derivatives = chua_dynamics(state, params, forcing)
    return state + dt * derivatives


@jit
def chua_step_rk4(
    state: jnp.ndarray,
    params: Dict[str, float],
    forcing: float,
    dt: float
) -> jnp.ndarray:
    """
    Single RK4 integration step (differentiable).
    
    More accurate than Euler, useful for training.
    
    Args:
        state: Current [x, y, z]
        params: Oscillator parameters
        forcing: External forcing
        dt: Time step
        
    Returns:
        New state [x, y, z]
    """
    k1 = chua_dynamics(state, params, forcing)
    k2 = chua_dynamics(state + 0.5 * dt * k1, params, forcing)
    k3 = chua_dynamics(state + 0.5 * dt * k2, params, forcing)
    k4 = chua_dynamics(state + dt * k3, params, forcing)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ============================================================================
# Trajectory Generation (Differentiable)
# ============================================================================

def chua_trajectory(
    initial_state: jnp.ndarray,
    params: Dict[str, float],
    forcing_sequence: jnp.ndarray,
    dt: float,
    n_steps: int,
    method: str = 'euler'
) -> jnp.ndarray:
    """
    Generate trajectory (differentiable wrt params and initial_state).
    
    This function is differentiable, allowing gradients to flow through
    the entire trajectory generation process.
    
    Args:
        initial_state: Initial [x, y, z]
        params: Oscillator parameters
        forcing_sequence: (n_steps,) forcing values
        dt: Time step
        n_steps: Number of steps
        method: 'euler' or 'rk4'
        
    Returns:
        (n_steps + 1, 3) trajectory array
    """
    step_fn = chua_step_euler if method == 'euler' else chua_step_rk4
    
    def scan_fn(state, forcing):
        next_state = step_fn(state, params, forcing, dt)
        return next_state, next_state
    
    # Use jax.lax.scan for efficient unrolling
    final_state, trajectory = jax.lax.scan(scan_fn, initial_state, forcing_sequence)
    
    # Prepend initial state
    trajectory = jnp.concatenate([initial_state[None, :], trajectory], axis=0)
    
    return trajectory


# ============================================================================
# Loss Functions for Training
# ============================================================================

@jit
def trajectory_mse_loss(
    pred_trajectory: jnp.ndarray,
    target_trajectory: jnp.ndarray
) -> float:
    """
    Mean squared error between trajectories.
    
    Args:
        pred_trajectory: (n_steps, 3) predicted
        target_trajectory: (n_steps, 3) target
        
    Returns:
        MSE loss
    """
    return jnp.mean((pred_trajectory - target_trajectory) ** 2)


@jit
def attractor_distance_loss(
    pred_trajectory: jnp.ndarray,
    target_trajectory: jnp.ndarray
) -> float:
    """
    Loss based on distance in phase space.
    
    Computes pairwise distances between points in attractors.
    Useful when exact trajectory matching is not needed.
    
    Args:
        pred_trajectory: (n_steps, 3) predicted
        target_trajectory: (n_steps, 3) target
        
    Returns:
        Attractor distance loss
    """
    # Compute pairwise distances within each trajectory
    pred_dists = jnp.linalg.norm(pred_trajectory[:, None] - pred_trajectory[None, :], axis=-1)
    target_dists = jnp.linalg.norm(target_trajectory[:, None] - target_trajectory[None, :], axis=-1)
    
    # Compare distance matrices
    return jnp.mean((pred_dists - target_dists) ** 2)


@jit
def lyapunov_loss(
    trajectory: jnp.ndarray,
    target_lyapunov: float = 2.0
) -> float:
    """
    Loss to encourage or discourage chaos.
    
    Approximates largest Lyapunov exponent and matches to target.
    Positive target → encourage chaos
    Negative target → discourage chaos
    
    Args:
        trajectory: (n_steps, 3) trajectory
        target_lyapunov: Target Lyapunov exponent
        
    Returns:
        Lyapunov matching loss
    """
    # Simple approximation: rate of divergence of nearby trajectories
    # Compute distances between consecutive points
    diffs = jnp.diff(trajectory, axis=0)
    distances = jnp.linalg.norm(diffs, axis=-1)
    
    # Estimate exponential growth rate
    log_distances = jnp.log(distances + 1e-8)
    estimated_lyapunov = jnp.mean(jnp.diff(log_distances))
    
    return (estimated_lyapunov - target_lyapunov) ** 2


@jit
def energy_landscape_loss(
    trajectory: jnp.ndarray,
    thrml_energy_fn: Callable
) -> float:
    """
    Align oscillator dynamics with THRML energy landscape.
    
    Args:
        trajectory: (n_steps, 3) trajectory
        thrml_energy_fn: Function that computes THRML energy
        
    Returns:
        Energy alignment loss
    """
    # Compute energy along trajectory
    energies = vmap(thrml_energy_fn)(trajectory)
    
    # Encourage low energy states
    return jnp.mean(energies)


# ============================================================================
# Parameter Optimization
# ============================================================================

class DifferentiableChuaOptimizer:
    """
    Optimizer for Chua oscillator parameters using gradients.
    
    Enables learning oscillator parameters to match target dynamics,
    optimize for specific behaviors, or align with THRML energies.
    """
    
    def __init__(
        self,
        initial_params: Dict[str, float],
        learning_rate: float = 0.01,
        optimizer: str = 'adam'
    ):
        """
        Initialize optimizer.
        
        Args:
            initial_params: Initial parameter values
            learning_rate: Learning rate
            optimizer: 'sgd' or 'adam'
        """
        self.params = initial_params
        self.lr = learning_rate
        self.optimizer_type = optimizer
        
        # Adam momentum
        if optimizer == 'adam':
            self.m = {k: 0.0 for k in initial_params.keys()}
            self.v = {k: 0.0 for k in initial_params.keys()}
            self.t = 0
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
    
    def compute_loss_and_grad(
        self,
        params: Dict[str, float],
        initial_state: jnp.ndarray,
        target_trajectory: jnp.ndarray,
        forcing_sequence: jnp.ndarray,
        dt: float,
        loss_fn: Callable
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute loss and gradients wrt parameters.
        
        Args:
            params: Current parameters
            initial_state: Initial state
            target_trajectory: Target trajectory
            forcing_sequence: Forcing sequence
            dt: Time step
            loss_fn: Loss function
            
        Returns:
            (loss, gradients)
        """
        n_steps = len(forcing_sequence)
        
        def loss_wrapper(params_dict):
            pred_traj = chua_trajectory(
                initial_state,
                params_dict,
                forcing_sequence,
                dt,
                n_steps
            )
            return loss_fn(pred_traj, target_trajectory)
        
        loss = loss_wrapper(params)
        grads = grad(loss_wrapper)(params)
        
        return loss, grads
    
    def step(
        self,
        initial_state: jnp.ndarray,
        target_trajectory: jnp.ndarray,
        forcing_sequence: jnp.ndarray,
        dt: float,
        loss_fn: Callable
    ) -> float:
        """
        Single optimization step.
        
        Args:
            initial_state: Initial state
            target_trajectory: Target trajectory
            forcing_sequence: Forcing sequence
            dt: Time step
            loss_fn: Loss function
            
        Returns:
            Loss value
        """
        loss, grads = self.compute_loss_and_grad(
            self.params,
            initial_state,
            target_trajectory,
            forcing_sequence,
            dt,
            loss_fn
        )
        
        # Update parameters
        if self.optimizer_type == 'sgd':
            for key in self.params.keys():
                self.params[key] -= self.lr * grads[key]
        
        elif self.optimizer_type == 'adam':
            self.t += 1
            for key in self.params.keys():
                # Update biased first moment
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                
                # Update biased second moment
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
                
                # Bias correction
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                self.params[key] -= self.lr * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
        
        return float(loss)
    
    def train(
        self,
        initial_state: jnp.ndarray,
        target_trajectory: jnp.ndarray,
        forcing_sequence: jnp.ndarray,
        dt: float,
        loss_fn: Callable,
        n_epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train parameters for multiple epochs.
        
        Args:
            initial_state: Initial state
            target_trajectory: Target trajectory
            forcing_sequence: Forcing sequence
            dt: Time step
            loss_fn: Loss function
            n_epochs: Number of epochs
            verbose: Print progress
            
        Returns:
            Training history
        """
        history = {'loss': [], 'params': []}
        
        for epoch in range(n_epochs):
            loss = self.step(
                initial_state,
                target_trajectory,
                forcing_sequence,
                dt,
                loss_fn
            )
            
            history['loss'].append(float(loss))
            history['params'].append(self.params.copy())
            
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch}: loss = {loss:.6f}")
        
        return history


# ============================================================================
# Batch Operations (Vectorized)
# ============================================================================

# Vectorize over multiple oscillators
batch_chua_step = vmap(chua_step_euler, in_axes=(0, None, 0, None))


@jit
def integrate_batch_differentiable(
    states: jnp.ndarray,
    params: Dict[str, float],
    forcings: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    Integrate batch of oscillators (differentiable).
    
    Args:
        states: (n_oscillators, 3) current states
        params: Shared parameters for all oscillators
        forcings: (n_oscillators,) forcing values
        dt: Time step
        
    Returns:
        (n_oscillators, 3) new states
    """
    return batch_chua_step(states, params, forcings, dt)


# ============================================================================
# Utility Functions
# ============================================================================

def params_to_vector(params: Dict[str, float]) -> jnp.ndarray:
    """Convert parameter dict to vector."""
    return jnp.array([params['alpha'], params['beta'], params['a'], params['b']])


def vector_to_params(vec: jnp.ndarray) -> Dict[str, float]:
    """Convert parameter vector to dict."""
    return {
        'alpha': float(vec[0]),
        'beta': float(vec[1]),
        'a': float(vec[2]),
        'b': float(vec[3])
    }


def default_params() -> Dict[str, float]:
    """Default Chua parameters (chaotic regime)."""
    return {
        'alpha': 15.6,
        'beta': 28.0,
        'a': -1.143,
        'b': -0.714
    }


def random_initial_state(key: jax.random.PRNGKey, scale: float = 0.1) -> jnp.ndarray:
    """Generate random initial state."""
    return jax.random.normal(key, (3,)) * scale


# ============================================================================
# Example Usage Functions
# ============================================================================

def example_parameter_optimization():
    """
    Example: Optimize parameters to match target trajectory.
    """
    # Generate target trajectory with known parameters
    target_params = default_params()
    initial_state = jnp.array([0.1, 0.1, 0.1])
    forcings = jnp.zeros(1000)
    dt = 0.01
    
    target_traj = chua_trajectory(
        initial_state,
        target_params,
        forcings,
        dt,
        1000
    )
    
    # Try to recover parameters from random initialization
    random_params = {
        'alpha': 10.0,
        'beta': 20.0,
        'a': -1.0,
        'b': -0.5
    }
    
    optimizer = DifferentiableChuaOptimizer(random_params, learning_rate=0.1)
    
    history = optimizer.train(
        initial_state,
        target_traj,
        forcings,
        dt,
        trajectory_mse_loss,
        n_epochs=100,
        verbose=True
    )
    
    print(f"\nTarget params: {target_params}")
    print(f"Learned params: {optimizer.params}")
    
    return history


if __name__ == '__main__':
    # Run example
    print("Running parameter optimization example...")
    history = example_parameter_optimization()
    print(f"Final loss: {history['loss'][-1]:.6f}")

