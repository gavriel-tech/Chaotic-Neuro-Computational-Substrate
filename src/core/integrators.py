"""
RK4 integration for Chua oscillator dynamics.

Implements 4th-order Runge-Kutta method for integrating the Chua circuit equations.
"""

import jax
import jax.numpy as jnp

# Chua circuit parameters
CHUA_A = 9.0
CHUA_B = 14.28
CHUA_C = 1.1


@jax.jit
def chua_derivatives(
    state_vec: jnp.ndarray,
    driving_F: jnp.ndarray,
    ebm_bias: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute derivatives [dx/dt, dy/dt, dz/dt] for Chua system.
    
    Equations:
        dx/dt = -y - z + driving_F + ebm_bias
        dy/dt = x + CHUA_A * y
        dz/dt = CHUA_B + z * (x - CHUA_C)
    
    Args:
        state_vec: (3,) array [x, y, z]
        driving_F: scalar driving force from GMCS pipeline
        ebm_bias: scalar bias from EBM feedback
        
    Returns:
        (3,) array of derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state_vec[0], state_vec[1], state_vec[2]
    
    dx = -y - z + driving_F + ebm_bias
    dy = x + CHUA_A * y
    dz = CHUA_B + z * (x - CHUA_C)
    
    return jnp.array([dx, dy, dz])


@jax.jit
def rk4_step_chua(
    state_vec: jnp.ndarray,
    driving_F: jnp.ndarray,
    ebm_bias: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    Single RK4 step for one Chua oscillator.
    
    Implements the classic 4th-order Runge-Kutta method:
        k1 = f(y_n, t_n)
        k2 = f(y_n + dt/2*k1, t_n + dt/2)
        k3 = f(y_n + dt/2*k2, t_n + dt/2)
        k4 = f(y_n + dt*k3, t_n + dt)
        y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    This function is vectorized via jax.vmap in the main simulation loop.
    
    Args:
        state_vec: (3,) array [x, y, z] current state
        driving_F: scalar driving force
        ebm_bias: scalar EBM feedback
        dt: time step size
        
    Returns:
        (3,) array [x, y, z] new state
    """
    # k1 = f(y_n)
    k1 = chua_derivatives(state_vec, driving_F, ebm_bias)
    
    # k2 = f(y_n + dt/2 * k1)
    k2 = chua_derivatives(state_vec + 0.5 * dt * k1, driving_F, ebm_bias)
    
    # k3 = f(y_n + dt/2 * k2)
    k3 = chua_derivatives(state_vec + 0.5 * dt * k2, driving_F, ebm_bias)
    
    # k4 = f(y_n + dt * k3)
    k4 = chua_derivatives(state_vec + dt * k3, driving_F, ebm_bias)
    
    # Combine using RK4 formula
    new_state = state_vec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    return new_state


@jax.jit
def integrate_all_oscillators(
    oscillator_states: jnp.ndarray,
    driving_forces: jnp.ndarray,
    ebm_biases: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    Integrate all oscillators in parallel using vmap.
    
    Args:
        oscillator_states: (N_MAX, 3) array of all oscillator states
        driving_forces: (N_MAX,) array of driving forces
        ebm_biases: (N_MAX,) array of EBM biases
        dt: time step size
        
    Returns:
        (N_MAX, 3) array of new oscillator states
    """
    # Vectorize over the first dimension (oscillator index)
    vmapped_rk4 = jax.vmap(rk4_step_chua, in_axes=(0, 0, 0, None))
    
    return vmapped_rk4(oscillator_states, driving_forces, ebm_biases, dt)


@jax.jit
def compute_energy(state_vec: jnp.ndarray) -> jnp.ndarray:
    """
    Compute approximate energy of Chua oscillator.
    
    This is not a conserved quantity for Chua (it's dissipative),
    but useful for monitoring system behavior.
    
    Args:
        state_vec: (3,) or (N, 3) array of states
        
    Returns:
        scalar or (N,) array of energy values
    """
    if state_vec.ndim == 1:
        # Single oscillator
        return jnp.sum(state_vec ** 2)
    else:
        # Multiple oscillators
        return jnp.sum(state_vec ** 2, axis=-1)


@jax.jit
def compute_attractor_bounds(oscillator_states: jnp.ndarray) -> tuple:
    """
    Compute bounding box of oscillator states in phase space.
    
    Useful for monitoring if oscillators remain on attractor.
    
    Args:
        oscillator_states: (N_MAX, 3) array
        
    Returns:
        (min_vals, max_vals) tuples of (3,) arrays
    """
    min_vals = jnp.min(oscillator_states, axis=0)
    max_vals = jnp.max(oscillator_states, axis=0)
    
    return min_vals, max_vals


def check_stability(oscillator_states: jnp.ndarray, max_value: float = 100.0) -> bool:
    """
    Check if oscillator states are within stable bounds.
    
    Args:
        oscillator_states: (N_MAX, 3) array
        max_value: Maximum allowed absolute value
        
    Returns:
        True if all states are bounded, False otherwise
    """
    max_abs = jnp.max(jnp.abs(oscillator_states))
    return bool(max_abs < max_value)

