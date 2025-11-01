"""
FDTD solver for 2D wave equation.

Implements finite-difference time-domain method for wave propagation with
Gaussian source terms.
"""

import jax
import jax.numpy as jnp
from jax.scipy import signal
import numpy as np


@jax.jit
def laplacian_2d(field: jnp.ndarray, dx: float = 1.0) -> jnp.ndarray:
    """
    Compute 2D Laplacian using 5-point stencil via convolution.
    
    Uses the standard 5-point finite difference stencil:
        ∇²f ≈ (f[i+1,j] + f[i-1,j] + f[i,j+1] + f[i,j-1] - 4*f[i,j]) / dx²
    
    Args:
        field: (H, W) array
        dx: grid spacing
        
    Returns:
        (H, W) array of Laplacian values
    """
    # 5-point stencil kernel
    kernel = jnp.array([
        [0., 1., 0.],
        [1., -4., 1.],
        [0., 1., 0.]
    ]) / (dx ** 2)
    
    # Convolve with 'same' mode to preserve shape
    # Use fill value 0 for boundaries (absorbing)
    laplacian = signal.convolve2d(
        field, kernel, mode='same', boundary='fill', fillvalue=0.0
    )
    
    return laplacian


@jax.jit
def fdtd_step_wave(
    p_t: jnp.ndarray,
    p_tm1: jnp.ndarray,
    source_S: jnp.ndarray,
    dt: float,
    c: float,
    dx: float = 1.0,
    thrml_energy: float = 0.0,
    thrml_temperature: float = 1.0
) -> jnp.ndarray:
    """
    One FDTD step for 2D wave equation with THRML energy modulation.
    
    Wave equation: ∂²P/∂t² = c²(E) ∇²P + S(r,t) - γ(T) ∂P/∂t
    
    THRML coupling:
    - Wave speed modulated by THRML energy: c_eff = c * (1 + α * E_THRML)
    - Damping modulated by THRML temperature: γ = γ₀ * (1 + β * T_THRML)
    
    This creates bidirectional coupling:
    - THRML energy landscapes shape wave propagation
    - Higher energy → faster waves (information spreading)
    - Higher temperature → more damping (entropy dissipation)
    
    Args:
        p_t: (GRID_W, GRID_H) current field
        p_tm1: (GRID_W, GRID_H) previous field
        source_S: (GRID_W, GRID_H) source term
        dt: time step
        c: base wave speed
        dx: grid spacing
        thrml_energy: THRML system energy (normalized)
        thrml_temperature: THRML sampling temperature
        
    Returns:
        (GRID_W, GRID_H) new field P^{t+1}
    """
    # Compute Laplacian
    laplacian = laplacian_2d(p_t, dx)
    
    # THRML-modulated wave speed
    # Higher THRML energy → faster wave propagation
    alpha_energy = 0.1  # Energy coupling strength
    c_effective = c * (1.0 + alpha_energy * jnp.tanh(thrml_energy))
    
    # THRML-modulated damping
    # Higher THRML temperature → more dissipation
    beta_temp = 0.05  # Temperature coupling strength
    gamma_base = 0.01  # Base damping
    damping = gamma_base * (1.0 + beta_temp * thrml_temperature)
    
    # Velocity term (for damping)
    velocity = (p_t - p_tm1) / dt
    
    # FDTD update formula with THRML modulation
    c_dt_sq = (c_effective * dt) ** 2
    dt_sq = dt ** 2
    
    p_tp1 = 2.0 * p_t - p_tm1 + c_dt_sq * laplacian + dt_sq * source_S - damping * dt * velocity
    
    # Apply absorbing boundaries (zero outer 2 pixels)
    p_tp1 = p_tp1.at[:2, :].set(0.0)  # Top
    p_tp1 = p_tp1.at[-2:, :].set(0.0)  # Bottom
    p_tp1 = p_tp1.at[:, :2].set(0.0)  # Left
    p_tp1 = p_tp1.at[:, -2:].set(0.0)  # Right
    
    return p_tp1


def compute_pde_source(
    x_states: jnp.ndarray,
    positions: jnp.ndarray,
    mask: jnp.ndarray,
    k_strengths: jnp.ndarray,
    alpha: float,
    grid_w: int,
    grid_h: int
) -> jnp.ndarray:
    """
    Compute PDE source term by splatting Gaussian kernels.
    
    For each active node i:
        S_i(r) = k_i * |x_i| * exp(-α * |r - r_i|²)
    
    Total source: S(r) = Σ_i S_i(r)
    
    Args:
        x_states: (N_MAX,) oscillator x values
        positions: (N_MAX, 2) [x_pos, y_pos] in grid coordinates
        mask: (N_MAX,) active node mask
        k_strengths: (N_MAX,) source strengths (audio controlled)
        alpha: Gaussian width parameter
        grid_w: grid width
        grid_h: grid height
        
    Returns:
        (grid_w, grid_h) source term grid
    """
    # Create meshgrid of coordinates
    gx = jnp.arange(grid_w, dtype=jnp.float32)
    gy = jnp.arange(grid_h, dtype=jnp.float32)
    xx, yy = jnp.meshgrid(gx, gy, indexing='ij')
    
    def single_node_source(x_i, pos, k_i, active):
        """Compute source contribution from single node."""
        # Distance squared from node position
        r_sq = (xx - pos[0]) ** 2 + (yy - pos[1]) ** 2
        
        # Gaussian kernel weighted by oscillator amplitude
        gaussian = jnp.exp(-alpha * r_sq)
        amplitude = k_i * jnp.abs(x_i)
        
        return active * amplitude * gaussian
    
    # Vectorize over all nodes
    all_sources = jax.vmap(single_node_source)(x_states, positions, k_strengths, mask)
    
    # Sum contributions from all nodes
    return jnp.sum(all_sources, axis=0)


@jax.jit
def check_cfl_condition(c: float, dt: float, dx: float) -> jnp.ndarray:
    """
    Check CFL (Courant-Friedrichs-Lewy) stability condition for 2D wave equation.
    
    For 2D: C = c * dt / dx <= 1/√2 ≈ 0.707
    
    Args:
        c: wave speed
        dt: time step
        dx: grid spacing
        
    Returns:
        Boolean array: True if stable
    """
    C = c * dt / dx
    max_C = 1.0 / jnp.sqrt(2.0)
    return C <= max_C


def get_cfl_number(c: float, dt: float, dx: float) -> float:
    """
    Compute CFL number.
    
    Args:
        c: wave speed
        dt: time step
        dx: grid spacing
        
    Returns:
        CFL number C
    """
    return float(c * dt / dx)


@jax.jit
def apply_damping(field: jnp.ndarray, damping: float = 0.999) -> jnp.ndarray:
    """
    Apply global damping to field to prevent energy buildup.
    
    Args:
        field: (GRID_W, GRID_H) field
        damping: damping coefficient (0-1, 1=no damping)
        
    Returns:
        Damped field
    """
    return field * damping


@jax.jit
def compute_field_energy(p_t: jnp.ndarray, p_tm1: jnp.ndarray, dt: float) -> jnp.ndarray:
    """
    Compute total energy in wave field.
    
    Energy = kinetic + potential
          ≈ (∂P/∂t)² + (∇P)²
    
    Args:
        p_t: current field
        p_tm1: previous field
        dt: time step
        
    Returns:
        Scalar total energy
    """
    # Approximate time derivative
    p_dot = (p_t - p_tm1) / dt
    kinetic = jnp.sum(p_dot ** 2)
    
    # Approximate spatial gradient
    laplacian = laplacian_2d(p_t)
    potential = jnp.sum(p_t * laplacian)  # Integration by parts approximation
    
    return kinetic + jnp.abs(potential)


def initialize_gaussian_pulse(
    grid_w: int,
    grid_h: int,
    center_x: float,
    center_y: float,
    sigma: float = 5.0,
    amplitude: float = 1.0
) -> jnp.ndarray:
    """
    Create initial Gaussian pulse for testing.
    
    Args:
        grid_w: grid width
        grid_h: grid height
        center_x: pulse center x
        center_y: pulse center y
        sigma: pulse width
        amplitude: pulse amplitude
        
    Returns:
        (grid_w, grid_h) field with Gaussian pulse
    """
    gx = np.arange(grid_w, dtype=np.float32)
    gy = np.arange(grid_h, dtype=np.float32)
    xx, yy = np.meshgrid(gx, gy, indexing='ij')
    
    r_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    pulse = amplitude * np.exp(-r_sq / (2 * sigma ** 2))
    
    return jnp.array(pulse)


@jax.jit
def sample_field_at_point(field: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
    """
    Sample field at position using bilinear interpolation.
    
    Args:
        field: (GRID_W, GRID_H) field
        pos: (2,) position [x, y] in grid coordinates
        
    Returns:
        Interpolated scalar value
    """
    grid_w, grid_h = field.shape
    
    x, y = pos[0], pos[1]
    
    # Clamp to grid bounds
    x = jnp.clip(x, 0, grid_w - 1.001)
    y = jnp.clip(y, 0, grid_h - 1.001)
    
    # Integer and fractional parts
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, grid_w - 1)
    y1 = jnp.minimum(y0 + 1, grid_h - 1)
    
    fx = x - x0
    fy = y - y0
    
    # Bilinear interpolation
    v00 = field[x0, y0]
    v10 = field[x1, y0]
    v01 = field[x0, y1]
    v11 = field[x1, y1]
    
    v0 = v00 * (1 - fx) + v10 * fx
    v1 = v01 * (1 - fx) + v11 * fx
    
    return v0 * (1 - fy) + v1 * fy


@jax.jit
def sample_field_at_nodes(
    field: jnp.ndarray,
    positions: jnp.ndarray
) -> jnp.ndarray:
    """
    Sample field at multiple node positions.
    
    Args:
        field: (GRID_W, GRID_H) field
        positions: (N_MAX, 2) positions
        
    Returns:
        (N_MAX,) sampled values
    """
    vmapped = jax.vmap(sample_field_at_point, in_axes=(None, 0))
    return vmapped(field, positions)

