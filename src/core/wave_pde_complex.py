"""
Complex-valued wave field propagation for photonic simulation.

Extends the real-valued FDTD solver to support complex fields for modeling
optical waveguides, dispersion, polarization, and optical gain/loss.
"""

from typing import Tuple
import jax
import jax.numpy as jnp


@jax.jit
def fdtd_step_wave_complex(
    P_current: jnp.ndarray,
    P_prev: jnp.ndarray,
    source_term: jnp.ndarray,
    c: float,
    dt: float,
    dx: float,
    gain: float = 0.0,
    loss: float = 0.0,
    dispersion_coeff: float = 0.0
) -> jnp.ndarray:
    """
    Complex-valued FDTD step for wave equation with gain/loss and dispersion.
    
    Wave equation with gain/loss:
    ∂²P/∂t² = c²∇²P + S(r,t) + γ∂P/∂t - α|P|²P
    
    Where:
    - γ: optical gain coefficient
    - α: nonlinear loss coefficient
    - Dispersion: wavelength-dependent propagation
    
    Args:
        P_current: (W, H) current complex field
        P_prev: (W, H) previous complex field
        source_term: (W, H) complex source term
        c: wave speed (m/s)
        dt: time step (s)
        dx: spatial step (m)
        gain: optical gain coefficient (1/s)
        loss: nonlinear loss coefficient
        dispersion_coeff: dispersion coefficient
        
    Returns:
        (W, H) next complex field P_next
    """
    # Compute Laplacian (∇²P) using 5-point stencil
    # Roll operations for periodic boundary conditions
    P_up = jnp.roll(P_current, -1, axis=0)
    P_down = jnp.roll(P_current, 1, axis=0)
    P_left = jnp.roll(P_current, -1, axis=1)
    P_right = jnp.roll(P_current, 1, axis=1)
    
    laplacian = (P_up + P_down + P_left + P_right - 4.0 * P_current) / (dx * dx)
    
    # Compute velocity (∂P/∂t) from finite difference
    velocity = (P_current - P_prev) / dt
    
    # Compute nonlinear loss term: -α|P|²P
    intensity = jnp.abs(P_current) ** 2
    nonlinear_loss = -loss * intensity * P_current
    
    # Dispersion term (simplified): β₂ * ∂²P/∂t²
    # Approximated as dispersion_coeff * laplacian
    dispersion_term = dispersion_coeff * laplacian
    
    # FDTD update with gain/loss
    # P_next = 2P - P_prev + (c²dt²/dx²)∇²P + dt²S + γdt(P - P_prev) + dt²(nonlinear_loss + dispersion)
    P_next = (
        2.0 * P_current
        - P_prev
        + (c * c * dt * dt) * laplacian
        + (dt * dt) * source_term
        + gain * dt * velocity
        + (dt * dt) * (nonlinear_loss + dispersion_term)
    )
    
    return P_next


@jax.jit
def compute_pde_source_complex(
    oscillator_states: jnp.ndarray,
    node_positions: jnp.ndarray,
    node_active_mask: jnp.ndarray,
    k_strengths: jnp.ndarray,
    grid_w: int,
    grid_h: int,
    sigma: float = 3.0
) -> jnp.ndarray:
    """
    Compute complex source term from oscillator states.
    
    Each oscillator creates a complex Gaussian source:
    S(r) = k_i * (x_i + i*y_i) * exp(-|r - r_i|²/(2σ²))
    
    Args:
        oscillator_states: (N_MAX, 3) [x, y, z] states
        node_positions: (N_MAX, 2) [x, y] grid positions
        node_active_mask: (N_MAX,) 1.0 for active, 0.0 for inactive
        k_strengths: (N_MAX,) source strength per node
        grid_w: Grid width
        grid_h: Grid height
        sigma: Gaussian width
        
    Returns:
        (grid_w, grid_h) complex source field
    """
    # Create grid coordinates
    x_grid = jnp.arange(grid_w, dtype=jnp.float32)
    y_grid = jnp.arange(grid_h, dtype=jnp.float32)
    X, Y = jnp.meshgrid(x_grid, y_grid, indexing='ij')
    
    def single_source(osc_state, pos, mask, k):
        """Compute source from single oscillator."""
        x_osc, y_osc, z_osc = osc_state
        pos_x, pos_y = pos
        
        # Distance squared from oscillator
        dx = X - pos_x
        dy = Y - pos_y
        r_sq = dx * dx + dy * dy
        
        # Gaussian envelope
        gaussian = jnp.exp(-r_sq / (2.0 * sigma * sigma))
        
        # Complex amplitude: x_osc + i*y_osc
        complex_amp = x_osc + 1j * y_osc
        
        # Source: k * complex_amp * gaussian * mask
        return k * complex_amp * gaussian * mask
    
    # Vectorize over all nodes
    vmapped = jax.vmap(single_source, in_axes=(0, 0, 0, 0))
    all_sources = vmapped(oscillator_states, node_positions, node_active_mask, k_strengths)
    
    # Sum all sources
    source_field = jnp.sum(all_sources, axis=0)
    
    return source_field


@jax.jit
def sample_field_at_nodes_complex(
    field: jnp.ndarray,
    node_positions: jnp.ndarray,
    node_active_mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Sample complex field at node positions using bilinear interpolation.
    
    Args:
        field: (W, H) complex field
        node_positions: (N_MAX, 2) [x, y] positions
        node_active_mask: (N_MAX,) active mask
        
    Returns:
        (N_MAX,) complex sampled values
    """
    grid_w, grid_h = field.shape
    
    def sample_single(pos, mask):
        """Sample field at single position."""
        x, y = pos
        
        # Clamp to grid bounds
        x = jnp.clip(x, 0.0, grid_w - 1.001)
        y = jnp.clip(y, 0.0, grid_h - 1.001)
        
        # Get integer and fractional parts
        x0 = jnp.floor(x).astype(jnp.int32)
        y0 = jnp.floor(y).astype(jnp.int32)
        x1 = jnp.minimum(x0 + 1, grid_w - 1)
        y1 = jnp.minimum(y0 + 1, grid_h - 1)
        
        fx = x - x0
        fy = y - y0
        
        # Bilinear interpolation
        f00 = field[x0, y0]
        f10 = field[x1, y0]
        f01 = field[x0, y1]
        f11 = field[x1, y1]
        
        f0 = f00 * (1.0 - fx) + f10 * fx
        f1 = f01 * (1.0 - fx) + f11 * fx
        
        value = f0 * (1.0 - fy) + f1 * fy
        
        return value * mask
    
    # Vectorize over all nodes
    vmapped = jax.vmap(sample_single, in_axes=(0, 0))
    samples = vmapped(node_positions, node_active_mask)
    
    return samples


@jax.jit
def compute_polarization_state(
    field_x: jnp.ndarray,
    field_y: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute polarization state from x and y field components.
    
    Returns:
    - Stokes parameters (S0, S1, S2, S3)
    - Polarization ellipse parameters (azimuth, ellipticity)
    
    Args:
        field_x: (W, H) x-polarized complex field
        field_y: (W, H) y-polarized complex field
        
    Returns:
        Tuple of:
        - stokes: (W, H, 4) Stokes parameters [S0, S1, S2, S3]
        - ellipse: (W, H, 2) [azimuth, ellipticity]
    """
    # Stokes parameters
    S0 = jnp.abs(field_x)**2 + jnp.abs(field_y)**2
    S1 = jnp.abs(field_x)**2 - jnp.abs(field_y)**2
    S2 = 2.0 * jnp.real(field_x * jnp.conj(field_y))
    S3 = 2.0 * jnp.imag(field_x * jnp.conj(field_y))
    
    stokes = jnp.stack([S0, S1, S2, S3], axis=-1)
    
    # Polarization ellipse
    azimuth = 0.5 * jnp.arctan2(S2, S1)
    ellipticity = 0.5 * jnp.arcsin(S3 / (S0 + 1e-10))
    
    ellipse = jnp.stack([azimuth, ellipticity], axis=-1)
    
    return stokes, ellipse


def check_cfl_condition_complex(c: float, dt: float, dx: float) -> bool:
    """
    Check CFL stability condition for complex-valued FDTD.
    
    For 2D: c*dt/dx <= 1/sqrt(2)
    
    Args:
        c: wave speed
        dt: time step
        dx: spatial step
        
    Returns:
        True if stable, False otherwise
    """
    cfl_number = c * dt / dx
    max_cfl = 1.0 / jnp.sqrt(2.0)
    return cfl_number <= max_cfl

