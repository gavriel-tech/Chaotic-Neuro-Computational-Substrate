"""
GMCS (Generative Modular Control System) signal processing pipeline.

Implements a chain of audio/signal processing algorithms that can be applied
in sequence to control signals.
"""

from typing import Tuple
import jax
import jax.numpy as jnp

# Algorithm IDs - Basic (0-6)
ALGO_NOP = 0
ALGO_LIMITER = 1
ALGO_COMPRESSOR = 2
ALGO_EXPANDER = 3
ALGO_THRESHOLD = 4
ALGO_PHASEMOD = 5
ALGO_FOLD = 6

# Algorithm IDs - Audio/Signal Processing (7-13)
ALGO_RESONATOR = 7
ALGO_HILBERT = 8
ALGO_RECTIFIER = 9
ALGO_QUANTIZER = 10
ALGO_SLEW_LIMITER = 11
ALGO_CROSS_MOD = 12
ALGO_BIPOLAR_FOLD = 13

# Algorithm IDs - Photonic (14-20)
ALGO_OPTICAL_KERR = 14
ALGO_ELECTRO_OPTIC = 15
ALGO_OPTICAL_SWITCH = 16
ALGO_FOUR_WAVE_MIXING = 17
ALGO_RAMAN_AMPLIFIER = 18
ALGO_SATURATION = 19
ALGO_OPTICAL_GAIN = 20


@jax.jit
def algo_nop(h: float, *_) -> float:
    """
    No operation - pass through.
    
    Args:
        h: input signal
        *_: ignored parameters
        
    Returns:
        h unchanged
    """
    return h


@jax.jit
def algo_limiter(h: float, A_max: float, *_) -> float:
    """
    Soft clipping limiter using tanh.
    
    Output: A_max * tanh(h / A_max)
    
    Args:
        h: input signal
        A_max: maximum output amplitude
        *_: ignored parameters
        
    Returns:
        Limited signal
    """
    return A_max * jnp.tanh(h / (A_max + 1e-12))


@jax.jit
def algo_compressor(h: float, R: float, T: float, *_) -> float:
    """
    Compressor - reduce dynamic range above threshold.
    
    If h > T: output = T + (h - T) / R
    If h <= T: output = h
    
    Args:
        h: input signal
        R: compression ratio (R > 1)
        T: threshold
        *_: ignored parameters
        
    Returns:
        Compressed signal
    """
    h_over = h - T
    return jnp.where(h_over > 0, T + h_over / R, h)


@jax.jit
def algo_expander(h: float, R: float, T: float, *_) -> float:
    """
    Expander - expand dynamic range below threshold.
    
    If h < T: output = T - (T - h) / R
    If h >= T: output = h
    
    Args:
        h: input signal
        R: expansion ratio (R > 1)
        T: threshold
        *_: ignored parameters
        
    Returns:
        Expanded signal
    """
    h_under = T - h
    return jnp.where(h_under > 0, T - h_under / R, h)


@jax.jit
def algo_threshold(h: float, T: float, V_low: float = 0.0, *_) -> float:
    """
    Hard threshold gate.
    
    If h > T: output = h
    If h <= T: output = V_low
    
    Args:
        h: input signal
        T: threshold
        V_low: low value (default 0)
        *_: ignored parameters
        
    Returns:
        Thresholded signal
    """
    return jnp.where(h > T, h, V_low)


@jax.jit
def algo_phasemod(h: float, t: float, Phi: float, omega: float, *_) -> float:
    """
    Phase modulation - amplitude modulation by sinusoid.
    
    Output: h * (1 + Φ * sin(ω * t))
    
    Args:
        h: input signal
        t: current time
        Phi: modulation depth
        omega: modulation frequency
        *_: ignored parameters
        
    Returns:
        Phase modulated signal
    """
    return h * (1.0 + Phi * jnp.sin(omega * t))


@jax.jit
def algo_fold(h: float, gamma: float, beta: float, *_) -> float:
    """
    Wave folding nonlinearity.
    
    Output: γ * arcsin(sin(β * h))
    
    This creates harmonic distortion and folding effects.
    
    Args:
        h: input signal
        gamma: output amplitude scale
        beta: folding frequency
        *_: ignored parameters
        
    Returns:
        Folded signal
    """
    return gamma * jnp.arcsin(jnp.sin(beta * h))


# ============================================================================
# Audio/Signal Processing Algorithms (7-13)
# ============================================================================

@jax.jit
def algo_resonator(h: float, f0: float, Q: float, t: float, *_) -> float:
    """
    Resonant bandpass filter (simplified 2nd-order).
    
    Output: h * exp(-decay*t) * sin(ω₀*t)
    
    Args:
        h: input signal
        f0: resonant frequency (Hz)
        Q: quality factor
        t: current time
        *_: ignored parameters
        
    Returns:
        Filtered signal
    """
    omega0 = 2.0 * jnp.pi * f0
    decay = omega0 / (2.0 * Q + 1e-6)
    impulse_response = jnp.exp(-decay * (t + 1e-6)) * jnp.sin(omega0 * (t + 1e-6))
    return h * impulse_response


@jax.jit
def algo_hilbert(h: float, *_) -> float:
    """
    Hilbert transform approximation (90° phase shift).
    
    Simplified version using all-pass filter approximation.
    
    Args:
        h: input signal
        *_: ignored parameters
        
    Returns:
        Phase-shifted signal
    """
    # Simplified Hilbert: multiply by sin(π/2) = 1 and shift
    return h * jnp.sin(jnp.pi / 2.0)


@jax.jit
def algo_rectifier(h: float, *_) -> float:
    """
    Full-wave rectification.
    
    Output: |h|
    
    Args:
        h: input signal
        *_: ignored parameters
        
    Returns:
        Rectified signal (absolute value)
    """
    return jnp.abs(h)


@jax.jit
def algo_quantizer(h: float, levels: float, *_) -> float:
    """
    Bit-depth reduction (quantization).
    
    Args:
        h: input signal in range [-1, 1]
        levels: number of quantization levels
        *_: ignored parameters
        
    Returns:
        Quantized signal
    """
    levels = jnp.maximum(levels, 2.0)  # At least 2 levels
    step = 2.0 / levels
    quantized = jnp.floor(h / step) * step
    return jnp.clip(quantized, -1.0, 1.0)


@jax.jit
def algo_slew_limiter(h: float, rate_limit: float, *_) -> float:
    """
    Slew rate limiter (derivative limiter).
    
    Simplified version that clips signal based on rate.
    
    Args:
        h: input signal
        rate_limit: maximum rate of change
        *_: ignored parameters
        
    Returns:
        Rate-limited signal
    """
    # Simplified: clip signal to rate limit range
    return jnp.clip(h, -rate_limit, rate_limit)


@jax.jit
def algo_cross_mod(h: float, mod_depth: float, mod_freq: float, t: float, *_) -> float:
    """
    Cross-modulation (ring modulation).
    
    Output: h * (1 + depth * sin(freq * t))
    
    Args:
        h: input signal
        mod_depth: modulation depth
        mod_freq: modulation frequency
        t: current time
        *_: ignored parameters
        
    Returns:
        Cross-modulated signal
    """
    modulator = 1.0 + mod_depth * jnp.sin(mod_freq * t)
    return h * modulator


@jax.jit
def algo_bipolar_fold(h: float, threshold: float, *_) -> float:
    """
    Bipolar wave folding (symmetric).
    
    Folds signal symmetrically around ±threshold.
    
    Args:
        h: input signal
        threshold: folding threshold
        *_: ignored parameters
        
    Returns:
        Folded signal
    """
    # Fold positive and negative separately
    abs_h = jnp.abs(h)
    sign_h = jnp.sign(h)
    
    # If |h| > threshold, fold back
    folded = jnp.where(
        abs_h > threshold,
        2.0 * threshold - abs_h,
        abs_h
    )
    
    return sign_h * folded


# ============================================================================
# Photonic Algorithms (14-20)
# ============================================================================

@jax.jit
def algo_optical_kerr(h: float, n2: float, length: float, *_) -> float:
    """
    Optical Kerr effect (χ³ nonlinearity, self-phase modulation).
    
    Phase shift: Δφ = n₂ * I * L
    Output: h * (1 + n₂ * h² * L)
    
    Args:
        h: electric field amplitude
        n2: nonlinear refractive index coefficient
        length: propagation length
        *_: ignored parameters
        
    Returns:
        Field with Kerr nonlinearity
    """
    intensity = h * h
    phase_shift = n2 * intensity * length
    return h * (1.0 + phase_shift)


@jax.jit
def algo_electro_optic(h: float, V: float, V_pi: float, *_) -> float:
    """
    Electro-optic modulation (Pockels effect).
    
    Phase shift: Δφ = π * V / V_π
    Output: h * cos(Δφ)
    
    Args:
        h: electric field
        V: applied voltage
        V_pi: half-wave voltage
        *_: ignored parameters
        
    Returns:
        Modulated field
    """
    phase_shift = jnp.pi * V / (V_pi + 1e-6)
    return h * jnp.cos(phase_shift)


@jax.jit
def algo_optical_switch(h: float, threshold: float, contrast: float, *_) -> float:
    """
    All-optical switch (saturable absorber).
    
    Transmission: T = T_min + contrast / (1 + I/I_sat)
    
    Args:
        h: input intensity
        threshold: saturation intensity
        contrast: transmission contrast
        *_: ignored parameters
        
    Returns:
        Transmitted intensity
    """
    T_min = 0.1
    intensity = jnp.abs(h)
    transmission = T_min + contrast / (1.0 + intensity / (threshold + 1e-6))
    return h * transmission


@jax.jit
def algo_four_wave_mixing(h: float, pump_power: float, gamma: float, *_) -> float:
    """
    Four-wave mixing (parametric amplification).
    
    Gain: G = γ * P_pump
    Output: h * (1 + G)
    
    Args:
        h: signal field
        pump_power: pump power
        gamma: nonlinear coefficient
        *_: ignored parameters
        
    Returns:
        Amplified signal
    """
    gain = gamma * pump_power
    return h * (1.0 + gain)


@jax.jit
def algo_raman_amplifier(h: float, pump_power: float, g_R: float, length: float, *_) -> float:
    """
    Stimulated Raman scattering amplification.
    
    Gain: G = exp(g_R * P_pump * L)
    
    Args:
        h: signal field
        pump_power: pump power (W)
        g_R: Raman gain coefficient
        length: fiber length
        *_: ignored parameters
        
    Returns:
        Amplified signal
    """
    gain_exp = g_R * pump_power * length
    gain = jnp.exp(jnp.clip(gain_exp, -10.0, 10.0))  # Clip for stability
    return h * gain


@jax.jit
def algo_saturation(h: float, sat_level: float, *_) -> float:
    """
    Soft saturation using tanh.
    
    Output: sat_level * tanh(h / sat_level)
    
    Args:
        h: input signal
        sat_level: saturation level
        *_: ignored parameters
        
    Returns:
        Saturated signal
    """
    return sat_level * jnp.tanh(h / (sat_level + 1e-6))


@jax.jit
def algo_optical_gain(h: float, gain_dB: float, *_) -> float:
    """
    Optical amplifier (linear gain).
    
    Gain (linear) = 10^(gain_dB/10)
    
    Args:
        h: input field
        gain_dB: gain in decibels
        *_: ignored parameters
        
    Returns:
        Amplified field
    """
    gain_linear = jnp.power(10.0, gain_dB / 10.0)
    return h * gain_linear


# ============================================================================
# Pipeline Functions
# ============================================================================

@jax.jit
def gmcs_pipeline_single_node(
    h_in: float,
    chain_ids: jnp.ndarray,
    params: jnp.ndarray,
    t_scalar: float
) -> float:
    """
    Apply GMCS algorithm chain to single node input.
    
    Processes input through a sequence of algorithms specified by chain_ids.
    Uses lax.scan for fixed-length iteration and lax.switch for dispatch.
    
    Args:
        h_in: raw field input
        chain_ids: (MAX_CHAIN_LEN,) array of algorithm IDs
        params: (16,) array of parameters [A_max, R_comp, T_comp, R_exp, T_exp, Phi, omega, gamma, beta,
                                           f0, Q, levels, rate_limit, n2, V, V_pi]
        t_scalar: current simulation time
        
    Returns:
        Final driving force F_i after processing
    """
    # Unpack parameters (basic 0-8)
    A_max = params[0]
    R_comp = params[1]
    T_comp = params[2]
    R_exp = params[3]
    T_exp = params[4]
    Phi = params[5]
    omega = params[6]
    gamma = params[7]
    beta = params[8]
    
    # Extended parameters (9-15)
    f0 = params[9]          # Resonator frequency
    Q = params[10]          # Resonator Q factor
    levels = params[11]     # Quantizer levels
    rate_limit = params[12] # Slew limiter rate
    n2 = params[13]         # Kerr coefficient
    V = params[14]          # Electro-optic voltage
    V_pi = params[15]       # Electro-optic half-wave voltage
    
    def scan_body(h_carry, chain_id):
        """Process one algorithm in the chain."""
        # Dispatch to appropriate algorithm using lax.switch
        h_out = jax.lax.switch(
            chain_id,
            [
                # Basic algorithms (0-6)
                lambda h: algo_nop(h),
                lambda h: algo_limiter(h, A_max),
                lambda h: algo_compressor(h, R_comp, T_comp),
                lambda h: algo_expander(h, R_exp, T_exp),
                lambda h: algo_threshold(h, T_comp, A_max),  # V_low = A_max
                lambda h: algo_phasemod(h, t_scalar, Phi, omega),
                lambda h: algo_fold(h, gamma, beta),
                # Audio/Signal Processing (7-13)
                lambda h: algo_resonator(h, f0, Q, t_scalar),
                lambda h: algo_hilbert(h),
                lambda h: algo_rectifier(h),
                lambda h: algo_quantizer(h, levels),
                lambda h: algo_slew_limiter(h, rate_limit),
                lambda h: algo_cross_mod(h, Phi, omega, t_scalar),  # Reuse Phi, omega
                lambda h: algo_bipolar_fold(h, T_comp),  # Reuse T_comp as threshold
                # Photonic (14-20)
                lambda h: algo_optical_kerr(h, n2, beta),  # Reuse beta as length
                lambda h: algo_electro_optic(h, V, V_pi),
                lambda h: algo_optical_switch(h, T_comp, gamma),  # Reuse T_comp, gamma
                lambda h: algo_four_wave_mixing(h, gamma, n2),  # Reuse gamma, n2
                lambda h: algo_raman_amplifier(h, gamma, n2, beta),  # Reuse params
                lambda h: algo_saturation(h, A_max),  # Reuse A_max
                lambda h: algo_optical_gain(h, gamma),  # Reuse gamma as gain_dB
            ],
            h_carry
        )
        return h_out, None
    
    # Apply chain using scan
    h_final, _ = jax.lax.scan(scan_body, h_in, chain_ids)
    
    return h_final


@jax.jit
def gmcs_pipeline(
    all_h_in: jnp.ndarray,
    all_chains: jnp.ndarray,
    all_params: jnp.ndarray,
    t_scalar: float
) -> jnp.ndarray:
    """
    Apply GMCS pipeline to all nodes in parallel via vmap.
    
    Args:
        all_h_in: (N_MAX,) raw field inputs
        all_chains: (N_MAX, MAX_CHAIN_LEN) algorithm ID chains
        all_params: (N_MAX, 16) parameter arrays
        t_scalar: current time
        
    Returns:
        (N_MAX,) driving forces after processing
    """
    vmapped = jax.vmap(
        gmcs_pipeline_single_node,
        in_axes=(0, 0, 0, None)
    )
    
    return vmapped(all_h_in, all_chains, all_params, t_scalar)


@jax.jit
def gmcs_pipeline_dual(
    all_h_in: jnp.ndarray,
    all_chains: jnp.ndarray,
    all_params: jnp.ndarray,
    t_scalar: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply GMCS pipeline to all nodes with dual outputs.
    
    Returns both continuous (for Chua oscillator) and discrete (for EBM) outputs.
    
    Args:
        all_h_in: (N_MAX,) raw field inputs
        all_chains: (N_MAX, MAX_CHAIN_LEN) algorithm ID chains
        all_params: (N_MAX, 16) parameter arrays
        t_scalar: current time
        
    Returns:
        Tuple of:
        - F_continuous: (N_MAX,) continuous driving forces for Chua
        - B_discrete: (N_MAX,) discrete bias values for EBM (binarized)
    """
    # Get continuous output
    F_continuous = gmcs_pipeline(all_h_in, all_chains, all_params, t_scalar)
    
    # Create discrete output by binarizing: +1 if F > 0, -1 otherwise
    B_discrete = jnp.where(F_continuous > 0.0, 1.0, -1.0)
    
    return F_continuous, B_discrete


def create_default_chain(chain_type: str = "limiter") -> jnp.ndarray:
    """
    Create a default algorithm chain.
    
    Args:
        chain_type: One of "limiter", "compressor", "expander", "full", "none"
        
    Returns:
        (MAX_CHAIN_LEN,) array of algorithm IDs
    """
    from src.core.state import MAX_CHAIN_LEN
    
    chain = jnp.zeros(MAX_CHAIN_LEN, dtype=jnp.int32)
    
    if chain_type == "limiter":
        chain = chain.at[0].set(ALGO_LIMITER)
    elif chain_type == "compressor":
        chain = chain.at[0].set(ALGO_COMPRESSOR)
    elif chain_type == "expander":
        chain = chain.at[0].set(ALGO_EXPANDER)
    elif chain_type == "full":
        # Example: Threshold -> Compressor -> Limiter
        chain = chain.at[0].set(ALGO_THRESHOLD)
        chain = chain.at[1].set(ALGO_COMPRESSOR)
        chain = chain.at[2].set(ALGO_LIMITER)
    # "none" or default: all zeros (NOP)
    
    return chain


def get_algorithm_name(algo_id: int) -> str:
    """Get human-readable name for algorithm ID."""
    names = {
        # Basic (0-6)
        ALGO_NOP: "No-Op",
        ALGO_LIMITER: "Limiter",
        ALGO_COMPRESSOR: "Compressor",
        ALGO_EXPANDER: "Expander",
        ALGO_THRESHOLD: "Threshold",
        ALGO_PHASEMOD: "Phase Mod",
        ALGO_FOLD: "Fold",
        # Audio/Signal (7-13)
        ALGO_RESONATOR: "Resonator",
        ALGO_HILBERT: "Hilbert",
        ALGO_RECTIFIER: "Rectifier",
        ALGO_QUANTIZER: "Quantizer",
        ALGO_SLEW_LIMITER: "Slew Limiter",
        ALGO_CROSS_MOD: "Cross Mod",
        ALGO_BIPOLAR_FOLD: "Bipolar Fold",
        # Photonic (14-20)
        ALGO_OPTICAL_KERR: "Optical Kerr",
        ALGO_ELECTRO_OPTIC: "Electro-Optic",
        ALGO_OPTICAL_SWITCH: "Optical Switch",
        ALGO_FOUR_WAVE_MIXING: "Four-Wave Mixing",
        ALGO_RAMAN_AMPLIFIER: "Raman Amplifier",
        ALGO_SATURATION: "Saturation",
        ALGO_OPTICAL_GAIN: "Optical Gain",
    }
    return names.get(algo_id, f"Unknown({algo_id})")


def describe_chain(chain: jnp.ndarray) -> str:
    """
    Get human-readable description of algorithm chain.
    
    Args:
        chain: (MAX_CHAIN_LEN,) array of algorithm IDs
        
    Returns:
        String like "Limiter -> Compressor -> No-Op"
    """
    chain_array = np.array(chain) if hasattr(chain, 'device') else chain
    active_algos = [get_algorithm_name(int(algo_id)) for algo_id in chain_array if algo_id != ALGO_NOP]
    
    if not active_algos:
        return "No processing"
    
    return " -> ".join(active_algos)


# Import numpy for describe_chain
import numpy as np

