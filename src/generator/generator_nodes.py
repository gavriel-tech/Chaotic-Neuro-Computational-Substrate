"""
Generator Nodes for GMCS Node Graph.

Provides signal and pattern generation capabilities:
- Noise Generator: Various noise types (white, pink, brown, Perlin)
- Pattern Generator: Test signals (sine, square, triangle, sawtooth)
- Sequence Generator: Structured sequences (arithmetic, geometric, fibonacci)

These nodes create signals for testing, driving oscillators, or seeding ML models.
"""

import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Noise Generator
# ============================================================================

class NoiseType(Enum):
    """Supported noise types."""
    WHITE = "white"
    PINK = "pink"
    BROWN = "brown"
    PERLIN = "perlin"


@dataclass
class NoiseGeneratorConfig:
    """Configuration for Noise Generator node."""
    type: str = 'white'
    amplitude: float = 1.0
    seed: int = 42
    sample_rate: float = 48000


class NoiseGenerator:
    """
    Generate various types of noise signals.
    
    Supports white (flat spectrum), pink (1/f), brown (1/f²), and Perlin (smooth) noise.
    """
    
    def __init__(self, config: NoiseGeneratorConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
        # For colored noise (pink/brown)
        self.noise_state = 0.0
        self.pink_rows = 16
        self.pink_state = np.zeros(self.pink_rows)
        self.pink_weights = np.array([
            0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
            0.0078125, 0.00390625, 0.001953125, 0.0009765625,
            0.00048828125, 0.000244140625, 0.0001220703125,
            0.00006103515625, 0.000030517578125, 0.0000152587890625
        ])
        
        # For Perlin noise
        self.perlin_x = 0.0
    
    def process(self, amplitude_mod: Optional[float] = None, shape: Optional[tuple] = None, seed: Optional[int] = None, **kwargs) -> Dict[str, any]:
        """
        Generate next noise sample or array.
        
        Args:
            amplitude_mod: Optional amplitude modulation (multiplier)
            shape: Optional shape for generating array of noise (for tests)
            seed: Optional seed for reproducibility (for tests)
            
        Returns:
            Dictionary with 'noise' key containing the sample or array
        """
        # If seed provided, temporarily set it
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        amplitude = self.config.amplitude
        if amplitude_mod is not None:
            amplitude *= amplitude_mod
        
        # If shape is provided, generate array
        if shape is not None:
            if self.config.type == 'white':
                noise_array = self.rng.randn(*shape) * amplitude
            elif self.config.type == 'pink':
                # For arrays, just use white noise (pink noise is per-sample)
                noise_array = self.rng.randn(*shape) * amplitude
            elif self.config.type == 'brown':
                # For arrays, use random walk
                noise_array = np.cumsum(self.rng.randn(*shape) * 0.1 * amplitude, axis=-1)
            elif self.config.type == 'perlin':
                # For arrays, use white noise (Perlin is per-sample)
                noise_array = self.rng.randn(*shape) * amplitude
            else:
                noise_array = self.rng.randn(*shape) * amplitude
            
            return {'noise': noise_array}
        
        # Single sample generation
        if self.config.type == 'white':
            noise = self._white_noise()
        elif self.config.type == 'pink':
            noise = self._pink_noise()
        elif self.config.type == 'brown':
            noise = self._brown_noise()
        elif self.config.type == 'perlin':
            noise = self._perlin_noise()
        else:
            noise = self._white_noise()
        
        return {'noise': float(noise * amplitude)}
    
    def _white_noise(self) -> float:
        """Generate white noise sample."""
        return self.rng.randn()
    
    def _pink_noise(self) -> float:
        """
        Generate pink (1/f) noise using Voss-McCartney algorithm.
        """
        # Update random subset of generators
        rand_bits = self.rng.randint(0, 2**self.pink_rows)
        for i in range(self.pink_rows):
            if rand_bits & (1 << i):
                self.pink_state[i] = self.rng.randn()
        
        # Sum weighted generators
        noise = np.sum(self.pink_state * self.pink_weights)
        return noise
    
    def _brown_noise(self) -> float:
        """
        Generate brown (1/f²) noise using random walk.
        """
        white = self.rng.randn() * 0.1
        self.noise_state = self.noise_state * 0.99 + white
        return self.noise_state
    
    def _perlin_noise(self) -> float:
        """
        Generate Perlin-like noise (smooth interpolated).
        """
        # Simplified 1D Perlin
        self.perlin_x += 0.01
        x = self.perlin_x
        
        # Integer and fractional parts
        xi = int(np.floor(x))
        xf = x - xi
        
        # Smooth interpolation
        u = xf * xf * (3.0 - 2.0 * xf)
        
        # Random gradients at integer points
        self.rng.seed(xi & 0xFFFFFF)
        g0 = self.rng.randn()
        self.rng.seed((xi + 1) & 0xFFFFFF)
        g1 = self.rng.randn()
        
        # Interpolate
        noise = g0 * (1 - u) + g1 * u
        return noise


# ============================================================================
# Pattern Generator
# ============================================================================

class PatternType(Enum):
    """Supported pattern types."""
    SINE = "sine"
    SQUARE = "square"
    TRIANGLE = "triangle"
    SAWTOOTH = "sawtooth"
    PULSE = "pulse"


@dataclass
class PatternGeneratorConfig:
    """Configuration for Pattern Generator node."""
    pattern: str = 'sine'
    frequency: float = 440.0
    amplitude: float = 1.0
    phase: float = 0.0
    sample_rate: float = 48000
    duty_cycle: float = 0.5  # For pulse/square waves


class PatternGenerator:
    """
    Generate standard test signals and patterns.
    
    Supports sine, square, triangle, sawtooth, and pulse waves
    with frequency and amplitude modulation.
    """
    
    def __init__(self, config: PatternGeneratorConfig):
        self.config = config
        self.phase_accumulator = config.phase
        self.sample_count = 0
    
    def process(
        self, 
        freq_mod: Optional[float] = None,
        amp_mod: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Generate next pattern sample.
        
        Args:
            freq_mod: Optional frequency modulation (Hz offset or multiplier)
            amp_mod: Optional amplitude modulation (multiplier)
            
        Returns:
            Dictionary with 'pattern' key containing the sample
        """
        # Apply modulation
        frequency = self.config.frequency
        if freq_mod is not None:
            frequency += freq_mod  # Additive FM
        
        amplitude = self.config.amplitude
        if amp_mod is not None:
            amplitude *= amp_mod
        
        # Generate pattern
        if self.config.pattern == 'sine':
            pattern = np.sin(2 * np.pi * self.phase_accumulator)
        elif self.config.pattern == 'square':
            pattern = 1.0 if (self.phase_accumulator % 1.0) < self.config.duty_cycle else -1.0
        elif self.config.pattern == 'triangle':
            phase = self.phase_accumulator % 1.0
            pattern = 2.0 * abs(2.0 * phase - 1.0) - 1.0
        elif self.config.pattern == 'sawtooth':
            pattern = 2.0 * (self.phase_accumulator % 1.0) - 1.0
        elif self.config.pattern == 'pulse':
            pattern = 1.0 if (self.phase_accumulator % 1.0) < self.config.duty_cycle else 0.0
        else:
            pattern = 0.0
        
        # Update phase
        phase_increment = frequency / self.config.sample_rate
        self.phase_accumulator += phase_increment
        self.sample_count += 1
        
        return {'pattern': float(pattern * amplitude)}


# ============================================================================
# Sequence Generator
# ============================================================================

class SequenceType(Enum):
    """Supported sequence types."""
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"
    FIBONACCI = "fibonacci"
    PRIMES = "primes"
    POWERS = "powers"


@dataclass
class SequenceGeneratorConfig:
    """Configuration for Sequence Generator node."""
    length: int = 100
    pattern_type: str = 'arithmetic'
    start: float = 0.0
    step: float = 1.0
    ratio: float = 2.0  # For geometric sequences


class SequenceGenerator:
    """
    Generate structured number sequences.
    
    Creates arithmetic progressions, geometric series, Fibonacci,
    primes, or powers. Useful for test data or rhythmic patterns.
    """
    
    def __init__(self, config: SequenceGeneratorConfig):
        self.config = config
        self.position = 0
        self.sequence = self._generate_sequence()
        
        # For Fibonacci
        self.fib_a = 0
        self.fib_b = 1
    
    def _generate_sequence(self) -> np.ndarray:
        """Pre-generate the sequence."""
        if self.config.pattern_type == 'arithmetic':
            return np.arange(
                self.config.start,
                self.config.start + self.config.length * self.config.step,
                self.config.step
            )
        elif self.config.pattern_type == 'geometric':
            return self.config.start * (self.config.ratio ** np.arange(self.config.length))
        elif self.config.pattern_type == 'fibonacci':
            return self._generate_fibonacci()
        elif self.config.pattern_type == 'primes':
            return self._generate_primes()
        elif self.config.pattern_type == 'powers':
            return np.arange(self.config.length) ** 2
        else:
            return np.zeros(self.config.length)
    
    def _generate_fibonacci(self) -> np.ndarray:
        """Generate Fibonacci sequence."""
        seq = [0, 1]
        for i in range(self.config.length - 2):
            seq.append(seq[-1] + seq[-2])
        return np.array(seq[:self.config.length])
    
    def _generate_primes(self) -> np.ndarray:
        """Generate prime numbers (simplified)."""
        primes = []
        candidate = 2
        while len(primes) < self.config.length:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return np.array(primes)
    
    def process(self, trigger: Optional[float] = None) -> Dict[str, any]:
        """
        Get next sequence value or reset on trigger.
        
        Args:
            trigger: If > 0.5, restart sequence
            
        Returns:
            Dictionary with 'sequence' (current value) and 'position'
        """
        # Check for trigger (restart)
        if trigger is not None and trigger > 0.5:
            self.position = 0
        
        # Get current value
        if self.position < len(self.sequence):
            value = self.sequence[self.position]
        else:
            # Loop or hold last value
            value = self.sequence[-1]
        
        # Advance position
        self.position = (self.position + 1) % len(self.sequence)
        
        return {
            'sequence': float(value),
            'position': float(self.position)
        }


# ============================================================================
# Node Wrappers for Test Compatibility
# ============================================================================

class NoiseGeneratorNode:
    """Wrapper for NoiseGenerator to match test calling convention."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        # Map 'noise_type' to 'type' for config
        if 'noise_type' in kwargs and 'type' not in kwargs:
            kwargs['type'] = kwargs.pop('noise_type')
        config = NoiseGeneratorConfig(**kwargs)
        self._impl = NoiseGenerator(config)
    
    def process(self, **inputs):
        result = self._impl.process(**inputs)
        # Add 'signal' alias for test compatibility
        if 'noise' in result and 'signal' not in result:
            result['signal'] = result['noise']
        return result


class PatternGeneratorNode:
    """Wrapper for PatternGenerator to match test calling convention."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        # Map 'pattern_type' to 'pattern' for config
        if 'pattern_type' in kwargs and 'pattern' not in kwargs:
            kwargs['pattern'] = kwargs.pop('pattern_type')
        config = PatternGeneratorConfig(**kwargs)
        self._impl = PatternGenerator(config)
    
    def process(self, **inputs):
        return self._impl.process(**inputs)


class SequenceGeneratorNode:
    """Wrapper for SequenceGenerator to match test calling convention."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        # Map 'sequence_length' to 'length' for config
        if 'sequence_length' in kwargs and 'length' not in kwargs:
            kwargs['length'] = kwargs.pop('sequence_length')
        config = SequenceGeneratorConfig(**kwargs)
        self._impl = SequenceGenerator(config)
    
    def process(self, **inputs):
        return self._impl.process(**inputs)


# ============================================================================
# Node Factory
# ============================================================================

def create_generator_node(node_type: str, config: dict):
    """
    Factory function to create generator nodes.
    
    Args:
        node_type: Type of generator node
        config: Configuration dictionary
        
    Returns:
        Generator node instance
    """
    # Try crypto RNG for cryptographic random generation
    if node_type in ['RandomNumberGenerator', 'Random Number Generator']:
        try:
            from src.processor.crypto_nodes import RandomNumberGenerator
            return RandomNumberGenerator(config)
        except ImportError:
            # Fall back to noise generator
            return NoiseGenerator(NoiseGeneratorConfig(**config))
    
    if node_type == 'Noise Generator':
        return NoiseGenerator(NoiseGeneratorConfig(**config))
    elif node_type == 'Pattern Generator':
        return PatternGenerator(PatternGeneratorConfig(**config))
    elif node_type == 'Sequence Generator':
        return SequenceGenerator(SequenceGeneratorConfig(**config))
    else:
        raise ValueError(f"Unknown generator node type: {node_type}")

