"""
Example Waveshaper Plugin for GMCS.

Demonstrates how to create a custom algorithm plugin.
"""

import jax.numpy as jnp
from src.plugins.plugin_base import AlgorithmPlugin, PluginMetadata


class WaveshaperPlugin(AlgorithmPlugin):
    """
    Custom waveshaper algorithm plugin.
    
    Applies various waveshaping functions to the input signal.
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="Waveshaper",
            version="1.0.0",
            author="GMCS Team",
            description="Custom waveshaping algorithm with multiple modes",
            category="algorithm",
            tags=["waveshaping", "distortion", "nonlinear"],
            parameters=[
                {
                    "name": "drive",
                    "type": "float",
                    "description": "Drive amount",
                    "range": [0.0, 10.0],
                    "default": 1.0,
                    "required": True
                },
                {
                    "name": "mode",
                    "type": "int",
                    "description": "Waveshaping mode (0=soft, 1=hard, 2=fold, 3=wrap)",
                    "range": [0, 3],
                    "default": 0,
                    "required": True
                },
                {
                    "name": "mix",
                    "type": "float",
                    "description": "Dry/wet mix",
                    "range": [0.0, 1.0],
                    "default": 1.0,
                    "required": False
                }
            ]
        )
    
    def initialize(self, config: dict):
        """Initialize plugin."""
        self.state = {
            "initialized": True,
            "config": config
        }
    
    def compute(
        self,
        input_signal: jnp.ndarray,
        params: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        """
        Compute waveshaper output.
        
        Args:
            input_signal: Input signal
            params: [drive, mode, mix, ...] parameters
            
        Returns:
            Shaped signal
        """
        drive = params[0]
        mode = jnp.int32(params[1])
        mix = params[2] if len(params) > 2 else 1.0
        
        # Apply drive
        driven = input_signal * drive
        
        # Apply waveshaping based on mode
        # Mode 0: Soft clipping (tanh)
        soft = jnp.tanh(driven)
        
        # Mode 1: Hard clipping
        hard = jnp.clip(driven, -1.0, 1.0)
        
        # Mode 2: Fold
        fold = jnp.where(
            jnp.abs(driven) > 1.0,
            2.0 - jnp.abs(driven),
            driven
        ) * jnp.sign(driven)
        
        # Mode 3: Wrap
        wrap = (driven + 1.0) % 2.0 - 1.0
        
        # Select mode using switch
        shaped = jnp.where(
            mode == 0, soft,
            jnp.where(
                mode == 1, hard,
                jnp.where(
                    mode == 2, fold,
                    wrap
                )
            )
        )
        
        # Apply mix
        output = input_signal * (1.0 - mix) + shaped * mix
        
        return output


class ChebyshevWaveshaperPlugin(AlgorithmPlugin):
    """
    Chebyshev polynomial waveshaper.
    
    Uses Chebyshev polynomials for harmonic generation.
    """
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="ChebyshevWaveshaper",
            version="1.0.0",
            author="GMCS Team",
            description="Chebyshev polynomial waveshaper for harmonic generation",
            category="algorithm",
            tags=["waveshaping", "harmonics", "chebyshev"],
            parameters=[
                {
                    "name": "order",
                    "type": "int",
                    "description": "Polynomial order (2-10)",
                    "range": [2, 10],
                    "default": 3,
                    "required": True
                },
                {
                    "name": "gain",
                    "type": "float",
                    "description": "Output gain",
                    "range": [0.1, 2.0],
                    "default": 1.0,
                    "required": False
                }
            ]
        )
    
    def initialize(self, config: dict):
        """Initialize plugin."""
        self.state = {"initialized": True}
    
    def compute(
        self,
        input_signal: jnp.ndarray,
        params: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        """
        Compute Chebyshev waveshaper output.
        
        Args:
            input_signal: Input signal (should be in [-1, 1])
            params: [order, gain, ...] parameters
            
        Returns:
            Shaped signal
        """
        order = jnp.int32(params[0])
        gain = params[1] if len(params) > 1 else 1.0
        
        # Clip input to [-1, 1]
        x = jnp.clip(input_signal, -1.0, 1.0)
        
        # Compute Chebyshev polynomials T_n(x)
        # T_0(x) = 1
        # T_1(x) = x
        # T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)
        
        T_prev = jnp.ones_like(x)  # T_0
        T_curr = x  # T_1
        
        # Compute up to order n
        for n in range(2, 11):  # Max order 10
            T_next = 2.0 * x * T_curr - T_prev
            T_prev = T_curr
            T_curr = jnp.where(n <= order, T_next, T_curr)
        
        # Apply gain and return
        return T_curr * gain

