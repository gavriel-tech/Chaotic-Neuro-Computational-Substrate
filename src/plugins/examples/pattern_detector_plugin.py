"""
Example Pattern Detector Plugin for GMCS.

Demonstrates stateful analysis plugin that detects patterns in oscillator dynamics.
"""

import jax.numpy as jnp
from typing import Any, List, Dict
from src.plugins.plugin_base import StatefulPlugin, PluginMetadata


class PatternDetectorPlugin(StatefulPlugin):
    """
    Pattern detection plugin.
    
    Analyzes oscillator state history to detect recurring patterns,
    phase transitions, and anomalies.
    """
    
    def __init__(self):
        """Initialize pattern detector."""
        super().__init__()
        self.max_history_length = 200
        self.patterns = {}
        self.detection_threshold = 0.85
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="PatternDetector",
            version="1.0.0",
            author="GMCS Team",
            description="Detects patterns and anomalies in oscillator dynamics",
            category="analyzer",
            tags=["pattern", "detection", "analysis", "temporal"],
            parameters=[
                {
                    "name": "window_size",
                    "type": "int",
                    "description": "Analysis window size",
                    "range": [10, 100],
                    "default": 50,
                    "required": True
                },
                {
                    "name": "threshold",
                    "type": "float",
                    "description": "Detection threshold",
                    "range": [0.5, 1.0],
                    "default": 0.85,
                    "required": False
                }
            ]
        )
    
    def initialize(self, config: Dict[str, Any]):
        """Initialize plugin."""
        self.detection_threshold = config.get("threshold", 0.85)
        self.state = {
            "initialized": True,
            "config": config,
            "detections": 0
        }
    
    def process_with_history(
        self,
        current_data: Any,
        history: List[Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process oscillator state with history.
        
        Args:
            current_data: Current oscillator states
            history: Historical states
            **kwargs: Additional parameters
            
        Returns:
            Analysis results
        """
        window_size = kwargs.get("window_size", 50)
        
        if len(history) < window_size:
            return {
                "status": "insufficient_data",
                "history_length": len(history),
                "required": window_size
            }
        
        # Get recent history
        recent = history[-window_size:]
        
        # Convert to array
        states_array = jnp.array(recent)
        
        # Detect patterns
        results = {
            "status": "success",
            "timestamp": len(history),
            "patterns": {}
        }
        
        # 1. Detect periodicity
        periodicity = self._detect_periodicity(states_array)
        results["patterns"]["periodicity"] = periodicity
        
        # 2. Detect phase transitions
        transitions = self._detect_transitions(states_array)
        results["patterns"]["transitions"] = transitions
        
        # 3. Detect anomalies
        anomalies = self._detect_anomalies(states_array, current_data)
        results["patterns"]["anomalies"] = anomalies
        
        # 4. Compute stability metrics
        stability = self._compute_stability(states_array)
        results["metrics"] = stability
        
        # Update detection count
        if anomalies["detected"]:
            self.state["detections"] += 1
        
        return results
    
    def _detect_periodicity(self, states: jnp.ndarray) -> Dict[str, Any]:
        """
        Detect periodic patterns using autocorrelation.
        
        Args:
            states: State history array
            
        Returns:
            Periodicity analysis
        """
        # Flatten states
        flat = states.reshape(len(states), -1)
        
        # Compute autocorrelation
        mean = jnp.mean(flat, axis=0)
        centered = flat - mean
        
        # Autocorrelation at different lags
        max_lag = min(50, len(flat) // 2)
        autocorr = []
        
        for lag in range(1, max_lag):
            corr = jnp.mean(
                jnp.sum(centered[:-lag] * centered[lag:], axis=1)
            )
            autocorr.append(float(corr))
        
        # Find peaks in autocorrelation
        autocorr_array = jnp.array(autocorr)
        peaks = []
        
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > self.detection_threshold * jnp.max(autocorr_array):
                    peaks.append({"lag": i + 1, "strength": float(autocorr[i])})
        
        return {
            "detected": len(peaks) > 0,
            "peaks": peaks,
            "max_autocorr": float(jnp.max(autocorr_array))
        }
    
    def _detect_transitions(self, states: jnp.ndarray) -> Dict[str, Any]:
        """
        Detect phase transitions.
        
        Args:
            states: State history
            
        Returns:
            Transition analysis
        """
        # Compute energy-like metric
        flat = states.reshape(len(states), -1)
        energy = jnp.sum(flat ** 2, axis=1)
        
        # Detect sudden changes
        energy_diff = jnp.abs(jnp.diff(energy))
        threshold = jnp.mean(energy_diff) + 2 * jnp.std(energy_diff)
        
        transitions = jnp.where(energy_diff > threshold)[0]
        
        return {
            "detected": len(transitions) > 0,
            "count": int(len(transitions)),
            "positions": [int(t) for t in transitions[:10]],  # First 10
            "threshold": float(threshold)
        }
    
    def _detect_anomalies(
        self,
        states: jnp.ndarray,
        current: Any
    ) -> Dict[str, Any]:
        """
        Detect anomalies in current state.
        
        Args:
            states: Historical states
            current: Current state
            
        Returns:
            Anomaly analysis
        """
        # Compute statistics from history
        flat = states.reshape(len(states), -1)
        mean = jnp.mean(flat, axis=0)
        std = jnp.std(flat, axis=0)
        
        # Check current state
        current_flat = jnp.array(current).flatten()
        
        # Z-score
        z_scores = jnp.abs((current_flat - mean) / (std + 1e-8))
        
        # Anomaly if any z-score > 3
        is_anomaly = jnp.any(z_scores > 3.0)
        
        return {
            "detected": bool(is_anomaly),
            "max_z_score": float(jnp.max(z_scores)),
            "anomaly_indices": [int(i) for i in jnp.where(z_scores > 3.0)[0]]
        }
    
    def _compute_stability(self, states: jnp.ndarray) -> Dict[str, Any]:
        """
        Compute stability metrics.
        
        Args:
            states: State history
            
        Returns:
            Stability metrics
        """
        flat = states.reshape(len(states), -1)
        
        # Variance over time
        variance = jnp.var(flat, axis=0)
        mean_variance = float(jnp.mean(variance))
        
        # Rate of change
        diffs = jnp.diff(flat, axis=0)
        mean_change = float(jnp.mean(jnp.abs(diffs)))
        
        # Lyapunov-like exponent (simplified)
        distances = jnp.linalg.norm(diffs, axis=1)
        lyapunov_approx = float(jnp.mean(jnp.log(distances + 1e-8)))
        
        return {
            "mean_variance": mean_variance,
            "mean_change_rate": mean_change,
            "lyapunov_estimate": lyapunov_approx,
            "stability_score": float(1.0 / (1.0 + mean_variance))
        }


class FrequencyAnalyzerPlugin(StatefulPlugin):
    """
    Frequency domain analyzer plugin.
    
    Performs FFT analysis on oscillator dynamics.
    """
    
    def __init__(self):
        """Initialize frequency analyzer."""
        super().__init__()
        self.max_history_length = 256  # Power of 2 for FFT
    
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="FrequencyAnalyzer",
            version="1.0.0",
            author="GMCS Team",
            description="FFT-based frequency domain analysis",
            category="analyzer",
            tags=["frequency", "fft", "spectrum", "analysis"],
            parameters=[
                {
                    "name": "n_bins",
                    "type": "int",
                    "description": "Number of frequency bins",
                    "range": [32, 512],
                    "default": 128,
                    "required": False
                }
            ]
        )
    
    def initialize(self, config: Dict[str, Any]):
        """Initialize plugin."""
        self.state = {"initialized": True}
    
    def process_with_history(
        self,
        current_data: Any,
        history: List[Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze frequency content.
        
        Args:
            current_data: Current state
            history: Historical states
            **kwargs: Additional parameters
            
        Returns:
            Frequency analysis results
        """
        n_bins = kwargs.get("n_bins", 128)
        
        if len(history) < n_bins:
            return {
                "status": "insufficient_data",
                "required": n_bins,
                "available": len(history)
            }
        
        # Get recent history
        recent = history[-n_bins:]
        states_array = jnp.array(recent)
        
        # Flatten and perform FFT
        flat = states_array.reshape(len(recent), -1)
        
        # FFT for each dimension
        fft_results = []
        for dim in range(flat.shape[1]):
            signal = flat[:, dim]
            fft = jnp.fft.fft(signal)
            magnitude = jnp.abs(fft[:n_bins//2])
            fft_results.append(magnitude)
        
        # Average across dimensions
        avg_spectrum = jnp.mean(jnp.array(fft_results), axis=0)
        
        # Find dominant frequencies
        top_k = 5
        top_indices = jnp.argsort(avg_spectrum)[-top_k:][::-1]
        
        dominant_freqs = [
            {
                "bin": int(idx),
                "magnitude": float(avg_spectrum[idx]),
                "frequency_ratio": float(idx) / n_bins
            }
            for idx in top_indices
        ]
        
        return {
            "status": "success",
            "spectrum": avg_spectrum.tolist(),
            "dominant_frequencies": dominant_freqs,
            "spectral_centroid": float(
                jnp.sum(jnp.arange(len(avg_spectrum)) * avg_spectrum) /
                jnp.sum(avg_spectrum)
            ),
            "spectral_energy": float(jnp.sum(avg_spectrum ** 2))
        }

