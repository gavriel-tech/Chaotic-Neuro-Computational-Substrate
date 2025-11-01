"""
Analysis Nodes for GMCS Node Graph.

Provides real-time signal analysis capabilities:
- FFT Analyzer: Frequency domain analysis
- Pattern Recognizer: Detect repeating patterns
- Lyapunov Calculator: Quantify chaos
- Attractor Analyzer: Characterize strange attractors

These nodes process signals from oscillators, THRML, or any source
and extract meaningful features for visualization, control, or ML.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# FFT Analyzer
# ============================================================================

@dataclass
class FFTAnalyzerConfig:
    """Configuration for FFT Analyzer node."""
    size: int = 2048
    window: str = 'hann'
    overlap: float = 0.5
    sample_rate: float = 48000
    
    def __post_init__(self):
        """Validate configuration."""
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        # Check if power of 2
        if (self.size & (self.size - 1)) != 0:
            raise ValueError(f"size must be a power of 2, got {self.size}")


class FFTAnalyzer:
    """
    Fast Fourier Transform analyzer for spectral analysis.
    
    Transforms time-domain signals to frequency domain, revealing
    harmonic structure and dominant frequencies.
    """
    
    def __init__(self, config: FFTAnalyzerConfig):
        self.config = config
        self.buffer = np.zeros(config.size)
        self.buffer_pos = 0
        
        # Pre-compute window
        if config.window == 'hann':
            self.window = np.hanning(config.size)
        elif config.window == 'hamming':
            self.window = np.hamming(config.size)
        elif config.window == 'blackman':
            self.window = np.blackman(config.size)
        else:
            self.window = np.ones(config.size)
    
    def process(self, signal: float) -> Dict[str, np.ndarray]:
        """
        Process incoming signal sample.
        
        Args:
            signal: Single sample value
            
        Returns:
            Dictionary with magnitude, phase, peak_freq, spectral_centroid
        """
        # Add to buffer
        self.buffer[self.buffer_pos] = signal
        self.buffer_pos = (self.buffer_pos + 1) % self.config.size
        
        # Only compute FFT when buffer is full
        if self.buffer_pos == 0:
            return self._compute_fft()
        
        return None
    
    def _compute_fft(self) -> Dict[str, np.ndarray]:
        """Compute FFT of current buffer."""
        # Apply window
        windowed = self.buffer * self.window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        
        # Find peak frequency
        peak_bin = np.argmax(magnitude)
        freqs = np.fft.rfftfreq(self.config.size, 1.0 / self.config.sample_rate)
        peak_freq = freqs[peak_bin]
        
        # Compute spectral centroid
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        
        return {
            'magnitude': magnitude,
            'phase': phase,
            'peak_freq': peak_freq,
            'spectral_centroid': spectral_centroid
        }


# ============================================================================
# Pattern Recognizer
# ============================================================================

@dataclass
class PatternRecognizerConfig:
    """Configuration for Pattern Recognizer node."""
    window_size: int = 100
    threshold: float = 0.8
    num_patterns: int = 10


class PatternRecognizer:
    """
    Detect repeating patterns in time series using autocorrelation.
    
    Identifies quasi-periodic orbits, bifurcations, and recurring motifs
    in chaotic or periodic signals.
    """
    
    def __init__(self, config: PatternRecognizerConfig):
        self.config = config
        self.buffer = np.zeros(config.window_size * 2)
        self.buffer_pos = 0
        self.patterns = []
    
    def process(self, signal: float) -> Optional[Dict[str, any]]:
        """
        Process incoming signal and detect patterns.
        
        Args:
            signal: Single sample value
            
        Returns:
            Dictionary with detected patterns, confidence, period
        """
        # Add to buffer
        self.buffer[self.buffer_pos] = signal
        self.buffer_pos = (self.buffer_pos + 1) % len(self.buffer)
        
        # Compute autocorrelation periodically
        if self.buffer_pos % self.config.window_size == 0:
            return self._detect_patterns()
        
        return None
    
    def _detect_patterns(self) -> Dict[str, any]:
        """Detect patterns using autocorrelation."""
        # Compute autocorrelation
        signal = self.buffer[:self.config.window_size]
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks (potential periods)
        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > self.config.threshold:
                    peaks.append((i, autocorr[i]))
        
        # Get strongest period
        if peaks:
            period, confidence = max(peaks, key=lambda x: x[1])
        else:
            period, confidence = 0, 0.0
        
        return {
            'patterns': len(peaks),
            'confidence': float(confidence),
            'period': float(period)
        }


# ============================================================================
# Lyapunov Calculator
# ============================================================================

@dataclass
class LyapunovConfig:
    """Configuration for Lyapunov Calculator node."""
    window: int = 1000
    neighbors: int = 10
    embedding_dim: int = 3


class LyapunovCalculator:
    """
    Compute largest Lyapunov exponent to quantify chaos.
    
    Positive exponents indicate chaotic dynamics (sensitive dependence
    on initial conditions).
    """
    
    def __init__(self, config: LyapunovConfig):
        self.config = config
        self.trajectory_buffer = []
        self.lyapunov = 0.0
    
    def process(self, trajectory_point: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Process trajectory point and estimate Lyapunov exponent.
        
        Args:
            trajectory_point: State vector (3D for typical oscillator)
            
        Returns:
            Dictionary with lyapunov exponent and is_chaotic flag
        """
        self.trajectory_buffer.append(trajectory_point.copy())
        
        # Keep only recent window
        if len(self.trajectory_buffer) > self.config.window:
            self.trajectory_buffer.pop(0)
        
        # Compute when buffer is full
        if len(self.trajectory_buffer) == self.config.window:
            self.lyapunov = self._compute_lyapunov()
            return {
                'lyapunov': float(self.lyapunov),
                'is_chaotic': 1.0 if self.lyapunov > 0.01 else 0.0
            }
        
        return None
    
    def _compute_lyapunov(self) -> float:
        """
        Compute largest Lyapunov exponent using Rosenstein's algorithm.
        
        Simplified implementation for real-time computation.
        """
        trajectory = np.array(self.trajectory_buffer)
        n_points = len(trajectory)
        
        # Find nearest neighbors
        divergences = []
        for i in range(n_points - 100):  # Leave room for time evolution
            # Current point
            point = trajectory[i]
            
            # Find nearest neighbor (excluding nearby points)
            distances = []
            for j in range(n_points - 100):
                if abs(i - j) > 50:  # Temporal separation
                    dist = np.linalg.norm(trajectory[j] - point)
                    distances.append((j, dist))
            
            if distances:
                # Get nearest neighbor
                j_nearest, initial_dist = min(distances, key=lambda x: x[1])
                
                # Track divergence over time
                for dt in range(1, min(50, n_points - max(i, j_nearest))):
                    final_dist = np.linalg.norm(
                        trajectory[i + dt] - trajectory[j_nearest + dt]
                    )
                    if initial_dist > 1e-10 and final_dist > 1e-10:
                        divergences.append(np.log(final_dist / initial_dist) / dt)
        
        # Average divergence rate
        if divergences:
            return float(np.mean(divergences))
        return 0.0


# ============================================================================
# Attractor Analyzer
# ============================================================================

@dataclass
class AttractorAnalyzerConfig:
    """Configuration for Attractor Analyzer node."""
    embedding_dim: int = 3
    time_delay: int = 10
    min_points: int = 1000


class AttractorAnalyzer:
    """
    Characterize strange attractors using phase space reconstruction.
    
    Estimates correlation dimension and entropy to classify
    dynamical regimes.
    """
    
    def __init__(self, config: AttractorAnalyzerConfig):
        self.config = config
        self.time_series = []
    
    def process(self, signal: float) -> Optional[Dict[str, float]]:
        """
        Process time series and analyze attractor.
        
        Args:
            signal: Scalar time series value
            
        Returns:
            Dictionary with correlation_dim and entropy
        """
        self.time_series.append(signal)
        
        # Keep only recent points
        if len(self.time_series) > self.config.min_points * 2:
            self.time_series.pop(0)
        
        # Compute when enough data
        if len(self.time_series) >= self.config.min_points:
            return self._analyze_attractor()
        
        return None
    
    def _analyze_attractor(self) -> Dict[str, float]:
        """
        Analyze attractor properties.
        
        Simplified implementation for real-time use.
        """
        # Time-delay embedding
        embedded = self._embed_time_series()
        
        # Estimate correlation dimension (simplified)
        corr_dim = self._estimate_correlation_dimension(embedded)
        
        # Estimate entropy (simplified)
        entropy = self._estimate_entropy(embedded)
        
        return {
            'correlation_dim': float(corr_dim),
            'entropy': float(entropy)
        }
    
    def _embed_time_series(self) -> np.ndarray:
        """Create time-delay embedding."""
        series = np.array(self.time_series[-self.config.min_points:])
        embedded_points = []
        
        for i in range(len(series) - (self.config.embedding_dim - 1) * self.config.time_delay):
            point = []
            for j in range(self.config.embedding_dim):
                point.append(series[i + j * self.config.time_delay])
            embedded_points.append(point)
        
        return np.array(embedded_points)
    
    def _estimate_correlation_dimension(self, embedded: np.ndarray) -> float:
        """
        Estimate correlation dimension using Grassberger-Procaccia algorithm.
        
        Simplified version for real-time computation.
        """
        # Sample random pairs
        n_pairs = min(1000, len(embedded) ** 2)
        distances = []
        
        for _ in range(n_pairs):
            i, j = np.random.choice(len(embedded), 2, replace=False)
            dist = np.linalg.norm(embedded[i] - embedded[j])
            distances.append(dist)
        
        # Sort distances
        distances = np.sort(distances)
        
        # Estimate slope in log-log plot (correlation dimension)
        if len(distances) > 10:
            # Use middle range to avoid noise
            mid_start = len(distances) // 4
            mid_end = 3 * len(distances) // 4
            
            log_r = np.log(distances[mid_start:mid_end] + 1e-10)
            log_c = np.log(np.arange(mid_start, mid_end) / len(distances))
            
            # Linear fit
            corr_dim = np.polyfit(log_r, log_c, 1)[0]
            return max(0.0, min(float(self.config.embedding_dim), corr_dim))
        
        return 0.0
    
    def _estimate_entropy(self, embedded: np.ndarray) -> float:
        """
        Estimate approximate entropy.
        
        Measures unpredictability in the time series.
        """
        # Simplified entropy based on state space coverage
        # Use histogram in embedded space
        
        # Discretize each dimension
        bins = 10
        hist, _ = np.histogramdd(embedded, bins=bins)
        
        # Compute entropy
        prob = hist / (np.sum(hist) + 1e-10)
        prob = prob[prob > 0]  # Remove zeros
        entropy = -np.sum(prob * np.log(prob))
        
        return float(entropy)


# ============================================================================
# Node Wrappers for Test Compatibility
# ============================================================================

class FFTAnalyzerNode:
    """Wrapper for FFTAnalyzer to match test calling convention."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        # Map 'fft_size' to 'size' for config
        if 'fft_size' in kwargs and 'size' not in kwargs:
            kwargs['size'] = kwargs.pop('fft_size')
        config = FFTAnalyzerConfig(**kwargs)
        self._impl = FFTAnalyzer(config)
        
        # Expose fft_size attribute for test compatibility
        self.fft_size = config.size
    
    def process(self, **inputs):
        # Handle array vs scalar signal
        if 'signal' in inputs:
            signal = inputs['signal']
            if isinstance(signal, np.ndarray):
                # For array input, process each sample
                outputs = []
                for sample in signal:
                    out = self._impl.process(signal=float(sample))
                    outputs.append(out)
                # Return aggregated result
                return outputs[-1] if outputs else {}
            else:
                # Scalar signal
                return self._impl.process(**inputs)
        return self._impl.process(**inputs)


class PatternAnalyzerNode:
    """Wrapper for PatternRecognizer to match test calling convention."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        config = PatternRecognizerConfig(**kwargs)
        self._impl = PatternRecognizer(config)
    
    def process(self, **inputs):
        return self._impl.process(**inputs)


class LyapunovAnalyzerNode:
    """Wrapper for LyapunovCalculator to match test calling convention."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        # Map 'dimensions' to 'embedding_dim' for config
        if 'dimensions' in kwargs and 'embedding_dim' not in kwargs:
            kwargs['embedding_dim'] = kwargs.pop('dimensions')
        config = LyapunovConfig(**kwargs)
        self._impl = LyapunovCalculator(config)
    
    def process(self, **inputs):
        return self._impl.process(**inputs)


class AttractorAnalyzerNode:
    """Wrapper for AttractorAnalyzer to match test calling convention."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        # Map 'dimensions' to 'embedding_dim' for config
        if 'dimensions' in kwargs and 'embedding_dim' not in kwargs:
            kwargs['embedding_dim'] = kwargs.pop('dimensions')
        config = AttractorAnalyzerConfig(**kwargs)
        self._impl = AttractorAnalyzer(config)
    
    def process(self, **inputs):
        return self._impl.process(**inputs)


# ============================================================================
# Node Factory
# ============================================================================

def create_analysis_node(node_type: str, config: dict):
    """
    Factory function to create analysis nodes.
    
    Args:
        node_type: Type of analysis node
        config: Configuration dictionary
        
    Returns:
        Analysis node instance
    """
    if node_type == 'FFT Analyzer':
        return FFTAnalyzer(FFTAnalyzerConfig(**config))
    elif node_type == 'Pattern Recognizer':
        return PatternRecognizer(PatternRecognizerConfig(**config))
    elif node_type == 'Lyapunov Calculator':
        return LyapunovCalculator(LyapunovConfig(**config))
    elif node_type == 'Attractor Analyzer':
        return AttractorAnalyzer(AttractorAnalyzerConfig(**config))
    else:
        raise ValueError(f"Unknown analysis node type: {node_type}")

