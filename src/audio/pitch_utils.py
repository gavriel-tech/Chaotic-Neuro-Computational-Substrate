"""
Pitch estimation algorithms for audio processing.

Implements YIN algorithm and FFT-based pitch detection.
"""

import numpy as np
from typing import Optional


def estimate_pitch_yin(
    audio: np.ndarray,
    samplerate: int,
    fmin: float = 80.0,
    fmax: float = 800.0,
    threshold: float = 0.1
) -> float:
    """
    YIN fundamental frequency estimator.
    
    The YIN algorithm is more accurate than autocorrelation for pitch detection,
    especially for complex musical signals.
    
    Args:
        audio: Audio samples (mono)
        samplerate: Sample rate in Hz
        fmin: Minimum frequency to search (Hz)
        fmax: Maximum frequency to search (Hz)
        threshold: YIN threshold (lower = more strict, typical: 0.1)
        
    Returns:
        Estimated frequency in Hz (0.0 if not detected)
    """
    # Compute lag range from frequency range
    min_lag = max(1, int(samplerate / fmax))
    max_lag = min(len(audio) // 2, int(samplerate / fmin))
    
    if max_lag <= min_lag:
        return 0.0
    
    # Ensure audio is float32
    audio = audio.astype(np.float32)
    
    # Step 1: Compute difference function
    diff = np.zeros(max_lag)
    for tau in range(1, max_lag):
        if tau >= len(audio):
            break
        diff[tau] = np.sum((audio[:-tau] - audio[tau:]) ** 2)
    
    # Step 2: Compute cumulative mean normalized difference
    cmnd = np.ones_like(diff)
    cumsum = 0.0
    for tau in range(1, max_lag):
        cumsum += diff[tau]
        if cumsum > 0:
            cmnd[tau] = diff[tau] * tau / cumsum
        else:
            cmnd[tau] = 1.0
    
    # Step 3: Find first minimum below threshold
    for tau in range(min_lag, max_lag):
        if cmnd[tau] < threshold:
            # Parabolic interpolation for sub-sample accuracy
            if tau > 0 and tau < len(cmnd) - 1:
                alpha = cmnd[tau - 1]
                beta = cmnd[tau]
                gamma = cmnd[tau + 1]
                
                # Avoid division by zero
                denom = alpha - 2 * beta + gamma
                if abs(denom) > 1e-10:
                    tau_estimate = tau + 0.5 * (alpha - gamma) / denom
                else:
                    tau_estimate = tau
            else:
                tau_estimate = tau
            
            # Convert lag to frequency
            if tau_estimate > 0:
                return float(samplerate / tau_estimate)
    
    # No pitch detected
    return 0.0


def estimate_pitch_fft(
    audio: np.ndarray,
    samplerate: int,
    fmin: float = 80.0,
    fmax: float = 800.0
) -> float:
    """
    Simple FFT-based pitch estimation.
    
    Faster but less accurate than YIN for complex signals.
    Good for quick estimates or simple tones.
    
    Args:
        audio: Audio samples (mono)
        samplerate: Sample rate in Hz
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)
        
    Returns:
        Estimated frequency in Hz (0.0 if not detected)
    """
    # Apply window to reduce spectral leakage
    windowed = audio * np.hanning(len(audio))
    
    # Compute FFT
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / samplerate)
    
    # Find peak in frequency range
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    
    masked_spectrum = spectrum.copy()
    masked_spectrum[~mask] = 0.0
    
    peak_idx = np.argmax(masked_spectrum)
    
    if masked_spectrum[peak_idx] < np.max(spectrum) * 0.1:
        # Peak too weak
        return 0.0
    
    return float(freqs[peak_idx])


def estimate_pitch_autocorrelation(
    audio: np.ndarray,
    samplerate: int,
    fmin: float = 80.0,
    fmax: float = 800.0
) -> float:
    """
    Autocorrelation-based pitch estimation.
    
    Classic method, works well for clean signals.
    
    Args:
        audio: Audio samples (mono)
        samplerate: Sample rate in Hz
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)
        
    Returns:
        Estimated frequency in Hz (0.0 if not detected)
    """
    # Compute lag range
    min_lag = max(1, int(samplerate / fmax))
    max_lag = min(len(audio) // 2, int(samplerate / fmin))
    
    if max_lag <= min_lag:
        return 0.0
    
    # Normalize audio
    audio = audio - np.mean(audio)
    audio = audio / (np.std(audio) + 1e-10)
    
    # Compute autocorrelation using FFT (faster)
    fft_audio = np.fft.fft(audio, n=2 * len(audio))
    autocorr = np.fft.ifft(fft_audio * np.conj(fft_audio)).real
    autocorr = autocorr[:len(audio)]
    
    # Normalize
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    
    # Find maximum in valid lag range
    valid_autocorr = autocorr[min_lag:max_lag]
    
    if len(valid_autocorr) == 0:
        return 0.0
    
    max_idx = np.argmax(valid_autocorr)
    lag = min_lag + max_idx
    
    # Check if peak is significant
    if autocorr[lag] < 0.3:
        # Weak periodicity
        return 0.0
    
    return float(samplerate / lag)


def compute_rms(audio: np.ndarray) -> float:
    """
    Compute RMS (Root Mean Square) amplitude.
    
    Args:
        audio: Audio samples
        
    Returns:
        RMS value
    """
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_spectral_centroid(
    audio: np.ndarray,
    samplerate: int
) -> float:
    """
    Compute spectral centroid (brightness measure).
    
    Args:
        audio: Audio samples
        samplerate: Sample rate in Hz
        
    Returns:
        Spectral centroid in Hz
    """
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / samplerate)
    
    # Compute weighted mean of frequencies
    if np.sum(spectrum) > 0:
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        return float(centroid)
    else:
        return 0.0


def detect_onset(
    audio: np.ndarray,
    prev_rms: float,
    threshold_ratio: float = 2.0
) -> bool:
    """
    Simple onset detection based on RMS change.
    
    Args:
        audio: Current audio block
        prev_rms: Previous RMS value
        threshold_ratio: Ratio threshold for onset
        
    Returns:
        True if onset detected
    """
    current_rms = compute_rms(audio)
    
    if prev_rms > 0 and current_rms / prev_rms > threshold_ratio:
        return True
    
    return False


class PitchTracker:
    """
    Stateful pitch tracker with smoothing.
    
    Maintains history and applies smoothing to reduce jitter.
    """
    
    def __init__(
        self,
        samplerate: int = 48000,
        method: str = 'yin',
        smoothing: float = 0.5
    ):
        """
        Initialize pitch tracker.
        
        Args:
            samplerate: Audio sample rate
            method: 'yin', 'fft', or 'autocorr'
            smoothing: Smoothing factor (0=no smoothing, 1=maximum smoothing)
        """
        self.samplerate = samplerate
        self.method = method
        self.smoothing = smoothing
        self.prev_pitch = 0.0
    
    def estimate(self, audio: np.ndarray) -> float:
        """
        Estimate pitch with smoothing.
        
        Args:
            audio: Audio samples
            
        Returns:
            Smoothed pitch estimate in Hz
        """
        # Choose method
        if self.method == 'yin':
            pitch = estimate_pitch_yin(audio, self.samplerate)
        elif self.method == 'fft':
            pitch = estimate_pitch_fft(audio, self.samplerate)
        elif self.method == 'autocorr':
            pitch = estimate_pitch_autocorrelation(audio, self.samplerate)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Apply smoothing
        if self.prev_pitch > 0 and pitch > 0:
            # Exponential moving average
            smoothed = self.smoothing * self.prev_pitch + (1 - self.smoothing) * pitch
        else:
            smoothed = pitch
        
        self.prev_pitch = smoothed
        return smoothed

