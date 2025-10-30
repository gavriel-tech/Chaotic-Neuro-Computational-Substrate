"""
Tests for pitch detection algorithms.
"""

import pytest
import numpy as np
from src.audio.pitch_utils import (
    estimate_pitch_yin,
    estimate_pitch_fft,
    estimate_pitch_autocorrelation,
    compute_rms,
    PitchTracker,
)


def generate_sine_wave(freq: float, duration: float, samplerate: int = 48000) -> np.ndarray:
    """Generate a pure sine wave for testing."""
    t = np.linspace(0, duration, int(samplerate * duration))
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_estimate_pitch_yin_440hz():
    """Test YIN pitch detection with 440 Hz sine wave."""
    audio = generate_sine_wave(440.0, 0.1, samplerate=48000)
    pitch = estimate_pitch_yin(audio, samplerate=48000)
    
    # Should detect close to 440 Hz
    assert 430 < pitch < 450


def test_estimate_pitch_fft_440hz():
    """Test FFT pitch detection with 440 Hz sine wave."""
    audio = generate_sine_wave(440.0, 0.1, samplerate=48000)
    pitch = estimate_pitch_fft(audio, samplerate=48000)
    
    # Should detect close to 440 Hz
    assert 430 < pitch < 450


def test_estimate_pitch_autocorr_440hz():
    """Test autocorrelation pitch detection with 440 Hz sine wave."""
    audio = generate_sine_wave(440.0, 0.1, samplerate=48000)
    pitch = estimate_pitch_autocorrelation(audio, samplerate=48000)
    
    # Should detect close to 440 Hz
    assert 400 < pitch < 480  # Slightly wider tolerance


def test_compute_rms():
    """Test RMS computation."""
    # Sine wave with amplitude 1 should have RMS ≈ 0.707
    audio = generate_sine_wave(440.0, 0.1)
    rms = compute_rms(audio)
    
    assert 0.6 < rms < 0.8


def test_pitch_tracker_smoothing():
    """Test pitch tracker with smoothing."""
    tracker = PitchTracker(samplerate=48000, method='yin', smoothing=0.5)
    
    # First estimate
    audio1 = generate_sine_wave(440.0, 0.05)
    pitch1 = tracker.estimate(audio1)
    
    # Second estimate (should be smoothed)
    audio2 = generate_sine_wave(450.0, 0.05)
    pitch2 = tracker.estimate(audio2)
    
    # Pitch should change but be smoothed
    assert pitch1 != pitch2
    assert abs(pitch2 - 450.0) > abs(450.0 - pitch1)  # Smoothing effect


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

