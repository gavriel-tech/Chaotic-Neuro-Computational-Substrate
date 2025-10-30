"""
Audio capture thread for real-time processing.

Captures audio from microphone and extracts features (pitch, RMS) in separate thread.
"""

import threading
import numpy as np
from typing import Dict, Optional
import sounddevice as sd

from .pitch_utils import estimate_pitch_yin, compute_rms


def start_audio_capture(
    shared_state: Dict[str, float],
    device: Optional[int] = None,
    samplerate: int = 48000,
    blocksize: int = 2048,
    method: str = 'yin'
) -> threading.Thread:
    """
    Start audio capture thread.
    
    Captures audio from input device and updates shared state dictionary
    with pitch and RMS values.
    
    Args:
        shared_state: Dictionary with keys 'rms' and 'pitch' (thread-safe for primitives)
        device: Audio input device ID (None for default)
        samplerate: Sample rate in Hz
        blocksize: Samples per callback
        method: Pitch detection method ('yin', 'fft', 'autocorr')
        
    Returns:
        Thread object (already started)
    """
    
    def audio_callback(indata, frames, time, status):
        """Called by sounddevice in separate thread."""
        if status:
            print(f"Audio status: {status}")
        
        try:
            # Extract mono channel
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)  # Average channels
            else:
                audio_data = indata[:, 0]
            
            audio_data = audio_data.astype(np.float32)
            
            # Compute RMS
            rms = compute_rms(audio_data)
            
            # Estimate pitch
            if method == 'yin':
                pitch = estimate_pitch_yin(audio_data, samplerate)
            elif method == 'fft':
                from .pitch_utils import estimate_pitch_fft
                pitch = estimate_pitch_fft(audio_data, samplerate)
            elif method == 'autocorr':
                from .pitch_utils import estimate_pitch_autocorrelation
                pitch = estimate_pitch_autocorrelation(audio_data, samplerate)
            else:
                pitch = 440.0
            
            # Update shared state (thread-safe for primitive types)
            shared_state['rms'] = float(rms)
            shared_state['pitch'] = float(pitch) if pitch > 0 else 440.0
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
    
    # Create and start stream
    stream = sd.InputStream(
        callback=audio_callback,
        device=device,
        channels=1,  # Mono
        samplerate=samplerate,
        blocksize=blocksize,
        dtype=np.float32
    )
    
    def run_stream():
        """Run audio stream in thread."""
        stream.start()
        try:
            while True:
                sd.sleep(1000)  # Sleep forever
        except KeyboardInterrupt:
            stream.stop()
            stream.close()
    
    thread = threading.Thread(target=run_stream, daemon=True)
    thread.start()
    
    return thread


def list_audio_devices():
    """List available audio input devices."""
    devices = sd.query_devices()
    print("Available audio devices:")
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  [{idx}] {device['name']} ({device['max_input_channels']} channels)")


def test_audio_input(duration: float = 2.0, device: Optional[int] = None):
    """
    Test audio input for a short duration.
    
    Args:
        duration: Test duration in seconds
        device: Device ID (None for default)
    """
    print(f"Testing audio input for {duration} seconds...")
    
    shared_state = {'rms': 0.0, 'pitch': 0.0}
    
    thread = start_audio_capture(shared_state, device=device)
    
    import time
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            time.sleep(0.1)
            print(f"RMS: {shared_state['rms']:.4f}, Pitch: {shared_state['pitch']:.1f} Hz")
    except KeyboardInterrupt:
        print("\nTest interrupted")
    
    print("Test complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_audio_devices()
    elif len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_audio_input()
    else:
        print("Usage:")
        print("  python -m src.audio.audio_thread --list   # List devices")
        print("  python -m src.audio.audio_thread --test   # Test audio input")

