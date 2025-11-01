"""
Audio File Node for GMCS.

Allows loading and playing audio files in the node graph.
"""

import numpy as np
from typing import Optional, Dict, Any


class AudioFileNode:
    """
    Audio file playback node for GMCS node graph.
    
    Loads audio files and provides sample-by-sample playback
    for integration into the simulation loop.
    
    Features:
    - Looping playback
    - Variable playback speed
    - Multiple file format support (via librosa)
    - Real-time resampling
    
    Example:
        >>> node = AudioFileNode("song.mp3", sample_rate=48000)
        >>> for _ in range(1000):
        >>>     sample = node.read_sample()
        >>>     # Use sample in simulation
    """
    
    def __init__(
        self,
        file_path: Optional[str] = None,
        sample_rate: int = 48000,
        loop: bool = True,
        speed: float = 1.0
    ):
        """
        Initialize audio file node.
        
        Args:
            file_path: Path to audio file (mp3, wav, flac, etc.)
            sample_rate: Target sample rate (Hz)
            loop: Whether to loop playback
            speed: Playback speed multiplier (1.0 = normal)
        """
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.loop = loop
        self.speed = speed
        
        self.audio_data: Optional[np.ndarray] = None
        self.position = 0
        self.duration = 0.0
        
        if file_path:
            self.load_file(file_path)
    
    def load_file(self, path: str) -> Dict[str, Any]:
        """
        Load audio file from disk.
        
        Args:
            path: File path
            
        Returns:
            Dict with file info (duration, channels, etc.)
        """
        try:
            import librosa
            
            # Load audio file
            self.audio_data, sr = librosa.load(
                path,
                sr=self.sample_rate,
                mono=True  # Convert to mono
            )
            
            self.file_path = path
            self.position = 0
            self.duration = len(self.audio_data) / self.sample_rate
            
            return {
                'success': True,
                'duration': self.duration,
                'sample_rate': self.sample_rate,
                'num_samples': len(self.audio_data),
                'path': path
            }
            
        except ImportError:
            print("[AudioFileNode] librosa not installed. Install with: pip install librosa")
            return {'success': False, 'error': 'librosa not installed'}
        except Exception as e:
            print(f"[AudioFileNode] Failed to load {path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def read_sample(self) -> float:
        """
        Read one audio sample at current position.
        
        Advances position and handles looping.
        
        Returns:
            Audio sample value (float in [-1, 1])
        """
        if self.audio_data is None or len(self.audio_data) == 0:
            return 0.0
        
        # Get current sample
        sample = float(self.audio_data[self.position])
        
        # Advance position (with speed multiplier)
        self.position += int(self.speed)
        
        # Handle end of file
        if self.position >= len(self.audio_data):
            if self.loop:
                self.position = 0  # Loop back to start
            else:
                self.position = len(self.audio_data) - 1  # Stick at end
        
        return sample
    
    def read_block(self, block_size: int) -> np.ndarray:
        """
        Read a block of audio samples.
        
        Args:
            block_size: Number of samples to read
            
        Returns:
            (block_size,) array of samples
        """
        if self.audio_data is None or len(self.audio_data) == 0:
            return np.zeros(block_size)
        
        block = []
        for _ in range(block_size):
            block.append(self.read_sample())
        
        return np.array(block)
    
    def seek(self, position: float):
        """
        Seek to position in audio file.
        
        Args:
            position: Time position (seconds)
        """
        if self.audio_data is not None:
            sample_pos = int(position * self.sample_rate)
            self.position = np.clip(sample_pos, 0, len(self.audio_data) - 1)
    
    def reset(self):
        """Reset playback to beginning."""
        self.position = 0
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get audio file information.
        
        Returns:
            Dict with duration, position, etc.
        """
        if self.audio_data is None:
            return {
                'loaded': False,
                'file_path': self.file_path
            }
        
        return {
            'loaded': True,
            'file_path': self.file_path,
            'duration': self.duration,
            'position': self.position / self.sample_rate,
            'sample_rate': self.sample_rate,
            'num_samples': len(self.audio_data),
            'loop': self.loop,
            'speed': self.speed
        }
    
    def set_speed(self, speed: float):
        """Set playback speed (1.0 = normal)."""
        self.speed = max(0.1, min(speed, 4.0))  # Clamp to reasonable range
    
    def set_loop(self, loop: bool):
        """Enable/disable looping."""
        self.loop = loop


# Helper functions for node registration

def create_audio_file_node(config: Dict[str, Any]) -> AudioFileNode:
    """
    Factory function to create AudioFileNode from config dict.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured AudioFileNode
    """
    return AudioFileNode(
        file_path=config.get('file_path'),
        sample_rate=config.get('sample_rate', 48000),
        loop=config.get('loop', True),
        speed=config.get('speed', 1.0)
    )

