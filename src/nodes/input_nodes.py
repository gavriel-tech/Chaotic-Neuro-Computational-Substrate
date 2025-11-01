"""
Input Nodes for GMCS Node Graph.

Provides various input sources for node graph execution:
- AudioInputNode: Live microphone input
- AudioFileNode: Audio file playback
- EEGInputNode: Simulated EEG data
- DataInputNode: Generic parameter/config input
- FragmentLibraryNode: Molecular fragment database
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import os

# Optional dependencies
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("[WARNING] sounddevice not available - AudioInputNode will not work")

try:
    from src.io.audio_file_node import AudioFileNode as AudioFileBackend
    AUDIO_FILE_AVAILABLE = True
except ImportError:
    AUDIO_FILE_AVAILABLE = False
    print("[WARNING] AudioFileNode backend not available")


# ============================================================================
# Audio Input Node (Live Microphone)
# ============================================================================

@dataclass
class AudioInputConfig:
    """Configuration for live audio input."""
    source: str = "microphone"
    sample_rate: int = 48000
    buffer_size: int = 512
    channels: int = 1
    device: Optional[int] = None


class AudioInputNode:
    """
    Live audio input from microphone or line-in.
    
    Uses sounddevice for real-time audio capture.
    Outputs single samples or buffers for processing.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions:
        # 1. AudioInputNode(config_dict)
        # 2. AudioInputNode(node_id, **kwargs)
        if isinstance(node_id_or_config, dict):
            # Config dict mode
            config = node_id_or_config
            self.node_id = config.get('node_id', 'audio_input')
        else:
            # node_id + kwargs mode (for tests)
            self.node_id = node_id_or_config
            config = {'source': 'microphone', **kwargs}
        
        self.config = AudioInputConfig(**config)
        self.buffer = np.zeros(self.config.buffer_size)
        self.buffer_pos = 0
        self.stream = None
        self.latest_sample = 0.0
        self.rms = 0.0
        self.latency = 0.0
        
        if SOUNDDEVICE_AVAILABLE:
            self._initialize_stream()
        else:
            print("[AudioInputNode] sounddevice not available, using simulated input")
    
    def _initialize_stream(self):
        """Initialize audio input stream."""
        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.buffer_size,
                device=self.config.device,
                callback=self._audio_callback
            )
            self.stream.start()
            print(f"[AudioInputNode] Started audio input: {self.config.sample_rate}Hz, {self.config.channels}ch")
        except Exception as e:
            print(f"[AudioInputNode ERROR] Failed to start audio stream: {e}")
            self.stream = None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input."""
        if status:
            print(f"[AudioInputNode] Stream status: {status}")
        
        # Store latest buffer
        if self.config.channels == 1:
            self.buffer = indata[:, 0].copy()
        else:
            # Mix to mono
            self.buffer = np.mean(indata, axis=1)
        
        # Update metrics
        self.latest_sample = float(self.buffer[-1])
        self.rms = float(np.sqrt(np.mean(self.buffer ** 2)))
        self.latency = time_info.input_buffer_adc_time - time_info.current_time if time_info else 0.0
    
    def process(self, **inputs) -> Dict[str, Any]:
        """
        Get current audio input.
        
        Returns:
            Dictionary with:
            - signal: Current sample value
            - buffer: Recent audio buffer
            - rms: RMS amplitude
            - latency: Input latency in seconds
        """
        if not SOUNDDEVICE_AVAILABLE or self.stream is None:
            # Simulated input (silence with occasional noise)
            self.latest_sample = np.random.randn() * 0.01
            self.rms = 0.01
        
        return {
            'signal': self.latest_sample,
            'buffer': self.buffer.copy() if isinstance(self.buffer, np.ndarray) else np.zeros(self.config.buffer_size),
            'rms': self.rms,
            'latency': self.latency
        }
    
    def close(self):
        """Clean up audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("[AudioInputNode] Closed audio stream")
    
    def __del__(self):
        self.close()


# ============================================================================
# Audio File Node (Playback)
# ============================================================================

class AudioFileNode:
    """
    Audio file playback node.
    
    Wraps the AudioFileBackend from src/io/audio_file_node.py
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'audio_file')
        else:
            self.node_id = node_id_or_config
            config = {'source': 'file', **kwargs}
        
        self.config = config
        
        if AUDIO_FILE_AVAILABLE:
            self.backend = AudioFileBackend(config)
        else:
            print("[AudioFileNode] Backend not available, using stub")
            self.backend = None
            self.position = 0
    
    def process(self, **inputs) -> Dict[str, Any]:
        """
        Read next audio sample.
        
        Returns:
            Dictionary with:
            - signal: Current sample
            - rms: RMS of recent block
            - pitch: Estimated pitch (if available)
        """
        if self.backend:
            sample = self.backend.read_sample()
            rms = self.backend.get_rms()
            return {
                'signal': sample,
                'rms': rms,
                'pitch': 0.0  # Pitch estimation is expensive, return 0 for now
            }
        else:
            # Stub - return silence
            self.position += 1
            return {
                'signal': 0.0,
                'rms': 0.0,
                'pitch': 0.0
            }


# ============================================================================
# EEG Input Node (Simulated)
# ============================================================================

@dataclass
class EEGInputConfig:
    """Configuration for EEG input."""
    channels: int = 64
    sample_rate: int = 1000
    source: str = "simulated"  # "simulated" or "live" (future: OpenBCI, Muse)


class EEGInputNode:
    """
    EEG input node for neuromapping.
    
    Currently simulated - generates realistic EEG-like signals.
    Future: integrate with OpenBCI, Muse, or other EEG devices.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'eeg')
        else:
            self.node_id = node_id_or_config
            config = kwargs
        
        self.config = EEGInputConfig(**config)
        self.time = 0.0
        self.dt = 1.0 / self.config.sample_rate
        
        # Initialize channel phases for realistic EEG-like signals
        self.channel_phases = np.random.rand(self.config.channels) * 2 * np.pi
        self.channel_freqs = {
            'delta': (0.5, 4),    # Deep sleep
            'theta': (4, 8),      # Drowsiness
            'alpha': (8, 13),     # Relaxed awareness
            'beta': (13, 30),     # Active thinking
            'gamma': (30, 100)    # Intense focus
        }
        
        print(f"[EEGInputNode] Initialized {self.config.channels}-channel simulated EEG")
    
    def process(self, **inputs) -> Dict[str, Any]:
        """
        Generate simulated EEG signals.
        
        Returns:
            Dictionary with:
            - signal: (channels,) array of current EEG values
            - bands: Dictionary of frequency band powers
        """
        # Generate multi-frequency signal for each channel
        signal = np.zeros(self.config.channels)
        
        for i in range(self.config.channels):
            # Mix of different frequency bands
            delta = 0.5 * np.sin(2 * np.pi * 2.0 * self.time + self.channel_phases[i])
            theta = 0.3 * np.sin(2 * np.pi * 6.0 * self.time + self.channel_phases[i] * 1.3)
            alpha = 0.4 * np.sin(2 * np.pi * 10.0 * self.time + self.channel_phases[i] * 1.7)
            beta = 0.2 * np.sin(2 * np.pi * 20.0 * self.time + self.channel_phases[i] * 2.1)
            
            # Add noise
            noise = np.random.randn() * 0.1
            
            signal[i] = delta + theta + alpha + beta + noise
        
        # Compute band powers (simplified - would use FFT in real implementation)
        bands = {
            'delta': np.mean(np.abs(signal)) * 0.5,
            'theta': np.mean(np.abs(signal)) * 0.3,
            'alpha': np.mean(np.abs(signal)) * 0.4,
            'beta': np.mean(np.abs(signal)) * 0.2,
            'gamma': np.mean(np.abs(signal)) * 0.1
        }
        
        self.time += self.dt
        
        return {
            'signal': signal,
            'bands': bands
        }


# ============================================================================
# Generic Data Input Node
# ============================================================================

class DataInputNode:
    """
    Generic data input node for parameters, configurations, seeds, etc.
    
    Can provide static values or sequences.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'data')
        else:
            self.node_id = node_id_or_config
            config = kwargs
        
        self.config = config
        self.data = config.get('data', {})
        self.current_index = 0
    
    def process(self, **inputs) -> Dict[str, Any]:
        """
        Return configured data.
        
        Returns:
            Dictionary with all configured key-value pairs
        """
        outputs = {}
        
        for key, value in self.data.items():
            if isinstance(value, list):
                # Sequence - cycle through values
                outputs[key] = value[self.current_index % len(value)]
            else:
                # Static value
                outputs[key] = value
        
        # Special handling for common keys
        if 'spec' in self.config:
            outputs['spec'] = self.config['spec']
        if 'parameters' in self.config:
            outputs['parameters'] = self.config['parameters']
        if 'value' in self.config:
            outputs['value'] = self.config['value']
        if 'seed' in self.config:
            outputs['value'] = self.config.get('seed', 12345)
        
        self.current_index += 1
        
        return outputs


# ============================================================================
# Fragment Library Node (Molecular)
# ============================================================================

@dataclass
class FragmentLibraryConfig:
    """Configuration for molecular fragment library."""
    library: str = "common_pharmacophores"
    num_fragments: int = 1000


class FragmentLibraryNode:
    """
    Molecular fragment library for drug design.
    
    Provides chemical fragments (SMILES strings) that can be
    assembled into drug candidates.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'fragments')
        else:
            self.node_id = node_id_or_config
            config = kwargs
        
        # Map 'library_size' to 'num_fragments' for config
        if 'library_size' in config and 'num_fragments' not in config:
            config['num_fragments'] = config.pop('library_size')
        
        self.config = FragmentLibraryConfig(**config)
        self.fragments = self._load_library()
        self.current_index = 0
        
        print(f"[FragmentLibraryNode] Loaded {len(self.fragments)} fragments")
    
    def _load_library(self) -> List[str]:
        """
        Load fragment library.
        
        In a real implementation, this would load from a database
        or chemical fragment file (SMILES, MOL, etc.).
        """
        # Simplified: generate placeholder SMILES strings
        # Common pharmacophore fragments
        common_fragments = [
            "c1ccccc1",      # Benzene
            "C1CCCCC1",      # Cyclohexane
            "c1ccc(O)cc1",   # Phenol
            "c1ccc(N)cc1",   # Aniline
            "CC(=O)O",       # Acetic acid
            "CCO",           # Ethanol
            "CN",            # Methylamine
            "C1CCNCC1",      # Piperidine
            "c1cnccn1",      # Pyrimidine
            "c1ccc2ccccc2c1" # Naphthalene
        ]
        
        # Extend to requested size by combining fragments
        fragments = common_fragments.copy()
        while len(fragments) < self.config.num_fragments:
            # Simple combination strategy
            frag1 = common_fragments[len(fragments) % len(common_fragments)]
            frag2 = common_fragments[(len(fragments) + 1) % len(common_fragments)]
            fragments.append(f"{frag1}.{frag2}")
        
        return fragments[:self.config.num_fragments]
    
    def process(self, **inputs) -> Dict[str, Any]:
        """
        Get fragments from library.
        
        Returns:
            Dictionary with:
            - fragments: List of fragment SMILES strings
            - current_fragment: Single fragment for sequential access
        """
        current_frag = self.fragments[self.current_index % len(self.fragments)]
        self.current_index += 1
        
        return {
            'fragments': self.fragments,
            'current_fragment': current_frag,
            'num_fragments': len(self.fragments)
        }


# ============================================================================
# Node Factory Registration Helper
# ============================================================================

def create_input_node(node_name: str, config: Dict[str, Any]) -> Any:
    """
    Factory function to create input nodes.
    
    Args:
        node_name: Name of the input node type
        config: Node configuration
        
    Returns:
        Input node instance
    """
    if 'Audio Input' in node_name:
        return AudioInputNode(config)
    elif 'Audio File' in node_name or config.get('source') == 'file':
        return AudioFileNode(config)
    elif 'EEG' in node_name:
        return EEGInputNode(config)
    elif 'Fragment' in node_name or 'Library' in node_name:
        return FragmentLibraryNode(config)
    else:
        # Generic data input
        return DataInputNode(config)


# ============================================================================
# Aliases for Test Compatibility
# ============================================================================

# Tests expect these exact class names
EEGNode = EEGInputNode
DataNode = DataInputNode