"""
Output Nodes for GMCS Node Graph.

Provides various output destinations for node graph execution:
- AudioOutputNode: Live audio playback
- MIDIOutputNode: MIDI note/CC output
- VideoOutputNode: Real-time video display/recording
- FileExportNode: Export data to files
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import os
import json

# Optional dependencies
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("[WARNING] sounddevice not available - AudioOutputNode will not work")

try:
    from src.io.midi_io import MIDIOutput, MIDIInput
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
    print("[WARNING] MIDI I/O not available")

try:
    from src.io.video_output import VideoOutput
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    print("[WARNING] VideoOutput not available")


# ============================================================================
# Audio Output Node (Live Playback)
# ============================================================================

@dataclass
class AudioOutputConfig:
    """Configuration for live audio output."""
    sample_rate: int = 48000
    buffer_size: int = 512
    channels: int = 2
    device: Optional[int] = None
    synthesis_mode: str = "additive"  # "additive", "direct", "wavetable"


class AudioOutputNode:
    """
    Live audio output for playback through speakers/headphones.
    
    Uses sounddevice for real-time audio playback.
    Supports various synthesis modes.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'audio_output')
        else:
            self.node_id = node_id_or_config
            config = kwargs
        
        self.config = AudioOutputConfig(**config)
        self.stream = None
        self.buffer = np.zeros((self.config.buffer_size, self.config.channels))
        self.buffer_pos = 0
        self.oscillator_phases = {}
        
        if SOUNDDEVICE_AVAILABLE:
            self._initialize_stream()
        else:
            print("[AudioOutputNode] sounddevice not available, audio will not play")
    
    def _initialize_stream(self):
        """Initialize audio output stream."""
        try:
            self.stream = sd.OutputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.buffer_size,
                device=self.config.device,
                callback=self._audio_callback
            )
            self.stream.start()
            print(f"[AudioOutputNode] Started audio output: {self.config.sample_rate}Hz, {self.config.channels}ch")
        except Exception as e:
            print(f"[AudioOutputNode ERROR] Failed to start audio stream: {e}")
            self.stream = None
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for audio output."""
        if status:
            print(f"[AudioOutputNode] Stream status: {status}")
        
        # Copy buffered audio to output
        outdata[:] = self.buffer[:frames]
        
        # Clear buffer for next samples
        self.buffer[:] = 0.0
        self.buffer_pos = 0
    
    def process(self, signal: Optional[Any] = None, **inputs) -> Dict[str, Any]:
        """
        Output audio signal.
        
        Args:
            signal: Audio signal to output (single sample or array)
            **inputs: Additional audio-related inputs (frequencies, amplitudes, etc.)
            
        Returns:
            Dictionary with status info
        """
        if signal is None:
            # No signal provided, check for synthesis parameters
            if 'audio' in inputs:
                signal = inputs['audio']
            else:
                signal = 0.0
        
        # Convert to numpy array if needed
        if not isinstance(signal, np.ndarray):
            signal = np.array([signal])
        
        # Handle single sample vs buffer
        if signal.ndim == 0:
            signal = np.array([float(signal)])
        elif signal.ndim > 1:
            signal = signal.flatten()
        
        # Add to output buffer
        num_samples = min(len(signal), self.config.buffer_size - self.buffer_pos)
        for i in range(num_samples):
            self.buffer[self.buffer_pos + i, 0] = signal[i]
            if self.config.channels == 2:
                self.buffer[self.buffer_pos + i, 1] = signal[i]  # Mono to stereo
        
        self.buffer_pos += num_samples
        
        # Clip to prevent distortion
        self.buffer = np.clip(self.buffer, -1.0, 1.0)
        
        return {
            'status': 'playing' if self.stream and self.stream.active else 'inactive',
            'buffer_fill': self.buffer_pos / self.config.buffer_size
        }
    
    def close(self):
        """Clean up audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("[AudioOutputNode] Closed audio stream")
    
    def __del__(self):
        self.close()


# ============================================================================
# MIDI Output Node
# ============================================================================

@dataclass
class MIDIOutputConfig:
    """Configuration for MIDI output."""
    device: str = "virtual"
    channel: int = 1
    velocity_mode: str = "dynamic"  # "dynamic", "fixed"
    fixed_velocity: int = 64


class MIDIOutputNode:
    """
    MIDI output node for sending notes and control changes.
    
    Wraps the MIDIOutput from src/io/midi_io.py
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'midi_output')
        else:
            self.node_id = node_id_or_config
            config = kwargs
        
        self.config = MIDIOutputConfig(**config)
        self.midi_out = None
        self.active_notes = {}  # Track note on/off
        
        if MIDI_AVAILABLE:
            try:
                virtual = (self.config.device == "virtual")
                port_name = self.config.device if not virtual else "GMCS MIDI Out"
                self.midi_out = MIDIOutput(port_name=port_name, virtual=virtual)
                print(f"[MIDIOutputNode] Opened MIDI output: {port_name}")
            except Exception as e:
                print(f"[MIDIOutputNode ERROR] Failed to open MIDI: {e}")
        else:
            print("[MIDIOutputNode] MIDI not available")
    
    def process(self, data: Optional[Any] = None, **inputs) -> Dict[str, Any]:
        """
        Send MIDI data.
        
        Args:
            data: MIDI note number or CC value
            **inputs: Can include 'note', 'velocity', 'cc_control', 'cc_value'
            
        Returns:
            Dictionary with status
        """
        if not self.midi_out:
            return {'status': 'inactive'}
        
        # Handle note input
        if 'notes' in inputs or data is not None:
            note = inputs.get('notes', data)
            if note is not None:
                note = int(np.clip(note, 0, 127))
                
                # Turn off previous note
                if note in self.active_notes:
                    self.midi_out.send_note_off(note, channel=self.config.channel - 1)
                
                # Turn on new note
                velocity = inputs.get('velocity', self.config.fixed_velocity)
                if self.config.velocity_mode == "dynamic" and 'velocity' in inputs:
                    velocity = int(np.clip(velocity * 127, 0, 127))
                else:
                    velocity = self.config.fixed_velocity
                
                self.midi_out.send_note_on(note, velocity=velocity, channel=self.config.channel - 1)
                self.active_notes[note] = True
        
        # Handle CC input
        if 'cc_control' in inputs and 'cc_value' in inputs:
            control = int(inputs['cc_control'])
            value = int(np.clip(inputs['cc_value'] * 127, 0, 127))
            self.midi_out.send_cc(control, value, channel=self.config.channel - 1)
        
        return {
            'status': 'active',
            'active_notes': len(self.active_notes)
        }
    
    def close(self):
        """Clean up MIDI."""
        if self.midi_out:
            self.midi_out.all_notes_off()
            self.midi_out.close()
            print("[MIDIOutputNode] Closed MIDI output")
    
    def __del__(self):
        self.close()


# ============================================================================
# Video Output Node
# ============================================================================

@dataclass
class VideoOutputConfig:
    """Configuration for video output."""
    resolution: List[int] = None  # [width, height]
    fps: int = 60
    output: str = "window"  # "window", "file", "both"
    fullscreen: bool = False
    output_file: Optional[str] = None


class VideoOutputNode:
    """
    Video output node for real-time display and recording.
    
    Wraps the VideoOutput from src/io/video_output.py
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config.copy()
            self.node_id = config.get('node_id', 'video_output')
        else:
            self.node_id = node_id_or_config
            config = kwargs.copy()
        
        if 'resolution' not in config or config['resolution'] is None:
            config['resolution'] = [1920, 1080]
        
        config_dict = config
        
        self.config = VideoOutputConfig(**config_dict)
        self.video_out = None
        self.frame_count = 0
        self.fps_actual = 0.0
        
        if VIDEO_AVAILABLE:
            try:
                output_file = None
                if self.config.output in ["file", "both"]:
                    output_file = self.config.output_file or "output_video.mp4"
                
                self.video_out = VideoOutput(
                    resolution=tuple(self.config.resolution),
                    fps=self.config.fps,
                    output_file=output_file,
                    display_window_name="GMCS Video Output"
                )
                print(f"[VideoOutputNode] Started video output: {self.config.resolution} @ {self.config.fps}fps")
            except Exception as e:
                print(f"[VideoOutputNode ERROR] Failed to start video: {e}")
        else:
            print("[VideoOutputNode] Video output not available")
    
    def process(self, data: Optional[np.ndarray] = None, **inputs) -> Dict[str, Any]:
        """
        Output video frame.
        
        Args:
            data: Video frame as numpy array (H, W, 3) or (H, W)
            **inputs: Can include 'image', 'field', etc.
            
        Returns:
            Dictionary with status and fps
        """
        if not self.video_out:
            return {'status': 'inactive', 'fps': 0.0}
        
        # Get frame data
        frame = data
        if frame is None:
            if 'image' in inputs:
                frame = inputs['image']
            elif 'data' in inputs:
                frame = inputs['data']
            else:
                # Generate black frame
                frame = np.zeros((self.config.resolution[1], self.config.resolution[0], 3), dtype=np.uint8)
        
        # Convert to proper format if needed
        if isinstance(frame, np.ndarray):
            # Normalize if float
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Convert grayscale to RGB if needed
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            
            # Write frame
            self.video_out.write_frame(frame)
            self.frame_count += 1
        
        return {
            'status': 'active',
            'fps': self.config.fps,  # Actual FPS measurement would require timing
            'frames': self.frame_count
        }
    
    def close(self):
        """Clean up video output."""
        if self.video_out:
            self.video_out.release()
            print("[VideoOutputNode] Closed video output")
    
    def __del__(self):
        self.close()


# ============================================================================
# File Export Node
# ============================================================================

@dataclass
class FileExportConfig:
    """Configuration for file export."""
    format: str = "json"  # "json", "csv", "sdf", "txt", "source_code"
    output_dir: str = "outputs"
    filename_prefix: str = "gmcs_output"
    auto_increment: bool = True


class FileExportNode:
    """
    Export data to files in various formats.
    
    Supports JSON, CSV, SDF (molecular), source code, and plain text.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'file_export')
        else:
            self.node_id = node_id_or_config
            config = kwargs
        
        # Map 'output_path' to 'output_dir' for config
        if 'output_path' in config and 'output_dir' not in config:
            config['output_dir'] = config.pop('output_path')
        
        self.config = FileExportConfig(**config)
        self.export_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f"[FileExportNode] Initialized, output dir: {self.config.output_dir}")
    
    def process(self, data: Any = None, **inputs) -> Dict[str, Any]:
        """
        Export data to file.
        
        Args:
            data: Data to export
            **inputs: Additional data fields
            
        Returns:
            Dictionary with export status and filename
        """
        # Collect all data
        export_data = {}
        if data is not None:
            export_data['data'] = data
        export_data.update(inputs)
        
        # Generate filename
        if self.config.auto_increment:
            filename = f"{self.config.filename_prefix}_{self.export_count:04d}.{self.config.format}"
        else:
            filename = f"{self.config.filename_prefix}.{self.config.format}"
        
        filepath = os.path.join(self.config.output_dir, filename)
        
        # Export based on format
        try:
            if self.config.format == "json":
                self._export_json(filepath, export_data)
            elif self.config.format == "csv":
                self._export_csv(filepath, export_data)
            elif self.config.format == "sdf":
                self._export_sdf(filepath, export_data)
            elif self.config.format == "txt" or self.config.format == "source_code":
                self._export_text(filepath, export_data)
            else:
                print(f"[FileExportNode] Unknown format: {self.config.format}")
                return {'status': 'error', 'message': f'Unknown format: {self.config.format}'}
            
            self.export_count += 1
            print(f"[FileExportNode] Exported: {filepath}")
            
            return {
                'status': 'success',
                'filename': filename,
                'filepath': filepath,
                'count': self.export_count
            }
            
        except Exception as e:
            print(f"[FileExportNode ERROR] Export failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _export_json(self, filepath: str, data: Dict[str, Any]):
        """Export as JSON."""
        # Convert numpy arrays to lists
        serializable_data = self._make_serializable(data)
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def _export_csv(self, filepath: str, data: Dict[str, Any]):
        """Export as CSV."""
        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(data.keys())
            
            # Write data (transpose if arrays)
            max_len = max(len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in data.values())
            for i in range(max_len):
                row = []
                for v in data.values():
                    if isinstance(v, (list, np.ndarray)):
                        row.append(v[i] if i < len(v) else '')
                    else:
                        row.append(v if i == 0 else '')
                writer.writerow(row)
    
    def _export_sdf(self, filepath: str, data: Dict[str, Any]):
        """Export as SDF (Structure Data File) for molecules."""
        # Simplified SDF export - real implementation would use RDKit
        with open(filepath, 'w') as f:
            f.write("Generated by GMCS\n\n\n")
            f.write("  0  0  0  0  0  0  0  0  0  0999 V2000\n")
            f.write("M  END\n")
            f.write("> <Data>\n")
            f.write(str(data))
            f.write("\n\n$$$$\n")
    
    def _export_text(self, filepath: str, data: Dict[str, Any]):
        """Export as plain text or source code."""
        with open(filepath, 'w') as f:
            for key, value in data.items():
                f.write(f"# {key}\n")
                f.write(str(value))
                f.write("\n\n")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        else:
            return obj


# ============================================================================
# Node Factory Registration Helper
# ============================================================================

def create_output_node(node_name: str, config: Dict[str, Any]) -> Any:
    """
    Factory function to create output nodes.
    
    Args:
        node_name: Name of the output node type
        config: Node configuration
        
    Returns:
        Output node instance
    """
    if 'Audio Output' in node_name or config.get('destination') == 'audio':
        return AudioOutputNode(config)
    elif 'MIDI' in node_name or config.get('destination') == 'midi':
        return MIDIOutputNode(config)
    elif 'Video' in node_name or config.get('destination') == 'video':
        return VideoOutputNode(config)
    elif 'Export' in node_name or 'File' in node_name:
        return FileExportNode(config)
    else:
        # Default to file export
        return FileExportNode(config)

