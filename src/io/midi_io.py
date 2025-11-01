"""
MIDI I/O Module for GMCS.

Provides MIDI output for sending note and control data.
"""

from typing import Optional, Set, Tuple, List, Dict, Any


class MIDIOutput:
    """
    MIDI output for sending notes and control changes.
    
    Features:
    - Virtual MIDI port creation
    - Note on/off with velocity
    - Control change (CC) messages
    - Program change
    - Pitch bend
    - Active note tracking
    
    Example:
        >>> midi = MIDIOutput('GMCS Virtual')
        >>> midi.send_note_on(60, velocity=100)  # Middle C
        >>> time.sleep(0.5)
        >>> midi.send_note_off(60)
        >>> midi.close()
    """
    
    def __init__(
        self,
        port_name: str = 'GMCS Virtual',
        virtual: bool = True,
        channel: int = 0
    ):
        """
        Initialize MIDI output.
        
        Args:
            port_name: Name of MIDI port
            virtual: Create virtual port (True) or use existing (False)
            channel: Default MIDI channel (0-15)
        """
        self.port_name = port_name
        self.virtual = virtual
        self.default_channel = channel
        self.port = None
        self.active_notes: Set[Tuple[int, int]] = set()  # (note, channel)
        
        try:
            import mido
            self.mido = mido
            self._init_port()
        except ImportError:
            print("[MIDI] mido not installed. Install with: pip install mido")
            print("[MIDI] For ALSA support: pip install python-rtmidi")
            self.mido = None
    
    def _init_port(self):
        """Initialize MIDI port."""
        if self.mido is None:
            return
        
        try:
            if self.virtual:
                self.port = self.mido.open_output(self.port_name, virtual=True)
                print(f"[MIDI] Created virtual port: {self.port_name}")
            else:
                # List available ports
                available_ports = self.mido.get_output_names()
                
                if self.port_name in available_ports:
                    self.port = self.mido.open_output(self.port_name)
                    print(f"[MIDI] Opened port: {self.port_name}")
                elif available_ports:
                    # Use first available port
                    self.port = self.mido.open_output(available_ports[0])
                    print(f"[MIDI] Using port: {available_ports[0]}")
                else:
                    print("[MIDI] No MIDI ports available")
                    self.port = None
                    
        except Exception as e:
            print(f"[MIDI] Failed to open port '{self.port_name}': {e}")
            self.port = None
    
    def send_note_on(
        self,
        note: int,
        velocity: int = 64,
        channel: Optional[int] = None
    ):
        """
        Send MIDI note on message.
        
        Args:
            note: MIDI note number (0-127)
            velocity: Note velocity (0-127)
            channel: MIDI channel (0-15), uses default if None
        """
        if self.port is None:
            return
        
        channel = channel if channel is not None else self.default_channel
        
        try:
            # Clamp values
            note = int(np.clip(note, 0, 127))
            velocity = int(np.clip(velocity, 0, 127))
            channel = int(np.clip(channel, 0, 15))
            
            msg = self.mido.Message(
                'note_on',
                note=note,
                velocity=velocity,
                channel=channel
            )
            self.port.send(msg)
            
            # Track active note
            self.active_notes.add((note, channel))
            
        except Exception as e:
            print(f"[MIDI] Failed to send note_on: {e}")
    
    def send_note_off(
        self,
        note: int,
        channel: Optional[int] = None
    ):
        """
        Send MIDI note off message.
        
        Args:
            note: MIDI note number (0-127)
            channel: MIDI channel (0-15), uses default if None
        """
        if self.port is None:
            return
        
        channel = channel if channel is not None else self.default_channel
        
        try:
            note = int(np.clip(note, 0, 127))
            channel = int(np.clip(channel, 0, 15))
            
            msg = self.mido.Message(
                'note_off',
                note=note,
                velocity=0,
                channel=channel
            )
            self.port.send(msg)
            
            # Remove from active notes
            self.active_notes.discard((note, channel))
            
        except Exception as e:
            print(f"[MIDI] Failed to send note_off: {e}")
    
    def send_cc(
        self,
        control: int,
        value: int,
        channel: Optional[int] = None
    ):
        """
        Send MIDI control change message.
        
        Args:
            control: CC number (0-127)
            value: CC value (0-127)
            channel: MIDI channel (0-15), uses default if None
        """
        if self.port is None:
            return
        
        channel = channel if channel is not None else self.default_channel
        
        try:
            import numpy as np
            control = int(np.clip(control, 0, 127))
            value = int(np.clip(value, 0, 127))
            channel = int(np.clip(channel, 0, 15))
            
            msg = self.mido.Message(
                'control_change',
                control=control,
                value=value,
                channel=channel
            )
            self.port.send(msg)
            
        except Exception as e:
            print(f"[MIDI] Failed to send CC: {e}")
    
    def send_program_change(
        self,
        program: int,
        channel: Optional[int] = None
    ):
        """
        Send MIDI program change (instrument selection).
        
        Args:
            program: Program number (0-127)
            channel: MIDI channel (0-15), uses default if None
        """
        if self.port is None:
            return
        
        channel = channel if channel is not None else self.default_channel
        
        try:
            import numpy as np
            program = int(np.clip(program, 0, 127))
            channel = int(np.clip(channel, 0, 15))
            
            msg = self.mido.Message(
                'program_change',
                program=program,
                channel=channel
            )
            self.port.send(msg)
            
        except Exception as e:
            print(f"[MIDI] Failed to send program_change: {e}")
    
    def send_pitch_bend(
        self,
        pitch: int,
        channel: Optional[int] = None
    ):
        """
        Send MIDI pitch bend message.
        
        Args:
            pitch: Pitch bend value (-8192 to 8191, 0 = center)
            channel: MIDI channel (0-15), uses default if None
        """
        if self.port is None:
            return
        
        channel = channel if channel is not None else self.default_channel
        
        try:
            import numpy as np
            pitch = int(np.clip(pitch, -8192, 8191))
            channel = int(np.clip(channel, 0, 15))
            
            msg = self.mido.Message(
                'pitchwheel',
                pitch=pitch,
                channel=channel
            )
            self.port.send(msg)
            
        except Exception as e:
            print(f"[MIDI] Failed to send pitch_bend: {e}")
    
    def all_notes_off(self, channel: Optional[int] = None):
        """
        Turn off all active notes.
        
        Args:
            channel: MIDI channel (0-15), all channels if None
        """
        if channel is not None:
            # Turn off notes on specific channel
            notes_to_stop = [(n, c) for n, c in self.active_notes if c == channel]
        else:
            # Turn off all notes
            notes_to_stop = list(self.active_notes)
        
        for note, ch in notes_to_stop:
            self.send_note_off(note, ch)
    
    def panic(self):
        """Send all notes off on all channels (MIDI panic)."""
        if self.port is None:
            return
        
        try:
            # Send all notes off CC (120) on all channels
            for ch in range(16):
                msg = self.mido.Message(
                    'control_change',
                    control=120,
                    value=0,
                    channel=ch
                )
                self.port.send(msg)
            
            self.active_notes.clear()
            
        except Exception as e:
            print(f"[MIDI] Failed to send panic: {e}")
    
    def close(self):
        """Close MIDI port and turn off all notes."""
        if self.port is not None:
            self.all_notes_off()
            self.port.close()
            print(f"[MIDI] Closed port: {self.port_name}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get MIDI output information."""
        return {
            'port_name': self.port_name,
            'virtual': self.virtual,
            'is_open': self.port is not None,
            'active_notes': len(self.active_notes),
            'default_channel': self.default_channel
        }
    
    @staticmethod
    def list_ports() -> List[str]:
        """
        List available MIDI output ports.
        
        Returns:
            List of port names
        """
        try:
            import mido
            return mido.get_output_names()
        except ImportError:
            print("[MIDI] mido not installed")
            return []
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup."""
        self.close()


# Helper functions

def note_name_to_number(note_name: str) -> int:
    """
    Convert note name to MIDI number.
    
    Args:
        note_name: Note name like 'C4', 'A#5', 'Bb3'
        
    Returns:
        MIDI note number (0-127)
    """
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    
    # Parse note name
    name = note_name[0].upper()
    octave = int(note_name[-1])
    
    # Handle sharps and flats
    if len(note_name) > 2:
        modifier = note_name[1]
        if modifier == '#':
            offset = 1
        elif modifier == 'b':
            offset = -1
        else:
            offset = 0
    else:
        offset = 0
    
    # Calculate MIDI number
    midi_num = note_map[name] + offset + (octave + 1) * 12
    
    return max(0, min(127, midi_num))


def frequency_to_note(freq: float) -> int:
    """
    Convert frequency (Hz) to nearest MIDI note number.
    
    Args:
        freq: Frequency in Hz
        
    Returns:
        MIDI note number (0-127)
    """
    import numpy as np
    
    # A4 = 440 Hz = MIDI note 69
    midi_num = 69 + 12 * np.log2(freq / 440.0)
    
    return int(np.clip(np.round(midi_num), 0, 127))

