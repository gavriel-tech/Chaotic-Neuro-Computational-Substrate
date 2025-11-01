"""
Music Processing Nodes for GMCS.

Provides music-specific analysis and generation capabilities:
- ChordDetector: Chromagram-based chord recognition
- BeatTracker: Onset detection and tempo estimation
- HarmonyAnalyzer: Key detection and harmonic analysis
- RhythmGenerator: Euclidean rhythm pattern generation

These nodes are used by music analysis and live music presets.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque


# ============================================================================
# Chord Detector
# ============================================================================

@dataclass
class ChordDetectorConfig:
    """Configuration for chord detection."""
    method: str = 'chromagram'
    root_detection: bool = True
    min_confidence: float = 0.6
    sample_rate: float = 48000
    n_fft: int = 2048
    hop_length: int = 512


class ChordDetector:
    """
    Detect chords from audio spectrum using chromagram analysis.
    
    Uses pitch class profiles and template matching to identify
    major, minor, 7th, and suspended chords.
    """
    
    # Chord templates (12 semitones, normalized)
    CHORD_TEMPLATES = {
        'maj': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]),  # Root, M3, P5
        'min': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]),  # Root, m3, P5
        'maj7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]),  # Root, M3, P5, M7
        'min7': np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]),  # Root, m3, P5, m7
        'dom7': np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),  # Root, M3, P5, m7
        'sus4': np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]),  # Root, P4, P5
        'sus2': np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]),  # Root, M2, P5
        'dim': np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]),   # Root, m3, d5
        'aug': np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),   # Root, M3, A5
    }
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def __init__(self, config: Dict[str, Any]):
        self.config = ChordDetectorConfig(**config)
        self.spectrum_buffer = deque(maxlen=4)  # Average over 4 frames
        
    def process(self, spectrum: np.ndarray) -> Dict[str, Any]:
        """
        Detect chord from frequency spectrum.
        
        Args:
            spectrum: FFT magnitude spectrum (N_FFT/2 + 1,)
            
        Returns:
            Dict with chord, confidence, root, bass
        """
        # Convert spectrum to chroma features
        chroma = self._spectrum_to_chroma(spectrum)
        
        # Add to buffer for temporal smoothing
        self.spectrum_buffer.append(chroma)
        
        # Average chroma over buffer
        avg_chroma = np.mean(list(self.spectrum_buffer), axis=0)
        
        # Normalize
        chroma_norm = avg_chroma / (np.sum(avg_chroma) + 1e-10)
        
        # Match against chord templates
        best_chord, confidence = self._match_chord(chroma_norm)
        
        # Extract root note
        root = self._extract_root(best_chord) if self.config.root_detection else None
        bass = root  # Simplified - full implementation would detect inversions
        
        return {
            'chord': best_chord,
            'confidence': float(confidence),
            'root': root,
            'bass': bass,
            'chroma': chroma_norm
        }
    
    def _spectrum_to_chroma(self, spectrum: np.ndarray) -> np.ndarray:
        """Convert FFT spectrum to 12-bin chromagram (pitch classes)."""
        n_bins = len(spectrum)
        chroma = np.zeros(12)
        
        # Map frequency bins to pitch classes
        # Using equal temperament: f = 440 * 2^((n-69)/12)
        for i in range(n_bins):
            freq = i * self.config.sample_rate / (2 * n_bins)
            
            if freq < 20:  # Skip very low frequencies
                continue
                
            # Convert frequency to MIDI note number
            if freq > 0:
                midi_note = 69 + 12 * np.log2(freq / 440.0)
                pitch_class = int(round(midi_note)) % 12
                
                # Add magnitude to corresponding pitch class
                if 0 <= pitch_class < 12:
                    chroma[pitch_class] += spectrum[i]
        
        return chroma
    
    def _match_chord(self, chroma: np.ndarray) -> Tuple[str, float]:
        """Match chroma profile against chord templates."""
        best_chord = 'N'  # No chord
        best_score = 0.0
        
        # Try all chord types at all 12 roots
        for root in range(12):
            for chord_type, template in self.CHORD_TEMPLATES.items():
                # Rotate template to current root
                rotated_template = np.roll(template, root)
                
                # Compute correlation
                score = np.dot(chroma, rotated_template)
                
                if score > best_score:
                    best_score = score
                    chord_name = f"{self.NOTE_NAMES[root]}{chord_type}"
                    best_chord = chord_name
        
        # Normalize score to [0, 1]
        confidence = min(best_score / (np.sum(chroma) + 1e-10), 1.0)
        
        # Return 'N' if confidence too low
        if confidence < self.config.min_confidence:
            return 'N', confidence
            
        return best_chord, confidence
    
    def _extract_root(self, chord: str) -> Optional[str]:
        """Extract root note from chord name."""
        if chord == 'N' or len(chord) < 1:
            return None
        
        # Handle sharps (C#, D#, etc)
        if len(chord) > 1 and chord[1] == '#':
            return chord[:2]
        else:
            return chord[0]


# ============================================================================
# Beat Tracker
# ============================================================================

@dataclass
class BeatTrackerConfig:
    """Configuration for beat tracking."""
    sample_rate: float = 48000
    hop_length: int = 512
    min_bpm: float = 60.0
    max_bpm: float = 180.0
    onset_threshold: float = 0.3


class BeatTracker:
    """
    Track beats using onset detection and tempo estimation.
    
    Uses spectral flux for onset detection and autocorrelation
    for tempo estimation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = BeatTrackerConfig(**config)
        
        # Buffers for analysis
        self.spectrum_history = deque(maxlen=200)  # ~4 seconds at 512 hop
        self.onset_history = deque(maxlen=200)
        self.beat_times = []
        self.last_beat_time = 0
        self.current_time = 0
        
        # Tempo tracking
        self.estimated_bpm = 120.0
        self.beat_phase = 0.0
        
    def process(self, spectrum: np.ndarray, time: float) -> Dict[str, Any]:
        """
        Process spectrum and detect beats.
        
        Args:
            spectrum: FFT magnitude spectrum
            time: Current time in seconds
            
        Returns:
            Dict with bpm, beat_detected, phase, onsets
        """
        self.current_time = time
        
        # Compute onset strength (spectral flux)
        onset_strength = self._compute_onset_strength(spectrum)
        
        # Add to history
        self.spectrum_history.append(spectrum)
        self.onset_history.append(onset_strength)
        
        # Detect beat
        beat_detected = self._detect_beat(onset_strength)
        
        # Estimate tempo periodically
        if len(self.onset_history) >= 100 and len(self.onset_history) % 50 == 0:
            self.estimated_bpm = self._estimate_tempo()
        
        # Update beat phase
        beat_interval = 60.0 / self.estimated_bpm
        self.beat_phase = (time - self.last_beat_time) / beat_interval
        
        return {
            'bpm': float(self.estimated_bpm),
            'beat_detected': bool(beat_detected),
            'phase': float(self.beat_phase % 1.0),
            'onset_strength': float(onset_strength),
            'beat_times': list(self.beat_times[-10:])  # Last 10 beats
        }
    
    def _compute_onset_strength(self, spectrum: np.ndarray) -> float:
        """Compute onset strength using spectral flux."""
        if len(self.spectrum_history) == 0:
            return 0.0
        
        # Spectral flux = sum of positive differences
        prev_spectrum = self.spectrum_history[-1]
        diff = spectrum - prev_spectrum
        
        # Half-wave rectification (only positive changes)
        positive_diff = np.maximum(diff, 0)
        
        # Sum and normalize
        flux = np.sum(positive_diff) / (np.sum(spectrum) + 1e-10)
        
        return flux
    
    def _detect_beat(self, onset_strength: float) -> bool:
        """Detect if current frame is a beat."""
        if len(self.onset_history) < 10:
            return False
        
        # Adaptive threshold
        recent_onsets = list(self.onset_history)[-50:]
        threshold = np.mean(recent_onsets) + self.config.onset_threshold * np.std(recent_onsets)
        
        # Peak detection with minimum interval
        min_interval = 60.0 / self.config.max_bpm
        time_since_last_beat = self.current_time - self.last_beat_time
        
        if onset_strength > threshold and time_since_last_beat > min_interval:
            self.beat_times.append(self.current_time)
            self.last_beat_time = self.current_time
            return True
        
        return False
    
    def _estimate_tempo(self) -> float:
        """Estimate tempo using autocorrelation of onset function."""
        if len(self.onset_history) < 100:
            return self.estimated_bpm
        
        onset_curve = np.array(list(self.onset_history))
        
        # Autocorrelation
        autocorr = np.correlate(onset_curve, onset_curve, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
        
        # Convert lag to BPM range
        frame_rate = self.config.sample_rate / self.config.hop_length
        min_lag = int(frame_rate * 60.0 / self.config.max_bpm)
        max_lag = int(frame_rate * 60.0 / self.config.min_bpm)
        
        # Find peak in valid range
        if max_lag < len(autocorr):
            valid_autocorr = autocorr[min_lag:max_lag]
            peak_lag = np.argmax(valid_autocorr) + min_lag
            
            # Convert lag to BPM
            bpm = 60.0 * frame_rate / peak_lag
            
            return float(np.clip(bpm, self.config.min_bpm, self.config.max_bpm))
        
        return self.estimated_bpm


# ============================================================================
# Harmony Analyzer
# ============================================================================

@dataclass
class HarmonyAnalyzerConfig:
    """Configuration for harmony analysis."""
    method: str = 'krumhansl_schmuckler'
    window_size: int = 100


class HarmonyAnalyzer:
    """
    Analyze harmonic content and detect musical key.
    
    Uses Krumhansl-Schmuckler algorithm for key detection
    based on pitch class profiles.
    """
    
    # Krumhansl-Schmuckler key profiles
    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def __init__(self, config: Dict[str, Any]):
        self.config = HarmonyAnalyzerConfig(**config)
        self.chroma_history = deque(maxlen=config.get('window_size', 100))
        
    def process(self, chroma: np.ndarray) -> Dict[str, Any]:
        """
        Analyze harmony from chroma features.
        
        Args:
            chroma: 12-bin chromagram
            
        Returns:
            Dict with key, mode, stability, tonic
        """
        # Add to history
        self.chroma_history.append(chroma)
        
        # Average chroma over window
        if len(self.chroma_history) < 10:
            avg_chroma = chroma
        else:
            avg_chroma = np.mean(list(self.chroma_history), axis=0)
        
        # Detect key
        key, mode, correlation = self._detect_key(avg_chroma)
        
        # Compute stability (how well it fits the key)
        stability = correlation
        
        # Tonic is the root of the key
        tonic = key
        
        return {
            'key': f"{key} {mode}",
            'mode': mode,
            'stability': float(stability),
            'tonic': tonic,
            'pitch_class_distribution': avg_chroma.tolist()
        }
    
    def _detect_key(self, chroma: np.ndarray) -> Tuple[str, str, float]:
        """Detect key using Krumhansl-Schmuckler algorithm."""
        best_key = 'C'
        best_mode = 'major'
        best_correlation = 0.0
        
        # Normalize chroma
        chroma_norm = chroma / (np.sum(chroma) + 1e-10)
        
        # Try all 24 keys (12 major + 12 minor)
        for root in range(12):
            # Major key
            rotated_major = np.roll(self.MAJOR_PROFILE, root)
            corr_major = self._pearson_correlation(chroma_norm, rotated_major)
            
            if corr_major > best_correlation:
                best_correlation = corr_major
                best_key = self.KEY_NAMES[root]
                best_mode = 'major'
            
            # Minor key
            rotated_minor = np.roll(self.MINOR_PROFILE, root)
            corr_minor = self._pearson_correlation(chroma_norm, rotated_minor)
            
            if corr_minor > best_correlation:
                best_correlation = corr_minor
                best_key = self.KEY_NAMES[root]
                best_mode = 'minor'
        
        return best_key, best_mode, best_correlation
    
    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
        
        if denominator < 1e-10:
            return 0.0
        
        return numerator / denominator


# ============================================================================
# Rhythm Generator
# ============================================================================

@dataclass
class RhythmGeneratorConfig:
    """Configuration for rhythm generation."""
    method: str = 'euclidean'
    default_steps: int = 16
    default_pulses: int = 4


class RhythmGenerator:
    """
    Generate rhythmic patterns using Euclidean and other algorithms.
    
    Euclidean rhythms distribute pulses as evenly as possible across steps,
    creating interesting polyrhythmic patterns used in world music.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = RhythmGeneratorConfig(**config)
        self.current_step = 0
        self.pattern = []
        
    def process(self, steps: Optional[int] = None, pulses: Optional[int] = None, 
                rotation: int = 0) -> Dict[str, Any]:
        """
        Generate rhythm pattern.
        
        Args:
            steps: Total number of steps in pattern
            pulses: Number of pulses (hits) to distribute
            rotation: Amount to rotate pattern
            
        Returns:
            Dict with pattern, current_step, active
        """
        steps = steps or self.config.default_steps
        pulses = pulses or self.config.default_pulses
        
        # Generate Euclidean rhythm
        if self.config.method == 'euclidean':
            pattern = self._euclidean_rhythm(int(steps), int(pulses))
        else:
            # Default to simple pattern
            pattern = [1 if i % (steps // pulses) == 0 else 0 for i in range(steps)]
        
        # Apply rotation
        if rotation != 0:
            pattern = pattern[rotation:] + pattern[:rotation]
        
        self.pattern = pattern
        
        # Current step active?
        active = self.pattern[self.current_step] if self.pattern else False
        
        # Advance step
        self.current_step = (self.current_step + 1) % len(self.pattern) if self.pattern else 0
        
        return {
            'pattern': pattern,
            'current_step': int(self.current_step),
            'active': bool(active),
            'steps': int(steps),
            'pulses': int(pulses)
        }
    
    def _euclidean_rhythm(self, steps: int, pulses: int) -> List[int]:
        """
        Generate Euclidean rhythm using Bjorklund's algorithm.
        
        This algorithm distributes k pulses across n steps as evenly as possible.
        """
        if pulses >= steps or pulses == 0:
            # Edge cases
            if pulses == 0:
                return [0] * steps
            return [1] * pulses + [0] * (steps - pulses)
        
        # Bjorklund's algorithm
        pattern = []
        counts = []
        remainders = []
        
        divisor = steps - pulses
        remainders.append(pulses)
        level = 0
        
        while True:
            counts.append(divisor // remainders[level])
            remainders.append(divisor % remainders[level])
            divisor = remainders[level]
            level += 1
            
            if remainders[level] <= 1:
                break
        
        counts.append(divisor)
        
        # Build pattern
        def build(level):
            if level == -1:
                pattern.append(0)
            elif level == -2:
                pattern.append(1)
            else:
                for _ in range(counts[level]):
                    build(level - 1)
                if remainders[level] != 0:
                    build(level - 2)
        
        build(level)
        
        # Ensure correct length
        pattern = pattern[:steps]
        while len(pattern) < steps:
            pattern.append(0)
        
        return pattern


# ============================================================================
# Factory Function
# ============================================================================

def create_music_node(node_type: str, config: Dict[str, Any]):
    """Factory function to create music nodes."""
    
    music_nodes = {
        'Chord Detector': ChordDetector,
        'ChordDetector': ChordDetector,
        'Beat Tracker': BeatTracker,
        'BeatTracker': BeatTracker,
        'Harmony Analyzer': HarmonyAnalyzer,
        'HarmonyAnalyzer': HarmonyAnalyzer,
        'Rhythm Generator': RhythmGenerator,
        'RhythmGenerator': RhythmGenerator,
    }
    
    node_class = music_nodes.get(node_type)
    if node_class is None:
        raise ValueError(f"Unknown music node type: {node_type}")
    
    return node_class(config)

