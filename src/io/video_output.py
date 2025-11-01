"""
Video Output Module for GMCS.

Provides real-time video rendering and recording capabilities.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class VideoOutput:
    """
    Real-time video output and recording for GMCS visualizations.
    
    Features:
    - Live window display
    - Video file recording (mp4, avi)
    - Configurable resolution and framerate
    - Frame-by-frame rendering
    
    Example:
        >>> video_out = VideoOutput(resolution=(1920, 1080), fps=60)
        >>> for frame_data in simulation:
        >>>     video_out.write_frame(frame_data)
        >>> video_out.close()
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 60,
        output_file: Optional[str] = None,
        codec: str = 'mp4v',
        window_name: str = 'GMCS Video Output'
    ):
        """
        Initialize video output.
        
        Args:
            resolution: (width, height) in pixels
            fps: Frames per second
            output_file: Path to save video (None for live display only)
            codec: Video codec ('mp4v', 'XVID', 'MJPG', 'H264')
            window_name: Name for display window
        """
        self.resolution = resolution
        self.fps = fps
        self.output_file = output_file
        self.codec = codec
        self.window_name = window_name
        
        self.writer = None
        self.frame_count = 0
        self.is_recording = output_file is not None
        
        # Initialize OpenCV writer and/or window
        try:
            import cv2
            self.cv2 = cv2
            
            if self.is_recording:
                self._init_writer()
            else:
                self._init_window()
                
        except ImportError:
            print("[VideoOutput] OpenCV not installed. Install with: pip install opencv-python")
            self.cv2 = None
    
    def _init_writer(self):
        """Initialize video writer for file recording."""
        if self.cv2 is None:
            return
        
        try:
            fourcc = self.cv2.VideoWriter_fourcc(*self.codec)
            self.writer = self.cv2.VideoWriter(
                self.output_file,
                fourcc,
                self.fps,
                self.resolution
            )
            
            if not self.writer.isOpened():
                print(f"[VideoOutput] Failed to open video writer: {self.output_file}")
                self.writer = None
                self.is_recording = False
            else:
                print(f"[VideoOutput] Recording to {self.output_file}")
                
        except Exception as e:
            print(f"[VideoOutput] Failed to initialize writer: {e}")
            self.writer = None
            self.is_recording = False
    
    def _init_window(self):
        """Initialize display window for live output."""
        if self.cv2 is None:
            return
        
        try:
            self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
            self.cv2.resizeWindow(self.window_name, self.resolution[0], self.resolution[1])
        except Exception as e:
            print(f"[VideoOutput] Failed to create window: {e}")
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a single frame.
        
        Args:
            frame: (H, W, 3) array in [0, 1] range (RGB)
                   or (H, W) grayscale in [0, 1]
        """
        if self.cv2 is None:
            return
        
        try:
            # Ensure frame is correct shape
            if frame.ndim == 2:
                # Grayscale → RGB
                frame = np.stack([frame, frame, frame], axis=-1)
            elif frame.shape[2] == 4:
                # RGBA → RGB (drop alpha)
                frame = frame[:, :, :3]
            
            # Resize if needed
            if frame.shape[:2][::-1] != self.resolution:
                frame = self.cv2.resize(frame, self.resolution)
            
            # Convert to uint8 [0, 255]
            if frame.dtype != np.uint8:
                frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame_uint8 = frame
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = self.cv2.cvtColor(frame_uint8, self.cv2.COLOR_RGB2BGR)
            
            # Write to file or display
            if self.writer is not None:
                self.writer.write(frame_bgr)
            else:
                self.cv2.imshow(self.window_name, frame_bgr)
                self.cv2.waitKey(1)  # Needed for display update
            
            self.frame_count += 1
            
        except Exception as e:
            print(f"[VideoOutput] Failed to write frame: {e}")
    
    def close(self):
        """Close video writer and windows."""
        if self.cv2 is None:
            return
        
        try:
            if self.writer is not None:
                self.writer.release()
                print(f"[VideoOutput] Saved {self.frame_count} frames to {self.output_file}")
            
            self.cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"[VideoOutput] Error during close: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get video output information."""
        return {
            'resolution': self.resolution,
            'fps': self.fps,
            'output_file': self.output_file,
            'is_recording': self.is_recording,
            'frame_count': self.frame_count,
            'duration': self.frame_count / self.fps if self.fps > 0 else 0.0
        }
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper cleanup."""
        self.close()


# Helper functions

def create_video_from_frames(
    frames: list,
    output_file: str,
    fps: int = 30,
    resolution: Optional[Tuple[int, int]] = None
):
    """
    Create a video file from a list of frames.
    
    Args:
        frames: List of (H, W, 3) numpy arrays
        output_file: Output video path
        fps: Frames per second
        resolution: Target resolution (uses first frame's size if None)
    """
    if not frames:
        print("[VideoOutput] No frames to write")
        return
    
    # Determine resolution
    if resolution is None:
        h, w = frames[0].shape[:2]
        resolution = (w, h)
    
    # Create video
    with VideoOutput(resolution=resolution, fps=fps, output_file=output_file) as video:
        for frame in frames:
            video.write_frame(frame)
    
    print(f"[VideoOutput] Created video with {len(frames)} frames")


def render_oscillator_field(
    oscillator_states: np.ndarray,
    wave_field: np.ndarray,
    resolution: Tuple[int, int] = (1920, 1080),
    colormap: str = 'viridis'
) -> np.ndarray:
    """
    Render oscillator states and wave field as a frame.
    
    Args:
        oscillator_states: (N, 3) oscillator states
        wave_field: (H, W) wave field
        resolution: Output resolution
        colormap: Color mapping ('viridis', 'hot', 'cool')
        
    Returns:
        (H, W, 3) RGB frame in [0, 1]
    """
    try:
        import cv2
        
        # Normalize wave field to [0, 1]
        field_norm = (wave_field - wave_field.min()) / (wave_field.max() - wave_field.min() + 1e-10)
        
        # Resize to target resolution
        field_resized = cv2.resize(field_norm, resolution)
        
        # Apply colormap
        if colormap == 'hot':
            r = field_resized
            g = np.clip(field_resized * 2 - 1, 0, 1)
            b = np.clip(field_resized * 4 - 3, 0, 1)
        elif colormap == 'cool':
            r = 1 - field_resized
            g = field_resized
            b = 1 - field_resized
        else:  # viridis-like
            r = np.sqrt(field_resized)
            g = field_resized ** 3
            b = 1 - field_resized ** 0.5
        
        frame = np.stack([r, g, b], axis=-1)
        
        return frame
        
    except ImportError:
        # Fallback: simple grayscale
        field_norm = (wave_field - wave_field.min()) / (wave_field.max() - wave_field.min() + 1e-10)
        frame = np.stack([field_norm, field_norm, field_norm], axis=-1)
        return frame

