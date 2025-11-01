"""
File Upload API Endpoints for GMCS.

Provides REST API for uploading audio files, videos, and other media.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import tempfile
import shutil
import os
from pathlib import Path

router = APIRouter(prefix="/api/files", tags=["files"])

# ============================================================================
# Configuration
# ============================================================================

# Temporary upload directory
UPLOAD_DIR = Path(tempfile.gettempdir()) / "gmcs_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

# Maximum file size (MB)
MAX_FILE_SIZE_MB = 100


# ============================================================================
# Models
# ============================================================================

class FileInfo(BaseModel):
    """File information response."""
    path: str
    filename: str
    size: int
    duration: float = None
    sample_rate: int = None
    width: int = None
    height: int = None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/audio/upload", response_model=FileInfo)
async def upload_audio(file: UploadFile = File(...)) -> FileInfo:
    """
    Upload an audio file for use in the node graph.
    
    Accepts: mp3, wav, flac, ogg, m4a, aac
    Max size: 100 MB
    
    Returns:
        FileInfo with path, duration, and sample rate
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio format. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
            )
        
        # Save file to temporary location
        file_path = UPLOAD_DIR / file.filename
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            file_path.unlink()  # Delete file
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE_MB} MB"
            )
        
        # Get audio metadata
        try:
            import librosa
            duration = librosa.get_duration(path=str(file_path))
            # Try to get sample rate
            try:
                _, sr = librosa.load(str(file_path), sr=None, duration=0.1)
                sample_rate = int(sr)
            except:
                sample_rate = 48000  # Default
        except ImportError:
            duration = 0.0
            sample_rate = 48000
        except Exception as e:
            print(f"[Audio Upload] Failed to get metadata: {e}")
            duration = 0.0
            sample_rate = 48000
        
        return FileInfo(
            path=str(file_path),
            filename=file.filename,
            size=file_size,
            duration=duration,
            sample_rate=sample_rate
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/video/upload", response_model=FileInfo)
async def upload_video(file: UploadFile = File(...)) -> FileInfo:
    """
    Upload a video file.
    
    Accepts: mp4, avi, mov, mkv, webm
    Max size: 100 MB
    
    Returns:
        FileInfo with path and dimensions
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video format. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
            )
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            file_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE_MB} MB"
            )
        
        # Get video metadata
        try:
            import cv2
            cap = cv2.VideoCapture(str(file_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()
        except Exception as e:
            print(f"[Video Upload] Failed to get metadata: {e}")
            width = 0
            height = 0
            duration = 0.0
        
        return FileInfo(
            path=str(file_path),
            filename=file.filename,
            size=file_size,
            duration=duration,
            width=width,
            height=height
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/image/upload", response_model=FileInfo)
async def upload_image(file: UploadFile = File(...)) -> FileInfo:
    """
    Upload an image file.
    
    Accepts: jpg, jpeg, png, gif, bmp
    Max size: 100 MB
    
    Returns:
        FileInfo with path and dimensions
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            )
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            file_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE_MB} MB"
            )
        
        # Get image dimensions
        try:
            import cv2
            img = cv2.imread(str(file_path))
            height, width = img.shape[:2]
        except Exception as e:
            print(f"[Image Upload] Failed to get dimensions: {e}")
            width = 0
            height = 0
        
        return FileInfo(
            path=str(file_path),
            filename=file.filename,
            size=file_size,
            width=width,
            height=height
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/list")
async def list_uploads() -> List[Dict[str, Any]]:
    """
    List all uploaded files.
    
    Returns:
        List of file information dicts
    """
    try:
        files = []
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                files.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': file_path.stat().st_mtime
                })
        
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.delete("/{filename}")
async def delete_file(filename: str) -> Dict[str, str]:
    """
    Delete an uploaded file.
    
    Args:
        filename: Name of file to delete
        
    Returns:
        Success message
    """
    try:
        file_path = UPLOAD_DIR / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path.unlink()
        
        return {"status": "success", "message": f"Deleted {filename}"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

