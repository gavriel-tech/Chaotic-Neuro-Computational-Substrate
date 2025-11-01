/**
 * Audio File Uploader Component for GMCS.
 * 
 * Allows users to upload audio files for use in node graphs.
 */

import React, { useState, useCallback } from 'react';
import { Box, Button, Typography, LinearProgress, Alert, IconButton, Chip } from '@mui/material';
import { Upload, AudioFile, Delete, PlayArrow, Pause } from '@mui/icons-material';

interface AudioFileInfo {
  path: string;
  filename: string;
  size: number;
  duration: number;
  sample_rate: number;
}

interface AudioUploaderProps {
  nodeId?: string;
  onFileUploaded?: (fileInfo: AudioFileInfo) => void;
  onFileSelected?: (file: File) => void;
}

export const AudioUploader: React.FC<AudioUploaderProps> = ({
  nodeId,
  onFileUploaded,
  onFileSelected
}) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFile, setUploadedFile] = useState<AudioFileInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/flac', 'audio/ogg', 'audio/x-m4a'];
    const validExtensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'];
    
    const isValidType = validTypes.includes(file.type);
    const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

    if (!isValidType && !hasValidExtension) {
      setError('Invalid file type. Please upload mp3, wav, flac, ogg, or m4a files.');
      return;
    }

    // Validate file size (max 100 MB)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
      setError(`File too large. Maximum size is 100 MB. Your file is ${(file.size / 1024 / 1024).toFixed(2)} MB.`);
      return;
    }

    setSelectedFile(file);
    setError(null);
    
    if (onFileSelected) {
      onFileSelected(file);
    }
  }, [onFileSelected]);

  const handleUpload = useCallback(async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = (event.loaded / event.total) * 100;
          setUploadProgress(progress);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText);
          setUploadedFile(response);
          setUploading(false);
          setUploadProgress(100);
          
          if (onFileUploaded) {
            onFileUploaded(response);
          }
        } else {
          const errorData = JSON.parse(xhr.responseText);
          setError(errorData.detail || 'Upload failed');
          setUploading(false);
        }
      });

      xhr.addEventListener('error', () => {
        setError('Network error during upload');
        setUploading(false);
      });

      xhr.open('POST', '/api/files/audio/upload');
      xhr.send(formData);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploading(false);
    }
  }, [selectedFile, onFileUploaded]);

  const handleRemove = useCallback(() => {
    setSelectedFile(null);
    setUploadedFile(null);
    setError(null);
    setUploadProgress(0);
  }, []);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
      <Typography variant="h6" gutterBottom>
        <AudioFile sx={{ verticalAlign: 'middle', mr: 1 }} />
        Audio File Upload
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {!uploadedFile ? (
        <>
          <input
            accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.aac"
            style={{ display: 'none' }}
            id={`audio-upload-${nodeId || 'default'}`}
            type="file"
            onChange={handleFileSelect}
          />
          <label htmlFor={`audio-upload-${nodeId || 'default'}`}>
            <Button
              variant="outlined"
              component="span"
              startIcon={<Upload />}
              fullWidth
              sx={{ mb: 2 }}
            >
              Select Audio File
            </Button>
          </label>

          {selectedFile && (
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="body2" sx={{ flex: 1, mr: 2 }}>
                  {selectedFile.name}
                </Typography>
                <Chip label={formatFileSize(selectedFile.size)} size="small" />
                <IconButton size="small" onClick={handleRemove} sx={{ ml: 1 }}>
                  <Delete />
                </IconButton>
              </Box>

              <Button
                variant="contained"
                onClick={handleUpload}
                disabled={uploading}
                fullWidth
              >
                {uploading ? 'Uploading...' : 'Upload'}
              </Button>

              {uploading && (
                <Box sx={{ mt: 2 }}>
                  <LinearProgress variant="determinate" value={uploadProgress} />
                  <Typography variant="caption" sx={{ mt: 0.5, display: 'block', textAlign: 'center' }}>
                    {uploadProgress.toFixed(0)}%
                  </Typography>
                </Box>
              )}
            </Box>
          )}
        </>
      ) : (
        <Box>
          <Alert severity="success" sx={{ mb: 2 }}>
            File uploaded successfully!
          </Alert>

          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1 }}>
            <Typography variant="body2" gutterBottom>
              <strong>Filename:</strong> {uploadedFile.filename}
            </Typography>
            <Typography variant="body2" gutterBottom>
              <strong>Duration:</strong> {formatDuration(uploadedFile.duration)}
            </Typography>
            <Typography variant="body2" gutterBottom>
              <strong>Sample Rate:</strong> {uploadedFile.sample_rate} Hz
            </Typography>
            <Typography variant="body2" gutterBottom>
              <strong>Size:</strong> {formatFileSize(uploadedFile.size)}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
              Path: {uploadedFile.path}
            </Typography>
          </Box>

          <Button
            variant="outlined"
            startIcon={<Delete />}
            onClick={handleRemove}
            fullWidth
            sx={{ mt: 2 }}
          >
            Remove File
          </Button>
        </Box>
      )}

      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
        Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC (max 100 MB)
      </Typography>
    </Box>
  );
};

export default AudioUploader;

