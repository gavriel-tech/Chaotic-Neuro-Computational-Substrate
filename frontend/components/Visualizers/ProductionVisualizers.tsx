'use client';

import React, { useEffect, useRef, useState } from 'react';
import { CustomSelect } from '../UI/CustomSelect';

interface VisualizerProps {
  width: number;
  height: number;
  nodeId?: number;
  data?: any;
  // Props for external control
  showMeasurements?: boolean;
  triggerMode?: 'auto' | 'normal' | 'single';
  showPeaks?: boolean;
  colorMap?: string;
}

// ============================================================================
// 1. Production Oscilloscope with Triggers and Measurements
// ============================================================================

interface OscilloscopeData {
  x: number[];
  y: number[];
  z: number[];
  time: number[];
}

export const ProductionOscilloscope: React.FC<VisualizerProps> = ({ 
  width, 
  height, 
  nodeId = 0,
  showMeasurements = true,
  triggerMode = 'auto'
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [data, setData] = useState<OscilloscopeData>({ x: [], y: [], z: [], time: [] });
  const [triggerLevel, setTriggerLevel] = useState(0.0);
  const [timeScale, setTimeScale] = useState(1.0);
  const [voltageScale, setVoltageScale] = useState(1.0);
  const [persistence, setPersistence] = useState(0);
  const historyRef = useRef<OscilloscopeData[]>([]);
  const maxHistory = 1000;

  // Fetch data from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/visualizer/oscilloscope/${nodeId}`);
        const result = await response.json();
        
        if (result.current) {
          const newPoint = {
            x: [result.current.x],
            y: [result.current.y],
            z: [result.current.z],
            time: [result.time]
          };
          
          // Add to history
          historyRef.current.push(newPoint);
          if (historyRef.current.length > maxHistory) {
            historyRef.current.shift();
          }
          
          // Flatten history for display
          const flatData: OscilloscopeData = {
            x: historyRef.current.flatMap(d => d.x),
            y: historyRef.current.flatMap(d => d.y),
            z: historyRef.current.flatMap(d => d.z),
            time: historyRef.current.flatMap(d => d.time)
          };
          
          setData(flatData);
        }
      } catch (err) {
        console.error('Failed to fetch oscilloscope data:', err);
      }
    };

    const interval = setInterval(fetchData, 33); // ~30 Hz
    return () => clearInterval(interval);
  }, [nodeId]);

  // Render oscilloscope
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.x.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear with persistence
    if (persistence > 0) {
      ctx.fillStyle = `rgba(13, 17, 23, ${1 - persistence / 100})`;
      ctx.fillRect(0, 0, width, height);
    } else {
      ctx.fillStyle = '#0d1117';
      ctx.fillRect(0, 0, width, height);
    }

    // Draw grid
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    const gridSpacing = 40;
    for (let x = 0; x < width; x += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw center lines
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw trigger level
    if (triggerMode !== 'auto') {
      ctx.strokeStyle = '#f85149';
      ctx.setLineDash([5, 5]);
      const triggerY = height / 2 - (triggerLevel * height / 4 * voltageScale);
      ctx.beginPath();
      ctx.moveTo(0, triggerY);
      ctx.lineTo(width, triggerY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw waveforms
    const drawWaveform = (values: number[], color: string, label: string) => {
      if (values.length < 2) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      const samplesPerPixel = Math.max(1, Math.floor(values.length / width));
      
      for (let i = 0; i < width; i++) {
        const idx = Math.floor(i * samplesPerPixel * timeScale);
        if (idx >= values.length) break;
        
        const value = values[idx];
        const x = i;
        const y = height / 2 - (value * height / 4 * voltageScale);
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Draw label
      ctx.fillStyle = color;
      ctx.font = '10px monospace';
      ctx.textBaseline = 'top';
      ctx.textAlign = 'start';
      ctx.fillText(label, 5, 5 + (label === 'X' ? 0 : label === 'Y' ? 12 : 24));
    };

    drawWaveform(data.x, '#58a6ff', 'X');
    drawWaveform(data.y, '#00ff99', 'Y');
    drawWaveform(data.z, '#f85149', 'Z');

    // Draw measurements
    if (showMeasurements && data.x.length > 0) {
      const measurements = calculateMeasurements(data.x);
      ctx.fillStyle = '#c9d1d9';
      ctx.font = '11px monospace';
      ctx.textBaseline = 'top';
      ctx.textAlign = 'start';
      ctx.fillText(`RMS: ${measurements.rms.toFixed(3)}`, width - 120, 5);
      ctx.fillText(`P-P: ${measurements.peakToPeak.toFixed(3)}`, width - 120, 17);
      ctx.fillText(`Freq: ${measurements.frequency.toFixed(1)} Hz`, width - 120, 29);
    }
  }, [data, width, height, triggerMode, triggerLevel, timeScale, voltageScale, showMeasurements, persistence]);

  const calculateMeasurements = (values: number[]) => {
    const rms = Math.sqrt(values.reduce((sum, v) => sum + v * v, 0) / values.length);
    const max = Math.max(...values);
    const min = Math.min(...values);
    const peakToPeak = max - min;
    
    // Simple frequency estimation via zero crossings
    let crossings = 0;
    for (let i = 1; i < values.length; i++) {
      if ((values[i-1] < 0 && values[i] >= 0) || (values[i-1] >= 0 && values[i] < 0)) {
        crossings++;
      }
    }
    const frequency = crossings / 2 / (data.time[data.time.length - 1] - data.time[0] || 1);

    return { rms, peakToPeak, frequency };
  };

  return <canvas ref={canvasRef} width={width} height={height} />;
};


// ============================================================================
// 2. Research-Grade Spectrogram with FFT Analysis
// ============================================================================

export const ResearchSpectrogram: React.FC<VisualizerProps> = ({ 
  width, 
  height, 
  nodeId = 0,
  showPeaks = true,
  colorMap = 'viridis'
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fftSize, setFftSize] = useState(1024);
  const [dbRange, setDbRange] = useState(80);
  const waterfallData = useRef<number[][]>([]);
  const maxWaterfallLines = 256;

  useEffect(() => {
    const fetchFFT = async () => {
      try {
        const response = await fetch(`http://localhost:8000/visualizer/fft/${nodeId}?fft_size=${fftSize}`);
        const result = await response.json();
        
        if (result.magnitudes) {
          // Convert to dB
          const dbMagnitudes = result.magnitudes.map((m: number) => 
            20 * Math.log10(Math.max(m, 1e-10))
          );
          
          // Add to waterfall
          waterfallData.current.push(dbMagnitudes);
          if (waterfallData.current.length > maxWaterfallLines) {
            waterfallData.current.shift();
          }
        }
      } catch (err) {
        console.error('Failed to fetch FFT data:', err);
      }
    };

    const interval = setInterval(fetchFFT, 100); // 10 Hz update
    return () => clearInterval(interval);
  }, [nodeId, fftSize]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || waterfallData.current.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    // Draw waterfall
    const lineHeight = height / maxWaterfallLines;
    waterfallData.current.forEach((line, lineIdx) => {
      const y = height - (lineIdx + 1) * lineHeight;
      
      line.forEach((db, freqIdx) => {
        const x = (freqIdx / line.length) * width;
        const normalized = Math.max(0, Math.min(1, (db + dbRange) / dbRange));
        const color = getColorFromMap(normalized, colorMap);
        
        ctx.fillStyle = color;
        ctx.fillRect(x, y, width / line.length + 1, lineHeight + 1);
      });
    });

    // Draw frequency axis
    ctx.fillStyle = '#c9d1d9';
    ctx.font = '10px monospace';
    ctx.textBaseline = 'bottom';
    ctx.textAlign = 'start';
    ctx.fillText('0 Hz', 5, height - 5);
    ctx.textAlign = 'end';
    ctx.fillText('Nyquist', width - 5, height - 5);

    // Draw peak markers
    if (showPeaks && waterfallData.current.length > 0) {
      const latestLine = waterfallData.current[waterfallData.current.length - 1];
      const peaks = findPeaks(latestLine, 3);
      
      ctx.strokeStyle = '#f85149';
      ctx.lineWidth = 1;
      peaks.forEach(peakIdx => {
        const x = (peakIdx / latestLine.length) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      });
    }
  }, [waterfallData.current.length, width, height, colorMap, dbRange, showPeaks]);

  const getColorFromMap = (value: number, map: string): string => {
    const v = Math.max(0, Math.min(1, value));
    
    switch (map) {
      case 'viridis':
        return `rgb(${Math.floor(68 + v * 185)}, ${Math.floor(1 + v * 208)}, ${Math.floor(84 + v * 106)})`;
      case 'plasma':
        return `rgb(${Math.floor(13 + v * 240)}, ${Math.floor(8 + v * 136)}, ${Math.floor(135 - v * 75)})`;
      case 'inferno':
        return `rgb(${Math.floor(0 + v * 252)}, ${Math.floor(0 + v * 255)}, ${Math.floor(4 + v * 160)})`;
      case 'grayscale':
        const gray = Math.floor(v * 255);
        return `rgb(${gray}, ${gray}, ${gray})`;
      default:
        return `rgb(${Math.floor(v * 255)}, ${Math.floor(v * 255)}, ${Math.floor(v * 255)})`;
    }
  };

  const findPeaks = (data: number[], threshold: number): number[] => {
    const peaks: number[] = [];
    for (let i = 1; i < data.length - 1; i++) {
      if (data[i] > data[i-1] && data[i] > data[i+1] && data[i] > threshold) {
        peaks.push(i);
      }
    }
    return peaks;
  };

  return <canvas ref={canvasRef} width={width} height={height} />;
};


// Export all visualizers
export {
  ProductionOscilloscope as Oscilloscope,
  ResearchSpectrogram as Spectrogram,
};

