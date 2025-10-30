'use client';

import React from 'react';
import { Oscilloscope, Spectrogram } from './ProductionVisualizers';
import { EnergyGraph, SpinMatrix, CorrelationMatrix } from './THRMLVisualizers';
import { PhaseSpace3D, XYPlot } from './PhaseSpaceVisualizer';
import { PBitMapperVisualizer } from './PBitVisualizers';

interface Connection {
  id: string;
  fromNodeId: string;
  fromPort: string;
  fromPortIndex: number;
  toNodeId: string;
  toPort: string;
  toPortIndex: number;
}

interface VisualizerProps {
  width: number;
  height: number;
  data?: any;
  nodeId?: string;
  connections?: Connection[];
}

// Re-export production visualizers with connection support
export const OscilloscopeVisualizer: React.FC<VisualizerProps> = ({ width, height, nodeId, connections = [] }) => {
  // Extract connected source node ID (if any)
  const sourceConnection = connections.find(c => c.toPort === 'signal' || c.toPort === 'x');
  const backendNodeId = sourceConnection ? parseInt(sourceConnection.fromNodeId.split('-')[1]) % 1024 : 0;
  
  return <Oscilloscope width={width} height={height} nodeId={backendNodeId} />;
};

export const SpectrogramVisualizer: React.FC<VisualizerProps> = ({ width, height, nodeId, connections = [] }) => {
  const sourceConnection = connections.find(c => c.toPort === 'signal');
  const backendNodeId = sourceConnection ? parseInt(sourceConnection.fromNodeId.split('-')[1]) % 1024 : 0;
  
  return <Spectrogram width={width} height={height} nodeId={backendNodeId} />;
};

export const PhaseSpace3DVisualizer: React.FC<VisualizerProps> = ({ width, height, nodeId, connections = [] }) => {
  const xConnection = connections.find(c => c.toPort === 'x');
  const backendNodeId = xConnection ? parseInt(xConnection.fromNodeId.split('-')[1]) % 1024 : 0;
  
  return <PhaseSpace3D width={width} height={height} nodeId={backendNodeId} />;
};

export const EnergyGraphVisualizer: React.FC<VisualizerProps> = ({ width, height, connections = [] }) => {
  // Can use connected THRML node's energy or global energy
  return <EnergyGraph width={width} height={height} />;
};

export const SpinStateMatrixVisualizer: React.FC<VisualizerProps> = ({ width, height, connections = [] }) => {
  // Can use connected THRML node's spins or global spins
  return <SpinMatrix width={width} height={height} />;
};

export const CorrelationMatrixVisualizer: React.FC<VisualizerProps> = ({ width, height, connections = [] }) => {
  // Can use connected correlation data or global correlations
  return <CorrelationMatrix width={width} height={height} />;
};

export const XYPlotVisualizer: React.FC<VisualizerProps> = ({ width, height, nodeId, connections = [] }) => {
  const xConnection = connections.find(c => c.toPort === 'x');
  const backendNodeId = xConnection ? parseInt(xConnection.fromNodeId.split('-')[1]) % 1024 : 0;
  
  return <XYPlot width={width} height={height} nodeId={backendNodeId} />;
};

// Simple waveform monitor (kept for compatibility)
export const WaveformMonitorVisualizer: React.FC<VisualizerProps> = ({ width, height, connections = [] }) => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [connected, setConnected] = React.useState(false);

  React.useEffect(() => {
    setConnected(connections.length > 0);
  }, [connections]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    if (!connected) {
      // Show "connect input" message
      ctx.fillStyle = '#6e7681';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Connect signal input', width / 2, height / 2);
      return;
    }

    // Center line
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Waveform
    ctx.strokeStyle = '#00ff99';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      const y = height / 2 + Math.sin(x * 0.1) * (height / 3);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }, [width, height, connected]);

  return <canvas ref={canvasRef} width={width} height={height} className="rounded" />;
};

export const PBitMapperVisualizerWrapper: React.FC<VisualizerProps> = ({ width, height, connections = [] }) => {
  return <PBitMapperVisualizer width={width} height={height} connections={connections} />;
};
