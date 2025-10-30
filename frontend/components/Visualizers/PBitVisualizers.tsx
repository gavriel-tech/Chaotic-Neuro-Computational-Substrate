'use client';

import React, { useRef, useEffect } from 'react';

interface PBitMapperProps {
  width?: number;
  height?: number;
  gridSize?: number;
  colorScheme?: 'red-green' | 'blue-yellow' | 'monochrome';
  updateRate?: number;
  connections?: { fromNodeId: string; fromPort: string }[];
}

const HISTORY_MAX = 128;

const createGrid = (size: number, fill: number = -1): number[][] =>
  Array.from({ length: size }, () => Array.from({ length: size }, () => fill));

const reshapeToGrid = (data: number[], size: number): number[][] => {
  const grid = createGrid(size, -1);
  const limit = Math.min(data.length, size * size);
  for (let i = 0; i < limit; i++) {
    const row = Math.floor(i / size);
    const col = i % size;
    const value = data[i];
    grid[row][col] = value >= 0 ? 1 : -1;
  }
  return grid;
};

const withAlpha = (hex: string, alpha: number) => {
  const sanitized = hex.replace('#', '');
  const bigint = parseInt(sanitized, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

export const PBitMapperVisualizer: React.FC<PBitMapperProps> = ({
  width = 300,
  height = 200,
  gridSize = 8,
  colorScheme = 'red-green',
  updateRate = 100,
  connections = []
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gridSizeRef = useRef<number>(gridSize);
  const currentGridRef = useRef<number[][]>(createGrid(gridSize));
  const historyRef = useRef<number[][][]>([]);
  const lastUpdateRef = useRef<number>(0);
  const statusRef = useRef<'waiting' | 'live' | 'error' | 'stale'>('waiting');

  // Sync default grid when prop changes and before data arrives
  useEffect(() => {
    gridSizeRef.current = gridSize;
    currentGridRef.current = createGrid(gridSize);
    historyRef.current = [];
    statusRef.current = 'waiting';
  }, [gridSize]);

  // Fetch THRML p-bit data on interval
  useEffect(() => {
    let cancelled = false;

    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/visualizer/thrml/pbits');
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const payload = await response.json();
        if (cancelled) return;

        if (Array.isArray(payload.current) && payload.current.length > 0) {
          const payloadSize = typeof payload.grid_size === 'number' && payload.grid_size > 0
            ? payload.grid_size
            : Math.max(gridSize, Math.round(Math.sqrt(payload.current.length)) || gridSize);

          gridSizeRef.current = payloadSize;
          currentGridRef.current = reshapeToGrid(payload.current, payloadSize);

          if (Array.isArray(payload.history)) {
            const trimmedHistory = payload.history.slice(-HISTORY_MAX);
            historyRef.current = trimmedHistory.map((entry: number[]) =>
              reshapeToGrid(entry, payloadSize)
            );
          }

          lastUpdateRef.current = Date.now();
          statusRef.current = 'live';
        } else {
          statusRef.current = 'waiting';
        }
      } catch (error) {
        if (!cancelled) {
          statusRef.current = 'error';
        }
      }
    };

    fetchData();
    const intervalId = window.setInterval(fetchData, updateRate);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [updateRate, gridSize, connections]);

  // Draw loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrame: number;

    const draw = () => {
      const size = Math.max(1, gridSizeRef.current);
      const currentGrid = currentGridRef.current;
      const history = historyRef.current;

      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, width, height);

      const cellWidth = width / size;
      const cellHeight = height / size;

      const historyDepth = history.length;
      let historyAverages: number[][] | null = null;

      if (historyDepth > 0) {
        historyAverages = createGrid(size, 0);
        for (let h = 0; h < historyDepth; h++) {
          const snapshot = history[h];
          for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
              const value = snapshot?.[row]?.[col] ?? -1;
              historyAverages[row][col] += value;
            }
          }
        }
        for (let row = 0; row < size; row++) {
          for (let col = 0; col < size; col++) {
            historyAverages[row][col] /= historyDepth;
          }
        }
      }

      ctx.shadowBlur = 0;

      const getColors = (value: number) => {
        switch (colorScheme) {
          case 'blue-yellow':
            return value >= 0
              ? { on: '#ffd700', off: '#1e90ff' }
              : { on: '#ffd700', off: '#1e90ff' };
          case 'monochrome':
            return { on: '#ffffff', off: '#1a1a1a' };
          case 'red-green':
          default:
            return { on: '#00ff99', off: '#f85149' };
        }
      };

      for (let row = 0; row < size; row++) {
        for (let col = 0; col < size; col++) {
          const value = currentGrid?.[row]?.[col] ?? -1;
          const mean = historyAverages ? historyAverages[row][col] : 0;
          const persistence = Math.max(0, Math.min(1, (mean + 1) / 2));
          const active = value >= 0 ? 1 : 0;

          const { on, off } = getColors(value);
          const baseColor = value >= 0 ? on : off;
          const alpha = 0.25 + 0.6 * (active * 0.6 + persistence * 0.4);

          const x = col * cellWidth;
          const y = row * cellHeight;

          ctx.fillStyle = withAlpha(baseColor, Math.min(1, alpha));
          ctx.fillRect(x + 1, y + 1, cellWidth - 2, cellHeight - 2);

          if (value >= 0) {
            ctx.shadowColor = withAlpha(on, Math.min(1, 0.4 + persistence));
            ctx.shadowBlur = 8 * (0.5 + persistence * 0.8);
            ctx.fillStyle = withAlpha(on, Math.min(1, alpha + 0.2));
            ctx.fillRect(x + 2, y + 2, cellWidth - 4, cellHeight - 4);
            ctx.shadowBlur = 0;
          }
        }
      }

      ctx.strokeStyle = 'rgba(0, 204, 119, 0.28)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= size; i++) {
        ctx.beginPath();
        ctx.moveTo(0, i * cellHeight);
        ctx.lineTo(width, i * cellHeight);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(i * cellWidth, 0);
        ctx.lineTo(i * cellWidth, height);
        ctx.stroke();
      }

      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';

      const elapsed = Date.now() - lastUpdateRef.current;
      const staleThreshold = updateRate * 3;
      if (statusRef.current === 'live' && elapsed > staleThreshold) {
        statusRef.current = 'stale';
      }

      const statusColor =
        statusRef.current === 'error'
          ? '#f85149'
          : statusRef.current === 'stale'
            ? '#d29922'
            : '#00cc77';

      ctx.fillStyle = statusColor;
      ctx.fillText(`status: ${statusRef.current}`, 8, 6);
      ctx.fillText(`grid: ${size}×${size}`, 8, 18);
      ctx.fillText(`history: ${history.length}`, 8, 30);

      if (statusRef.current === 'error' || statusRef.current === 'waiting') {
        ctx.fillStyle = 'rgba(248, 81, 73, 0.85)';
        ctx.fillText('waiting for THRML data…', 8, height - 18);
      }

      animationFrame = window.requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.cancelAnimationFrame(animationFrame);
    };
  }, [width, height, colorScheme, updateRate]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="w-full rounded border border-[#00cc77]/40"
      style={{ imageRendering: 'pixelated' }}
    />
  );
};

