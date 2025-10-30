'use client';

import React, { useEffect, useRef, useState } from 'react';

interface VisualizerProps {
  width: number;
  height: number;
  data?: any;
  showHistogram?: boolean;
  showMovingAvg?: boolean;
  showValues?: boolean;
  showColorBar?: boolean;
}

// ============================================================================
// THRML Energy Graph with Convergence Indicators
// ============================================================================

export const THRMLEnergyGraph: React.FC<VisualizerProps> = ({ 
  width, 
  height, 
  showHistogram = false, 
  showMovingAvg = true 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [energyHistory, setEnergyHistory] = useState<number[]>([]);
  const maxHistory = 1000;

  useEffect(() => {
    const fetchEnergy = async () => {
      try {
        const response = await fetch('http://localhost:8000/visualizer/thrml/energy');
        const result = await response.json();
        
        if (result.current_energy !== undefined) {
          setEnergyHistory(prev => {
            const newHistory = [...prev, result.current_energy];
            return newHistory.slice(-maxHistory);
          });
        }
      } catch (err) {
        console.error('Failed to fetch THRML energy:', err);
      }
    };

    const interval = setInterval(fetchEnergy, 100); // 10 Hz
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || energyHistory.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    if (showHistogram) {
      // Draw histogram
      const bins = 50;
      const min = Math.min(...energyHistory);
      const max = Math.max(...energyHistory);
      const binWidth = (max - min) / bins;
      const histogram = new Array(bins).fill(0);
      
      energyHistory.forEach(e => {
        const binIdx = Math.min(bins - 1, Math.floor((e - min) / binWidth));
        histogram[binIdx]++;
      });
      
      const maxCount = Math.max(...histogram);
      const barWidth = width / bins;
      
      ctx.fillStyle = '#58a6ff';
      histogram.forEach((count, i) => {
        const barHeight = (count / maxCount) * height * 0.8;
        ctx.fillRect(i * barWidth, height - barHeight, barWidth - 1, barHeight);
      });
      
      // Draw mean line
      const mean = energyHistory.reduce((a, b) => a + b, 0) / energyHistory.length;
      const meanX = ((mean - min) / (max - min)) * width;
      ctx.strokeStyle = '#f85149';
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(meanX, 0);
      ctx.lineTo(meanX, height);
      ctx.stroke();
      ctx.setLineDash([]);
      
    } else {
      // Draw time series
      // Grid
      ctx.strokeStyle = '#21262d';
      ctx.lineWidth = 1;
      for (let i = 0; i < width; i += 50) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, height);
        ctx.stroke();
      }
      for (let i = 0; i < height; i += 40) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(width, i);
        ctx.stroke();
      }

      // Energy line
      const min = Math.min(...energyHistory);
      const max = Math.max(...energyHistory);
      const range = max - min || 1;
      
      ctx.strokeStyle = '#58a6ff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      energyHistory.forEach((energy, i) => {
        const x = (i / energyHistory.length) * width;
        const y = height - ((energy - min) / range) * height * 0.9 - height * 0.05;
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Moving average
      if (showMovingAvg && energyHistory.length > 10) {
        const windowSize = 20;
        ctx.strokeStyle = '#f85149';
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        for (let i = windowSize; i < energyHistory.length; i++) {
          const avg = energyHistory.slice(i - windowSize, i).reduce((a, b) => a + b, 0) / windowSize;
          const x = (i / energyHistory.length) * width;
          const y = height - ((avg - min) / range) * height * 0.9 - height * 0.05;
          
          if (i === windowSize) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // Convergence indicator
      if (energyHistory.length > 50) {
        const recent = energyHistory.slice(-50);
        const variance = recent.reduce((sum, e) => {
          const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
          return sum + Math.pow(e - mean, 2);
        }, 0) / recent.length;
        
        const converged = variance < 0.01;
        ctx.fillStyle = converged ? '#00ff99' : '#f85149';
        ctx.beginPath();
        ctx.arc(width - 15, 15, 5, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.fillStyle = '#c9d1d9';
        ctx.font = '10px monospace';
        ctx.textBaseline = 'top';
        ctx.textAlign = 'end';
        ctx.fillText(converged ? 'CONVERGED' : 'EVOLVING', width - 10, 5);
      }
    }

    // Draw stats
    ctx.fillStyle = '#c9d1d9';
    ctx.font = '11px monospace';
    ctx.textBaseline = 'top';
    ctx.textAlign = 'start';
    const current = energyHistory[energyHistory.length - 1];
    const mean = energyHistory.reduce((a, b) => a + b, 0) / energyHistory.length;
    ctx.fillText(`Current: ${current.toFixed(2)}`, 5, 5);
    ctx.fillText(`Mean: ${mean.toFixed(2)}`, 5, 17);
    ctx.fillText(`Samples: ${energyHistory.length}`, 5, 29);

  }, [energyHistory, width, height, showHistogram, showMovingAvg]);

  return <canvas ref={canvasRef} width={width} height={height} />;
};


// ============================================================================
// THRML Spin State Matrix with Correlation Highlighting
// ============================================================================

export const THRMLSpinMatrix: React.FC<VisualizerProps> = ({ 
  width, 
  height, 
  showValues = false 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [spins, setSpins] = useState<number[][]>([]);
  const [selectedSpin, setSelectedSpin] = useState<[number, number] | null>(null);

  useEffect(() => {
    const fetchSpins = async () => {
      try {
        const response = await fetch('http://localhost:8000/visualizer/thrml/spins');
        const result = await response.json();
        
        if (result.spins && result.spins.length > 0) {
          setSpins(result.spins);
        }
      } catch (err) {
        console.error('Failed to fetch spin states:', err);
      }
    };

    const interval = setInterval(fetchSpins, 100); // 10 Hz
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || spins.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rows = spins.length;
    const cols = spins[0]?.length || 0;
    const cellWidth = width / cols;
    const cellHeight = height / rows;

    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    // Draw spins
    spins.forEach((row, i) => {
      row.forEach((spin, j) => {
        const x = j * cellWidth;
        const y = i * cellHeight;
        
        // Color based on spin value
        const isSelected = selectedSpin && selectedSpin[0] === i && selectedSpin[1] === j;
        const alpha = isSelected ? 1.0 : 0.7;
        
        if (spin > 0) {
          ctx.fillStyle = `rgba(88, 166, 255, ${alpha})`;
        } else {
          ctx.fillStyle = `rgba(248, 81, 73, ${alpha})`;
        }
        
        ctx.fillRect(x + 1, y + 1, cellWidth - 2, cellHeight - 2);
        
        // Draw value if enabled
        if (showValues && cellWidth > 20) {
          ctx.fillStyle = '#c9d1d9';
          ctx.font = `${Math.min(cellWidth / 3, 10)}px monospace`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(spin.toFixed(1), x + cellWidth / 2, y + cellHeight / 2);
        }
        
        // Highlight selected
        if (isSelected) {
          ctx.strokeStyle = '#00ff99';
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, cellWidth, cellHeight);
        }
      });
    });

    // Draw grid
    ctx.strokeStyle = '#21262d';
    ctx.lineWidth = 1;
    for (let i = 0; i <= cols; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellWidth, 0);
      ctx.lineTo(i * cellWidth, height);
      ctx.stroke();
    }
    for (let i = 0; i <= rows; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * cellHeight);
      ctx.lineTo(width, i * cellHeight);
      ctx.stroke();
    }

  }, [spins, width, height, selectedSpin, showValues]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (spins.length === 0) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const cols = spins[0]?.length || 0;
    const rows = spins.length;
    const cellWidth = width / cols;
    const cellHeight = height / rows;
    
    const col = Math.floor(x / cellWidth);
    const row = Math.floor(y / cellHeight);
    
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      setSelectedSpin([row, col]);
    }
  };

  return (
    <canvas 
      ref={canvasRef} 
      width={width} 
      height={height} 
      className="cursor-pointer"
      onClick={handleClick}
    />
  );
};


// ============================================================================
// THRML Correlation Matrix
// ============================================================================

export const THRMLCorrelationMatrix: React.FC<VisualizerProps> = ({ 
  width, 
  height, 
  showColorBar = true 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [correlations, setCorrelations] = useState<number[][]>([]);

  useEffect(() => {
    const fetchCorrelations = async () => {
      try {
        const response = await fetch('http://localhost:8000/visualizer/thrml/correlations');
        const result = await response.json();
        
        if (result.correlations && result.correlations.length > 0) {
          setCorrelations(result.correlations);
        }
      } catch (err) {
        console.error('Failed to fetch correlations:', err);
      }
    };

    const interval = setInterval(fetchCorrelations, 200); // 5 Hz
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || correlations.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const n = correlations.length;
    const cellSize = Math.min(width, height) / n;
    const matrixSize = cellSize * n;

    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    // Draw correlation matrix
    correlations.forEach((row, i) => {
      row.forEach((corr, j) => {
        const x = j * cellSize;
        const y = i * cellSize;
        
        // Color based on correlation (-1 to 1)
        const normalized = (corr + 1) / 2; // Map to 0-1
        
        if (corr > 0) {
          ctx.fillStyle = `rgba(88, 166, 255, ${normalized})`;
        } else {
          ctx.fillStyle = `rgba(248, 81, 73, ${Math.abs(normalized - 0.5) * 2})`;
        }
        
        ctx.fillRect(x, y, cellSize, cellSize);
      });
    });

    // Draw grid
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 1;
    for (let i = 0; i <= n; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cellSize, 0);
      ctx.lineTo(i * cellSize, matrixSize);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(0, i * cellSize);
      ctx.lineTo(matrixSize, i * cellSize);
      ctx.stroke();
    }

    // Draw color bar
    if (showColorBar) {
      const barWidth = 20;
      const barHeight = height * 0.6;
      const barX = width - barWidth - 10;
      const barY = (height - barHeight) / 2;
      
      for (let i = 0; i < barHeight; i++) {
        const value = 1 - (i / barHeight); // 1 to -1
        const normalized = (value + 1) / 2;
        
        if (value > 0) {
          ctx.fillStyle = `rgba(88, 166, 255, ${normalized})`;
        } else {
          ctx.fillStyle = `rgba(248, 81, 73, ${Math.abs(normalized - 0.5) * 2})`;
        }
        
        ctx.fillRect(barX, barY + i, barWidth, 1);
      }
      
      ctx.strokeStyle = '#c9d1d9';
      ctx.strokeRect(barX, barY, barWidth, barHeight);
      
      ctx.fillStyle = '#c9d1d9';
      ctx.font = '10px monospace';
      ctx.fillText('+1', barX + barWidth + 5, barY + 10);
      ctx.fillText('0', barX + barWidth + 5, barY + barHeight / 2);
      ctx.fillText('-1', barX + barWidth + 5, barY + barHeight - 5);
    }

  }, [correlations, width, height, showColorBar]);

  return (
    <div className="flex flex-col h-full bg-black/60 rounded border border-[#00cc77]">
      <div className="flex items-center justify-between px-2 py-1 border-b border-[#00cc77] bg-black/60">
        <span className="text-xs font-semibold text-[#00ff99]">CORRELATION MATRIX</span>
        <button
          onClick={() => setShowColorBar(!showColorBar)}
          className="px-2 py-0.5 text-xs bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
        >
          {showColorBar ? 'Hide' : 'Show'} Scale
        </button>
      </div>
      <div className="flex-1 relative">
        <canvas ref={canvasRef} width={width} height={height} className="absolute inset-0" />
      </div>
    </div>
  );
};


export {
  THRMLEnergyGraph as EnergyGraph,
  THRMLSpinMatrix as SpinMatrix,
  THRMLCorrelationMatrix as CorrelationMatrix,
};

