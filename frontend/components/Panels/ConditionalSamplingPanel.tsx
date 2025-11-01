/**
 * Conditional Sampling Panel (Inpainting UI)
 * 
 * Interactive UI for clamping nodes and performing conditional THRML sampling
 */

'use client';

import React, { useState, useCallback, useMemo } from 'react';
import { useConditionalSampling } from '@/lib/hooks/useTHRML';

interface ConditionalSamplingPanelProps {
  className?: string;
  gridSize?: number;
}

type BrushMode = 'positive' | 'negative' | 'erase';

export const ConditionalSamplingPanel: React.FC<ConditionalSamplingPanelProps> = ({
  className = '',
  gridSize = 8
}) => {
  const {
    clampedNodes,
    addClampedNode,
    removeClampedNode,
    applyMask,
    clearMask,
    sampleConditional
  } = useConditionalSampling();

  const [brushMode, setBrushMode] = useState<BrushMode>('positive');
  const [brushSize, setBrushSize] = useState(1);
  const [nSteps, setNSteps] = useState(100);
  const [sampling, setSampling] = useState(false);
  const [result, setResult] = useState<number[] | null>(null);

  // Convert node ID to grid coords
  const nodeToCoords = useCallback((nodeId: number) => {
    const row = Math.floor(nodeId / gridSize);
    const col = nodeId % gridSize;
    return { row, col };
  }, [gridSize]);

  // Convert grid coords to node ID
  const coordsToNode = useCallback((row: number, col: number) => {
    if (row < 0 || row >= gridSize || col < 0 || col >= gridSize) return -1;
    return row * gridSize + col;
  }, [gridSize]);

  // Handle cell click
  const handleCellClick = useCallback((nodeId: number) => {
    if (nodeId < 0) return;

    if (brushMode === 'erase') {
      removeClampedNode(nodeId);
    } else {
      const value = brushMode === 'positive' ? 1 : -1;
      addClampedNode(nodeId, value);
      
      // Apply brush size
      if (brushSize > 1) {
        const { row, col } = nodeToCoords(nodeId);
        const radius = Math.floor(brushSize / 2);
        
        for (let dr = -radius; dr <= radius; dr++) {
          for (let dc = -radius; dc <= radius; dc++) {
            const neighborId = coordsToNode(row + dr, col + dc);
            if (neighborId >= 0) {
              addClampedNode(neighborId, value);
            }
          }
        }
      }
    }
  }, [brushMode, brushSize, nodeToCoords, coordsToNode, addClampedNode, removeClampedNode]);

  // Handle sampling
  const handleSample = useCallback(async () => {
    setSampling(true);
    try {
      const result = await sampleConditional(nSteps);
      if (result && result.states) {
        setResult(result.states);
      }
    } catch (err) {
      console.error('Sampling error:', err);
    } finally {
      setSampling(false);
    }
  }, [sampleConditional, nSteps]);

  // Render grid
  const gridCells = useMemo(() => {
    const cells = [];
    const totalNodes = gridSize * gridSize;
    
    for (let i = 0; i < totalNodes; i++) {
      const isClamped = clampedNodes.has(i);
      const clampedValue = isClamped ? clampedNodes.get(i) : null;
      const resultValue = result && result[i] !== undefined ? result[i] : null;
      
      let bgColor = 'bg-black/50';
      let borderColor = 'border-cyber-cyan-muted/30';
      
      if (isClamped) {
        if (clampedValue === 1) {
          bgColor = 'bg-[#00ff99]/30';
          borderColor = 'border-[#00ff99]';
        } else if (clampedValue === -1) {
          bgColor = 'bg-cyber-magenta-glow/30';
          borderColor = 'border-cyber-magenta-glow';
        }
      } else if (resultValue !== null) {
        if (resultValue > 0) {
          bgColor = 'bg-cyan-500/20';
        } else {
          bgColor = 'bg-magenta-500/20';
        }
      }
      
      cells.push(
        <div
          key={i}
          onClick={() => handleCellClick(i)}
          className={`
            ${bgColor} ${borderColor} 
            border cursor-pointer transition-all duration-150
            hover:border-cyber-cyan-core hover:shadow-[0_0_4px_currentColor]
            aspect-square
          `}
          title={`Node ${i}${isClamped ? ` (clamped: ${clampedValue})` : ''}`}
        />
      );
    }
    
    return cells;
  }, [gridSize, clampedNodes, result, handleCellClick]);

  return (
    <div className={`panel p-3 md:p-4 space-y-3 md:space-y-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-cyber-cyan-core font-bold text-sm md:text-lg">
          Conditional Sampling
        </h3>
        <div className="text-[10px] md:text-xs text-cyber-cyan-muted">
          {clampedNodes.size} nodes clamped
        </div>
      </div>

      {/* Instructions */}
      <div className="text-[10px] md:text-xs text-cyber-cyan-muted/70 bg-black/50 rounded p-2 md:p-3 border border-cyber-cyan-muted/20">
        <div className="font-semibold text-cyber-cyan-core mb-1">How to use:</div>
        <ol className="list-decimal list-inside space-y-0.5">
          <li>Select brush mode and size below</li>
          <li>Click cells to clamp them</li>
          <li>Click &quot;Sample&quot; to generate</li>
        </ol>
      </div>

      {/* Brush Controls */}
      <div className="space-y-2">
        <label className="text-[10px] md:text-xs text-cyber-cyan-muted uppercase tracking-wide">
          Brush Mode
        </label>
        <div className="grid grid-cols-3 gap-1.5 md:gap-2">
          <button
            onClick={() => setBrushMode('positive')}
            className={`
              px-2 md:px-3 py-1.5 md:py-2 rounded border transition-all duration-200
              ${brushMode === 'positive'
                ? 'bg-[#00ff99]/10 border-[#00ff99] text-[#00ff99] shadow-[0_0_8px_currentColor]'
                : 'bg-black/30 border-cyber-cyan-muted/30 text-cyber-cyan-muted hover:border-[#00ff99]/50'
              }
              text-[10px] md:text-sm font-medium
            `}
          >
            <div className="flex flex-col items-center gap-0.5 md:gap-1">
              <span className="text-lg md:text-xl">+</span>
              <span className="text-[9px] md:text-xs">Positive</span>
            </div>
          </button>

          <button
            onClick={() => setBrushMode('negative')}
            className={`
              px-2 md:px-3 py-1.5 md:py-2 rounded border transition-all duration-200
              ${brushMode === 'negative'
                ? 'bg-cyber-magenta-glow/10 border-cyber-magenta-glow text-cyber-magenta-glow shadow-[0_0_8px_currentColor]'
                : 'bg-black/30 border-cyber-cyan-muted/30 text-cyber-cyan-muted hover:border-cyber-magenta-glow/50'
              }
              text-[10px] md:text-sm font-medium
            `}
          >
            <div className="flex flex-col items-center gap-0.5 md:gap-1">
              <span className="text-lg md:text-xl">−</span>
              <span className="text-[9px] md:text-xs">Negative</span>
            </div>
          </button>

          <button
            onClick={() => setBrushMode('erase')}
            className={`
              px-2 md:px-3 py-1.5 md:py-2 rounded border transition-all duration-200
              ${brushMode === 'erase'
                ? 'bg-red-500/10 border-red-500 text-red-500 shadow-[0_0_8px_currentColor]'
                : 'bg-black/30 border-cyber-cyan-muted/30 text-cyber-cyan-muted hover:border-red-500/50'
              }
              text-[10px] md:text-sm font-medium
            `}
          >
            <div className="flex flex-col items-center gap-0.5 md:gap-1">
              <span className="text-lg md:text-xl">×</span>
              <span className="text-[9px] md:text-xs">Erase</span>
            </div>
          </button>
        </div>
      </div>

      {/* Brush Size */}
      <div className="space-y-2">
        <label className="text-[10px] md:text-xs text-cyber-cyan-muted uppercase tracking-wide flex items-center justify-between">
          <span>Brush Size</span>
          <span className="text-cyber-cyan-core font-mono">{brushSize}×{brushSize}</span>
        </label>
        <input
          type="range"
          min="1"
          max="3"
          step="1"
          value={brushSize}
          onChange={(e) => setBrushSize(parseInt(e.target.value))}
          className="w-full h-2 bg-black/50 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* Sampling Steps */}
      <div className="space-y-2">
        <label className="text-[10px] md:text-xs text-cyber-cyan-muted uppercase tracking-wide flex items-center justify-between">
          <span>Gibbs Steps</span>
          <span className="text-cyber-cyan-core font-mono">{nSteps}</span>
        </label>
        <input
          type="range"
          min="10"
          max="500"
          step="10"
          value={nSteps}
          onChange={(e) => setNSteps(parseInt(e.target.value))}
          className="w-full h-2 bg-black/50 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* Grid */}
      <div className="space-y-2">
        <label className="text-[10px] md:text-xs text-cyber-cyan-muted uppercase tracking-wide">
          P-bit Grid ({gridSize}×{gridSize})
        </label>
        <div 
          className="grid gap-0.5 md:gap-1 bg-black/70 p-1 md:p-2 rounded border border-cyber-cyan-muted/30"
          style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}
        >
          {gridCells}
        </div>
        <div className="flex items-center justify-between text-[9px] md:text-xs text-cyber-cyan-muted/70">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-[#00ff99]/30 border border-[#00ff99] rounded" />
            <span>Positive (+1)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-cyber-magenta-glow/30 border border-cyber-magenta-glow rounded" />
            <span>Negative (-1)</span>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-2">
        <button
          onClick={handleSample}
          disabled={sampling || clampedNodes.size === 0}
          className={`
            flex-1 px-3 md:px-4 py-2 md:py-3 rounded font-semibold text-xs md:text-sm transition-all duration-200
            ${sampling || clampedNodes.size === 0
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : 'bg-cyber-cyan-core text-black hover:shadow-[0_0_12px_currentColor] hover:scale-[1.02]'
            }
          `}
        >
          {sampling ? 'Sampling...' : 'Sample'}
        </button>

        <button
          onClick={clearMask}
          disabled={clampedNodes.size === 0}
          className={`
            px-3 md:px-4 py-2 md:py-3 rounded font-semibold text-xs md:text-sm transition-all duration-200
            ${clampedNodes.size === 0
              ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
              : 'bg-red-500/20 border border-red-500/50 text-red-400 hover:bg-red-500/30'
            }
          `}
        >
          Clear
        </button>
      </div>

      {/* Result Display */}
      {result && (
        <div className="pt-2 md:pt-3 border-t border-cyber-cyan-muted/20">
          <div className="text-[10px] md:text-xs text-cyber-cyan-muted uppercase mb-2">
            Result
          </div>
          <div className="bg-black/50 rounded p-2 md:p-3 border border-[#00ff99]/30">
            <div className="text-[10px] md:text-xs text-[#00ff99] font-mono">
              Generated {result.length} states
            </div>
            <div className="text-[9px] md:text-[10px] text-cyber-cyan-muted/70 mt-1">
              Green cells show sampled values
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConditionalSamplingPanel;

