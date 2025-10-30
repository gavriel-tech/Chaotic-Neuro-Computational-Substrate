'use client';

import React, { useEffect, useState } from 'react';
import { useSimulationStore } from '@/lib/stores/simulation';

interface EnergyHistory {
  timestamp: number;
  energy: number;
}

export const THRMLVisualizer: React.FC = () => {
  const { thrml } = useSimulationStore();
  const [energyHistory, setEnergyHistory] = useState<EnergyHistory[]>([]);
  const [correlations, setCorrelations] = useState<number[][]>([]);

  useEffect(() => {
    if (thrml.energy !== null) {
      setEnergyHistory(prev => {
        const newHistory = [...prev, { timestamp: Date.now(), energy: thrml.energy! }];
        return newHistory.slice(-50); // Keep last 50 points
      });
    }
  }, [thrml.energy]);

  const maxEnergy = Math.max(...energyHistory.map(h => h.energy), 0);
  const minEnergy = Math.min(...energyHistory.map(h => h.energy), 0);
  const range = maxEnergy - minEnergy || 1;

  return (
    <div className="flex flex-col h-full bg-black/60">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[#00cc77] bg-black/60">
        <h2 className="text-sm font-semibold text-[#00ff99] uppercase tracking-wide">THRML Visualizer</h2>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
        {/* Energy Graph */}
        <div className="bg-black/60 border border-[#00cc77] rounded p-3">
          <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-2">
            Energy Landscape
          </h3>
          <div className="relative h-32 bg-black/60 rounded border border-[#00cc77]">
            <svg className="w-full h-full">
              {energyHistory.length > 1 && (
                <polyline
                  points={energyHistory
                    .map((h, i) => {
                      const x = (i / (energyHistory.length - 1)) * 100;
                      const y = 100 - ((h.energy - minEnergy) / range) * 90;
                      return `${x}%,${y}%`;
                    })
                    .join(' ')}
                  fill="none"
                  stroke="#f85149"
                  strokeWidth="2"
                  vectorEffect="non-scaling-stroke"
                />
              )}
            </svg>
            <div className="absolute bottom-2 right-2 text-[10px] text-[#00cc77] font-mono">
              {thrml.energy !== null ? thrml.energy.toFixed(3) : '---'}
            </div>
          </div>
        </div>

        {/* Sampling Stats */}
        <div className="bg-black/60 border border-[#00cc77] rounded p-3">
          <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-2">
            Sampling Statistics
          </h3>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-black/60 rounded p-2 border border-[#00cc77]">
              <div className="text-[10px] text-[#00cc77] mb-1">Gibbs Steps</div>
              <div className="text-sm text-[#00ff99] font-mono">{thrml.gibbsSteps}</div>
            </div>
            <div className="bg-black/60 rounded p-2 border border-[#00cc77]">
              <div className="text-[10px] text-[#00cc77] mb-1">CD-k</div>
              <div className="text-sm text-[#00ff99] font-mono">{thrml.cdK}</div>
            </div>
            <div className="bg-black/60 rounded p-2 border border-[#00cc77]">
              <div className="text-[10px] text-[#00cc77] mb-1">Temperature</div>
              <div className="text-sm text-[#f85149] font-mono">{thrml.temperature.toFixed(2)}</div>
            </div>
            <div className="bg-black/60 rounded p-2 border border-[#00cc77]">
              <div className="text-[10px] text-[#00cc77] mb-1">Mode</div>
              <div className="text-sm text-[#00ff99] font-mono uppercase">{thrml.performanceMode}</div>
            </div>
          </div>
        </div>

        {/* State Visualization */}
        <div className="bg-black/60 border border-[#00cc77] rounded p-3">
          <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-2">
            Spin States
          </h3>
          <div className="grid grid-cols-8 gap-1">
            {Array.from({ length: 64 }).map((_, i) => (
              <div
                key={i}
                className="aspect-square rounded"
                style={{
                  backgroundColor: Math.random() > 0.5 ? '#58a6ff' : '#f85149',
                  opacity: 0.3 + Math.random() * 0.7,
                }}
              />
            ))}
          </div>
        </div>

        {/* Correlation Matrix */}
        <div className="bg-black/60 border border-[#00cc77] rounded p-3">
          <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-2">
            Correlation Matrix
          </h3>
          <div className="grid grid-cols-8 gap-0.5">
            {Array.from({ length: 64 }).map((_, i) => {
              const val = Math.random();
              return (
                <div
                  key={i}
                  className="aspect-square"
                  style={{
                    backgroundColor: val > 0.5 ? `rgba(88, 166, 255, ${val})` : `rgba(248, 81, 73, ${1 - val})`,
                  }}
                />
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

