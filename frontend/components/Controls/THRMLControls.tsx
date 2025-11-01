'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useSimulationStore } from '@/lib/stores/simulation';

interface THRMLControlsProps {
  className?: string;
}

type PerformanceMode = 'speed' | 'accuracy' | 'research';

interface PerformanceModeConfig {
  gibbs_steps: number;
  temperature: number;
  learning_rate: number;
  cd_k_steps: number;
  weight_update_freq: number;
  use_jit: boolean;
  description: string;
}

const MODE_ICONS = {
  speed: 'SPD',
  accuracy: 'ACC',
  research: 'RES'
};

const MODE_DESCRIPTIONS = {
  speed: 'Optimized for real-time visualization',
  accuracy: 'Balanced for production applications',
  research: 'High-quality for scientific experiments'
};

export const THRMLControls: React.FC<THRMLControlsProps> = ({ className = '' }) => {
  const { thrml, setThrmlMode, setThrmlTemperature, updateThrmlEnergy } = useSimulationStore();
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modeConfig, setModeConfig] = useState<PerformanceModeConfig | null>(null);
  const [tempInput, setTempInput] = useState(thrml.temperature.toString());

  // Fetch current mode configuration
  useEffect(() => {
    fetchModeConfig();
  }, [thrml.performanceMode]);

  // Update temp input when store changes
  useEffect(() => {
    setTempInput(thrml.temperature.toFixed(2));
  }, [thrml.temperature]);

  const fetchModeConfig = async () => {
    try {
      const response = await fetch('http://localhost:8000/thrml/performance-mode');
      const data = await response.json();
      setModeConfig(data.config);
    } catch (err) {
      console.error('Failed to fetch THRML config:', err);
    }
  };

  const fetchEnergy = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/thrml/energy');
      const data = await response.json();
      updateThrmlEnergy(data.energy);
    } catch (err) {
      // Silently fail if no nodes active
      console.debug('Energy fetch failed:', err);
    }
  }, [updateThrmlEnergy]);

  // Fetch energy periodically
  useEffect(() => {
    const interval = setInterval(fetchEnergy, 2000);
    return () => clearInterval(interval);
  }, [fetchEnergy]);

  const handleModeChange = async (mode: PerformanceMode) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/thrml/performance-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      });
      
      if (!response.ok) {
        throw new Error('Failed to update performance mode');
      }
      
      const data = await response.json();
      setThrmlMode(mode);
      setModeConfig(data.config);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleTemperatureChange = async (temp: number) => {
    try {
      const response = await fetch('http://localhost:8000/thrml/temperature', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ temperature: temp })
      });
      
      if (!response.ok) {
        throw new Error('Failed to update temperature');
      }
      
      setThrmlTemperature(temp);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  const handleTempInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTempInput(e.target.value);
  };

  const handleTempInputBlur = () => {
    const temp = parseFloat(tempInput);
    if (!isNaN(temp) && temp > 0 && temp <= 10) {
      handleTemperatureChange(temp);
    } else {
      setTempInput(thrml.temperature.toFixed(2));
    }
  };

  const handleTempInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleTempInputBlur();
    }
  };

  return (
    <div className={`panel p-3 md:p-4 space-y-3 md:space-y-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h3 className="text-cyber-cyan-core font-bold text-sm md:text-lg">THRML Performance</h3>
        {thrml.enabled && (
          <div className="text-[10px] md:text-xs text-cyber-cyan-muted whitespace-nowrap">
            {thrml.gibbsSteps} steps · CD-{thrml.cdK}
          </div>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/20 border border-red-500/50 rounded px-2 md:px-3 py-1.5 md:py-2 text-[10px] md:text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Performance Mode Tabs */}
      <div className="space-y-2">
        <label className="text-[10px] md:text-xs text-cyber-cyan-muted uppercase tracking-wide">
          Performance Mode
        </label>
        <div className="grid grid-cols-3 gap-1.5 md:gap-2">
          {(['speed', 'accuracy', 'research'] as PerformanceMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => handleModeChange(mode)}
              disabled={loading}
              className={`
                relative px-2 md:px-3 py-1.5 md:py-2 rounded border transition-all duration-200
                ${thrml.performanceMode === mode
                  ? 'bg-cyber-cyan-core/10 border-cyber-cyan-core text-cyber-cyan-core shadow-[0_0_8px_currentColor]'
                  : 'bg-black/30 border-cyber-cyan-muted/30 text-cyber-cyan-muted hover:border-cyber-cyan-core/50'
                }
                ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                text-[10px] md:text-sm font-medium
              `}
            >
              <div className="flex flex-col items-center gap-0.5 md:gap-1">
                <span className="text-xs md:text-sm font-bold">{MODE_ICONS[mode]}</span>
                <span className="capitalize text-[9px] md:text-xs">{mode}</span>
              </div>
            </button>
          ))}
        </div>
        <p className="text-[9px] md:text-xs text-cyber-cyan-muted/70 italic">
          {MODE_DESCRIPTIONS[thrml.performanceMode]}
        </p>
      </div>

      {/* Temperature Control */}
      <div className="space-y-2">
        <label className="text-[10px] md:text-xs text-cyber-cyan-muted uppercase tracking-wide flex items-center justify-between">
          <span>Temperature</span>
          <span className="text-cyber-magenta-glow font-mono text-xs md:text-sm">{thrml.temperature.toFixed(2)}</span>
        </label>
        
        <input
          type="range"
          min="0.1"
          max="5.0"
          step="0.1"
          value={thrml.temperature}
          onChange={(e) => handleTemperatureChange(parseFloat(e.target.value))}
          className="w-full h-2 bg-black/50 rounded-lg appearance-none cursor-pointer
                     [&::-webkit-slider-thumb]:appearance-none
                     [&::-webkit-slider-thumb]:w-3
                     [&::-webkit-slider-thumb]:h-3
                     [&::-webkit-slider-thumb]:rounded-full
                     [&::-webkit-slider-thumb]:bg-cyber-magenta-glow
                     [&::-webkit-slider-thumb]:shadow-[0_0_6px_currentColor]
                     [&::-webkit-slider-thumb]:cursor-pointer
                     [&::-moz-range-thumb]:w-3
                     [&::-moz-range-thumb]:h-3
                     [&::-moz-range-thumb]:rounded-full
                     [&::-moz-range-thumb]:bg-cyber-magenta-glow
                     [&::-moz-range-thumb]:border-0
                     [&::-moz-range-thumb]:cursor-pointer"
        />
        
        <div className="flex items-center gap-2">
          <input
            type="number"
            value={tempInput}
            onChange={handleTempInputChange}
            onBlur={handleTempInputBlur}
            onKeyDown={handleTempInputKeyDown}
            min="0.1"
            max="10.0"
            step="0.1"
            className="flex-1 bg-black/50 border border-cyber-cyan-muted/30 rounded px-2 py-1 text-xs md:text-sm
                       text-cyber-cyan-core font-mono focus:outline-none focus:border-cyber-cyan-core"
          />
          <span className="text-[9px] md:text-xs text-cyber-cyan-muted whitespace-nowrap">
            {thrml.temperature < 1.0 ? 'Cold (Deterministic)' : thrml.temperature > 2.0 ? 'Hot (Stochastic)' : 'Balanced'}
          </span>
        </div>
      </div>

      {/* Energy Display */}
      <div className="space-y-2">
        <label className="text-[10px] md:text-xs text-cyber-cyan-muted uppercase tracking-wide">
          System Energy
        </label>
        <div className="bg-black/50 border border-cyber-magenta-glow/30 rounded px-2 md:px-3 py-1.5 md:py-2">
          <div className="text-lg md:text-2xl font-mono text-cyber-magenta-glow">
            {thrml.energy !== null ? thrml.energy.toFixed(3) : '---'}
          </div>
          <div className="text-[9px] md:text-xs text-cyber-cyan-muted mt-1">
            Lower energy = more probable states
          </div>
        </div>
      </div>

      {/* Configuration Details (Collapsible) */}
      {modeConfig && (
        <details className="text-[10px] md:text-xs text-cyber-cyan-muted/70">
          <summary className="cursor-pointer hover:text-cyber-cyan-core transition-colors">
            Advanced Configuration
          </summary>
          <div className="mt-2 space-y-1 pl-3 md:pl-4 border-l border-cyber-cyan-muted/20">
            <div className="flex justify-between gap-2">
              <span>Gibbs Steps:</span>
              <span className="font-mono">{modeConfig.gibbs_steps}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span>Learning Rate:</span>
              <span className="font-mono">{modeConfig.learning_rate}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span>CD-k Steps:</span>
              <span className="font-mono">{modeConfig.cd_k_steps}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span>Update Frequency:</span>
              <span className="font-mono">{modeConfig.weight_update_freq} steps</span>
            </div>
            <div className="flex justify-between gap-2">
              <span>JIT Compilation:</span>
              <span className="font-mono">{modeConfig.use_jit ? 'Enabled' : 'Disabled'}</span>
            </div>
          </div>
        </details>
      )}

      {/* Status Indicator */}
      <div className="flex items-center gap-2 text-[10px] md:text-xs">
        <div className={`w-2 h-2 rounded-full ${thrml.enabled ? 'bg-[#00ff99] animate-pulse' : 'bg-gray-500'}`} />
        <span className="text-cyber-cyan-muted">
          {thrml.enabled ? 'THRML Active' : 'THRML Inactive'}
        </span>
      </div>
    </div>
  );
};

