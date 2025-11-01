/**
 * THRML Real-Time Diagnostics Panel
 * 
 * Displays comprehensive THRML performance metrics and diagnostics in real-time
 */

'use client';

import React, { useMemo } from 'react';
import { useTHRML, useTHRMLPerformance } from '@/lib/hooks/useTHRML';

interface THRMLDiagnosticsPanelProps {
  className?: string;
  compact?: boolean;
}

export const THRMLDiagnosticsPanel: React.FC<THRMLDiagnosticsPanelProps> = ({
  className = '',
  compact = false
}) => {
  const { health, diagnostics, latestSample, loading, error } = useTHRML();
  const { history, current } = useTHRMLPerformance();

  // Compute performance grades
  const performanceGrade = useMemo(() => {
    if (!diagnostics) return { letter: '?', color: 'text-gray-500', description: 'No data' };
    
    const { samples_per_sec, ess_per_sec } = diagnostics;
    
    if (samples_per_sec >= 500 && ess_per_sec >= 50) {
      return { letter: 'A', color: 'text-[#00ff99]', description: 'Excellent' };
    } else if (samples_per_sec >= 200 && ess_per_sec >= 20) {
      return { letter: 'B', color: 'text-cyan-400', description: 'Good' };
    } else if (samples_per_sec >= 50 && ess_per_sec >= 10) {
      return { letter: 'C', color: 'text-yellow-400', description: 'Fair' };
    } else {
      return { letter: 'D', color: 'text-red-400', description: 'Poor' };
    }
  }, [diagnostics]);

  // Mixing quality assessment
  const mixingQuality = useMemo(() => {
    if (!diagnostics) return { quality: '?', color: 'text-gray-500' };
    
    const { tau_int, lag1_autocorr } = diagnostics;
    
    if (tau_int < 10 && lag1_autocorr < 0.5) {
      return { quality: 'Excellent', color: 'text-[#00ff99]' };
    } else if (tau_int < 50 && lag1_autocorr < 0.8) {
      return { quality: 'Good', color: 'text-cyan-400' };
    } else if (tau_int < 100 && lag1_autocorr < 0.9) {
      return { quality: 'Fair', color: 'text-yellow-400' };
    } else {
      return { quality: 'Poor', color: 'text-red-400' };
    }
  }, [diagnostics]);

  // Trend analysis
  const trends = useMemo(() => {
    if (history.samples_per_sec.length < 2) {
      return { samples: '→', ess: '→', energy: '→' };
    }

    const recent = history.samples_per_sec.slice(-5);
    const older = history.samples_per_sec.slice(-10, -5);
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const olderAvg = older.reduce((a, b) => a + b, 0) / older.length || recentAvg;
    
    const samplesTrend = recentAvg > olderAvg * 1.1 ? '↑' : recentAvg < olderAvg * 0.9 ? '↓' : '→';
    
    const essRecent = history.ess_per_sec.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const essOlder = history.ess_per_sec.slice(-10, -5).reduce((a, b) => a + b, 0) / 5 || essRecent;
    const essTrend = essRecent > essOlder * 1.1 ? '↑' : essRecent < essOlder * 0.9 ? '↓' : '→';
    
    const energyRecent = history.energy.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const energyOlder = history.energy.slice(-10, -5).reduce((a, b) => a + b, 0) / 5 || energyRecent;
    const energyTrend = energyRecent > energyOlder * 1.05 ? '↑' : energyRecent < energyOlder * 0.95 ? '↓' : '→';
    
    return { samples: samplesTrend, ess: essTrend, energy: energyTrend };
  }, [history]);

  if (compact) {
    return (
      <div className={`panel p-2 md:p-3 ${className}`}>
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <div className={`text-xl font-bold ${performanceGrade.color}`}>
              {performanceGrade.letter}
            </div>
            <div className="text-[9px] md:text-xs text-cyber-cyan-muted">
              {diagnostics?.samples_per_sec.toFixed(0) ?? '---'} smp/s
            </div>
          </div>
          <div className={`w-2 h-2 rounded-full ${health?.healthy ? 'bg-[#00ff99]' : 'bg-red-500'} animate-pulse`} />
        </div>
      </div>
    );
  }

  return (
    <div className={`panel p-3 md:p-4 space-y-3 md:space-y-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-cyber-cyan-core font-bold text-sm md:text-lg">
          THRML Diagnostics
        </h3>
        <div className="flex items-center gap-2">
          <div className={`text-2xl md:text-3xl font-bold ${performanceGrade.color}`}>
            {performanceGrade.letter}
          </div>
          <div className={`w-2 md:w-3 h-2 md:h-3 rounded-full ${health?.healthy ? 'bg-[#00ff99]' : 'bg-red-500'} animate-pulse`} />
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/20 border border-red-500/50 rounded px-2 md:px-3 py-1.5 md:py-2 text-[10px] md:text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Main Metrics Grid */}
      <div className="grid grid-cols-2 gap-2 md:gap-3">
        {/* Samples/sec */}
        <div className="bg-black/50 border border-cyber-cyan-muted/30 rounded p-2 md:p-3">
          <div className="text-[9px] md:text-xs text-cyber-cyan-muted uppercase">Samples/sec {trends.samples}</div>
          <div className="text-lg md:text-2xl font-mono text-cyber-cyan-core mt-1">
            {diagnostics?.samples_per_sec.toFixed(1) ?? '---'}
          </div>
          <div className="text-[8px] md:text-[10px] text-cyber-cyan-muted/70 mt-1">
            {diagnostics && diagnostics.samples_per_sec >= 500 ? 'Excellent' : 
             diagnostics && diagnostics.samples_per_sec >= 100 ? 'Good' : 'Fair'}
          </div>
        </div>

        {/* ESS/sec */}
        <div className="bg-black/50 border border-cyber-magenta-glow/30 rounded p-2 md:p-3">
          <div className="text-[9px] md:text-xs text-cyber-magenta-glow/70 uppercase">ESS/sec {trends.ess}</div>
          <div className="text-lg md:text-2xl font-mono text-cyber-magenta-glow mt-1">
            {diagnostics?.ess_per_sec.toFixed(1) ?? '---'}
          </div>
          <div className="text-[8px] md:text-[10px] text-cyber-magenta-glow/50 mt-1">
            {diagnostics && diagnostics.ess_per_sec >= 50 ? 'Excellent' : 
             diagnostics && diagnostics.ess_per_sec >= 10 ? 'Good' : 'Fair'}
          </div>
        </div>

        {/* Energy */}
        <div className="bg-black/50 border border-cyber-cyan-muted/30 rounded p-2 md:p-3">
          <div className="text-[9px] md:text-xs text-cyber-cyan-muted uppercase">Mean Energy {trends.energy}</div>
          <div className="text-lg md:text-2xl font-mono text-cyber-cyan-core mt-1">
            {diagnostics?.mean_energy.toFixed(2) ?? '---'}
          </div>
          <div className="text-[8px] md:text-[10px] text-cyber-cyan-muted/70 mt-1">
            σ = {diagnostics?.energy_std.toFixed(2) ?? '---'}
          </div>
        </div>

        {/* Mixing Quality */}
        <div className="bg-black/50 border border-cyber-magenta-glow/30 rounded p-2 md:p-3">
          <div className="text-[9px] md:text-xs text-cyber-magenta-glow/70 uppercase">Mixing</div>
          <div className={`text-lg md:text-2xl font-bold mt-1 ${mixingQuality.color}`}>
            {mixingQuality.quality}
          </div>
          <div className="text-[8px] md:text-[10px] text-cyber-magenta-glow/50 mt-1">
            τ_int = {diagnostics?.tau_int.toFixed(1) ?? '---'}
          </div>
        </div>
      </div>

      {/* Detailed Metrics */}
      <div className="space-y-1.5 md:space-y-2 text-[10px] md:text-xs">
        <div className="flex justify-between gap-2">
          <span className="text-cyber-cyan-muted">Total Samples:</span>
          <span className="font-mono text-cyber-cyan-core">
            {diagnostics?.total_samples.toLocaleString() ?? '---'}
          </span>
        </div>
        
        <div className="flex justify-between gap-2">
          <span className="text-cyber-cyan-muted">Lag-1 Autocorr:</span>
          <span className="font-mono text-cyber-cyan-core">
            {diagnostics?.lag1_autocorr.toFixed(3) ?? '---'}
          </span>
        </div>
        
        <div className="flex justify-between gap-2">
          <span className="text-cyber-cyan-muted">Blocking Strategy:</span>
          <span className="font-mono text-cyber-cyan-core capitalize">
            {diagnostics?.blocking_strategy ?? '---'}
          </span>
        </div>
        
        <div className="flex justify-between gap-2">
          <span className="text-cyber-cyan-muted">Device:</span>
          <span className="font-mono text-cyber-cyan-core uppercase">
            {diagnostics?.device_type ?? '---'}
          </span>
        </div>
        
        <div className="flex justify-between gap-2">
          <span className="text-cyber-cyan-muted">Chains:</span>
          <span className="font-mono text-cyber-cyan-core">
            {diagnostics?.n_chains ?? '---'}
          </span>
        </div>
      </div>

      {/* Health Status */}
      {health && (
        <div className="pt-2 md:pt-3 border-t border-cyber-cyan-muted/20">
          <div className="text-[10px] md:text-xs text-cyber-cyan-muted uppercase mb-2">
            System Health
          </div>
          <div className="space-y-1 text-[9px] md:text-xs">
            <div className="flex justify-between gap-2">
              <span>Nodes:</span>
              <span className="font-mono">{health.n_nodes}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span>Edges:</span>
              <span className="font-mono">{health.n_edges}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span>Errors:</span>
              <span className={`font-mono ${health.error_count > 0 ? 'text-red-400' : 'text-[#00ff99]'}`}>
                {health.error_count}
              </span>
            </div>
            {health.last_error && (
              <div className="mt-2 p-2 bg-red-900/20 border border-red-500/50 rounded">
                <div className="text-[9px] text-red-400">{health.last_error}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Mini Sparklines (if history available) */}
      {history.samples_per_sec.length > 2 && (
        <div className="pt-2 md:pt-3 border-t border-cyber-cyan-muted/20">
          <div className="text-[10px] md:text-xs text-cyber-cyan-muted uppercase mb-2">
            Performance History
          </div>
          <div className="h-12 md:h-16 bg-black/50 rounded border border-cyber-cyan-muted/20 p-1 md:p-2">
            <svg width="100%" height="100%" viewBox="0 0 100 40" preserveAspectRatio="none">
              {/* Samples/sec line */}
              <polyline
                points={history.samples_per_sec.map((val, idx) => {
                  const x = (idx / (history.samples_per_sec.length - 1)) * 100;
                  const max = Math.max(...history.samples_per_sec) || 1;
                  const y = 40 - (val / max) * 35;
                  return `${x},${y}`;
                }).join(' ')}
                fill="none"
                stroke="currentColor"
                strokeWidth="0.5"
                className="text-cyber-cyan-core"
              />
              {/* ESS/sec line */}
              <polyline
                points={history.ess_per_sec.map((val, idx) => {
                  const x = (idx / (history.ess_per_sec.length - 1)) * 100;
                  const max = Math.max(...history.ess_per_sec) || 1;
                  const y = 40 - (val / max) * 35;
                  return `${x},${y}`;
                }).join(' ')}
                fill="none"
                stroke="currentColor"
                strokeWidth="0.5"
                className="text-cyber-magenta-glow opacity-70"
              />
            </svg>
          </div>
          <div className="flex items-center justify-between mt-1 text-[8px] md:text-[10px]">
            <span className="text-cyber-cyan-core">Samples/sec</span>
            <span className="text-cyber-magenta-glow">ESS/sec</span>
          </div>
        </div>
      )}

      {/* Loading Indicator */}
      {loading && (
        <div className="flex items-center gap-2 text-[10px] md:text-xs text-cyber-cyan-muted">
          <div className="w-2 h-2 bg-cyber-cyan-core rounded-full animate-pulse" />
          <span>Updating...</span>
        </div>
      )}
    </div>
  );
};

export default THRMLDiagnosticsPanel;

