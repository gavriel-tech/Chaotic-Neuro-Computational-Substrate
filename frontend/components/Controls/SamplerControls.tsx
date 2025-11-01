'use client';

import React, { useState, useEffect } from 'react';

interface Backend {
  name: string;
  available: boolean;
  capabilities: Record<string, any>;
}

interface ChainConfig {
  current_chains: number;
  auto_detected_optimal: number;
  has_gpu: boolean;
  enabled: boolean;
}

interface ClampedNodes {
  clamped_count: number;
  node_ids: number[];
  values: number[];
}

export const SamplerControls: React.FC = () => {
  // Backend selection
  const [backends, setBackends] = useState<Backend[]>([]);
  const [currentBackend, setCurrentBackend] = useState<string>('thrml');
  
  // Multi-chain configuration
  const [chainConfig, setChainConfig] = useState<ChainConfig | null>(null);
  const [chainSliderValue, setChainSliderValue] = useState<number>(-1);
  
  // Blocking strategy
  const [currentStrategy, setCurrentStrategy] = useState<string>('checkerboard');
  const [availableStrategies, setAvailableStrategies] = useState<string[]>([]);
  
  // Conditional sampling (clamping)
  const [clampedNodes, setClampedNodes] = useState<ClampedNodes | null>(null);
  const [clampNodeId, setClampNodeId] = useState<string>('');
  const [clampValue, setClampValue] = useState<string>('1.0');
  
  // UI state
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('');

  // Fetch available backends
  useEffect(() => {
    const fetchBackends = async () => {
      try {
        const response = await fetch('http://localhost:8000/sampler/backends');
        const data = await response.json();
        setBackends(data.backends || []);
      } catch (err) {
        console.error('Failed to fetch backends:', err);
      }
    };
    fetchBackends();
  }, []);

  // Fetch chain configuration
  useEffect(() => {
    const fetchChainConfig = async () => {
      try {
        const response = await fetch('http://localhost:8000/sampler/chains');
        const data = await response.json();
        setChainConfig(data);
        setChainSliderValue(data.current_chains);
      } catch (err) {
        console.error('Failed to fetch chain config:', err);
      }
    };
    fetchChainConfig();
    const interval = setInterval(fetchChainConfig, 2000);
    return () => clearInterval(interval);
  }, []);

  // Fetch blocking strategy
  useEffect(() => {
    const fetchStrategy = async () => {
      try {
        const response = await fetch('http://localhost:8000/thrml/blocking-strategy');
        const data = await response.json();
        setCurrentStrategy(data.current);
        setAvailableStrategies(data.available || []);
      } catch (err) {
        console.error('Failed to fetch blocking strategy:', err);
      }
    };
    fetchStrategy();
  }, []);

  // Fetch clamped nodes
  useEffect(() => {
    const fetchClamped = async () => {
      try {
        const response = await fetch('http://localhost:8000/sampler/clamp');
        const data = await response.json();
        setClampedNodes(data);
      } catch (err) {
        console.error('Failed to fetch clamped nodes:', err);
      }
    };
    fetchClamped();
    const interval = setInterval(fetchClamped, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleBackendChange = async (backendType: string) => {
    try {
      const response = await fetch('http://localhost:8000/sampler/backend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend_type: backendType })
      });
      const data = await response.json();
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
      } else {
        setCurrentBackend(backendType);
        setStatusMessage(data.message);
      }
    } catch (err) {
      setStatusMessage('Failed to switch backend');
    }
  };

  const handleChainChange = async (numChains: number) => {
    setChainSliderValue(numChains);
    try {
      const response = await fetch(`http://localhost:8000/sampler/chains?num_chains=${numChains}`, {
        method: 'POST'
      });
      const data = await response.json();
      setStatusMessage(data.message);
    } catch (err) {
      setStatusMessage('Failed to set chain count');
    }
  };

  const handleStrategyChange = async (strategy: string) => {
    try {
      const response = await fetch(`http://localhost:8000/thrml/blocking-strategy?strategy_name=${strategy}`, {
        method: 'POST'
      });
      const data = await response.json();
      if (data.error) {
        setStatusMessage(`Error: ${data.error}`);
      } else {
        setCurrentStrategy(strategy);
        setStatusMessage(data.message);
      }
    } catch (err) {
      setStatusMessage('Failed to set blocking strategy');
    }
  };

  const handleClampNode = async () => {
    const nodeId = parseInt(clampNodeId);
    const value = parseFloat(clampValue);
    
    if (isNaN(nodeId) || isNaN(value)) {
      setStatusMessage('Invalid node ID or value');
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/sampler/clamp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_ids: [nodeId], values: [value] })
      });
      const data = await response.json();
      setStatusMessage(data.message);
      setClampNodeId('');
      setClampValue('1.0');
    } catch (err) {
      setStatusMessage('Failed to clamp node');
    }
  };

  const handleClearClamped = async () => {
    try {
      const response = await fetch('http://localhost:8000/sampler/clamp', {
        method: 'DELETE'
      });
      const data = await response.json();
      setStatusMessage(data.message);
    } catch (err) {
      setStatusMessage('Failed to clear clamped nodes');
    }
  };

  const getChainLabel = (value: number) => {
    if (value === -1) return 'Auto';
    if (value === 1) return 'Single';
    return `${value} chains`;
  };

  return (
    <div className="bg-[#1a1a1a] border border-[#00cc77]/40 rounded p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-[#00cc77] font-bold text-sm">SAMPLER CONTROLS</h3>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs text-[#00cc77]/60 hover:text-[#00cc77] transition-colors"
        >
          {showAdvanced ? '▼ Hide Advanced' : '▶ Show Advanced'}
        </button>
      </div>

      {/* Status Message */}
      {statusMessage && (
        <div className="text-xs text-[#00cc77]/80 bg-[#00cc77]/10 p-2 rounded">
          {statusMessage}
        </div>
      )}

      {/* Backend Selector */}
      <div className="space-y-2">
        <label className="text-xs text-[#00cc77]/80 uppercase tracking-wide">Backend</label>
        <select
          value={currentBackend}
          onChange={(e) => handleBackendChange(e.target.value)}
          className="w-full bg-[#0a0a0a] border border-[#00cc77]/30 text-[#00cc77] px-3 py-2 rounded text-sm focus:outline-none focus:border-[#00cc77]"
        >
          {backends.map((backend) => (
            <option key={backend.name} value={backend.name} disabled={!backend.available}>
              {backend.name} {!backend.available && '(unavailable)'}
            </option>
          ))}
        </select>
      </div>

      {/* Multi-Chain Slider */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <label className="text-xs text-[#00cc77]/80 uppercase tracking-wide">Parallel Chains</label>
          <span className="text-xs text-[#00cc77] font-mono">
            {getChainLabel(chainSliderValue)}
            {chainConfig?.has_gpu && ' (GPU)'}
          </span>
        </div>
        <input
          type="range"
          min="-1"
          max="16"
          step="1"
          value={chainSliderValue}
          onChange={(e) => handleChainChange(parseInt(e.target.value))}
          className="w-full accent-[#00cc77]"
        />
        <div className="flex justify-between text-xs text-[#00cc77]/50">
          <span>Auto</span>
          <span>1</span>
          <span>4</span>
          <span>8</span>
          <span>16</span>
        </div>
        {chainConfig && chainConfig.auto_detected_optimal > 1 && (
          <div className="text-xs text-[#00cc77]/60">
            Recommended: {chainConfig.auto_detected_optimal} chains
          </div>
        )}
      </div>

      {/* Advanced Controls */}
      {showAdvanced && (
        <>
          {/* Blocking Strategy Dropdown */}
          <div className="space-y-2 pt-2 border-t border-[#00cc77]/20">
            <label className="text-xs text-[#00cc77]/80 uppercase tracking-wide">Blocking Strategy</label>
            <select
              value={currentStrategy}
              onChange={(e) => handleStrategyChange(e.target.value)}
              className="w-full bg-[#0a0a0a] border border-[#00cc77]/30 text-[#00cc77] px-3 py-2 rounded text-sm focus:outline-none focus:border-[#00cc77]"
            >
              {availableStrategies.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </select>
            <div className="text-xs text-[#00cc77]/50">
              {currentStrategy === 'checkerboard' && 'Spatial alternating pattern (best for grids)'}
              {currentStrategy === 'random' && 'Random independent sets (general purpose)'}
              {currentStrategy === 'stripes' && 'Horizontal/vertical stripes (good for 2D)'}
              {currentStrategy === 'supercell' && 'Large block partitioning (fast, less mixing)'}
              {currentStrategy === 'graph-coloring' && 'Graph-based independence (optimal)'}
            </div>
          </div>

          {/* Conditional Sampling (Clamping) Interface */}
          <div className="space-y-2 pt-2 border-t border-[#00cc77]/20">
            <label className="text-xs text-[#00cc77]/80 uppercase tracking-wide">
              Conditional Sampling
              {clampedNodes && clampedNodes.clamped_count > 0 && (
                <span className="ml-2 text-[#f85149]">({clampedNodes.clamped_count} clamped)</span>
              )}
            </label>
            
            <div className="flex gap-2">
              <input
                type="number"
                placeholder="Node ID"
                value={clampNodeId}
                onChange={(e) => setClampNodeId(e.target.value)}
                className="flex-1 bg-[#0a0a0a] border border-[#00cc77]/30 text-[#00cc77] px-3 py-2 rounded text-sm focus:outline-none focus:border-[#00cc77]"
              />
              <input
                type="number"
                placeholder="Value"
                value={clampValue}
                onChange={(e) => setClampValue(e.target.value)}
                step="0.1"
                className="w-20 bg-[#0a0a0a] border border-[#00cc77]/30 text-[#00cc77] px-3 py-2 rounded text-sm focus:outline-none focus:border-[#00cc77]"
              />
              <button
                onClick={handleClampNode}
                className="px-4 py-2 bg-[#00cc77]/20 hover:bg-[#00cc77]/30 text-[#00cc77] rounded text-sm transition-colors"
              >
                Clamp
              </button>
            </div>

            {clampedNodes && clampedNodes.clamped_count > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-[#00cc77]/60 max-h-20 overflow-y-auto">
                  {clampedNodes.node_ids.map((id, idx) => (
                    <div key={id} className="flex justify-between">
                      <span>Node {id}:</span>
                      <span className="font-mono">{clampedNodes.values[idx].toFixed(2)}</span>
                    </div>
                  ))}
                </div>
                <button
                  onClick={handleClearClamped}
                  className="w-full px-3 py-1 bg-[#f85149]/20 hover:bg-[#f85149]/30 text-[#f85149] rounded text-xs transition-colors"
                >
                  Clear All Clamped Nodes
                </button>
              </div>
            )}

            <div className="text-xs text-[#00cc77]/50">
              Clamp nodes to fix their values during sampling (for inpainting, constrained synthesis, etc.)
            </div>
          </div>
        </>
      )}
    </div>
  );
};

