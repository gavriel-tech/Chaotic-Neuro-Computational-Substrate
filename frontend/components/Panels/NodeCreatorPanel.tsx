'use client';

import React, { useState, useEffect } from 'react';
import { notify } from '../UI/Notification';

interface NodeCreatorPanelProps {
  onClose: () => void;
  onCreated?: (response: NodeResponsePayload) => void;
}

interface NodeResponsePayload {
  status: string;
  node_id?: number;
  message?: string;
  data?: Record<string, unknown>;
}

interface Algorithm {
  id: number;
  name: string;
  category: string;
  description: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export const NodeCreatorPanel: React.FC<NodeCreatorPanelProps> = ({ onClose, onCreated }) => {
  const [nodeName, setNodeName] = useState('');
  const [position, setPosition] = useState({ x: 128, y: 128 });
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<number[]>([]);
  const [parameters, setParameters] = useState<{ [key: string]: number }>({});
  const [initialPerturbation, setInitialPerturbation] = useState(0.1);
  const [categoryFilter, setCategoryFilter] = useState<string>('all');

  useEffect(() => {
    fetchAlgorithms();
  }, []);

  const fetchAlgorithms = async () => {
    try {
      const response = await fetch(`${API_BASE}/algorithms/list`);
      const data = await response.json();
      setAlgorithms(data.algorithms || []);
    } catch (err) {
      console.error('Failed to fetch algorithms:', err);
    }
  };

  const addParameter = () => {
    // TODO: Replace with custom dialog
    notify.info('Parameter creation: Coming soon! Use default node parameters for now.');
    // const paramName = prompt('Enter parameter name (e.g., A_max, R, T):');
    // if (paramName && !parameters[paramName]) {
    //   setParameters({ ...parameters, [paramName]: 1.0 });
    // }
  };

  const updateParameter = (key: string, value: number) => {
    setParameters({ ...parameters, [key]: value });
  };

  const removeParameter = (key: string) => {
    const newParams = { ...parameters };
    delete newParams[key];
    setParameters(newParams);
  };

  const toggleAlgorithm = (algoId: number) => {
    if (selectedAlgorithms.includes(algoId)) {
      setSelectedAlgorithms(selectedAlgorithms.filter(id => id !== algoId));
    } else {
      if (selectedAlgorithms.length >= 8) {
        notify.error('Maximum 8 algorithms per chain');
        return;
      }
      setSelectedAlgorithms([...selectedAlgorithms, algoId]);
    }
  };

  const moveAlgorithmUp = (index: number) => {
    if (index === 0) return;
    const newChain = [...selectedAlgorithms];
    [newChain[index - 1], newChain[index]] = [newChain[index], newChain[index - 1]];
    setSelectedAlgorithms(newChain);
  };

  const moveAlgorithmDown = (index: number) => {
    if (index === selectedAlgorithms.length - 1) return;
    const newChain = [...selectedAlgorithms];
    [newChain[index + 1], newChain[index]] = [newChain[index], newChain[index + 1]];
    setSelectedAlgorithms(newChain);
  };

  const createNode = async () => {
    try {
      const response = await fetch(`${API_BASE}/node/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          position: [position.x, position.y],
          config: parameters,
          chain: selectedAlgorithms,
          initial_perturbation: initialPerturbation
        })
      });

      if (response.ok) {
        const result: NodeResponsePayload = await response.json();
        notify.success(result.message ?? `Node created successfully! ID: ${result.node_id}`);
        onCreated?.(result);
        onClose();
      } else {
        const error = await response.text();
        notify.error(`Failed to create node: ${error}`);
      }
    } catch (err) {
      console.error('Failed to create node:', err);
      notify.error('Error creating node');
    }
  };

  const filteredAlgorithms = categoryFilter === 'all'
    ? algorithms
    : algorithms.filter(algo => algo.category === categoryFilter);

  const getAlgorithmName = (id: number) => {
    const algo = algorithms.find(a => a.id === id);
    return algo ? algo.name : `Algo ${id}`;
  };

  return (
    <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-black/95 border border-[#00cc77] rounded-lg shadow-glow max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77]">
          <h2 className="text-lg font-semibold text-[#00ff99]">Node Creator</h2>
          <button onClick={onClose} className="text-[#00cc77] hover:text-[#00ff99] text-2xl">&times;</button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Column: Node Configuration */}
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-[#00ff99] uppercase tracking-wide">Node Configuration</h3>

              {/* Node Name */}
              <div>
                <label className="text-xs text-[#00cc77] block mb-2">Node Name (Optional)</label>
                <input
                  type="text"
                  value={nodeName}
                  onChange={(e) => setNodeName(e.target.value)}
                  placeholder="e.g., My Custom Processor"
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-sm"
                />
              </div>

              {/* Position */}
              <div>
                <label className="text-xs text-[#00cc77] block mb-2">Position (X, Y)</label>
                <div className="grid grid-cols-2 gap-2">
                  <input
                    type="number"
                    value={position.x}
                    onChange={(e) => setPosition({ ...position, x: parseFloat(e.target.value) })}
                    className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-sm"
                  />
                  <input
                    type="number"
                    value={position.y}
                    onChange={(e) => setPosition({ ...position, y: parseFloat(e.target.value) })}
                    className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-sm"
                  />
                </div>
                <div className="text-xs text-[#00cc77] mt-1">Grid: 0-256 (X), 0-256 (Y)</div>
              </div>

              {/* Initial Perturbation */}
              <div>
                <label className="text-xs text-[#00cc77] block mb-2">
                  Initial Perturbation: {initialPerturbation.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="-1"
                  max="1"
                  step="0.01"
                  value={initialPerturbation}
                  onChange={(e) => setInitialPerturbation(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-[#00cc77]">
                  <span>-1.0</span>
                  <span>0.0</span>
                  <span>1.0</span>
                </div>
              </div>

              {/* Custom Parameters */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs text-[#00cc77]">Custom Parameters</label>
                  <button
                    onClick={addParameter}
                    className="px-2 py-1 text-xs bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition"
                  >
                    + Add Parameter
                  </button>
                </div>
                <div className="space-y-2 max-h-48 overflow-auto">
                  {Object.keys(parameters).length === 0 ? (
                    <div className="text-xs text-[#00cc77]/60 text-center py-4">
                      No custom parameters. Click &quot;Add Parameter&quot; to define variables like A_max, R, T, etc.
                    </div>
                  ) : (
                    Object.entries(parameters).map(([key, value]) => (
                      <div key={key} className="bg-black/60 border border-[#00cc77] rounded p-2">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-[#00ff99] font-mono">{key}</span>
                          <button
                            onClick={() => removeParameter(key)}
                            className="text-[#f85149] hover:text-[#ff6b6b] text-sm"
                          >
                            ×
                          </button>
                        </div>
                        <input
                          type="number"
                          step="0.01"
                          value={value}
                          onChange={(e) => updateParameter(key, parseFloat(e.target.value))}
                          className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-sm"
                        />
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Algorithm Chain */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs text-[#00cc77]">Algorithm Chain ({selectedAlgorithms.length}/8)</label>
                </div>
                <div className="space-y-2 max-h-64 overflow-auto">
                  {selectedAlgorithms.length === 0 ? (
                    <div className="text-xs text-[#00cc77]/60 text-center py-4">
                      No algorithms selected. Choose from the library on the right →
                    </div>
                  ) : (
                    selectedAlgorithms.map((algoId, index) => (
                      <div key={`${algoId}-${index}`} className="bg-[#00cc77]/10 border border-[#00cc77] rounded p-2 flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-[#00cc77] font-mono">#{index + 1}</span>
                          <span className="text-xs text-[#00ff99] font-semibold">{getAlgorithmName(algoId)}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => moveAlgorithmUp(index)}
                            disabled={index === 0}
                            className="px-2 py-1 text-xs text-[#00cc77] hover:text-[#00ff99] disabled:opacity-30 disabled:cursor-not-allowed"
                          >
                            ↑
                          </button>
                          <button
                            onClick={() => moveAlgorithmDown(index)}
                            disabled={index === selectedAlgorithms.length - 1}
                            className="px-2 py-1 text-xs text-[#00cc77] hover:text-[#00ff99] disabled:opacity-30 disabled:cursor-not-allowed"
                          >
                            ↓
                          </button>
                          <button
                            onClick={() => toggleAlgorithm(algoId)}
                            className="px-2 py-1 text-xs text-[#f85149] hover:text-[#ff6b6b]"
                          >
                            ×
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Right Column: Algorithm Library */}
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-[#00ff99] uppercase tracking-wide">Algorithm Library</h3>

              {/* Category Filter */}
              <div 
                className="overflow-x-auto pb-2 custom-scrollbar"
                onWheel={(e) => {
                  // Enable horizontal scrolling with mouse wheel
                  if (e.currentTarget.scrollWidth > e.currentTarget.clientWidth) {
                    e.preventDefault();
                    e.currentTarget.scrollLeft += e.deltaY;
                  }
                }}
              >
                <div className="flex items-center gap-2 min-w-max">
                {['all', 'basic', 'audio', 'photonic'].map(cat => (
                  <button
                    key={cat}
                    onClick={() => setCategoryFilter(cat)}
                      className={`px-3 py-1 text-xs rounded transition whitespace-nowrap ${categoryFilter === cat
                      ? 'bg-[#00cc77] text-black'
                      : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] hover:bg-[#00cc77]/20'
                      }`}
                  >
                    {cat.toUpperCase()}
                  </button>
                ))}
                </div>
              </div>

              {/* Algorithm Grid */}
              <div className="grid grid-cols-1 gap-2 max-h-[500px] overflow-auto">
                {filteredAlgorithms.map(algo => {
                  const isSelected = selectedAlgorithms.includes(algo.id);
                  return (
                    <button
                      key={algo.id}
                      onClick={() => toggleAlgorithm(algo.id)}
                      className={`text-left p-3 rounded border transition ${isSelected
                        ? 'bg-[#00cc77]/20 border-[#00ff99]'
                        : 'bg-black/60 border-[#00cc77] hover:bg-[#00cc77]/10'
                        }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="font-semibold text-[#00ff99] text-sm mb-1">{algo.name}</div>
                          <div className="text-xs text-[#00cc77]/80">{algo.description}</div>
                        </div>
                        {isSelected && (
                          <span className="ml-2 text-[#00ff99] text-sm">✓</span>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-[#00cc77]">
          <div className="text-xs text-[#00cc77]">
            {selectedAlgorithms.length > 0
              ? `${selectedAlgorithms.length} algorithm${selectedAlgorithms.length !== 1 ? 's' : ''} in chain`
              : 'Select algorithms to create a processing chain'
            }
          </div>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77] rounded text-xs hover:bg-[#00cc77]/20 transition"
            >
              Cancel
            </button>
            <button
              onClick={createNode}
              disabled={selectedAlgorithms.length === 0}
              className="px-4 py-2 bg-[#00cc77] text-black rounded text-xs font-semibold hover:bg-[#00ff99] transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Create Node
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

