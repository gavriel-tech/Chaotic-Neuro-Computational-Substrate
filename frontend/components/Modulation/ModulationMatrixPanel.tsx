'use client';
import { notify } from '../UI/Notification';

import React, { useState } from 'react';

interface ModulationRoute {
  id: string;
  source: string;
  target: string;
  amount: number;
}

export const ModulationMatrixPanel: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [routes, setRoutes] = useState<ModulationRoute[]>([]);
  const [showAddRoute, setShowAddRoute] = useState(false);

  const sources = [
    'Audio Pitch', 'Audio RMS', 'THRML Energy', 'THRML Spins',
    'Oscillator X', 'Oscillator Y', 'Oscillator Z', 'Wave Field'
  ];

  const targets = [
    'THRML Temperature', 'THRML Bias', 'Oscillator Alpha', 'Oscillator Beta',
    'Algorithm Param 1', 'Algorithm Param 2', 'Wave Coupling', 'Field Diffusion'
  ];

  const addRoute = (source: string, target: string) => {
    setRoutes([...routes, {
      id: `route-${Date.now()}`,
      source,
      target,
      amount: 0.5
    }]);
    setShowAddRoute(false);
  };

  const deleteRoute = (id: string) => {
    setRoutes(routes.filter(r => r.id !== id));
  };

  const updateAmount = (id: string, amount: number) => {
    setRoutes(routes.map(r => r.id === id ? { ...r, amount } : r));
  };

  const loadPreset = async (presetName: string) => {
    try {
      const response = await fetch(`http://localhost:8000/modulation/presets/${presetName}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        const data = await response.json();
        notify.error(`Preset "${presetName}" loaded successfully!`);
        // Fetch updated routes from server
        fetchRoutes();
      } else {
        notify.error(`Failed to load preset "${presetName}"`);
      }
    } catch (err) {
      console.error('Failed to load preset:', err);
      notify.error('Error loading preset');
    }
  };

  const fetchRoutes = async () => {
    try {
      const response = await fetch('http://localhost:8000/modulation/routes');
      const data = await response.json();
      if (data.routes) {
        setRoutes(data.routes);
      }
    } catch (err) {
      console.error('Failed to fetch routes:', err);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="w-[800px] max-h-[85vh] bg-black/60 border border-[#00cc77] rounded-lg shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77]">
          <h2 className="text-sm font-semibold text-[#00ff99]">Modulation Matrix</h2>
          <button
            onClick={onClose}
            className="text-[#00cc77] hover:text-[#00ff99] text-xl"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-4 overflow-y-auto max-h-[calc(85vh-120px)] custom-scrollbar">
          <div className="mb-4">
            <button
              onClick={() => setShowAddRoute(true)}
              className="px-3 py-2 text-xs bg-[#00cc77] text-white rounded hover:bg-[#2ea043] transition font-semibold"
            >
              + Add Route
            </button>
          </div>

          {/* Add Route Modal */}
          {showAddRoute && (
            <div className="mb-4 p-4 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded">
              <h3 className="text-xs font-semibold text-[#00ff99] mb-3">New Modulation Route</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-[#00cc77] block mb-2">Source</label>
                  <div className="space-y-1">
                    {sources.map(source => (
                      <button
                        key={source}
                        onClick={() => {
                          const target = targets[0];
                          addRoute(source, target);
                        }}
                        className="w-full text-left px-2 py-1 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
                      >
                        {source}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="text-xs text-[#00cc77] block mb-2">Target</label>
                  <div className="space-y-1">
                    {targets.map(target => (
                      <div key={target} className="px-2 py-1 text-xs text-[#00cc77]">
                        {target}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <button
                onClick={() => setShowAddRoute(false)}
                className="mt-3 px-3 py-1 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
              >
                Cancel
              </button>
            </div>
          )}

          {/* Routes List */}
          <div className="space-y-2">
            {routes.length === 0 ? (
              <div className="text-center py-8 text-[#00cc77] text-sm">
                No modulation routes configured
              </div>
            ) : (
              routes.map(route => (
                <div key={route.id} className="p-3 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-[#00ff99] font-semibold">{route.source}</span>
                      <span className="text-[#00cc77]">→</span>
                      <span className="text-[#00ff99] font-semibold">{route.target}</span>
                    </div>
                    <button
                      onClick={() => deleteRoute(route.id)}
                      className="text-[#f85149] hover:text-[#ff7b72] text-sm"
                    >
                      ×
                    </button>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-[#00cc77]">Amount</span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={route.amount}
                      onChange={(e) => updateAmount(route.id, parseFloat(e.target.value))}
                      className="flex-1"
                    />
                    <span className="text-xs text-[#00ff99] font-mono w-12">
                      {route.amount.toFixed(2)}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Presets */}
          <div className="mt-6 pt-4 border-t border-[#00cc77]">
            <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-3">
              Presets
            </h3>
            <div className="grid grid-cols-2 gap-2">
              <button 
                onClick={() => loadPreset('audio-reactive')}
                className="px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-left"
              >
                <div className="font-semibold">Audio Reactive</div>
                <div className="text-[10px] text-[#00cc77] mt-1">Pitch → Temperature, RMS → Bias</div>
              </button>
              <button 
                onClick={() => loadPreset('feedback')}
                className="px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-left"
              >
                <div className="font-semibold">Feedback Loop</div>
                <div className="text-[10px] text-[#00cc77] mt-1">Energy → Alpha, Spins → Beta</div>
              </button>
              <button 
                onClick={() => loadPreset('cross-modulation')}
                className="px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-left"
              >
                <div className="font-semibold">Cross-Modulation</div>
                <div className="text-[10px] text-[#00cc77] mt-1">X → Y, Y → Z, Z → X</div>
              </button>
              <button 
                onClick={() => loadPreset('wave-coupling')}
                className="px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-left"
              >
                <div className="font-semibold">Wave Coupling</div>
                <div className="text-[10px] text-[#00cc77] mt-1">Field → All Oscillators</div>
              </button>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-[#00cc77] bg-black/60">
          <div className="text-xs text-[#00cc77]">
            {routes.length} active route{routes.length !== 1 ? 's' : ''}
          </div>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
            >
              Close
            </button>
            <button className="px-3 py-1.5 text-xs bg-[#00cc77] text-white rounded hover:bg-[#2ea043] transition font-semibold">
              Apply
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

