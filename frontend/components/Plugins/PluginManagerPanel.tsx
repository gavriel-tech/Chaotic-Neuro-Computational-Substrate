'use client';

import React, { useState } from 'react';

interface Plugin {
  id: string;
  name: string;
  description: string;
  author: string;
  version: string;
  enabled: boolean;
}

export const PluginManagerPanel: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [plugins, setPlugins] = useState<Plugin[]>([
    {
      id: 'waveshaper',
      name: 'Waveshaper Plugin',
      description: 'Custom waveshaping algorithm with adjustable curve',
      author: 'GMCS Team',
      version: '1.0.0',
      enabled: true
    },
    {
      id: 'pattern-detector',
      name: 'Pattern Detector',
      description: 'Stateful pattern detection across oscillator states',
      author: 'GMCS Team',
      version: '1.0.0',
      enabled: false
    }
  ]);

  const togglePlugin = (id: string) => {
    setPlugins(plugins.map(p => p.id === id ? { ...p, enabled: !p.enabled } : p));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="w-[700px] max-h-[85vh] bg-black/60 border border-[#00cc77] rounded-lg shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77]">
          <h2 className="text-sm font-semibold text-[#00ff99]">Plugin Manager</h2>
          <button
            onClick={onClose}
            className="text-[#00cc77] hover:text-[#00ff99] text-xl"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-4 overflow-y-auto max-h-[calc(85vh-120px)] custom-scrollbar">
          <div className="mb-4 flex gap-2">
            <button className="px-3 py-2 text-xs bg-[#00cc77] text-white rounded hover:bg-[#2ea043] transition font-semibold">
              + Load Plugin
            </button>
            <button className="px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition">
              Browse Library
            </button>
          </div>

          {/* Installed Plugins */}
          <div className="space-y-3">
            {plugins.map(plugin => (
              <div key={plugin.id} className="p-4 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <h3 className="text-sm font-semibold text-[#00ff99]">{plugin.name}</h3>
                    <p className="text-xs text-[#00cc77] mt-1">{plugin.description}</p>
                  </div>
                  <button
                    onClick={() => togglePlugin(plugin.id)}
                    className={`px-3 py-1 text-xs rounded font-semibold transition ${
                      plugin.enabled
                        ? 'bg-[#00cc77] text-white hover:bg-[#2ea043]'
                        : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77] hover:bg-[#00cc77]/20'
                    }`}
                  >
                    {plugin.enabled ? 'Enabled' : 'Disabled'}
                  </button>
                </div>
                <div className="flex items-center gap-4 text-xs text-[#00cc77] mt-3">
                  <span>v{plugin.version}</span>
                  <span>•</span>
                  <span>{plugin.author}</span>
                  <span>•</span>
                  <button className="text-[#00ff99] hover:text-[#79c0ff]">Configure</button>
                  <button className="text-[#f85149] hover:text-[#ff7b72]">Uninstall</button>
                </div>
              </div>
            ))}
          </div>

          {/* Plugin Development */}
          <div className="mt-6 pt-4 border-t border-[#00cc77]">
            <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-3">
              Plugin Development
            </h3>
            <div className="p-4 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded">
              <p className="text-xs text-[#00ff99] mb-3">
                Create custom algorithms and extend GMCS functionality
              </p>
              <div className="flex gap-2">
                <button className="px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition">
                  View Documentation
                </button>
                <button className="px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition">
                  Example Templates
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-[#00cc77] bg-black/60">
          <div className="text-xs text-[#00cc77]">
            {plugins.filter(p => p.enabled).length} of {plugins.length} plugins enabled
          </div>
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

