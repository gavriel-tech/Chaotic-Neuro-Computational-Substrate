/**
 * Plugin Browser Component
 * 
 * Browse, manage, and execute plugins.
 */

'use client';

import { useState, useEffect, useCallback } from 'react';

interface Plugin {
  plugin_id: string;
  metadata: {
    name: string;
    version: string;
    author: string;
    description: string;
    category: string;
    tags: string[];
  };
  enabled: boolean;
}

export default function PluginBrowser() {
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [loading, setLoading] = useState(true);

  const fetchPlugins = useCallback(async () => {
    try {
      const url = selectedCategory === 'all' 
        ? 'http://localhost:8000/plugins/list'
        : `http://localhost:8000/plugins/list?category=${selectedCategory}`;
      const response = await fetch(url);
      const data = await response.json();
      setPlugins(data.plugins);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch plugins:', error);
      setLoading(false);
    }
  }, [selectedCategory]);

  useEffect(() => {
    fetchPlugins();
  }, [fetchPlugins]);

  const togglePlugin = async (pluginId: string, enabled: boolean) => {
    try {
      const endpoint = enabled ? 'disable' : 'enable';
      await fetch(`http://localhost:8000/plugins/${pluginId}/${endpoint}`, {
        method: 'POST'
      });
      await fetchPlugins();
    } catch (error) {
      console.error('Failed to toggle plugin:', error);
    }
  };

  const discoverPlugins = async () => {
    try {
      await fetch('http://localhost:8000/plugins/discover', { method: 'POST' });
      await fetchPlugins();
    } catch (error) {
      console.error('Failed to discover plugins:', error);
    }
  };

  return (
    <div className="plugin-browser bg-gray-900 border border-purple-500/30 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-purple-400 text-xl font-bold">Plugin Browser</h3>
        <button
          onClick={discoverPlugins}
          className="px-4 py-2 bg-purple-500 rounded text-gray-900 font-medium hover:bg-purple-400 transition-all"
        >
          🔍 Discover Plugins
        </button>
      </div>

      {/* Category Filter */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {['all', 'algorithm', 'processor', 'analyzer', 'visualizer'].map(cat => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-3 py-1 rounded text-sm font-medium transition-all ${
              selectedCategory === cat
                ? 'bg-purple-500 text-gray-900'
                : 'bg-gray-800 text-purple-400 border border-purple-500/30 hover:border-purple-400'
            }`}
          >
            {cat.charAt(0).toUpperCase() + cat.slice(1)}
          </button>
        ))}
      </div>

      {/* Plugin Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {loading ? (
          <div className="col-span-full text-purple-400 text-center py-8">Loading plugins...</div>
        ) : plugins.length === 0 ? (
          <div className="col-span-full text-purple-400/60 text-center py-8">
            No plugins found. Click &quot;Discover Plugins&quot; to scan for new plugins.
          </div>
        ) : (
          plugins.map(plugin => (
            <div
              key={plugin.plugin_id}
              className="bg-gray-800 border border-purple-500/20 rounded-lg p-4 hover:border-purple-400 transition-all"
            >
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h4 className="text-purple-300 font-bold">{plugin.metadata.name}</h4>
                  <p className="text-purple-400/60 text-xs">v{plugin.metadata.version} by {plugin.metadata.author}</p>
                </div>
                <button
                  onClick={() => togglePlugin(plugin.plugin_id, plugin.enabled)}
                  className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                    plugin.enabled
                      ? 'bg-[#00ff99]/20 text-green-400 border border-green-500/30'
                      : 'bg-gray-700 text-gray-400 border border-gray-600'
                  }`}
                >
                  {plugin.enabled ? 'Enabled' : 'Disabled'}
                </button>
              </div>
              
              <p className="text-purple-400/80 text-sm mb-3">{plugin.metadata.description}</p>
              
              <div className="flex flex-wrap gap-1 mb-3">
                {plugin.metadata.tags.map(tag => (
                  <span
                    key={tag}
                    className="px-2 py-0.5 bg-purple-500/20 text-purple-400 text-xs rounded"
                  >
                    {tag}
                  </span>
                ))}
              </div>
              
              <div className="flex gap-2">
                <button className="flex-1 px-3 py-1 bg-gray-700 border border-purple-500/30 rounded text-purple-400 text-sm hover:border-purple-400 transition-all">
                  Configure
                </button>
                <button className="flex-1 px-3 py-1 bg-purple-500/20 border border-purple-500/30 rounded text-purple-400 text-sm hover:border-purple-400 transition-all">
                  Execute
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

