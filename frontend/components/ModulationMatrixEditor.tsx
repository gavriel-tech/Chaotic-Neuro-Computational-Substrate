/**
 * Modulation Matrix Editor Component
 * 
 * Visual editor for creating and managing modulation routes between
 * different system components.
 */

'use client';

import { useState, useEffect } from 'react';

interface ModulationRoute {
  id: number;
  source_type: string;
  source_node_id: number;
  target_type: string;
  target_node_id: number;
  strength: number;
  mode: string;
}

interface ModulationSource {
  id: string;
  name: string;
}

interface ModulationTarget {
  id: string;
  name: string;
}

export default function ModulationMatrixEditor() {
  const [routes, setRoutes] = useState<ModulationRoute[]>([]);
  const [sources, setSources] = useState<ModulationSource[]>([]);
  const [targets, setTargets] = useState<ModulationTarget[]>([]);
  const [showAddDialog, setShowAddDialog] = useState(false);
  
  // New route form state
  const [newRoute, setNewRoute] = useState({
    source_type: '',
    source_node_id: 0,
    target_type: '',
    target_node_id: 0,
    strength: 1.0,
    mode: 'multiply'
  });

  useEffect(() => {
    fetchRoutes();
    fetchSources();
    fetchTargets();
  }, []);

  const fetchRoutes = async () => {
    try {
      const response = await fetch('http://localhost:8000/modulation/routes');
      const data = await response.json();
      setRoutes(data.routes.map((r: any, i: number) => ({ ...r.route, id: i })));
    } catch (error) {
      console.error('Failed to fetch routes:', error);
    }
  };

  const fetchSources = async () => {
    try {
      const response = await fetch('http://localhost:8000/modulation/sources');
      const data = await response.json();
      setSources(data.sources);
    } catch (error) {
      console.error('Failed to fetch sources:', error);
    }
  };

  const fetchTargets = async () => {
    try {
      const response = await fetch('http://localhost:8000/modulation/targets');
      const data = await response.json();
      setTargets(data.targets);
    } catch (error) {
      console.error('Failed to fetch targets:', error);
    }
  };

  const addRoute = async () => {
    try {
      const response = await fetch('http://localhost:8000/modulation/routes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newRoute)
      });
      
      if (response.ok) {
        await fetchRoutes();
        setShowAddDialog(false);
        setNewRoute({
          source_type: '',
          source_node_id: 0,
          target_type: '',
          target_node_id: 0,
          strength: 1.0,
          mode: 'multiply'
        });
      }
    } catch (error) {
      console.error('Failed to add route:', error);
    }
  };

  const deleteRoute = async (routeId: number) => {
    try {
      await fetch(`http://localhost:8000/modulation/routes/${routeId}`, {
        method: 'DELETE'
      });
      await fetchRoutes();
    } catch (error) {
      console.error('Failed to delete route:', error);
    }
  };

  const loadPreset = async (presetName: string) => {
    try {
      await fetch(`http://localhost:8000/modulation/presets/${presetName}`, {
        method: 'POST'
      });
      await fetchRoutes();
    } catch (error) {
      console.error('Failed to load preset:', error);
    }
  };

  return (
    <div className="modulation-matrix-editor bg-gray-900 border border-magenta-500/30 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-magenta-400 text-xl font-bold">Modulation Matrix</h3>
        <div className="flex gap-2">
          <button
            onClick={() => loadPreset('audio-reactive')}
            className="px-3 py-1 bg-gray-800 border border-magenta-500/30 rounded text-magenta-400 text-sm hover:border-magenta-400 transition-all"
          >
            Audio Preset
          </button>
          <button
            onClick={() => loadPreset('feedback')}
            className="px-3 py-1 bg-gray-800 border border-magenta-500/30 rounded text-magenta-400 text-sm hover:border-magenta-400 transition-all"
          >
            Feedback Preset
          </button>
          <button
            onClick={() => setShowAddDialog(true)}
            className="px-4 py-1 bg-magenta-500 rounded text-gray-900 font-medium hover:bg-magenta-400 transition-all"
          >
            + Add Route
          </button>
        </div>
      </div>

      {/* Routes Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-magenta-500/30">
              <th className="text-left text-magenta-400 font-medium py-2 px-3">Source</th>
              <th className="text-left text-magenta-400 font-medium py-2 px-3">Node</th>
              <th className="text-center text-magenta-400 font-medium py-2 px-3">→</th>
              <th className="text-left text-magenta-400 font-medium py-2 px-3">Target</th>
              <th className="text-left text-magenta-400 font-medium py-2 px-3">Node</th>
              <th className="text-left text-magenta-400 font-medium py-2 px-3">Strength</th>
              <th className="text-left text-magenta-400 font-medium py-2 px-3">Mode</th>
              <th className="text-right text-magenta-400 font-medium py-2 px-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {routes.length === 0 ? (
              <tr>
                <td colSpan={8} className="text-center text-magenta-400/60 py-8">
                  No modulation routes configured
                </td>
              </tr>
            ) : (
              routes.map((route) => (
                <tr key={route.id} className="border-b border-magenta-500/10 hover:bg-gray-800/50">
                  <td className="py-3 px-3 text-magenta-300">{route.source_type}</td>
                  <td className="py-3 px-3 text-magenta-300">{route.source_node_id}</td>
                  <td className="py-3 px-3 text-center text-magenta-500">→</td>
                  <td className="py-3 px-3 text-magenta-300">{route.target_type}</td>
                  <td className="py-3 px-3 text-magenta-300">{route.target_node_id}</td>
                  <td className="py-3 px-3 text-magenta-300">{route.strength.toFixed(2)}</td>
                  <td className="py-3 px-3 text-magenta-300">{route.mode}</td>
                  <td className="py-3 px-3 text-right">
                    <button
                      onClick={() => deleteRoute(route.id)}
                      className="text-red-400 hover:text-red-300 transition-colors"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Add Route Dialog */}
      {showAddDialog && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-magenta-500/50 rounded-lg p-6 w-full max-w-md">
            <h4 className="text-magenta-400 text-lg font-bold mb-4">Add Modulation Route</h4>
            
            <div className="space-y-4">
              <div>
                <label className="block text-magenta-400 text-sm mb-1">Source Type</label>
                <select
                  value={newRoute.source_type}
                  onChange={(e) => setNewRoute({ ...newRoute, source_type: e.target.value })}
                  className="w-full bg-gray-800 border border-magenta-500/30 rounded px-3 py-2 text-magenta-100"
                >
                  <option value="">Select source...</option>
                  {sources.map(s => (
                    <option key={s.id} value={s.id}>{s.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-magenta-400 text-sm mb-1">Source Node ID</label>
                <input
                  type="number"
                  value={newRoute.source_node_id}
                  onChange={(e) => setNewRoute({ ...newRoute, source_node_id: parseInt(e.target.value) })}
                  className="w-full bg-gray-800 border border-magenta-500/30 rounded px-3 py-2 text-magenta-100"
                />
              </div>

              <div>
                <label className="block text-magenta-400 text-sm mb-1">Target Type</label>
                <select
                  value={newRoute.target_type}
                  onChange={(e) => setNewRoute({ ...newRoute, target_type: e.target.value })}
                  className="w-full bg-gray-800 border border-magenta-500/30 rounded px-3 py-2 text-magenta-100"
                >
                  <option value="">Select target...</option>
                  {targets.map(t => (
                    <option key={t.id} value={t.id}>{t.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-magenta-400 text-sm mb-1">Target Node ID</label>
                <input
                  type="number"
                  value={newRoute.target_node_id}
                  onChange={(e) => setNewRoute({ ...newRoute, target_node_id: parseInt(e.target.value) })}
                  className="w-full bg-gray-800 border border-magenta-500/30 rounded px-3 py-2 text-magenta-100"
                />
              </div>

              <div>
                <label className="block text-magenta-400 text-sm mb-1">Strength</label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={newRoute.strength}
                  onChange={(e) => setNewRoute({ ...newRoute, strength: parseFloat(e.target.value) })}
                  className="w-full"
                />
                <span className="text-magenta-300 text-sm">{newRoute.strength.toFixed(2)}</span>
              </div>

              <div>
                <label className="block text-magenta-400 text-sm mb-1">Mode</label>
                <select
                  value={newRoute.mode}
                  onChange={(e) => setNewRoute({ ...newRoute, mode: e.target.value })}
                  className="w-full bg-gray-800 border border-magenta-500/30 rounded px-3 py-2 text-magenta-100"
                >
                  <option value="multiply">Multiply</option>
                  <option value="add">Add</option>
                  <option value="replace">Replace</option>
                </select>
              </div>
            </div>

            <div className="flex gap-2 mt-6">
              <button
                onClick={addRoute}
                className="flex-1 bg-magenta-500 rounded py-2 text-gray-900 font-medium hover:bg-magenta-400 transition-all"
              >
                Add Route
              </button>
              <button
                onClick={() => setShowAddDialog(false)}
                className="flex-1 bg-gray-800 border border-magenta-500/30 rounded py-2 text-magenta-400 hover:border-magenta-400 transition-all"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

