'use client';
import { notify } from '../UI/Notification';

import React, { useState, useEffect } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// ============================================================================
// Heterogeneous Node Configuration Panel
// ============================================================================

interface HeterogeneousNodePanelProps {
  onClose: () => void;
}

export const HeterogeneousNodePanel: React.FC<HeterogeneousNodePanelProps> = ({ onClose }) => {
  const [nodeConfigs, setNodeConfigs] = useState<{
    node_id: number;
    type: 'spin' | 'continuous' | 'discrete';
    min_val?: number;
    max_val?: number;
    num_states?: number;
  }[]>([]);
  
  const [newNode, setNewNode] = useState({
    node_id: 0,
    type: 'spin' as 'spin' | 'continuous' | 'discrete',
    min_val: -1,
    max_val: 1,
    num_states: 10
  });

  const addNodeConfig = () => {
    setNodeConfigs([...nodeConfigs, { ...newNode }]);
    setNewNode({ ...newNode, node_id: newNode.node_id + 1 });
  };

  const removeNodeConfig = (index: number) => {
    setNodeConfigs(nodeConfigs.filter((_, i) => i !== index));
  };

  const applyConfiguration = async () => {
    if (nodeConfigs.length === 0) {
      notify.warning('Add at least one node type to configure.');
      return;
    }

    const nodeTypes = [...nodeConfigs]
      .sort((a, b) => a.node_id - b.node_id)
      .map((cfg) => {
        switch (cfg.type) {
          case 'continuous':
            return 1;
          case 'discrete':
            return 2;
          case 'spin':
          default:
            return 0;
        }
      });

    try {
      const response = await fetch(`${API_BASE}/thrml/heterogeneous/configure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_types: nodeTypes })
      });
      
      if (response.ok) {
        notify.success('Heterogeneous configuration applied successfully!');
        onClose();
      } else {
        notify.error('Failed to apply configuration');
      }
    } catch (err) {
      console.error('Failed to configure heterogeneous nodes:', err);
      notify.error('Error applying configuration');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-black/95 border border-[#00cc77] rounded-lg shadow-glow max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77]">
          <h2 className="text-lg font-semibold text-[#00ff99]">Heterogeneous Node Configuration</h2>
          <button onClick={onClose} className="text-[#00cc77] hover:text-[#00ff99] text-2xl">&times;</button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4 space-y-4">
          <p className="text-xs text-[#00cc77]">
            Configure mixed node types (spin/continuous/discrete) for advanced THRML modeling.
          </p>

          {/* Add New Node Config */}
          <div className="bg-black/60 border border-[#00cc77] rounded p-3 space-y-3">
            <h3 className="text-sm font-semibold text-[#00ff99]">Add Node Configuration</h3>
            
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-[#00cc77]">Node ID</label>
                <input
                  type="number"
                  value={newNode.node_id}
                  onChange={(e) => setNewNode({ ...newNode, node_id: parseInt(e.target.value) })}
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs mt-1"
                />
              </div>
              
              <div>
                <label className="text-xs text-[#00cc77]">Type</label>
                <select
                  value={newNode.type}
                  onChange={(e) => setNewNode({ ...newNode, type: e.target.value as any })}
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs mt-1"
                >
                  <option value="spin">Spin (±1)</option>
                  <option value="continuous">Continuous</option>
                  <option value="discrete">Discrete</option>
                </select>
              </div>
              
              {newNode.type === 'continuous' && (
                <>
                  <div>
                    <label className="text-xs text-[#00cc77]">Min Value</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newNode.min_val}
                      onChange={(e) => setNewNode({ ...newNode, min_val: parseFloat(e.target.value) })}
                      className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs mt-1"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-[#00cc77]">Max Value</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newNode.max_val}
                      onChange={(e) => setNewNode({ ...newNode, max_val: parseFloat(e.target.value) })}
                      className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs mt-1"
                    />
                  </div>
                </>
              )}
              
              {newNode.type === 'discrete' && (
                <div>
                  <label className="text-xs text-[#00cc77]">Number of States</label>
                  <input
                    type="number"
                    value={newNode.num_states}
                    onChange={(e) => setNewNode({ ...newNode, num_states: parseInt(e.target.value) })}
                    className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs mt-1"
                  />
                </div>
              )}
            </div>
            
            <button
              onClick={addNodeConfig}
              className="w-full px-3 py-2 bg-[#00cc77] text-black rounded text-xs font-semibold hover:bg-[#00ff99] transition"
            >
              Add Node
            </button>
          </div>

          {/* Current Configurations */}
          <div>
            <h3 className="text-sm font-semibold text-[#00ff99] mb-2">Configured Nodes ({nodeConfigs.length})</h3>
            <div className="space-y-2 max-h-64 overflow-auto">
              {nodeConfigs.map((config, i) => (
                <div key={i} className="bg-black/60 border border-[#00cc77] rounded p-2 flex items-center justify-between">
                  <div className="text-xs text-[#00ff99]">
                    Node {config.node_id}: <span className="text-[#00cc77]">{config.type}</span>
                    {config.type === 'continuous' && ` [${config.min_val}, ${config.max_val}]`}
                    {config.type === 'discrete' && ` (${config.num_states} states)`}
                  </div>
                  <button
                    onClick={() => removeNodeConfig(i)}
                    className="text-[#f85149] hover:text-[#ff6b6b] text-xl"
                  >
                    &times;
                  </button>
                </div>
              ))}
              {nodeConfigs.length === 0 && (
                <div className="text-xs text-[#00cc77] text-center py-4">No nodes configured</div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-[#00cc77]">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77] rounded text-xs hover:bg-[#00cc77]/20 transition"
          >
            Cancel
          </button>
          <button
            onClick={applyConfiguration}
            disabled={nodeConfigs.length === 0}
            className="px-4 py-2 bg-[#00cc77] text-black rounded text-xs font-semibold hover:bg-[#00ff99] transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Apply Configuration
          </button>
        </div>
      </div>
    </div>
  );
};


// ============================================================================
// Conditional Sampling Panel
// ============================================================================

interface ConditionalSamplingPanelProps {
  onClose: () => void;
}

export const ConditionalSamplingPanel: React.FC<ConditionalSamplingPanelProps> = ({ onClose }) => {
  const [clampedNodes, setClampedNodes] = useState<{ node_id: number; value: number }[]>([]);
  const [newClamp, setNewClamp] = useState({ node_id: 0, value: 1.0 });
  const [currentClamped, setCurrentClamped] = useState<{ node_id: number; value: number }[]>([]);

  useEffect(() => {
    fetchClampedNodes();
  }, []);

  const fetchClampedNodes = async () => {
    try {
      const response = await fetch(`${API_BASE}/thrml/clamped-nodes`);
      const data = await response.json();
      const mapped = Array.isArray(data?.clamped_nodes)
        ? data.clamped_nodes.map((nodeId: number, index: number) => ({
            node_id: nodeId,
            value: Array.isArray(data?.values) ? data.values[index] ?? 0 : 0,
          }))
        : [];
      setCurrentClamped(mapped);
    } catch (err) {
      console.error('Failed to fetch clamped nodes:', err);
    }
  };

  const addClamp = () => {
    setClampedNodes([...clampedNodes, { ...newClamp }]);
    setNewClamp({ ...newClamp, node_id: newClamp.node_id + 1 });
  };

  const removeClamp = (index: number) => {
    setClampedNodes(clampedNodes.filter((_, i) => i !== index));
  };

  const applyClamps = async () => {
    if (clampedNodes.length === 0) {
      notify.warning('Add at least one clamp before applying.');
      return;
    }

    const nodeIds = clampedNodes.map((c) => c.node_id);
    const values = clampedNodes.map((c) => c.value);

    try {
      const response = await fetch(`${API_BASE}/thrml/clamp-nodes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_ids: nodeIds, values })
      });
      
      if (response.ok) {
        notify.success('Node clamping applied successfully!');
        fetchClampedNodes();
      } else {
        notify.error('Failed to apply clamping');
      }
    } catch (err) {
      console.error('Failed to clamp nodes:', err);
      notify.error('Error applying clamping');
    }
  };

  const sampleConditional = async () => {
    try {
      const response = await fetch(`${API_BASE}/thrml/sample-conditional`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (response.ok) {
        const result = await response.json();
        notify.success('Conditional sample generated!');
        if (Array.isArray(result.sample)) {
          console.debug('Conditional sample:', result.sample);
        }
      } else {
        notify.error('Failed to generate sample');
      }
    } catch (err) {
      console.error('Failed to sample conditionally:', err);
      notify.error('Error generating sample');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-black/95 border border-[#00cc77] rounded-lg shadow-glow max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77]">
          <h2 className="text-lg font-semibold text-[#00ff99]">Conditional Sampling</h2>
          <button onClick={onClose} className="text-[#00cc77] hover:text-[#00ff99] text-2xl">&times;</button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4 space-y-4">
          <p className="text-xs text-[#00cc77]">
            Fix specific nodes at values for targeted generation. Clamped nodes remain fixed during sampling.
          </p>

          {/* Add New Clamp */}
          <div className="bg-black/60 border border-[#00cc77] rounded p-3 space-y-3">
            <h3 className="text-sm font-semibold text-[#00ff99]">Clamp Node</h3>
            
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-[#00cc77]">Node ID</label>
                <input
                  type="number"
                  value={newClamp.node_id}
                  onChange={(e) => setNewClamp({ ...newClamp, node_id: parseInt(e.target.value) })}
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs mt-1"
                />
              </div>
              
              <div>
                <label className="text-xs text-[#00cc77]">Value</label>
                <input
                  type="number"
                  step="0.1"
                  value={newClamp.value}
                  onChange={(e) => setNewClamp({ ...newClamp, value: parseFloat(e.target.value) })}
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs mt-1"
                />
              </div>
            </div>
            
            <button
              onClick={addClamp}
              className="w-full px-3 py-2 bg-[#00cc77] text-black rounded text-xs font-semibold hover:bg-[#00ff99] transition"
            >
              Add Clamp
            </button>
          </div>

          {/* Current Clamps */}
          <div>
            <h3 className="text-sm font-semibold text-[#00ff99] mb-2">Pending Clamps ({clampedNodes.length})</h3>
            <div className="space-y-2 max-h-32 overflow-auto">
              {clampedNodes.map((clamp, i) => (
                <div key={i} className="bg-black/60 border border-[#00cc77] rounded p-2 flex items-center justify-between">
                  <div className="text-xs text-[#00ff99]">
                    Node {clamp.node_id}: <span className="text-[#00cc77]">{clamp.value.toFixed(2)}</span>
                  </div>
                  <button
                    onClick={() => removeClamp(i)}
                    className="text-[#f85149] hover:text-[#ff6b6b] text-xl"
                  >
                    &times;
                  </button>
                </div>
              ))}
              {clampedNodes.length === 0 && (
                <div className="text-xs text-[#00cc77] text-center py-2">No clamps pending</div>
              )}
            </div>
          </div>

          {/* Currently Active Clamps */}
          <div>
            <h3 className="text-sm font-semibold text-[#00ff99] mb-2">Active Clamps ({currentClamped.length})</h3>
            <div className="space-y-2 max-h-32 overflow-auto">
              {currentClamped.map((clamp, i) => (
                <div key={i} className="bg-black/60 border border-[#00cc77] rounded p-2">
                  <div className="text-xs text-[#00ff99]">
                    Node {clamp.node_id}: <span className="text-[#00cc77]">{clamp.value.toFixed(2)}</span>
                  </div>
                </div>
              ))}
              {currentClamped.length === 0 && (
                <div className="text-xs text-[#00cc77] text-center py-2">No active clamps</div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-[#00cc77]">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77] rounded text-xs hover:bg-[#00cc77]/20 transition"
          >
            Close
          </button>
          <button
            onClick={sampleConditional}
            disabled={clampedNodes.length === 0}
            className="px-4 py-2 bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded text-xs font-semibold hover:bg-[#00cc77]/20 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Generate Sample
          </button>
          <button
            onClick={applyClamps}
            disabled={clampedNodes.length === 0}
            className="px-4 py-2 bg-[#00cc77] text-black rounded text-xs font-semibold hover:bg-[#00ff99] transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Apply Clamps
          </button>
        </div>
      </div>
    </div>
  );
};


// ============================================================================
// Higher-Order Interactions Panel
// ============================================================================

interface HigherOrderInteractionsPanelProps {
  onClose: () => void;
}

export const HigherOrderInteractionsPanel: React.FC<HigherOrderInteractionsPanelProps> = ({ onClose }) => {
  const [interactions, setInteractions] = useState<any[]>([]);
  const [newInteraction, setNewInteraction] = useState({
    node_ids: '',
    coupling_strength: 1.0
  });

  useEffect(() => {
    fetchInteractions();
  }, []);

  const fetchInteractions = async () => {
    try {
      const response = await fetch(`${API_BASE}/thrml/interactions/list`);
      const data = await response.json();
      setInteractions(data.interactions || []);
    } catch (err) {
      console.error('Failed to fetch interactions:', err);
    }
  };

  const addInteraction = async () => {
    const nodeIds = newInteraction.node_ids.split(',').map(id => parseInt(id.trim())).filter(id => !isNaN(id));
    
    if (nodeIds.length < 3) {
      notify.error('Please enter at least 3 node IDs separated by commas');
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/thrml/interactions/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          node_ids: nodeIds,
          coupling_strength: newInteraction.coupling_strength
        })
      });
      
      if (response.ok) {
        notify.success('Interaction added successfully!');
        setNewInteraction({ node_ids: '', coupling_strength: 1.0 });
        fetchInteractions();
      } else {
        notify.error('Failed to add interaction');
      }
    } catch (err) {
      console.error('Failed to add interaction:', err);
      notify.error('Error adding interaction');
    }
  };

  const deleteInteraction = async (interactionId: string) => {
    try {
      const response = await fetch(`${API_BASE}/thrml/interactions/${interactionId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        notify.success('Interaction deleted successfully!');
        fetchInteractions();
      } else {
        notify.error('Failed to delete interaction');
      }
    } catch (err) {
      console.error('Failed to delete interaction:', err);
      notify.error('Error deleting interaction');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-black/95 border border-[#00cc77] rounded-lg shadow-glow max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77]">
          <h2 className="text-lg font-semibold text-[#00ff99]">Higher-Order Interactions</h2>
          <button onClick={onClose} className="text-[#00cc77] hover:text-[#00ff99] text-2xl">&times;</button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4 space-y-4">
          <p className="text-xs text-[#00cc77]">
            Configure 3-way and 4-way THRML coupling beyond standard pairwise interactions.
          </p>

          {/* Add New Interaction */}
          <div className="bg-black/60 border border-[#00cc77] rounded p-3 space-y-3">
            <h3 className="text-sm font-semibold text-[#00ff99]">Add Interaction</h3>
            
            <div>
              <label className="text-xs text-[#00cc77]">Node IDs (comma-separated, min 3)</label>
              <input
                type="text"
                placeholder="e.g., 0, 1, 2"
                value={newInteraction.node_ids}
                onChange={(e) => setNewInteraction({ ...newInteraction, node_ids: e.target.value })}
                className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs mt-1"
              />
            </div>
            
            <div>
              <label className="text-xs text-[#00cc77]">Coupling Strength</label>
              <input
                type="number"
                step="0.1"
                value={newInteraction.coupling_strength}
                onChange={(e) => setNewInteraction({ ...newInteraction, coupling_strength: parseFloat(e.target.value) })}
                className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs mt-1"
              />
            </div>
            
            <button
              onClick={addInteraction}
              className="w-full px-3 py-2 bg-[#00cc77] text-black rounded text-xs font-semibold hover:bg-[#00ff99] transition"
            >
              Add Interaction
            </button>
          </div>

          {/* Current Interactions */}
          <div>
            <h3 className="text-sm font-semibold text-[#00ff99] mb-2">Active Interactions ({interactions.length})</h3>
            <div className="space-y-2 max-h-64 overflow-auto">
              {interactions.map((interaction, i) => (
                <div key={i} className="bg-black/60 border border-[#00cc77] rounded p-2 flex items-center justify-between">
                  <div className="text-xs text-[#00ff99]">
                    Nodes [{interaction.node_ids?.join(', ')}]: 
                    <span className="text-[#00cc77]"> strength = {interaction.coupling_strength?.toFixed(2)}</span>
                  </div>
                  <button
                    onClick={() => deleteInteraction(interaction.id)}
                    className="text-[#f85149] hover:text-[#ff6b6b] text-xl"
                  >
                    &times;
                  </button>
                </div>
              ))}
              {interactions.length === 0 && (
                <div className="text-xs text-[#00cc77] text-center py-4">No higher-order interactions configured</div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-[#00cc77]">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77] rounded text-xs hover:bg-[#00cc77]/20 transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};


// ============================================================================
// Energy Factors Panel
// ============================================================================

interface EnergyFactorsPanelProps {
  onClose: () => void;
}

export const EnergyFactorsPanel: React.FC<EnergyFactorsPanelProps> = ({ onClose }) => {
  const [factors, setFactors] = useState<any[]>([]);
  const [factorLibrary, setFactorLibrary] = useState<string[]>([]);
  const [selectedType, setSelectedType] = useState('photonic_coupling');
  const [factorParams, setFactorParams] = useState<any>({});

  useEffect(() => {
    fetchFactors();
    fetchLibrary();
  }, []);

  const fetchFactors = async () => {
    try {
      const response = await fetch(`${API_BASE}/thrml/factors/list`);
      const data = await response.json();
      setFactors(data.factors || []);
    } catch (err) {
      console.error('Failed to fetch factors:', err);
    }
  };

  const fetchLibrary = async () => {
    try {
      const response = await fetch(`${API_BASE}/thrml/factors/library`);
      const data = await response.json();
      setFactorLibrary(data.available_factors || []);
    } catch (err) {
      console.error('Failed to fetch factor library:', err);
    }
  };

  const addFactor = async () => {
    try {
      const response = await fetch(`${API_BASE}/thrml/factors/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          factor_type: selectedType,
          weight: 1.0,
          params: factorParams
        })
      });
      
      if (response.ok) {
        notify.success('Energy factor added successfully!');
        setFactorParams({});
        fetchFactors();
      } else {
        notify.error('Failed to add factor');
      }
    } catch (err) {
      console.error('Failed to add factor:', err);
      notify.error('Error adding factor');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-black/95 border border-[#00cc77] rounded-lg shadow-glow max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77]">
          <h2 className="text-lg font-semibold text-[#00ff99]">Custom Energy Factors</h2>
          <button onClick={onClose} className="text-[#00cc77] hover:text-[#00ff99] text-2xl">&times;</button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4 space-y-4">
          <p className="text-xs text-[#00cc77]">
            Add custom energy factors: photonic coupling, audio harmony, ML regularization, and more.
          </p>

          {/* Add New Factor */}
          <div className="bg-black/60 border border-[#00cc77] rounded p-3 space-y-3">
            <h3 className="text-sm font-semibold text-[#00ff99]">Add Energy Factor</h3>
            
            <div>
              <label className="text-xs text-[#00cc77]">Factor Type</label>
              <select
                value={selectedType}
                onChange={(e) => setSelectedType(e.target.value)}
                className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs mt-1"
              >
                {factorLibrary.map(type => (
                  <option key={type} value={type}>{type.replace(/_/g, ' ').toUpperCase()}</option>
                ))}
              </select>
            </div>
            
            <button
              onClick={addFactor}
              className="w-full px-3 py-2 bg-[#00cc77] text-black rounded text-xs font-semibold hover:bg-[#00ff99] transition"
            >
              Add Factor
            </button>
          </div>

          {/* Current Factors */}
          <div>
            <h3 className="text-sm font-semibold text-[#00ff99] mb-2">Active Factors ({factors.length})</h3>
            <div className="space-y-2 max-h-64 overflow-auto">
              {factors.map((factor, i) => (
                <div key={i} className="bg-black/60 border border-[#00cc77] rounded p-2">
                  <div className="text-xs text-[#00ff99]">
                    {factor.factor_type?.replace(/_/g, ' ').toUpperCase()}
                    <span className="text-[#00cc77]"> (weight: {factor.weight?.toFixed(2)})</span>
                  </div>
                </div>
              ))}
              {factors.length === 0 && (
                <div className="text-xs text-[#00cc77] text-center py-4">No custom energy factors active</div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-[#00cc77]">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77] rounded text-xs hover:bg-[#00cc77]/20 transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

