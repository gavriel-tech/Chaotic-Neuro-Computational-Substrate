'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { NodeCreatorPanel } from '@/components/Panels/NodeCreatorPanel';

interface Algorithm {
  id: string;
  name: string;
  category: string;
  description: string;
  parameters: { [key: string]: any };
}

export default function AlgorithmsPage() {
  const [activeTab, setActiveTab] = useState<'browse' | 'createAlgo' | 'createNode'>('browse');
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<Algorithm | null>(null);
  const [loading, setLoading] = useState(true);
  const [showNodeCreator, setShowNodeCreator] = useState(false);
  
  // Create Algorithm state
  const [newAlgoName, setNewAlgoName] = useState('');
  const [newAlgoCategory, setNewAlgoCategory] = useState('basic');
  const [newAlgoDescription, setNewAlgoDescription] = useState('');
  const [newAlgoCode, setNewAlgoCode] = useState('');
  const [newAlgoParams, setNewAlgoParams] = useState<{[key: string]: any}>({});

  useEffect(() => {
    fetchAlgorithms();
    fetchCategories();
  }, []);

  const fetchAlgorithms = async () => {
    try {
      const response = await fetch('http://localhost:8000/algorithms/list');
      const data = await response.json();
      setAlgorithms(data.algorithms || []);
      setLoading(false);
    } catch (err) {
      console.error('Failed to fetch algorithms:', err);
      setLoading(false);
    }
  };

  const fetchCategories = async () => {
    try {
      const response = await fetch('http://localhost:8000/algorithms/categories');
      const data = await response.json();
      const categoryNames = (data.categories || []).map((cat: any) => cat.name);
      setCategories(['all', ...categoryNames]);
    } catch (err) {
      console.error('Failed to fetch categories:', err);
    }
  };

  const fetchAlgorithmDetails = async (algoId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/algorithms/${algoId}`);
      const data = await response.json();
      setSelectedAlgorithm(data);
    } catch (err) {
      console.error('Failed to fetch algorithm details:', err);
    }
  };

  const filteredAlgorithms = selectedCategory === 'all'
    ? algorithms
    : algorithms.filter(algo => algo.category === selectedCategory);

  const addAlgoParameter = () => {
    const paramName = prompt('Parameter name (e.g., amplitude, frequency):');
    if (paramName && paramName.trim()) {
      const paramValue = prompt('Default value:');
      setNewAlgoParams(prev => ({
        ...prev,
        [paramName.trim()]: paramValue || 0
      }));
    }
  };

  const removeAlgoParameter = (key: string) => {
    const newParams = { ...newAlgoParams };
    delete newParams[key];
    setNewAlgoParams(newParams);
  };

  const saveAlgorithm = async () => {
    if (!newAlgoName.trim()) {
      alert('Please enter an algorithm name');
      return;
    }
    
    if (!newAlgoCode.trim()) {
      alert('Please enter the algorithm code');
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/algorithms/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newAlgoName,
          category: newAlgoCategory,
          description: newAlgoDescription,
          code: newAlgoCode,
          parameters: newAlgoParams
        })
      });

      if (response.ok) {
        alert('Algorithm created successfully!');
        // Reset form
        setNewAlgoName('');
        setNewAlgoCategory('basic');
        setNewAlgoDescription('');
        setNewAlgoCode('');
        setNewAlgoParams({});
        // Refresh algorithms list
        fetchAlgorithms();
        setActiveTab('browse');
      } else {
        const error = await response.text();
        alert(`Failed to create algorithm: ${error}`);
      }
    } catch (err) {
      console.error('Failed to create algorithm:', err);
      alert('Error creating algorithm');
    }
  };

  return (
    <div className="min-h-screen bg-black text-[#00ff99] font-mono p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-3xl font-bold text-[#00ff99] glow-text">Algorithm Library & Node Creator</h1>
          <Link 
            href="/"
            className="px-4 py-2 bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-sm"
          >
            ← Back to App
          </Link>
        </div>
        <p className="text-sm text-[#00cc77]">
          Browse algorithms, create custom algorithms, or design custom nodes for your simulation.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="flex gap-2 border-b border-[#00cc77]">
          <button
            onClick={() => setActiveTab('browse')}
            className={`px-4 py-2 text-sm font-semibold transition ${
              activeTab === 'browse'
                ? 'text-[#00ff99] border-b-2 border-[#00ff99]'
                : 'text-[#00cc77] hover:text-[#00ff99]'
            }`}
          >
            Browse Algorithms
          </button>
          <button
            onClick={() => setActiveTab('createAlgo')}
            className={`px-4 py-2 text-sm font-semibold transition ${
              activeTab === 'createAlgo'
                ? 'text-[#00ff99] border-b-2 border-[#00ff99]'
                : 'text-[#00cc77] hover:text-[#00ff99]'
            }`}
          >
            Create Algorithm
          </button>
          <button
            onClick={() => setActiveTab('createNode')}
            className={`px-4 py-2 text-sm font-semibold transition ${
              activeTab === 'createNode'
                ? 'text-[#00ff99] border-b-2 border-[#00ff99]'
                : 'text-[#00cc77] hover:text-[#00ff99]'
            }`}
          >
            Create Node
          </button>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'browse' && (
        <>
          {/* Category Filter */}
          <div className="max-w-7xl mx-auto mb-6">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs text-[#00cc77] uppercase tracking-wide">Filter by Category:</span>
              {categories.map(cat => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-3 py-1 text-xs rounded transition ${
                    selectedCategory === cat
                      ? 'bg-[#00cc77] text-black'
                      : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] hover:bg-[#00cc77]/20'
                  }`}
                >
                  {cat.replace(/_/g, ' ').toUpperCase()}
                </button>
              ))}
            </div>
          </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto">
        {loading ? (
          <div className="text-center py-12">
            <div className="text-[#00cc77] text-sm">Loading algorithms...</div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Algorithm List */}
            <div className="space-y-3">
              <h2 className="text-lg font-semibold text-[#00ff99] mb-3">
                {filteredAlgorithms.length} Algorithm{filteredAlgorithms.length !== 1 ? 's' : ''}
              </h2>
              <div className="space-y-2">
                {filteredAlgorithms.map(algo => (
                  <button
                    key={algo.id}
                    onClick={() => fetchAlgorithmDetails(algo.id)}
                    className={`w-full text-left p-4 rounded border transition ${
                      selectedAlgorithm?.id === algo.id
                        ? 'bg-[#00cc77]/20 border-[#00ff99]'
                        : 'bg-black/60 border-[#00cc77] hover:bg-[#00cc77]/10'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="font-semibold text-[#00ff99] mb-1">{algo.name}</div>
                        <div className="text-xs text-[#00cc77] mb-2">
                          Category: {algo.category?.replace(/_/g, ' ').toUpperCase() || 'N/A'}
                        </div>
                        <div className="text-xs text-[#00cc77]/80 line-clamp-2">
                          {algo.description || 'No description available'}
                        </div>
                      </div>
                      <div className="ml-2 text-[#00cc77]">→</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Algorithm Details */}
            <div className="lg:sticky lg:top-6 self-start">
              {selectedAlgorithm ? (
                <div className="bg-black/60 border border-[#00cc77] rounded p-6 shadow-glow">
                  <h2 className="text-xl font-bold text-[#00ff99] mb-2">
                    {selectedAlgorithm.name}
                  </h2>
                  <div className="text-xs text-[#00cc77] mb-4">
                    ID: <span className="font-mono">{selectedAlgorithm.id}</span>
                  </div>
                  <div className="text-xs text-[#00cc77] mb-4">
                    Category: <span className="font-semibold">{selectedAlgorithm.category?.replace(/_/g, ' ').toUpperCase() || 'N/A'}</span>
                  </div>
                  
                  {selectedAlgorithm.description && (
                    <div className="mb-6">
                      <h3 className="text-sm font-semibold text-[#00ff99] mb-2">Description</h3>
                      <p className="text-xs text-[#00cc77]/90">{selectedAlgorithm.description}</p>
                    </div>
                  )}
                  
                  {selectedAlgorithm.parameters && Object.keys(selectedAlgorithm.parameters).length > 0 && (
                    <div>
                      <h3 className="text-sm font-semibold text-[#00ff99] mb-3">Parameters</h3>
                      <div className="space-y-3">
                        {Object.entries(selectedAlgorithm.parameters).map(([key, value]) => (
                          <div key={key} className="bg-black/60 border border-[#00cc77]/50 rounded p-3">
                            <div className="font-mono text-xs text-[#00ff99] mb-1">{key}</div>
                            <div className="text-xs text-[#00cc77]/80">
                              {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {(!selectedAlgorithm.parameters || Object.keys(selectedAlgorithm.parameters).length === 0) && (
                    <div className="text-xs text-[#00cc77]/60 italic">
                      No parameters defined for this algorithm.
                    </div>
                  )}
                </div>
              ) : (
                <div className="bg-black/60 border border-[#00cc77] rounded p-12 text-center shadow-glow">
                  <div className="text-[#00cc77]/60 text-sm">
                    Select an algorithm to view details
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
        </>
      )}

      {/* Create Algorithm Tab */}
      {activeTab === 'createAlgo' && (
        <div className="max-w-4xl mx-auto">
          <div className="bg-black/60 border border-[#00cc77] rounded-lg p-6">
            <h2 className="text-xl font-semibold text-[#00ff99] mb-6">Create Custom Algorithm</h2>
            
            <div className="space-y-6">
              <div>
                <label className="text-sm text-[#00cc77] block mb-2">Algorithm Name *</label>
                <input
                  type="text"
                  value={newAlgoName}
                  onChange={(e) => setNewAlgoName(e.target.value)}
                  placeholder="e.g., Custom Wave Transform"
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-sm"
                />
              </div>

              <div>
                <label className="text-sm text-[#00cc77] block mb-2">Category *</label>
                <select 
                  value={newAlgoCategory}
                  onChange={(e) => setNewAlgoCategory(e.target.value)}
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-sm"
                >
                  <option value="basic">Basic</option>
                  <option value="audio">Audio/Signal</option>
                  <option value="photonic">Photonic</option>
                  <option value="neural">Neural</option>
                </select>
              </div>

              <div>
                <label className="text-sm text-[#00cc77] block mb-2">Description</label>
                <textarea
                  value={newAlgoDescription}
                  onChange={(e) => setNewAlgoDescription(e.target.value)}
                  placeholder="Describe what this algorithm does..."
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-sm h-24"
                />
              </div>

              <div>
                <label className="text-sm text-[#00cc77] block mb-2">Algorithm Code * (Python/JAX)</label>
                <textarea
                  value={newAlgoCode}
                  onChange={(e) => setNewAlgoCode(e.target.value)}
                  placeholder={"def process(x, params):\n    # Your algorithm code here\n    # x: input state (array)\n    # params: dict of parameters\n    return x * params.get('amplitude', 1.0)"}
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-sm h-48 font-mono"
                  spellCheck={false}
                />
                <div className="text-xs text-[#00cc77]/70 mt-1">
                  Write a Python function that processes input state and returns output
                </div>
              </div>

              <div>
                <label className="text-sm text-[#00cc77] block mb-3">Parameters</label>
                <div className="bg-black/60 border border-[#00cc77]/50 rounded p-4">
                  {Object.keys(newAlgoParams).length === 0 ? (
                    <p className="text-xs text-[#00cc77]/70 mb-4">
                      Define custom parameters for your algorithm. These will be configurable when the algorithm is used in a node.
                    </p>
                  ) : (
                    <div className="space-y-2 mb-4">
                      {Object.entries(newAlgoParams).map(([key, value]) => (
                        <div key={key} className="flex items-center justify-between bg-black/60 border border-[#00cc77] rounded p-2">
                          <div className="flex items-center gap-3">
                            <span className="text-xs text-[#00ff99] font-mono">{key}</span>
                            <span className="text-xs text-[#00cc77]">=</span>
                            <span className="text-xs text-[#00ff99]">{String(value)}</span>
                          </div>
                          <button
                            onClick={() => removeAlgoParameter(key)}
                            className="text-xs text-[#f85149] hover:text-[#ff6b6b] px-2"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                  <button 
                    onClick={addAlgoParameter}
                    className="px-4 py-2 bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition text-sm font-semibold"
                  >
                    + Add Parameter
                  </button>
                </div>
              </div>

              <div className="flex justify-end gap-3 pt-4">
                <button 
                  onClick={() => setActiveTab('browse')}
                  className="px-6 py-2 bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-sm"
                >
                  Cancel
                </button>
                <button 
                  onClick={saveAlgorithm}
                  className="px-6 py-2 bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition text-sm font-semibold"
                >
                  Save Algorithm
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Create Node Tab */}
      {activeTab === 'createNode' && (
        <div className="max-w-7xl mx-auto -mt-4">
          <NodeCreatorPanel onClose={() => setActiveTab('browse')} />
        </div>
      )}

      <style jsx global>{`
        .glow-text {
          text-shadow: 0 0 10px rgba(0, 255, 153, 0.5);
        }
        .shadow-glow {
          box-shadow: 0 0 20px rgba(0, 255, 153, 0.2);
        }
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </div>
  );
}

