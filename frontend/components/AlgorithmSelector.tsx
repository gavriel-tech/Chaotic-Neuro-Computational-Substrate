/**
 * Algorithm Selector Component
 * 
 * Allows users to browse and select GMCS algorithms with filtering and search.
 */

'use client';

import { useState, useEffect } from 'react';
import { useSimulationStore } from '@/lib/stores/simulation';

interface Algorithm {
  id: number;
  name: string;
  category: string;
  description: string;
}

interface AlgorithmSelectorProps {
  nodeId?: number;
  onSelect?: (algorithmId: number) => void;
}

export default function AlgorithmSelector({ nodeId, onSelect }: AlgorithmSelectorProps) {
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAlgorithms();
  }, []);

  const fetchAlgorithms = async () => {
    try {
      const response = await fetch('http://localhost:8000/algorithms/list');
      const data = await response.json();
      setAlgorithms(data.algorithms);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch algorithms:', error);
      setLoading(false);
    }
  };

  const filteredAlgorithms = algorithms.filter(algo => {
    const matchesCategory = selectedCategory === 'all' || algo.category === selectedCategory;
    const matchesSearch = algo.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         algo.description.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const categories = ['all', 'basic', 'audio', 'photonic'];

  return (
    <div className="algorithm-selector bg-gray-900 border border-[#00ff99] 500/30 rounded-lg p-4">
      <h3 className="text-[#00ff99] 400 text-lg font-bold mb-4">Algorithm Selector</h3>
      
      {/* Search */}
      <input
        type="text"
        placeholder="Search algorithms..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="w-full bg-gray-800 border border-[#00ff99] 500/30 rounded px-3 py-2 text-[#00ff99] 100 mb-4 focus:border-[#00ff99] 400 focus:outline-none"
      />

      {/* Category Filter */}
      <div className="flex gap-2 mb-4 flex-wrap">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-3 py-1 rounded text-sm font-medium transition-all ${
              selectedCategory === cat
                ? 'bg-[#00ff99] 500 text-gray-900'
                : 'bg-gray-800 text-[#00ff99] 400 border border-[#00ff99] 500/30 hover:border-[#00ff99] 400'
            }`}
          >
            {cat.charAt(0).toUpperCase() + cat.slice(1)}
          </button>
        ))}
      </div>

      {/* Algorithm List */}
      <div className="space-y-2 max-h-96 overflow-y-auto custom-scrollbar">
        {loading ? (
          <div className="text-[#00ff99] 400 text-center py-8">Loading algorithms...</div>
        ) : filteredAlgorithms.length === 0 ? (
          <div className="text-[#00ff99] 400/60 text-center py-8">No algorithms found</div>
        ) : (
          filteredAlgorithms.map(algo => (
            <div
              key={algo.id}
              onClick={() => onSelect?.(algo.id)}
              className="bg-gray-800 border border-[#00ff99] 500/20 rounded p-3 cursor-pointer hover:border-[#00ff99] 400 hover:bg-gray-750 transition-all group"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-[#00ff99] 300 font-medium group-hover:text-[#00ff99] 200">
                  {algo.name}
                </span>
                <span className="text-xs px-2 py-1 rounded bg-[#00ff99] 500/20 text-[#00ff99] 400">
                  {algo.category}
                </span>
              </div>
              <p className="text-[#00ff99] 400/70 text-sm">{algo.description}</p>
            </div>
          ))
        )}
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(6, 182, 212, 0.1);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(6, 182, 212, 0.3);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(6, 182, 212, 0.5);
        }
      `}</style>
    </div>
  );
}

