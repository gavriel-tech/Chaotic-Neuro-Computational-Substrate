/**
 * Model Browser for GMCS
 * 
 * Browse and load pre-trained ML models:
 * - Search HuggingFace Hub
 * - Filter by task/framework
 * - One-click loading
 * - Model previews
 */

import React, { useState } from 'react';

// ============================================================================
// Types
// ============================================================================

interface ModelInfo {
  model_id: string;
  name: string;
  model_type: string;
  framework: string;
  task: string;
  architecture: string;
  parameters: number;
  source: string;
  hub_id?: string;
  description?: string;
  tags?: string[];
  license?: string;
}

// ============================================================================
// Sample Models (would come from model_registry.py via API)
// ============================================================================

const SAMPLE_MODELS: ModelInfo[] = [
  {
    model_id: 'bert-base',
    name: 'BERT Base',
    model_type: 'transformer',
    framework: 'pytorch',
    task: 'embedding',
    architecture: 'bert',
    parameters: 110_000_000,
    source: 'huggingface',
    hub_id: 'bert-base-uncased',
    description: 'Base BERT model for text embeddings',
    tags: ['nlp', 'embedding', 'bert'],
    license: 'apache-2.0'
  },
  {
    model_id: 'gpt2',
    name: 'GPT-2',
    model_type: 'transformer',
    framework: 'pytorch',
    task: 'generation',
    architecture: 'gpt2',
    parameters: 117_000_000,
    source: 'huggingface',
    hub_id: 'gpt2',
    description: 'GPT-2 for text generation',
    tags: ['nlp', 'generation', 'gpt'],
    license: 'mit'
  },
  {
    model_id: 'distilbert',
    name: 'DistilBERT',
    model_type: 'transformer',
    framework: 'pytorch',
    task: 'embedding',
    architecture: 'distilbert',
    parameters: 66_000_000,
    source: 'huggingface',
    hub_id: 'distilbert-base-uncased',
    description: 'Distilled BERT for fast embeddings',
    tags: ['nlp', 'embedding', 'fast'],
    license: 'apache-2.0'
  },
  {
    model_id: 't5-small',
    name: 'T5 Small',
    model_type: 'transformer',
    framework: 'pytorch',
    task: 'seq2seq',
    architecture: 't5',
    parameters: 60_000_000,
    source: 'huggingface',
    hub_id: 't5-small',
    description: 'Small T5 model for sequence-to-sequence',
    tags: ['nlp', 'seq2seq', 't5'],
    license: 'apache-2.0'
  }
];

// ============================================================================
// Model Browser Component
// ============================================================================

export const ModelBrowser: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterTask, setFilterTask] = useState<string>('all');
  const [filterFramework, setFilterFramework] = useState<string>('all');
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);

  // Filter models
  const filteredModels = SAMPLE_MODELS.filter(model => {
    const matchesSearch = searchQuery === '' || 
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesTask = filterTask === 'all' || model.task === filterTask;
    const matchesFramework = filterFramework === 'all' || model.framework === filterFramework;

    return matchesSearch && matchesTask && matchesFramework;
  });

  const handleLoadModel = async (model: ModelInfo) => {
    setLoading(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));
    setLoading(false);
    notify.error(`Model ${model.name} loaded successfully!`);
  };

  return (
    <div style={{
      padding: '20px',
      backgroundColor: 'rgba(0,0,0,0.9)',
      borderRadius: '12px',
      border: '1px solid rgba(255,255,255,0.2)',
      color: 'white',
      fontFamily: 'monospace',
      maxHeight: '90vh',
      overflowY: 'auto'
    }}>
      <h2 style={{ marginBottom: '20px', fontSize: '20px' }}>
        📦 Model Browser
      </h2>

      {/* Search and Filters */}
      <div style={{ marginBottom: '20px' }}>
        <input
          type="text"
          placeholder="🔍 Search models..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            width: '100%',
            padding: '12px',
            backgroundColor: 'rgba(255,255,255,0.1)',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '6px',
            color: 'white',
            fontSize: '14px',
            marginBottom: '12px'
          }}
        />

        <div style={{ display: 'flex', gap: '12px' }}>
          <select
            value={filterTask}
            onChange={(e) => setFilterTask(e.target.value)}
            style={{
              flex: 1,
              padding: '10px',
              backgroundColor: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '6px',
              color: 'white',
              fontSize: '12px'
            }}
          >
            <option value="all">All Tasks</option>
            <option value="embedding">Embedding</option>
            <option value="generation">Generation</option>
            <option value="seq2seq">Seq2Seq</option>
            <option value="classification">Classification</option>
          </select>

          <select
            value={filterFramework}
            onChange={(e) => setFilterFramework(e.target.value)}
            style={{
              flex: 1,
              padding: '10px',
              backgroundColor: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '6px',
              color: 'white',
              fontSize: '12px'
            }}
          >
            <option value="all">All Frameworks</option>
            <option value="pytorch">PyTorch</option>
            <option value="tensorflow">TensorFlow</option>
            <option value="jax">JAX</option>
          </select>
        </div>
      </div>

      {/* Model Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
        gap: '16px',
        marginBottom: '20px'
      }}>
        {filteredModels.map(model => (
          <ModelCard
            key={model.model_id}
            model={model}
            onClick={() => setSelectedModel(model)}
            onLoad={() => handleLoadModel(model)}
            isSelected={selectedModel?.model_id === model.model_id}
          />
        ))}
      </div>

      {filteredModels.length === 0 && (
        <div style={{
          textAlign: 'center',
          padding: '40px',
          opacity: 0.5
        }}>
          No models found. Try different search terms or filters.
        </div>
      )}

      {/* Model Details Panel */}
      {selectedModel && (
        <ModelDetailsPanel
          model={selectedModel}
          onClose={() => setSelectedModel(null)}
          onLoad={() => handleLoadModel(selectedModel)}
          loading={loading}
        />
      )}
    </div>
  );
};

// ============================================================================
// Model Card Component
// ============================================================================

const ModelCard: React.FC<{
  model: ModelInfo;
  onClick: () => void;
  onLoad: () => void;
  isSelected: boolean;
}> = ({ model, onClick, onLoad, isSelected }) => {
  const formatNumber = (num: number) => {
    if (num >= 1_000_000_000) return `${(num / 1_000_000_000).toFixed(1)}B`;
    if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(0)}M`;
    return num.toString();
  };

  return (
    <div
      style={{
        padding: '16px',
        backgroundColor: isSelected ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255,255,255,0.05)',
        border: isSelected ? '2px solid #3b82f6' : '1px solid rgba(255,255,255,0.2)',
        borderRadius: '8px',
        cursor: 'pointer',
        transition: 'all 0.2s'
      }}
      onClick={onClick}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
        <h3 style={{ fontSize: '16px', margin: 0 }}>{model.name}</h3>
        <span style={{
          padding: '2px 8px',
          backgroundColor: getTaskColor(model.task),
          borderRadius: '4px',
          fontSize: '10px'
        }}>
          {model.task}
        </span>
      </div>

      <p style={{ fontSize: '12px', opacity: 0.8, marginBottom: '12px', minHeight: '40px' }}>
        {model.description}
      </p>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: '11px', opacity: 0.7 }}>
          <div>{formatNumber(model.parameters)} params</div>
          <div>{model.framework}</div>
        </div>

        <button
          onClick={(e) => {
            e.stopPropagation();
            onLoad();
          }}
          style={{
            padding: '6px 12px',
            backgroundColor: '#10b981',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '12px'
          }}
        >
          Load
        </button>
      </div>

      {/* Tags */}
      {model.tags && (
        <div style={{ display: 'flex', gap: '4px', marginTop: '8px', flexWrap: 'wrap' }}>
          {model.tags.map(tag => (
            <span
              key={tag}
              style={{
                padding: '2px 6px',
                backgroundColor: 'rgba(255,255,255,0.1)',
                borderRadius: '3px',
                fontSize: '10px'
              }}
            >
              {tag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Model Details Panel
// ============================================================================

const ModelDetailsPanel: React.FC<{
  model: ModelInfo;
  onClose: () => void;
  onLoad: () => void;
  loading: boolean;
}> = ({ model, onClose, onLoad, loading }) => {
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.8)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 1000
    }}>
      <div style={{
        width: '600px',
        maxHeight: '80vh',
        backgroundColor: 'rgba(20,20,20,0.95)',
        border: '2px solid rgba(255,255,255,0.3)',
        borderRadius: '12px',
        padding: '24px',
        color: 'white',
        overflowY: 'auto'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
          <h2 style={{ fontSize: '24px', margin: 0 }}>{model.name}</h2>
          <button
            onClick={onClose}
            style={{
              padding: '8px 12px',
              backgroundColor: 'transparent',
              color: 'white',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            ✕
          </button>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <InfoRow label="Model ID" value={model.model_id} />
          <InfoRow label="Architecture" value={model.architecture} />
          <InfoRow label="Parameters" value={`${(model.parameters / 1_000_000).toFixed(0)}M`} />
          <InfoRow label="Framework" value={model.framework} />
          <InfoRow label="Task" value={model.task} />
          <InfoRow label="Source" value={model.source} />
          {model.hub_id && <InfoRow label="Hub ID" value={model.hub_id} />}
          {model.license && <InfoRow label="License" value={model.license} />}
        </div>

        {model.description && (
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ fontSize: '14px', marginBottom: '8px' }}>Description</h3>
            <p style={{ fontSize: '13px', opacity: 0.8, lineHeight: 1.6 }}>
              {model.description}
            </p>
          </div>
        )}

        <button
          onClick={onLoad}
          disabled={loading}
          style={{
            width: '100%',
            padding: '12px',
            backgroundColor: loading ? '#6b7280' : '#10b981',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: 'bold'
          }}
        >
          {loading ? '⏳ Loading...' : '📥 Load Model'}
        </button>
      </div>
    </div>
  );
};

// ============================================================================
// Helper Components
// ============================================================================

const InfoRow: React.FC<{ label: string; value: string }> = ({ label, value }) => {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      padding: '8px 0',
      borderBottom: '1px solid rgba(255,255,255,0.1)',
      fontSize: '13px'
    }}>
      <span style={{ opacity: 0.7 }}>{label}:</span>
      <span style={{ fontFamily: 'monospace' }}>{value}</span>
    </div>
  );
};

const getTaskColor = (task: string): string => {
  const colors: Record<string, string> = {
    'embedding': '#3b82f6',
    'generation': '#8b5cf6',
    'seq2seq': '#10b981',
    'classification': '#f59e0b'
  };
  return colors[task] || '#6b7280';
};

export default ModelBrowser;

