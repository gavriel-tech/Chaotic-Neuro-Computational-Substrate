/**
 * ML Node Components for GMCS Node Graph
 * 
 * React components for ML nodes in the visual node graph.
 * Each ML model type has its own visual representation.
 */

import React, { useState } from 'react';

// ============================================================================
// Base ML Node Component
// ============================================================================

interface MLNodeProps {
  nodeId: string;
  nodeType: string;
  position: { x: number; y: number };
  onUpdate?: (nodeId: string, data: any) => void;
  data?: any;
}

export const BaseMLNode: React.FC<MLNodeProps> = ({ 
  nodeId, 
  nodeType, 
  position, 
  onUpdate,
  data = {}
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getNodeColor = (type: string) => {
    const colors: Record<string, string> = {
      'transformer': '#3b82f6',
      'diffusion': '#8b5cf6',
      'gan': '#ec4899',
      'supervised': '#10b981',
      'reinforcement': '#f59e0b',
      'differentiable_oscillator': '#ef4444'
    };
    return colors[type] || '#6b7280';
  };

  const getNodeIcon = (type: string) => {
    const icons: Record<string, string> = {
      'transformer': '🤖',
      'diffusion': '🌊',
      'gan': '🎨',
      'supervised': '📊',
      'reinforcement': '🎮',
      'differentiable_oscillator': '⚡'
    };
    return icons[type] || '🔮';
  };

  return (
    <div
      className="ml-node"
      style={{
        position: 'absolute',
        left: position.x,
        top: position.y,
        backgroundColor: getNodeColor(nodeType),
        border: '2px solid rgba(255,255,255,0.3)',
        borderRadius: '12px',
        padding: '12px',
        minWidth: '180px',
        color: 'white',
        fontFamily: 'monospace',
        cursor: 'move',
        boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
        transition: 'all 0.2s'
      }}
      onClick={() => setIsExpanded(!isExpanded)}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ fontSize: '24px' }}>{getNodeIcon(nodeType)}</span>
        <div>
          <div style={{ fontWeight: 'bold', fontSize: '14px' }}>
            {nodeType.toUpperCase()}
          </div>
          <div style={{ fontSize: '10px', opacity: 0.8 }}>
            {nodeId}
          </div>
        </div>
      </div>
      
      {/* Connectors */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px' }}>
        <div 
          className="connector input"
          style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            backgroundColor: 'rgba(255,255,255,0.8)',
            border: '2px solid white'
          }}
          title="Input"
        />
        <div 
          className="connector output"
          style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            backgroundColor: 'rgba(255,255,255,0.8)',
            border: '2px solid white'
          }}
          title="Output"
        />
      </div>

      {/* Expanded details */}
      {isExpanded && data && (
        <div style={{ 
          marginTop: '8px', 
          padding: '8px', 
          backgroundColor: 'rgba(0,0,0,0.2)',
          borderRadius: '6px',
          fontSize: '11px'
        }}>
          {Object.entries(data).map(([key, value]) => (
            <div key={key} style={{ margin: '4px 0' }}>
              <strong>{key}:</strong> {String(value)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Transformer Node
// ============================================================================

export const TransformerNode: React.FC<MLNodeProps> = (props) => {
  return (
    <BaseMLNode 
      {...props}
      nodeType="transformer"
      data={{
        model: props.data?.model || 'BERT',
        hidden_size: props.data?.hidden_size || 768,
        num_layers: props.data?.num_layers || 12,
        ...props.data
      }}
    />
  );
};

// ============================================================================
// Diffusion Node
// ============================================================================

export const DiffusionNode: React.FC<MLNodeProps> = (props) => {
  return (
    <BaseMLNode 
      {...props}
      nodeType="diffusion"
      data={{
        timesteps: props.data?.timesteps || 1000,
        beta_schedule: props.data?.beta_schedule || 'cosine',
        data_shape: props.data?.data_shape || '(1, 256)',
        ...props.data
      }}
    />
  );
};

// ============================================================================
// GAN Node
// ============================================================================

export const GANNode: React.FC<MLNodeProps> = (props) => {
  return (
    <BaseMLNode 
      {...props}
      nodeType="gan"
      data={{
        latent_dim: props.data?.latent_dim || 100,
        output_length: props.data?.output_length || 256,
        loss_type: props.data?.loss_type || 'bce',
        ...props.data
      }}
    />
  );
};

// ============================================================================
// Supervised Learning Node
// ============================================================================

export const SupervisedNode: React.FC<MLNodeProps> = (props) => {
  return (
    <BaseMLNode 
      {...props}
      nodeType="supervised"
      data={{
        architecture: props.data?.architecture || 'MLP',
        input_dim: props.data?.input_dim || 10,
        output_dim: props.data?.output_dim || 1,
        ...props.data
      }}
    />
  );
};

// ============================================================================
// RL Controller Node
// ============================================================================

export const RLControllerNode: React.FC<MLNodeProps> = (props) => {
  return (
    <BaseMLNode 
      {...props}
      nodeType="reinforcement"
      data={{
        algorithm: props.data?.algorithm || 'PPO',
        state_dim: props.data?.state_dim || 3,
        action_dim: props.data?.action_dim || 1,
        ...props.data
      }}
    />
  );
};

// ============================================================================
// Differentiable Oscillator Node
// ============================================================================

export const DifferentiableOscillatorNode: React.FC<MLNodeProps> = (props) => {
  return (
    <BaseMLNode 
      {...props}
      nodeType="differentiable_oscillator"
      data={{
        alpha: props.data?.alpha || 15.6,
        beta: props.data?.beta || 28.0,
        differentiable: true,
        ...props.data
      }}
    />
  );
};

// ============================================================================
// ML Node Factory
// ============================================================================

export const createMLNode = (
  nodeType: string,
  nodeId: string,
  position: { x: number; y: number },
  data?: any
) => {
  const props = { nodeId, nodeType, position, data };

  switch (nodeType) {
    case 'transformer':
      return <TransformerNode {...props} />;
    case 'diffusion':
      return <DiffusionNode {...props} />;
    case 'gan':
      return <GANNode {...props} />;
    case 'supervised':
      return <SupervisedNode {...props} />;
    case 'reinforcement':
      return <RLControllerNode {...props} />;
    case 'differentiable_oscillator':
      return <DifferentiableOscillatorNode {...props} />;
    default:
      return <BaseMLNode {...props} />;
  }
};

// ============================================================================
// ML Node Palette (for adding nodes)
// ============================================================================

export const MLNodePalette: React.FC<{ onAddNode: (type: string) => void }> = ({ onAddNode }) => {
  const nodeTypes = [
    { type: 'transformer', label: '🤖 Transformer', color: '#3b82f6' },
    { type: 'diffusion', label: '🌊 Diffusion', color: '#8b5cf6' },
    { type: 'gan', label: '🎨 GAN', color: '#ec4899' },
    { type: 'supervised', label: '📊 Supervised', color: '#10b981' },
    { type: 'reinforcement', label: '🎮 RL', color: '#f59e0b' },
    { type: 'differentiable_oscillator', label: '⚡ Diff Osc', color: '#ef4444' }
  ];

  return (
    <div style={{
      padding: '16px',
      backgroundColor: 'rgba(0,0,0,0.8)',
      borderRadius: '8px',
      border: '1px solid rgba(255,255,255,0.2)'
    }}>
      <h3 style={{ color: 'white', marginBottom: '12px', fontSize: '14px' }}>
        ML Nodes
      </h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        {nodeTypes.map(({ type, label, color }) => (
          <button
            key={type}
            onClick={() => onAddNode(type)}
            style={{
              padding: '8px',
              backgroundColor: color,
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '12px',
              fontFamily: 'monospace',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'scale(1.05)';
              e.currentTarget.style.boxShadow = '0 4px 8px rgba(0,0,0,0.3)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'scale(1)';
              e.currentTarget.style.boxShadow = 'none';
            }}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
};

