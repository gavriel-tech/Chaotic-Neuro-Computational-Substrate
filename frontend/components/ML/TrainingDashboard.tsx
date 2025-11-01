/**
 * Training Dashboard for ML Models
 * 
 * Real-time visualization of ML model training:
 * - Loss curves
 * - Gradient magnitudes
 * - Performance metrics
 * - Training controls
 */

import React, { useState, useEffect } from 'react';

// ============================================================================
// Types
// ============================================================================

interface TrainingMetrics {
  epoch: number;
  loss: number;
  val_loss?: number;
  accuracy?: number;
  val_accuracy?: number;
  learning_rate: number;
  gradient_norm?: number;
  timestamp: number;
}

interface TrainingState {
  isTraining: boolean;
  modelId: string;
  totalEpochs: number;
  currentEpoch: number;
  metrics: TrainingMetrics[];
}

// ============================================================================
// Training Dashboard Component
// ============================================================================

export const TrainingDashboard: React.FC = () => {
  const [trainingState, setTrainingState] = useState<TrainingState>({
    isTraining: false,
    modelId: 'model_1',
    totalEpochs: 100,
    currentEpoch: 0,
    metrics: []
  });

  const [config, setConfig] = useState({
    batchSize: 32,
    learningRate: 0.001,
    optimizer: 'adam',
    lossFunction: 'mse'
  });

  // Simulate receiving training updates
  useEffect(() => {
    if (!trainingState.isTraining) return;

    const interval = setInterval(() => {
      setTrainingState(prev => {
        if (prev.currentEpoch >= prev.totalEpochs) {
          return { ...prev, isTraining: false };
        }

        const newMetric: TrainingMetrics = {
          epoch: prev.currentEpoch + 1,
          loss: 1.0 / (prev.currentEpoch + 1) + Math.random() * 0.1,
          val_loss: 1.2 / (prev.currentEpoch + 1) + Math.random() * 0.1,
          accuracy: Math.min(0.95, (prev.currentEpoch + 1) * 0.01),
          val_accuracy: Math.min(0.93, (prev.currentEpoch + 1) * 0.009),
          learning_rate: config.learningRate * Math.pow(0.99, prev.currentEpoch),
          gradient_norm: 1.0 + Math.random(),
          timestamp: Date.now()
        };

        return {
          ...prev,
          currentEpoch: prev.currentEpoch + 1,
          metrics: [...prev.metrics, newMetric]
        };
      });
    }, 100);

    return () => clearInterval(interval);
  }, [trainingState.isTraining, config.learningRate]);

  const startTraining = () => {
    setTrainingState(prev => ({
      ...prev,
      isTraining: true,
      currentEpoch: 0,
      metrics: []
    }));
  };

  const stopTraining = () => {
    setTrainingState(prev => ({ ...prev, isTraining: false }));
  };

  const resetTraining = () => {
    setTrainingState(prev => ({
      ...prev,
      isTraining: false,
      currentEpoch: 0,
      metrics: []
    }));
  };

  // Get latest metrics
  const latestMetrics = trainingState.metrics[trainingState.metrics.length - 1];
  const progress = (trainingState.currentEpoch / trainingState.totalEpochs) * 100;

  return (
    <div style={{
      padding: '20px',
      backgroundColor: 'rgba(0,0,0,0.9)',
      borderRadius: '12px',
      border: '1px solid rgba(255,255,255,0.2)',
      color: 'white',
      fontFamily: 'monospace'
    }}>
      <h2 style={{ marginBottom: '20px', fontSize: '20px' }}>
        🎓 Training Dashboard
      </h2>

      {/* Training Controls */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '12px',
        marginBottom: '20px'
      }}>
        <div>
          <label style={{ fontSize: '12px', opacity: 0.8 }}>Batch Size</label>
          <input
            type="number"
            value={config.batchSize}
            onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
            disabled={trainingState.isTraining}
            style={{
              width: '100%',
              padding: '8px',
              backgroundColor: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '4px',
              color: 'white',
              marginTop: '4px'
            }}
          />
        </div>
        <div>
          <label style={{ fontSize: '12px', opacity: 0.8 }}>Learning Rate</label>
          <input
            type="number"
            step="0.0001"
            value={config.learningRate}
            onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
            disabled={trainingState.isTraining}
            style={{
              width: '100%',
              padding: '8px',
              backgroundColor: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '4px',
              color: 'white',
              marginTop: '4px'
            }}
          />
        </div>
        <div>
          <label style={{ fontSize: '12px', opacity: 0.8 }}>Optimizer</label>
          <select
            value={config.optimizer}
            onChange={(e) => setConfig({ ...config, optimizer: e.target.value })}
            disabled={trainingState.isTraining}
            style={{
              width: '100%',
              padding: '8px',
              backgroundColor: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '4px',
              color: 'white',
              marginTop: '4px'
            }}
          >
            <option value="adam">Adam</option>
            <option value="sgd">SGD</option>
            <option value="rmsprop">RMSprop</option>
          </select>
        </div>
        <div>
          <label style={{ fontSize: '12px', opacity: 0.8 }}>Loss Function</label>
          <select
            value={config.lossFunction}
            onChange={(e) => setConfig({ ...config, lossFunction: e.target.value })}
            disabled={trainingState.isTraining}
            style={{
              width: '100%',
              padding: '8px',
              backgroundColor: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '4px',
              color: 'white',
              marginTop: '4px'
            }}
          >
            <option value="mse">MSE</option>
            <option value="mae">MAE</option>
            <option value="lyapunov">Lyapunov</option>
            <option value="combined">Combined</option>
          </select>
        </div>
      </div>

      {/* Control Buttons */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
        {!trainingState.isTraining ? (
          <button
            onClick={startTraining}
            style={{
              padding: '10px 20px',
              backgroundColor: '#10b981',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            ▶️ Start Training
          </button>
        ) : (
          <button
            onClick={stopTraining}
            style={{
              padding: '10px 20px',
              backgroundColor: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            ⏸️ Stop Training
          </button>
        )}
        <button
          onClick={resetTraining}
          style={{
            padding: '10px 20px',
            backgroundColor: '#6b7280',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          🔄 Reset
        </button>
      </div>

      {/* Progress Bar */}
      <div style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
          <span style={{ fontSize: '14px' }}>
            Epoch {trainingState.currentEpoch} / {trainingState.totalEpochs}
          </span>
          <span style={{ fontSize: '14px' }}>
            {progress.toFixed(1)}%
          </span>
        </div>
        <div style={{
          width: '100%',
          height: '8px',
          backgroundColor: 'rgba(255,255,255,0.1)',
          borderRadius: '4px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${progress}%`,
            height: '100%',
            backgroundColor: trainingState.isTraining ? '#3b82f6' : '#10b981',
            transition: 'width 0.3s'
          }} />
        </div>
      </div>

      {/* Current Metrics */}
      {latestMetrics && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '12px',
          marginBottom: '20px'
        }}>
          <MetricCard label="Loss" value={latestMetrics.loss?.toFixed(4) || 'N/A'} color="#ef4444" />
          <MetricCard label="Val Loss" value={latestMetrics.val_loss?.toFixed(4) || 'N/A'} color="#f59e0b" />
          <MetricCard label="Accuracy" value={`${((latestMetrics.accuracy || 0) * 100).toFixed(1)}%`} color="#10b981" />
          <MetricCard label="Val Acc" value={`${((latestMetrics.val_accuracy || 0) * 100).toFixed(1)}%`} color="#3b82f6" />
          <MetricCard label="LR" value={latestMetrics.learning_rate?.toExponential(2) || 'N/A'} color="#8b5cf6" />
          <MetricCard label="Grad Norm" value={latestMetrics.gradient_norm?.toFixed(3) || 'N/A'} color="#ec4899" />
        </div>
      )}

      {/* Loss Curve (Simple SVG visualization) */}
      <div style={{
        backgroundColor: 'rgba(255,255,255,0.05)',
        borderRadius: '8px',
        padding: '16px',
        marginBottom: '16px'
      }}>
        <h3 style={{ fontSize: '14px', marginBottom: '12px' }}>Loss Curves</h3>
        <LossCurveChart metrics={trainingState.metrics} />
      </div>

      {/* Metrics Table */}
      <div style={{
        backgroundColor: 'rgba(255,255,255,0.05)',
        borderRadius: '8px',
        padding: '16px',
        maxHeight: '200px',
        overflowY: 'auto'
      }}>
        <h3 style={{ fontSize: '14px', marginBottom: '12px' }}>Training History</h3>
        <table style={{ width: '100%', fontSize: '12px' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.2)' }}>
              <th style={{ padding: '8px', textAlign: 'left' }}>Epoch</th>
              <th style={{ padding: '8px', textAlign: 'right' }}>Loss</th>
              <th style={{ padding: '8px', textAlign: 'right' }}>Val Loss</th>
              <th style={{ padding: '8px', textAlign: 'right' }}>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {trainingState.metrics.slice(-10).reverse().map((m) => (
              <tr key={m.epoch} style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                <td style={{ padding: '8px' }}>{m.epoch}</td>
                <td style={{ padding: '8px', textAlign: 'right' }}>{m.loss?.toFixed(4) || '-'}</td>
                <td style={{ padding: '8px', textAlign: 'right' }}>{m.val_loss?.toFixed(4) || '-'}</td>
                <td style={{ padding: '8px', textAlign: 'right' }}>{((m.accuracy || 0) * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ============================================================================
// Metric Card Component
// ============================================================================

const MetricCard: React.FC<{ label: string; value: string; color: string }> = ({ label, value, color }) => {
  return (
    <div style={{
      padding: '12px',
      backgroundColor: 'rgba(255,255,255,0.05)',
      borderRadius: '8px',
      borderLeft: `4px solid ${color}`
    }}>
      <div style={{ fontSize: '11px', opacity: 0.7, marginBottom: '4px' }}>
        {label}
      </div>
      <div style={{ fontSize: '18px', fontWeight: 'bold' }}>
        {value}
      </div>
    </div>
  );
};

// ============================================================================
// Loss Curve Chart (Simple SVG)
// ============================================================================

const LossCurveChart: React.FC<{ metrics: TrainingMetrics[] }> = ({ metrics }) => {
  if (metrics.length === 0) {
    return <div style={{ textAlign: 'center', opacity: 0.5, padding: '40px' }}>
      No data yet. Start training to see loss curves.
    </div>;
  }

  const width = 600;
  const height = 200;
  const padding = 40;

  const maxLoss = Math.max(...metrics.map(m => Math.max(m.loss, m.val_loss || 0)));
  const minLoss = Math.min(...metrics.map(m => Math.min(m.loss, m.val_loss || 0)));

  const xScale = (epoch: number) => padding + (epoch / metrics.length) * (width - 2 * padding);
  const yScale = (loss: number) => height - padding - ((loss - minLoss) / (maxLoss - minLoss + 0.001)) * (height - 2 * padding);

  const lossPath = metrics.map((m, i) => 
    `${i === 0 ? 'M' : 'L'} ${xScale(m.epoch)} ${yScale(m.loss)}`
  ).join(' ');

  const valLossPath = metrics.filter(m => m.val_loss).map((m, i) => 
    `${i === 0 ? 'M' : 'L'} ${xScale(m.epoch)} ${yScale(m.val_loss!)}`
  ).join(' ');

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map((ratio) => (
        <line
          key={ratio}
          x1={padding}
          y1={padding + ratio * (height - 2 * padding)}
          x2={width - padding}
          y2={padding + ratio * (height - 2 * padding)}
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="1"
        />
      ))}

      {/* Loss curve */}
      <path
        d={lossPath}
        fill="none"
        stroke="#ef4444"
        strokeWidth="2"
      />

      {/* Validation loss curve */}
      {valLossPath && (
        <path
          d={valLossPath}
          fill="none"
          stroke="#f59e0b"
          strokeWidth="2"
          strokeDasharray="5,5"
        />
      )}

      {/* Legend */}
      <text x={padding} y={20} fill="white" fontSize="12">
        Loss Curves
      </text>
      <line x1={padding + 80} y1={15} x2={padding + 110} y2={15} stroke="#ef4444" strokeWidth="2" />
      <text x={padding + 115} y={20} fill="white" fontSize="10">Train</text>
      <line x1={padding + 160} y1={15} x2={padding + 190} y2={15} stroke="#f59e0b" strokeWidth="2" strokeDasharray="5,5" />
      <text x={padding + 195} y={20} fill="white" fontSize="10">Val</text>
    </svg>
  );
};

export default TrainingDashboard;

