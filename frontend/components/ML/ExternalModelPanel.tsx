'use client';
import { notify } from '../UI/Notification';

import React, { useState } from 'react';

interface ExternalModel {
  id: string;
  name: string;
  framework: 'pytorch' | 'tensorflow' | 'huggingface';
  status: 'connected' | 'disconnected' | 'loading';
  inputShape: string;
  outputShape: string;
  path: string;
}

export const ExternalModelPanel: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [models, setModels] = useState<ExternalModel[]>([]);
  const [showAdd, setShowAdd] = useState(false);
  const [selectedFramework, setSelectedFramework] = useState<'pytorch' | 'tensorflow' | 'huggingface' | null>(null);
  const [modelName, setModelName] = useState('');
  const [modelPath, setModelPath] = useState('');
  const [inputShape, setInputShape] = useState('');
  const [outputShape, setOutputShape] = useState('');

  const connectModel = async () => {
    if (!selectedFramework || !modelName.trim() || !modelPath.trim()) {
      notify.error('Please fill in all fields');
      return;
    }

    const newModel: ExternalModel = {
      id: `model-${Date.now()}`,
      name: modelName,
      framework: selectedFramework,
      status: 'connected',
      inputShape: inputShape || '[1, 64]',
      outputShape: outputShape || '[1, 32]',
      path: modelPath
    };

    try {
      // Save model configuration to backend
      const response = await fetch('http://localhost:8000/ml/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newModel)
      });

      if (response.ok) {
        setModels(prev => [...prev, newModel]);
        notify.error(`${modelName} connected successfully and added to node library!`);
        // Reset form
        setModelName('');
        setModelPath('');
        setInputShape('');
        setOutputShape('');
        setSelectedFramework(null);
        setShowAdd(false);
      } else {
        notify.error('Failed to connect model');
      }
    } catch (err) {
      console.error('Failed to connect model:', err);
      setModels(prev => [...prev, newModel]);
      notify.error(`${modelName} saved locally and added to node library!`);
      setModelName('');
      setModelPath('');
      setInputShape('');
      setOutputShape('');
      setSelectedFramework(null);
      setShowAdd(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="w-[700px] max-h-[85vh] bg-black/60 border border-[#00cc77] rounded-lg shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77]">
          <h2 className="text-sm font-semibold text-[#00ff99]">External ML Models</h2>
          <button onClick={onClose} className="text-[#00cc77] hover:text-[#00ff99] text-xl">×</button>
        </div>

        <div className="p-4 overflow-y-auto max-h-[calc(85vh-120px)] custom-scrollbar">
          <div className="mb-4">
            <button
              onClick={() => setShowAdd(true)}
              className="px-3 py-2 text-xs bg-[#00cc77] text-white rounded hover:bg-[#2ea043] transition font-semibold"
            >
              + Connect Model
            </button>
          </div>

          {/* Framework Options */}
          {showAdd && !selectedFramework && (
            <div className="mb-4 p-4 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded space-y-3">
              <h3 className="text-xs font-semibold text-[#00ff99]">Select Framework</h3>
              
              <button 
                onClick={() => setSelectedFramework('pytorch')}
                className="w-full text-left p-3 bg-[#1a1a1a] border border-[#00cc77] rounded hover:bg-[#00cc77]/20 transition"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold text-[#00ff99]">PyTorch</div>
                    <div className="text-xs text-[#00cc77] mt-1">Connect PyTorch models for feature extraction</div>
                  </div>
                  <div className="text-[#00ff99]">→</div>
                </div>
              </button>

              <button 
                onClick={() => setSelectedFramework('tensorflow')}
                className="w-full text-left p-3 bg-[#1a1a1a] border border-[#00cc77] rounded hover:bg-[#00cc77]/20 transition"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold text-[#00ff99]">TensorFlow</div>
                    <div className="text-xs text-[#00cc77] mt-1">Integrate TensorFlow/Keras models</div>
                  </div>
                  <div className="text-[#00ff99]">→</div>
                </div>
              </button>

              <button 
                onClick={() => setSelectedFramework('huggingface')}
                className="w-full text-left p-3 bg-[#1a1a1a] border border-[#00cc77] rounded hover:bg-[#00cc77]/20 transition"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold text-[#00ff99]">HuggingFace</div>
                    <div className="text-xs text-[#00cc77] mt-1">Use transformer models from HuggingFace Hub</div>
                  </div>
                  <div className="text-[#00ff99]">→</div>
                </div>
              </button>

              <button
                onClick={() => setShowAdd(false)}
                className="px-3 py-1 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
              >
                Cancel
              </button>
            </div>
          )}

          {/* Model Configuration Form */}
          {showAdd && selectedFramework && (
            <div className="mb-4 p-4 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold text-[#00ff99]">Configure {selectedFramework} Model</h3>
                <button
                  onClick={() => setSelectedFramework(null)}
                  className="text-xs text-[#00cc77] hover:text-[#00ff99]"
                >
                  ← Back
                </button>
              </div>
              
              <div>
                <label className="text-xs text-[#00cc77] block mb-2">Model Name *</label>
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="e.g., ResNet50 Feature Extractor"
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-xs"
                />
              </div>

              <div>
                <label className="text-xs text-[#00cc77] block mb-2">Model Path/URL *</label>
                <input
                  type="text"
                  value={modelPath}
                  onChange={(e) => setModelPath(e.target.value)}
                  placeholder={selectedFramework === 'huggingface' ? 'e.g., bert-base-uncased' : 'e.g., /models/resnet50.pth'}
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-xs"
                />
              </div>

              <div>
                <label className="text-xs text-[#00cc77] block mb-2">Input Shape</label>
                <input
                  type="text"
                  value={inputShape}
                  onChange={(e) => setInputShape(e.target.value)}
                  placeholder="e.g., [1, 3, 224, 224] or [1, 64]"
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-xs"
                />
              </div>

              <div>
                <label className="text-xs text-[#00cc77] block mb-2">Output Shape</label>
                <input
                  type="text"
                  value={outputShape}
                  onChange={(e) => setOutputShape(e.target.value)}
                  placeholder="e.g., [1, 1000] or [1, 32]"
                  className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-xs"
                />
              </div>

              <div className="flex justify-end gap-2">
                <button
                  onClick={() => {
                    setSelectedFramework(null);
                    setShowAdd(false);
                  }}
                  className="px-4 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
                >
                  Cancel
                </button>
                <button
                  onClick={connectModel}
                  className="px-4 py-2 text-xs bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition font-semibold"
                >
                  Connect Model
                </button>
              </div>
            </div>
          )}

          {/* Connected Models */}
          {models.length === 0 ? (
            <div className="text-center py-8 text-[#00cc77] text-sm">
              No external models connected
            </div>
          ) : (
            <div className="space-y-2">
              {models.map(model => (
                <div key={model.id} className="p-3 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-semibold text-[#00ff99]">{model.name}</div>
                      <div className="text-xs text-[#00cc77] mt-1">{model.framework}</div>
                    </div>
                    <div className={`px-2 py-1 text-xs rounded ${
                      model.status === 'connected' ? 'bg-[#00cc77]/20 text-[#00ff99]' :
                      model.status === 'loading' ? 'bg-[#d29922]/20 text-[#d29922]' :
                      'bg-[#f85149]/20 text-[#f85149]'
                    }`}>
                      {model.status}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Integration Examples */}
          <div className="mt-6 pt-4 border-t border-[#00cc77]">
            <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-3">
              Use Cases
            </h3>
            <div className="space-y-2 text-xs text-[#00cc77]">
              <div className="flex items-start gap-2">
                <span className="text-[#00ff99]">•</span>
                <span>Feature extraction from chaotic dynamics</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-[#00ff99]">•</span>
                <span>Data augmentation using GMCS-generated patterns</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-[#00ff99]">•</span>
                <span>Hybrid models combining neural networks with EBMs</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-[#00ff99]">•</span>
                <span>Transfer learning with GMCS-modulated embeddings</span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-[#00cc77] bg-black/60">
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

