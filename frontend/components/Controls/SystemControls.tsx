'use client';

import React, { useState } from 'react';
import { ModulationMatrixPanel } from '../Modulation/ModulationMatrixPanel';
import { PluginManagerPanel } from '../Plugins/PluginManagerPanel';
import { ExternalModelPanel } from '../ML/ExternalModelPanel';
import { SessionManagerPanel } from '../Session/SessionManagerPanel';
import { CustomSelect } from '../UI/CustomSelect';
import {
  HeterogeneousNodePanel,
  ConditionalSamplingPanel,
  HigherOrderInteractionsPanel,
  EnergyFactorsPanel
} from '../Panels/THRMLAdvancedPanels';

export const SystemControls: React.FC = () => {
  const [activeSection, setActiveSection] = useState<'system' | 'thrml' | 'audio' | 'advanced'>('system');
  const [showModulation, setShowModulation] = useState(false);
  const [showPlugins, setShowPlugins] = useState(false);
  const [showModels, setShowModels] = useState(false);
  const [showSessions, setShowSessions] = useState(false);
  const [showHeterogeneousNodes, setShowHeterogeneousNodes] = useState(false);
  const [showConditionalSampling, setShowConditionalSampling] = useState(false);
  const [showHigherOrderInteractions, setShowHigherOrderInteractions] = useState(false);
  const [showEnergyFactors, setShowEnergyFactors] = useState(false);
  
  // System state
  const [simSpeed, setSimSpeed] = useState(1.0);
  const [timeStep, setTimeStep] = useState(0.01);
  const [multiGPU, setMultiGPU] = useState(false);
  const [jitEnabled, setJitEnabled] = useState(true);
  
  // THRML state
  const [temperature, setTemperature] = useState(1.0);
  const [gibbsSteps, setGibbsSteps] = useState(5);
  const [cdkSteps, setCdkSteps] = useState(1);
  const [perfMode, setPerfMode] = useState<'speed' | 'accuracy' | 'research'>('speed');
  const [thrmlEnabled, setThrmlEnabled] = useState(true);
  const [learningRate, setLearningRate] = useState(0.01);
  const [updateFreq, setUpdateFreq] = useState(10);
  const [thrmlNodes, setThrmlNodes] = useState(64);
  const [thrmlActivity, setThrmlActivity] = useState(0); // 0-100 percentage
  const [currentEnergy, setCurrentEnergy] = useState<number | null>(null);
  
  // Audio state
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [audioInputGain, setAudioInputGain] = useState(1.0);
  const [audioOutputVolume, setAudioOutputVolume] = useState(0.7);
  const [audioInputDevice, setAudioInputDevice] = useState('default');
  const [audioSmoothing, setAudioSmoothing] = useState(0.8);
  const [audioFFTSize, setAudioFFTSize] = useState(2048);
  const [audioLowCut, setAudioLowCut] = useState(20);
  const [audioHighCut, setAudioHighCut] = useState(20000);
  
  // Simulation state
  const [simRunning, setSimRunning] = useState(false); // Start stopped by default
  const [activeNodes, setActiveNodes] = useState(0);
  const [connections, setConnections] = useState(0);
  
  // System health state
  const [systemHealth, setSystemHealth] = useState<{
    cpu_percent: number;
    memory_percent: number;
    gpu_memory_used_gb: number;
    gpu_memory_total_gb: number;
    uptime_seconds: number;
  } | null>(null);
  
  // Fetch simulation status on mount
  React.useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/simulation/status');
        const data = await response.json();
        setSimRunning(data.running);
        setActiveNodes(data.active_nodes || 0);
      } catch (err) {
        console.error('Failed to fetch simulation status:', err);
      }
    };
    
    fetchStatus();
    const interval = setInterval(fetchStatus, 1000); // Update every second
    return () => clearInterval(interval);
  }, []);
  
  // Simulate THRML activity meter (in production, this would come from backend)
  React.useEffect(() => {
    if (!thrmlEnabled || !simRunning) {
      setThrmlActivity(0);
      return;
    }
    
    const updateActivity = () => {
      // Simulate activity based on Gibbs steps and temperature
      // Higher Gibbs steps and temperature = more activity
      const baseActivity = (gibbsSteps / 100) * 100;
      const tempFactor = Math.min(temperature / 2, 1);
      const randomVariation = Math.random() * 20 - 10;
      const activity = Math.max(0, Math.min(100, baseActivity * tempFactor + randomVariation));
      setThrmlActivity(Math.round(activity));
    };
    
    updateActivity();
    const interval = setInterval(updateActivity, 500); // Update twice per second
    return () => clearInterval(interval);
  }, [thrmlEnabled, simRunning, gibbsSteps, temperature]);
  
  // Fetch system health data
  React.useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        const data = await response.json();
        if (data.system && data.gpu) {
          setSystemHealth({
            cpu_percent: data.system.cpu_percent || 0,
            memory_percent: data.system.memory_percent || 0,
            gpu_memory_used_gb: data.gpu.memory_used_gb || 0,
            gpu_memory_total_gb: data.gpu.memory_total_gb || 0,
            uptime_seconds: data.uptime_seconds || 0
          });
        }
      } catch (err) {
        console.error('Failed to fetch system health:', err);
      }
    };
    
    fetchHealth();
    const interval = setInterval(fetchHealth, 2000); // Update every 2 seconds
    return () => clearInterval(interval);
  }, []);
  
  // Fetch THRML energy data
  React.useEffect(() => {
    const fetchEnergy = async () => {
      try {
        const response = await fetch('http://localhost:8000/thrml/energy');
        const data = await response.json();
        
        // Check for energy in multiple possible response formats
        if (data.energy !== undefined) {
          setCurrentEnergy(data.energy);
        } else if (data.current_energy !== undefined) {
          setCurrentEnergy(data.current_energy);
        } else if (data.detail) {
          // Backend returned an error message (like "not initialized")
          console.log('THRML not initialized:', data.detail);
          setCurrentEnergy(null);
        }
      } catch (err) {
        console.error('Failed to fetch THRML energy:', err);
        setCurrentEnergy(null);
      }
    };
    
    // Fetch immediately
    fetchEnergy();
    
    // Then fetch every 500ms
    const interval = setInterval(fetchEnergy, 500);
    return () => clearInterval(interval);
  }, []); // Remove dependencies - always fetch
  
  // Multi-GPU state
  const [availableGPUs, setAvailableGPUs] = useState<Array<{id: number, name: string, memory: string}>>([
    { id: 0, name: 'NVIDIA RTX 4090', memory: '24GB' },
    { id: 1, name: 'NVIDIA RTX 4080', memory: '16GB' }
  ]);
  const [selectedGPUs, setSelectedGPUs] = useState<number[]>([0]);
  const [gpuLoadBalancing, setGpuLoadBalancing] = useState<'auto' | 'manual'>('auto');
  const [showGPUConfig, setShowGPUConfig] = useState(false);
  
  // Handlers
  const handleStartSimulation = async () => {
    try {
      const response = await fetch('http://localhost:8000/simulation/start', {
        method: 'POST'
      });
      const data = await response.json();
      setSimRunning(data.running);
    } catch (err) {
      console.error('Failed to start simulation:', err);
    }
  };
  
  const handleStopSimulation = async () => {
    try {
      const response = await fetch('http://localhost:8000/simulation/stop', {
        method: 'POST'
      });
      const data = await response.json();
      setSimRunning(data.running);
    } catch (err) {
      console.error('Failed to stop simulation:', err);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[#00cc77] bg-black/60">
        <h2 className="text-sm font-semibold text-[#00ff99] uppercase tracking-wide">System Controls</h2>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-[#00cc77] bg-black/60">
        <button
          onClick={() => setActiveSection('system')}
          className={`flex-1 px-3 py-2 text-xs font-semibold transition ${
            activeSection === 'system'
              ? 'text-[#00ff99] border-b-2 border-[#00ff99]'
              : 'text-[#00cc77] hover:text-[#00ff99]'
          }`}
        >
          System
        </button>
        <button
          onClick={() => setActiveSection('thrml')}
          className={`flex-1 px-3 py-2 text-xs font-semibold transition ${
            activeSection === 'thrml'
              ? 'text-[#00ff99] border-b-2 border-[#00ff99]'
              : 'text-[#00cc77] hover:text-[#00ff99]'
          }`}
        >
          THRML
        </button>
        <button
          onClick={() => setActiveSection('audio')}
          className={`flex-1 px-3 py-2 text-xs font-semibold transition ${
            activeSection === 'audio'
              ? 'text-[#00ff99] border-b-2 border-[#00ff99]'
              : 'text-[#00cc77] hover:text-[#00ff99]'
          }`}
        >
          Audio
        </button>
        <button
          onClick={() => setActiveSection('advanced')}
          className={`flex-1 px-3 py-2 text-xs font-semibold transition ${
            activeSection === 'advanced'
              ? 'text-[#00ff99] border-b-2 border-[#00ff99]'
              : 'text-[#00cc77] hover:text-[#00ff99]'
          }`}
        >
          Advanced
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
        {activeSection === 'system' && (
          <>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-[#00cc77] uppercase tracking-wide">Simulation Speed</label>
                <span className="text-xs text-[#00ff99] font-mono">{simSpeed.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={simSpeed}
                onChange={(e) => setSimSpeed(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-[#00cc77]">
                <span>0.1x</span>
                <span>1.0x</span>
                <span>2.0x</span>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-[#00cc77] uppercase tracking-wide">Integration Method</label>
              <CustomSelect
                value="rk4"
                onChange={() => {}}
                options={[
                  { value: 'rk4', label: 'RK4 (Runge-Kutta 4th Order)' },
                  { value: 'euler', label: 'Euler' },
                  { value: 'rk2', label: 'RK2 (Midpoint)' }
                ]}
                className="w-full"
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs text-[#00cc77] uppercase tracking-wide">Time Step (dt)</label>
              <input
                type="number"
                value={timeStep}
                onChange={(e) => setTimeStep(parseFloat(e.target.value))}
                step="0.001"
                className="w-full"
              />
            </div>

            <div className="border-t border-[#00cc77] pt-3 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-[#00cc77]">Multi-GPU</span>
                <div className="flex items-center gap-2">
                  <button 
                    onClick={() => setMultiGPU(!multiGPU)}
                    className={`px-2 py-1 text-xs rounded transition ${
                      multiGPU 
                        ? 'bg-[#00cc77] text-white' 
                        : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77]'
                    }`}
                  >
                    {multiGPU ? 'ON' : 'OFF'}
                  </button>
                  {multiGPU && (
                    <button
                      onClick={() => setShowGPUConfig(!showGPUConfig)}
                      className="px-2 py-1 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
                    >
                      Configure
                    </button>
                  )}
                </div>
              </div>
              
              {/* GPU Configuration Panel */}
              {multiGPU && showGPUConfig && (
                <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-3 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-[#00ff99]">GPU Configuration</span>
                    <button
                      onClick={() => setShowGPUConfig(false)}
                      className="text-[#00cc77] hover:text-[#00ff99] text-lg leading-none"
                    >
                      ×
                    </button>
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-xs text-[#00cc77] uppercase tracking-wide">Load Balancing</label>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={() => setGpuLoadBalancing('auto')}
                        className={`px-2 py-1 text-xs rounded transition ${
                          gpuLoadBalancing === 'auto'
                            ? 'bg-[#00ff99] text-white'
                            : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99]'
                        }`}
                      >
                        Auto
                      </button>
                      <button
                        onClick={() => setGpuLoadBalancing('manual')}
                        className={`px-2 py-1 text-xs rounded transition ${
                          gpuLoadBalancing === 'manual'
                            ? 'bg-[#00ff99] text-white'
                            : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99]'
                        }`}
                      >
                        Manual
                      </button>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-xs text-[#00cc77] uppercase tracking-wide">
                      Available GPUs ({selectedGPUs.length} selected)
                    </label>
                    <div className="space-y-2">
                      {availableGPUs.map((gpu) => (
                        <div
                          key={gpu.id}
                          className={`p-2 rounded border transition cursor-pointer ${
                            selectedGPUs.includes(gpu.id)
                              ? 'bg-[#00ff99]/10 border-[#00ff99]'
                              : 'bg-black/60 border-[#00cc77] hover:border-[#00ff99]'
                          }`}
                          onClick={() => {
                            if (selectedGPUs.includes(gpu.id)) {
                              setSelectedGPUs(selectedGPUs.filter(id => id !== gpu.id));
                            } else {
                              setSelectedGPUs([...selectedGPUs, gpu.id]);
                            }
                          }}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <input
                                type="checkbox"
                                checked={selectedGPUs.includes(gpu.id)}
                                onChange={() => {}}
                                className="w-3 h-3"
                              />
                              <div>
                                <div className="text-xs font-semibold text-[#00ff99]">
                                  GPU {gpu.id}: {gpu.name}
                                </div>
                                <div className="text-[10px] text-[#00cc77]">{gpu.memory} VRAM</div>
                              </div>
                            </div>
                            <div className={`w-2 h-2 rounded-full ${
                              selectedGPUs.includes(gpu.id) ? 'bg-[#00ff99] animate-pulse' : 'bg-gray-500'
                            }`} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="text-[10px] text-[#00cc77] bg-black/60 p-2 rounded">
                    <div className="font-semibold mb-1">JAX pmap Distribution</div>
                    <div>Workload will be sharded across {selectedGPUs.length} GPU{selectedGPUs.length !== 1 ? 's' : ''}</div>
                    {gpuLoadBalancing === 'auto' && (
                      <div className="mt-1">Auto mode: JAX handles distribution automatically</div>
                    )}
                  </div>
                </div>
              )}
              
              <div className="flex items-center justify-between">
                <span className="text-xs text-[#00cc77]">JIT Compilation</span>
                <button 
                  onClick={() => setJitEnabled(!jitEnabled)}
                  className={`px-2 py-1 text-xs rounded transition ${
                    jitEnabled 
                      ? 'bg-[#00cc77] text-white' 
                      : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77]'
                  }`}
                >
                  {jitEnabled ? 'ON' : 'OFF'}
                </button>
              </div>
            </div>
          </>
        )}

        {activeSection === 'thrml' && (
          <>
            {/* THRML Enable/Disable */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-[#00cc77] uppercase tracking-wide font-semibold">THRML System</span>
                <button 
                  onClick={() => setThrmlEnabled(!thrmlEnabled)}
                  className={`px-3 py-1 text-xs rounded transition ${
                    thrmlEnabled 
                      ? 'bg-[#00cc77] text-white' 
                      : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77]'
                  }`}
                >
                  {thrmlEnabled ? 'ENABLED' : 'DISABLED'}
                </button>
              </div>
              
              {/* Activity Meter */}
              {thrmlEnabled && (
                <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-2">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-[#00cc77]">Activity</span>
                    <span className="text-xs text-[#00ff99] font-mono">{thrmlActivity}%</span>
                  </div>
                  <div className="h-2 bg-black/60 rounded overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-[#58a6ff] to-[#00ff99] transition-all duration-300"
                      style={{ width: `${thrmlActivity}%` }}
                    />
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <div className={`w-2 h-2 rounded-full ${thrmlActivity > 10 ? 'bg-[#00ff99] animate-pulse' : 'bg-gray-500'}`} />
                    <span className="text-xs text-[#00cc77]">
                      {thrmlActivity > 50 ? 'High Activity' : thrmlActivity > 10 ? 'Active' : 'Idle'}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {thrmlEnabled && (
              <>
                {/* Performance Mode */}
                <div className="space-y-2">
                  <label className="text-xs text-[#00cc77] uppercase tracking-wide">Performance Mode</label>
                  <div className="grid grid-cols-3 gap-2">
                    <button 
                      onClick={() => setPerfMode('speed')}
                      className={`px-2 py-1 text-xs rounded transition ${
                        perfMode === 'speed' 
                          ? 'bg-[#00ff99] text-white' 
                          : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99]'
                      }`}
                    >
                      Speed
                    </button>
                    <button 
                      onClick={() => setPerfMode('accuracy')}
                      className={`px-2 py-1 text-xs rounded transition ${
                        perfMode === 'accuracy' 
                          ? 'bg-[#00ff99] text-white' 
                          : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99]'
                      }`}
                    >
                      Accuracy
                    </button>
                    <button 
                      onClick={() => setPerfMode('research')}
                      className={`px-2 py-1 text-xs rounded transition ${
                        perfMode === 'research' 
                          ? 'bg-[#00ff99] text-white' 
                          : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99]'
                      }`}
                    >
                      Research
                    </button>
                  </div>
                  <div className="text-xs text-[#00cc77]">
                    {perfMode === 'speed' && 'Optimized for real-time visualization'}
                    {perfMode === 'accuracy' && 'Balanced for production applications'}
                    {perfMode === 'research' && 'High-quality for scientific experiments'}
                  </div>
                </div>

                {/* Model Configuration */}
                <div className="border-t border-[#00cc77] pt-3 space-y-2">
                  <div className="text-xs font-semibold text-[#00ff99] uppercase tracking-wide">
                    Model Configuration
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-xs text-[#00cc77]">Node Count</label>
                      <span className="text-xs text-[#00ff99] font-mono">{thrmlNodes}</span>
                    </div>
                    <input
                      type="range"
                      min="16"
                      max="256"
                      step="16"
                      value={thrmlNodes}
                      onChange={(e) => setThrmlNodes(parseInt(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-[#00cc77]">
                      <span>16</span>
                      <span>128</span>
                      <span>256</span>
                    </div>
                  </div>
                </div>

                {/* Sampling Parameters */}
                <div className="border-t border-[#00cc77] pt-3 space-y-3">
                  <div className="text-xs font-semibold text-[#00ff99] uppercase tracking-wide">
                    Sampling Parameters
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-xs text-[#00cc77]">Temperature</label>
                      <span className="text-xs text-[#00ff99] font-mono">{temperature.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="5.0"
                      step="0.1"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs">
                      <span className="text-[#00cc77]">Cold (Det.)</span>
                      <span className="text-[#00cc77]">
                        {temperature < 1.0 ? 'Deterministic' : temperature > 2.0 ? 'Stochastic' : 'Balanced'}
                      </span>
                      <span className="text-[#00cc77]">Hot (Stoch.)</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs text-[#00cc77]">Gibbs Steps</label>
                    <input
                      type="number"
                      value={gibbsSteps}
                      onChange={(e) => setGibbsSteps(parseInt(e.target.value))}
                      min="1"
                      max="100"
                      className="w-full bg-black/60 backdrop-blur-md border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs"
                    />
                    <div className="text-xs text-[#00cc77]">Iterations per sample</div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs text-[#00cc77]">CD-k Steps</label>
                    <input
                      type="number"
                      value={cdkSteps}
                      onChange={(e) => setCdkSteps(parseInt(e.target.value))}
                      min="1"
                      max="10"
                      className="w-full bg-black/60 backdrop-blur-md border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs"
                    />
                    <div className="text-xs text-[#00cc77]">Contrastive Divergence steps</div>
                  </div>
                </div>

                {/* Learning Parameters */}
                <div className="border-t border-[#00cc77] pt-3 space-y-3">
                  <div className="text-xs font-semibold text-[#00ff99] uppercase tracking-wide">
                    Learning Parameters
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-xs text-[#00cc77]">Learning Rate</label>
                      <span className="text-xs text-[#00ff99] font-mono">{learningRate.toFixed(4)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0001"
                      max="0.1"
                      step="0.0001"
                      value={learningRate}
                      onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-[#00cc77]">
                      <span>0.0001</span>
                      <span>0.01</span>
                      <span>0.1</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-xs text-[#00cc77]">Update Frequency</label>
                      <span className="text-xs text-[#00ff99] font-mono">{updateFreq} steps</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="100"
                      step="1"
                      value={updateFreq}
                      onChange={(e) => setUpdateFreq(parseInt(e.target.value))}
                      className="w-full"
                    />
                    <div className="text-xs text-[#00cc77]">Weight updates every {updateFreq} simulation steps</div>
                  </div>
                </div>

                {/* Energy Display */}
                <div className="border-t border-[#00cc77] pt-3">
                  <div className="text-xs text-[#00cc77] mb-2 uppercase tracking-wide">System Energy</div>
                  <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-3">
                    <div className="text-2xl font-mono text-[#f85149] mb-1">
                      {currentEnergy !== null ? currentEnergy.toFixed(3) : '---'}
                    </div>
                    <div className="text-xs text-[#00cc77]">
                      {currentEnergy !== null ? 'Lower energy = more probable states' : 'Waiting for data'}
                    </div>
                  </div>
                </div>

                {/* Advanced THRML Features */}
                <div className="border-t border-[#00cc77] pt-3 space-y-3">
                  <div className="text-xs font-semibold text-[#00ff99] uppercase tracking-wide">
                    Advanced Features
                  </div>

                  {/* Heterogeneous Node Types */}
                  <button
                    onClick={() => setShowHeterogeneousNodes(true)}
                    className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-left"
                  >
                    <div className="font-semibold">Heterogeneous Nodes</div>
                    <div className="text-[10px] text-[#00cc77] mt-1">Mixed spin/continuous/discrete types</div>
                  </button>

                  {/* Conditional Sampling */}
                  <button
                    onClick={() => setShowConditionalSampling(true)}
                    className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-left"
                  >
                    <div className="font-semibold">Conditional Sampling</div>
                    <div className="text-[10px] text-[#00cc77] mt-1">Clamp nodes for targeted generation</div>
                  </button>

                  {/* Higher-Order Interactions */}
                  <button
                    onClick={() => setShowHigherOrderInteractions(true)}
                    className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-left"
                  >
                    <div className="font-semibold">Higher-Order Interactions</div>
                    <div className="text-[10px] text-[#00cc77] mt-1">3-way & 4-way coupling beyond pairwise</div>
                  </button>

                  {/* Custom Energy Factors */}
                  <button
                    onClick={() => setShowEnergyFactors(true)}
                    className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-left"
                  >
                    <div className="font-semibold">Energy Factors</div>
                    <div className="text-[10px] text-[#00cc77] mt-1">Photonic, audio, ML regularization</div>
                  </button>
                </div>
              </>
            )}
          </>
        )}

        {activeSection === 'audio' && (
          <>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-[#00cc77] uppercase tracking-wide">Audio Input</span>
                <button 
                  onClick={() => setAudioEnabled(!audioEnabled)}
                  className={`px-2 py-1 text-xs rounded transition ${
                    audioEnabled 
                      ? 'bg-[#00cc77] text-white' 
                      : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77]'
                  }`}
                >
                  {audioEnabled ? 'ON' : 'OFF'}
                </button>
              </div>
              {audioEnabled && (
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${audioEnabled ? 'bg-[#00ff99] animate-pulse' : 'bg-gray-500'}`} />
                  <span className="text-xs text-[#00cc77]">Monitoring Active</span>
                </div>
              )}
            </div>

            {audioEnabled && (
              <>
                <div className="space-y-2">
                  <label className="text-xs text-[#00cc77] uppercase tracking-wide">Input Device</label>
                  <CustomSelect
                    value={audioInputDevice}
                    onChange={setAudioInputDevice}
                    options={[
                      { value: 'default', label: 'Default Microphone' },
                      { value: 'system', label: 'System Audio (Loopback)' },
                      { value: 'line', label: 'Line In' },
                      { value: 'virtual', label: 'Virtual Audio Cable' }
                    ]}
                    className="w-full"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-[#00cc77] uppercase tracking-wide">Input Gain</label>
                    <span className="text-xs text-[#00ff99] font-mono">{audioInputGain.toFixed(2)}x</span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="4.0"
                    step="0.1"
                    value={audioInputGain}
                    onChange={(e) => setAudioInputGain(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-[#00cc77]">
                    <span>0x (Mute)</span>
                    <span>1x (Unity)</span>
                    <span>4x (Boost)</span>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs text-[#00cc77] uppercase tracking-wide">Output Volume</label>
                    <span className="text-xs text-[#00ff99] font-mono">{Math.round(audioOutputVolume * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.01"
                    value={audioOutputVolume}
                    onChange={(e) => setAudioOutputVolume(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-[#00cc77]">
                    <span>Silent</span>
                    <span>50%</span>
                    <span>100%</span>
                  </div>
                </div>

                <div className="border-t border-[#00cc77] pt-3 space-y-3">
                  <div className="text-xs font-semibold text-[#00ff99] uppercase tracking-wide">
                    Signal Processing
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-xs text-[#00cc77]">Smoothing</label>
                      <span className="text-xs text-[#00ff99] font-mono">{audioSmoothing.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="0.99"
                      step="0.01"
                      value={audioSmoothing}
                      onChange={(e) => setAudioSmoothing(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <div className="text-[10px] text-[#00cc77]">
                      Higher values = smoother, slower response
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs text-[#00cc77]">FFT Size</label>
                    <select
                      value={audioFFTSize}
                      onChange={(e) => setAudioFFTSize(parseInt(e.target.value))}
                      className="w-full bg-black/60 backdrop-blur-md border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs"
                    >
                      <option value="512">512 (Low Latency)</option>
                      <option value="1024">1024 (Balanced)</option>
                      <option value="2048">2048 (High Quality)</option>
                      <option value="4096">4096 (Max Resolution)</option>
                      <option value="8192">8192 (Ultra)</option>
                    </select>
                    <div className="text-[10px] text-[#00cc77]">
                      Frequency resolution: {(44100 / audioFFTSize).toFixed(2)} Hz/bin
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <label className="text-xs text-[#00cc77]">Low Cut (Hz)</label>
                      <input
                        type="number"
                        value={audioLowCut}
                        onChange={(e) => setAudioLowCut(parseInt(e.target.value))}
                        min="20"
                        max="1000"
                        className="w-full bg-black/60 backdrop-blur-md border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs text-[#00cc77]">High Cut (Hz)</label>
                      <input
                        type="number"
                        value={audioHighCut}
                        onChange={(e) => setAudioHighCut(parseInt(e.target.value))}
                        min="1000"
                        max="20000"
                        className="w-full bg-black/60 backdrop-blur-md border border-[#00cc77] text-[#00ff99] rounded px-2 py-1 text-xs"
                      />
                    </div>
                  </div>
                  <div className="text-[10px] text-[#00cc77]">
                    Bandpass filter: {audioLowCut} Hz - {audioHighCut} Hz
                  </div>
                </div>

                <div className="border-t border-[#00cc77] pt-3 space-y-2">
                  <div className="text-xs font-semibold text-[#00ff99] uppercase tracking-wide">
                    Level Meters
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[#00cc77]">RMS Level</span>
                      <span className="text-[#00ff99] font-mono">-12 dB</span>
                    </div>
                    <div className="h-2 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500" style={{width: '60%'}} />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[#00cc77]">Peak Level</span>
                      <span className="text-[#00ff99] font-mono">-6 dB</span>
                    </div>
                    <div className="h-2 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500" style={{width: '80%'}} />
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-[10px] text-center">
                    <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-2">
                      <div className="text-[#00cc77] mb-1">Pitch</div>
                      <div className="text-[#00ff99] font-mono">440 Hz</div>
                    </div>
                    <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-2">
                      <div className="text-[#00cc77] mb-1">Centroid</div>
                      <div className="text-[#00ff99] font-mono">2.1 kHz</div>
                    </div>
                    <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-2">
                      <div className="text-[#00cc77] mb-1">Flux</div>
                      <div className="text-[#00ff99] font-mono">0.42</div>
                    </div>
                  </div>
                </div>
              </>
            )}

            <div className="space-y-2">
              <label className="text-xs text-[#00cc77] uppercase tracking-wide">Pitch Tracking</label>
              <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-2">
                <div className="text-sm font-mono text-[#00ff99]">440.0 Hz</div>
                <div className="text-xs text-[#00cc77] mt-1">A4</div>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-[#00cc77] uppercase tracking-wide">RMS Level</label>
              <div className="h-2 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded overflow-hidden">
                <div className="h-full bg-[#00ff99]" style={{ width: '45%' }} />
              </div>
            </div>
          </>
        )}

        {activeSection === 'advanced' && (
          <>
            <div className="space-y-2">
              <label className="text-xs text-[#00cc77] uppercase tracking-wide">Modulation Matrix</label>
              <button
                onClick={() => setShowModulation(true)}
                className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
              >
                Configure Routes
              </button>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-[#00cc77] uppercase tracking-wide">Plugin System</label>
              <button
                onClick={() => setShowPlugins(true)}
                className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
              >
                Manage Plugins
              </button>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-[#00cc77] uppercase tracking-wide">Session Management</label>
              <button
                onClick={() => setShowSessions(true)}
                className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
              >
                Save / Load Sessions
              </button>
            </div>

            <div className="border-t border-[#00cc77] pt-3 space-y-2">
              <div className="text-xs text-[#00cc77] uppercase tracking-wide mb-2">External Models</div>
              <button
                onClick={() => setShowModels(true)}
                className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-left"
              >
                <div className="font-semibold">Connect ML Models</div>
                <div className="text-[10px] text-[#00cc77] mt-1">PyTorch, TensorFlow, HuggingFace</div>
              </button>
            </div>

            {/* Algorithm Library */}
            <div className="border-t border-[#00cc77] pt-3 space-y-2">
              <div className="text-xs text-[#00cc77] uppercase tracking-wide mb-2">Algorithm Library</div>
              <button
                onClick={() => window.open('/algorithms', '_blank')}
                className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition text-left"
              >
                <div className="font-semibold">Browse All 21 Algorithms</div>
                <div className="text-[10px] text-[#00cc77] mt-1">Basic, Audio/Signal, Photonic categories</div>
              </button>
            </div>

            {/* System Health */}
            <div className="border-t border-[#00cc77] pt-3 space-y-2">
              <div className="text-xs font-semibold text-[#00ff99] uppercase tracking-wide mb-2">
                System Health
              </div>
              
              <div className="space-y-2">
                <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-2">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-[#00cc77]">CPU Usage</span>
                    <span className="text-[#00ff99] font-mono">
                      {systemHealth ? `${systemHealth.cpu_percent.toFixed(1)}%` : '---%'}
                    </span>
                  </div>
                  <div className="h-1.5 bg-black/60 rounded overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-[#00ff99] to-[#58a6ff] transition-all duration-300" 
                      style={{width: systemHealth ? `${systemHealth.cpu_percent}%` : '0%'}} 
                    />
                  </div>
                </div>

                <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-2">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-[#00cc77]">Memory Usage</span>
                    <span className="text-[#00ff99] font-mono">
                      {systemHealth ? `${systemHealth.memory_percent.toFixed(1)}%` : '---%'}
                    </span>
                  </div>
                  <div className="h-1.5 bg-black/60 rounded overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-[#00ff99] to-[#58a6ff] transition-all duration-300" 
                      style={{width: systemHealth ? `${systemHealth.memory_percent}%` : '0%'}} 
                    />
                  </div>
                </div>

                <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-2">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-[#00cc77]">GPU Memory</span>
                    <span className="text-[#00ff99] font-mono">
                      {systemHealth 
                        ? `${systemHealth.gpu_memory_used_gb.toFixed(1)} / ${systemHealth.gpu_memory_total_gb.toFixed(1)} GB`
                        : '--- / --- GB'
                      }
                    </span>
                  </div>
                  <div className="h-1.5 bg-black/60 rounded overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-[#00ff99] to-[#f85149] transition-all duration-300" 
                      style={{
                        width: systemHealth 
                          ? `${(systemHealth.gpu_memory_used_gb / systemHealth.gpu_memory_total_gb) * 100}%` 
                          : '0%'
                      }} 
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[10px]">
                  <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-2 text-center">
                    <div className="text-[#00cc77] mb-1">WebSocket</div>
                    <div className="flex items-center justify-center gap-1">
                      <div className="w-1.5 h-1.5 rounded-full bg-[#00ff99] animate-pulse" />
                      <span className="text-[#00ff99] font-mono">CONNECTED</span>
                    </div>
                  </div>
                  <div className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded p-2 text-center">
                    <div className="text-[#00cc77] mb-1">Uptime</div>
                    <div className="text-[#00ff99] font-mono">
                      {systemHealth 
                        ? (() => {
                            const hours = Math.floor(systemHealth.uptime_seconds / 3600);
                            const minutes = Math.floor((systemHealth.uptime_seconds % 3600) / 60);
                            const seconds = Math.floor(systemHealth.uptime_seconds % 60);
                            return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                          })()
                        : '--:--:--'
                      }
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Modals */}
        {showModulation && <ModulationMatrixPanel onClose={() => setShowModulation(false)} />}
        {showPlugins && <PluginManagerPanel onClose={() => setShowPlugins(false)} />}
        {showModels && <ExternalModelPanel onClose={() => setShowModels(false)} />}
        {showSessions && <SessionManagerPanel onClose={() => setShowSessions(false)} />}
        {showHeterogeneousNodes && <HeterogeneousNodePanel onClose={() => setShowHeterogeneousNodes(false)} />}
        {showConditionalSampling && <ConditionalSamplingPanel onClose={() => setShowConditionalSampling(false)} />}
        {showHigherOrderInteractions && <HigherOrderInteractionsPanel onClose={() => setShowHigherOrderInteractions(false)} />}
        {showEnergyFactors && <EnergyFactorsPanel onClose={() => setShowEnergyFactors(false)} />}
      </div>

      {/* Footer */}
      <div className="border-t border-[#00cc77] bg-black/60 px-4 py-3 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-[#00cc77]">Active Nodes</span>
          <span className="text-[#00ff99] font-mono">{activeNodes}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-[#00cc77]">Connections</span>
          <span className="text-[#00ff99] font-mono">{connections}</span>
        </div>
        <div className="flex items-center gap-2 mt-2">
          <div className={`w-2 h-2 rounded-full ${simRunning ? 'bg-[#00ff99] animate-pulse' : 'bg-gray-500'}`} />
          <span className="text-xs text-[#00cc77] flex-1">
            {simRunning ? 'Running' : 'Stopped'}
          </span>
        </div>
        <div className="flex gap-2">
          <button 
            onClick={handleStartSimulation}
            disabled={simRunning}
            className={`flex-1 px-3 py-2 text-xs rounded transition ${
              simRunning 
                ? 'bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77] cursor-not-allowed'
                : 'bg-[#00cc77] text-black hover:bg-[#00ff99]'
            }`}
          >
            Start Simulation
          </button>
          <button 
            onClick={handleStopSimulation}
            disabled={!simRunning}
            className={`px-3 py-2 text-xs rounded transition ${
              !simRunning
                ? 'bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77] cursor-not-allowed'
                : 'bg-[#00cc77] text-black hover:bg-[#00ff99]'
            }`}
          >
            Stop
          </button>
        </div>
      </div>
    </div>
  );
};

