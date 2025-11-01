'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { NODE_PRESETS, NodePreset, GRAPH_PRESETS, GraphPreset } from './NodePresets';
import {
  OscilloscopeVisualizer,
  SpectrogramVisualizer,
  PhaseSpace3DVisualizer,
  EnergyGraphVisualizer,
  SpinStateMatrixVisualizer,
  CorrelationMatrixVisualizer,
  WaveformMonitorVisualizer,
  XYPlotVisualizer,
  PBitMapperVisualizerWrapper,
} from '../Visualizers/EmbeddedVisualizers';
import { notify } from '../UI/Notification';
import { confirm } from '../UI/ConfirmDialog';
import { useSimulationStore } from '@/lib/stores/simulation';
import { useBackendDataStore } from '@/lib/stores/backendData';

// Import all 16 JSON presets directly
import chaosCrypto from '../../presets/chaos_crypto.json';
import emergentLogic from '../../presets/emergent_logic.json';
import gameCode from '../../presets/game_code.json';
import liveMusic from '../../presets/live_music.json';
import liveVideo from '../../presets/live_video.json';
import molecularDesign from '../../presets/molecular_design.json';
import musicAnalysis from '../../presets/music_analysis.json';
import nasDiscovery from '../../presets/nas_discovery.json';
import neuralSim from '../../presets/neural_sim.json';
import neuromapping from '../../presets/neuromapping.json';
import photonicSim from '../../presets/photonic_sim.json';
import pixelArt from '../../presets/pixel_art.json';
import quantumOpt from '../../presets/quantum_opt.json';
import rlBoost from '../../presets/rl_boost.json';
import solarOpt from '../../presets/solar_opt.json';
import worldGen from '../../presets/world_gen.json';

const JSON_PRESETS = [
  chaosCrypto, emergentLogic, gameCode, liveMusic, liveVideo, molecularDesign,
  musicAnalysis, nasDiscovery, neuralSim, neuromapping, photonicSim, pixelArt,
  quantumOpt, rlBoost, solarOpt, worldGen
] as any[];

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

interface BackendNodePayload {
  node_id: number;
  active: boolean;
  position?: number[];
  oscillator_state?: number[];
  config: Record<string, number>;
  chain: number[];
}

interface NodeResponsePayload {
  status: string;
  node_id?: number;
  message?: string;
  data?: Record<string, unknown>;
}

const extractPosition = (position: number[] | undefined, fallbackX: number, fallbackY: number) => {
  if (Array.isArray(position) && position.length >= 2) {
    const [x, y] = position;
    if (Number.isFinite(x) && Number.isFinite(y)) {
      return { x, y };
    }
  }
  return { x: fallbackX, y: fallbackY };
};

const getDefaultPreset = (): NodePreset => {
  if (Array.isArray(NODE_PRESETS.oscillators) && NODE_PRESETS.oscillators.length > 0) {
    return NODE_PRESETS.oscillators[0];
  }
  if (Array.isArray(NODE_PRESETS.system) && NODE_PRESETS.system.length > 0) {
    return NODE_PRESETS.system[0];
  }
  const categories = Object.values(NODE_PRESETS) as NodePreset[][];
  for (const presets of categories) {
    if (Array.isArray(presets) && presets.length > 0) {
      return presets[0];
    }
  }
  throw new Error('NODE_PRESETS is empty; cannot determine default preset');
};

interface Node {
  id: string;
  preset: NodePreset;
  x: number;
  y: number;
  config: Record<string, any>;
  backendId?: number;
}

interface Connection {
  id: string;
  fromNodeId: string;
  fromPort: string;
  fromPortIndex: number;
  toNodeId: string;
  toPort: string;
  toPortIndex: number;
}

interface DraggingWire {
  fromNodeId: string;
  fromPort: string;
  fromPortIndex: number;
  fromX: number;
  fromY: number;
  currentX: number;
  currentY: number;
  isOutput: boolean;
}

interface DeleteConfirmation {
  connectionId: string;
  x: number;
  y: number;
}

const VisualizerComponents: Record<string, React.FC<any>> = {
  'Oscilloscope': OscilloscopeVisualizer,
  'Spectrogram': SpectrogramVisualizer,
  'Phase Space 3D': PhaseSpace3DVisualizer,
  'Energy Graph': EnergyGraphVisualizer,
  'Spin State Matrix': SpinStateMatrixVisualizer,
  'Correlation Matrix': CorrelationMatrixVisualizer,
  'Waveform Monitor': WaveformMonitorVisualizer,
  'XY Plot': XYPlotVisualizer,
  'P-Bit Mapper': PBitMapperVisualizerWrapper,
};

const STUB_NODE_TYPES = new Set(['ml', 'analysis', 'control', 'generator', 'custom']);

export const NodeGraphSimple: React.FC = () => {
  const [nodes, setNodes] = useState<Node[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('gmcs_nodes');
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch (e) {
          console.error('Failed to parse saved nodes:', e);
        }
      }
    }
    return [];
  });

  const [connections, setConnections] = useState<Connection[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('gmcs_connections');
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch (e) {
          console.error('Failed to parse saved connections:', e);
        }
      }
    }
    return [];
  });
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [showNodeModal, setShowNodeModal] = useState(false);
  const [showGraphPresets, setShowGraphPresets] = useState(false);
  const [backendPresets, setBackendPresets] = useState<any[]>([]);
  const [loadingPresets, setLoadingPresets] = useState(false);
  const [presetCategory, setPresetCategory] = useState<string>('oscillators');
  const [draggingNode, setDraggingNode] = useState<{ id: string; offsetX: number; offsetY: number } | null>(null);
  const [showNodeInfo, setShowNodeInfo] = useState<string | null>(null);
  const [draggingWire, setDraggingWire] = useState<DraggingWire | null>(null);
  const [hoveredPort, setHoveredPort] = useState<{ nodeId: string; portIndex: number; isOutput: boolean } | null>(null);
  const [deleteConfirmation, setDeleteConfirmation] = useState<DeleteConfirmation | null>(null);
  const [hoveredConnection, setHoveredConnection] = useState<string | null>(null);
  const simRunning = useSimulationStore((state) => state.simulationRunning);
  const { processors, backendNodes, startPolling, stopPolling } = useBackendDataStore((state) => ({
    processors: state.processors,
    backendNodes: state.nodes,
    startPolling: state.startPolling,
    stopPolling: state.stopPolling,
  }));

  const [activeNodes, setActiveNodes] = useState<Set<string>>(new Set());
  const [activeConnections, setActiveConnections] = useState<Set<string>>(new Set());

  const nodesRef = useRef<Node[]>(nodes);
  const pendingBackendIdsRef = useRef<Set<number>>(new Set());

  useEffect(() => {
    nodesRef.current = nodes;
  }, [nodes]);

  // Ensure backend polling is active while the node graph is mounted
  useEffect(() => {
    startPolling();
    return () => stopPolling();
  }, [startPolling, stopPolling]);

  useEffect(() => {
    const backendMap = new Map<number, BackendNodePayload>();
    backendNodes.forEach((backendNode) => {
      backendMap.set(backendNode.node_id, backendNode);
      pendingBackendIdsRef.current.delete(backendNode.node_id);
    });

    if (!backendMap.size && pendingBackendIdsRef.current.size === 0 && nodesRef.current.length === 0) {
      return;
    }

    setNodes((prevNodes) => {
      const updatedNodes: Node[] = [];
      const removedLocalIds = new Set<string>();

      prevNodes.forEach((node) => {
        if (node.backendId != null) {
          const backendNode = backendMap.get(node.backendId);
          if (backendNode) {
            const { x, y } = extractPosition(backendNode.position, node.x, node.y);
            updatedNodes.push({
              ...node,
              x,
              y,
              config: { ...node.config, ...backendNode.config },
            });
            backendMap.delete(node.backendId);
          } else if (!pendingBackendIdsRef.current.has(node.backendId)) {
            removedLocalIds.add(node.id);
          } else {
            updatedNodes.push(node);
          }
        } else {
          updatedNodes.push(node);
        }
      });

      backendMap.forEach((backendNode) => {
        const preset = getDefaultPreset();
        const { x, y } = extractPosition(backendNode.position, 100 + updatedNodes.length * 50, 100);
        let generatedId = `node-${backendNode.node_id}`;
        if (updatedNodes.some((node) => node.id === generatedId)) {
          generatedId = `${generatedId}-${Date.now()}`;
        }

        updatedNodes.push({
          id: generatedId,
          preset,
          x,
          y,
          config: { ...preset.config, ...backendNode.config },
          backendId: backendNode.node_id,
        });
      });

      if (removedLocalIds.size > 0) {
        setConnections((prevConnections) =>
          prevConnections.filter(
            (conn) => !removedLocalIds.has(conn.fromNodeId) && !removedLocalIds.has(conn.toNodeId),
          ),
        );
      }

      return updatedNodes;
    });
  }, [backendNodes]);

  // Update activity maps based on backend processor data and local graph
  useEffect(() => {
    if (!simRunning) {
      setActiveNodes(new Set());
      setActiveConnections(new Set());
      return;
    }

    const newActiveNodes = new Set<string>();
    const newActiveConnections = new Set<string>();

    processors.forEach((proc) => {
      if (proc?.active) {
        const nodeId = String(proc.node_id);
        newActiveNodes.add(nodeId);
        connections
          .filter((conn) => conn.fromNodeId === nodeId)
          .forEach((conn) => newActiveConnections.add(conn.id));
      }
    });

    nodes.forEach((node) => {
      const outputConnections = connections.filter((conn) => conn.fromNodeId === node.id);
      if (
        outputConnections.length > 0 &&
        (node.preset.type === 'oscillator' || node.preset.type === 'algorithm')
      ) {
        newActiveNodes.add(node.id);
        outputConnections.forEach((conn) => newActiveConnections.add(conn.id));
      }
    });

    nodes.forEach((node) => {
      const inputConnections = connections.filter((conn) => conn.toNodeId === node.id);
      const hasActiveInput = inputConnections.some((conn) => newActiveConnections.has(conn.id));
      if (hasActiveInput) {
        newActiveNodes.add(node.id);
      }
    });

    setActiveNodes(newActiveNodes);
    setActiveConnections(newActiveConnections);
  }, [simRunning, processors, nodes, connections]);

  // Load backend presets when modal opens
  useEffect(() => {
    if (showGraphPresets && backendPresets.length === 0) {
      loadBackendPresets();
    }
  }, [showGraphPresets]);

  const loadBackendPresets = async () => {
    setLoadingPresets(true);
    try {
      // Frontend GRAPH_PRESETS (3 demo presets)
      const frontendPresetList = GRAPH_PRESETS.map((p, idx) => ({
        id: `frontend_${idx}`,
        name: p.name,
        description: p.description,
        category: p.category,
        node_count: p.nodes.length,
        connection_count: p.connections?.length || 0,
        tags: [],
        _frontendPreset: p,
        _isJsonPreset: false
      }));

      // All 16 JSON presets directly imported
      const jsonPresetList = JSON_PRESETS.map((p: any) => ({
        id: p.id,
        name: p.name,
        description: p.description,
        category: p.category,
        tags: p.tags || [],
        node_count: p.nodes?.length || 0,
        connection_count: p.connections?.length || 0,
        _frontendPreset: null,
        _jsonPreset: p,
        _isJsonPreset: true
      }));

      // Merge: 3 frontend presets + 16 JSON presets = 19 total
      setBackendPresets([...frontendPresetList, ...jsonPresetList]);
    } catch (error) {
      console.error('Failed to load presets:', error);
    } finally {
      setLoadingPresets(false);
    }
  };
  // Save to localStorage whenever nodes or connections change
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('gmcs_nodes', JSON.stringify(nodes));
    }
  }, [nodes]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('gmcs_connections', JSON.stringify(connections));
    }
  }, [connections]);

  const canvasRef = useCallback((node: HTMLDivElement | null) => {
    if (node) {
      // Store reference to canvas for coordinate calculations
    }
  }, []);

  const handleAddNode = useCallback(async (preset: NodePreset) => {
    const currentNodes = nodesRef.current;
    const fallbackPosition = {
      x: 100 + currentNodes.length * 50,
      y: 100 + (currentNodes.length % 3) * 150,
    };

    try {
      const response = await fetch(`${API_BASE}/node/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          position: [fallbackPosition.x, fallbackPosition.y],
          config: preset.config ?? {},
          chain: [],
          initial_perturbation: preset.config?.initial_perturbation ?? 0.1,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        notify.error(`Failed to create node: ${errorText}`);
        return;
      }

      const result: NodeResponsePayload = await response.json();
      const backendNode = (result.data?.node ?? null) as BackendNodePayload | null;
      const backendId = backendNode?.node_id ?? result.node_id ?? null;

      const { x, y } = backendNode
        ? extractPosition(backendNode.position, fallbackPosition.x, fallbackPosition.y)
        : fallbackPosition;

      const config = backendNode
        ? { ...preset.config, ...backendNode.config }
        : { ...preset.config };

      const localIdBase = backendId != null ? `node-${backendId}` : `node-${Date.now()}`;
      let localId = localIdBase;
      if (nodesRef.current.some((node) => node.id === localId)) {
        localId = `${localIdBase}-${Date.now()}`;
      }

      if (typeof backendId === 'number') {
        pendingBackendIdsRef.current.add(backendId);
      }

      setNodes((prev) => [...prev, {
        id: localId,
        preset,
        x,
        y,
        config,
        backendId: backendId ?? undefined,
      }]);

      setShowNodeModal(false);

      if (preset.category === 'THRML' || preset.type === 'thrml') {
        const processorId = backendId != null ? String(backendId) : localId;
        try {
          const procResponse = await fetch(`${API_BASE}/processor/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              node_id: processorId,
              processor_type: 'thrml',
              nodes: preset.config?.nodes || 64,
              temperature: preset.config?.temperature || 1.0,
              gibbs_steps: preset.config?.gibbs_steps || 5,
              config: preset.config,
            }),
          });

          if (!procResponse.ok) {
            const errorText = await procResponse.text();
            console.warn('Failed to create THRML processor on backend:', errorText);
            notify.error(`THRML processor creation failed: ${errorText}`);
          } else {
            const data = await procResponse.json();
            console.log('THRML processor created on backend:', data);
            notify.success(`THRML processor created: ${data.n_nodes} nodes at T=${data.temperature}`);
          }
        } catch (error: any) {
          console.error('Error creating THRML processor:', error);
          notify.error(`Failed to connect to backend for THRML processor: ${error.message}`);
        }
      } else {
        notify.success(result.message ?? `Node created successfully!${backendId != null ? ` ID: ${backendId}` : ''}`);
      }
    } catch (error) {
      console.error('Failed to create node:', error);
      notify.error('Error creating node');
    }
  }, [setNodes, setShowNodeModal]);

  const handleLoadGraphPreset = useCallback((graphPreset: GraphPreset) => {
    // Clear existing nodes and connections
    setNodes([]);
    setConnections([]);

    // Create nodes from preset
    const newNodes: Node[] = graphPreset.nodes.map((nodeData, index) => {
      const preset = NODE_PRESETS[nodeData.presetKey]?.[nodeData.presetIndex];
      if (!preset) {
        console.error(`Preset not found: ${nodeData.presetKey}[${nodeData.presetIndex}]`);
        return null;
      }
      return {
        id: `node-${Date.now()}-${index}`,
        preset,
        x: nodeData.x,
        y: nodeData.y,
        config: { ...preset.config, ...nodeData.config },
      };
    }).filter(Boolean) as Node[];

    setNodes(newNodes);

    // Create connections after a short delay to ensure nodes are rendered
    setTimeout(() => {
      if (graphPreset.connections) {
        const newConnections: Connection[] = graphPreset.connections.map((conn, index) => {
          const fromNode = newNodes[conn.fromNode];
          const toNode = newNodes[conn.toNode];
          if (!fromNode || !toNode) return null;

          return {
            id: `conn-${Date.now()}-${index}`,
            fromNodeId: fromNode.id,
            fromPort: fromNode.preset.outputs[conn.fromPort]?.name || '',
            fromPortIndex: conn.fromPort,
            toNodeId: toNode.id,
            toPort: toNode.preset.inputs[conn.toPort]?.name || '',
            toPortIndex: conn.toPort,
          };
        }).filter(Boolean) as Connection[];

        setConnections(newConnections);
      }
    }, 100);

    setShowGraphPresets(false);
  }, []);

  const handleLoadBackendPreset = useCallback(async (presetId: string) => {
    try {
      // Fetch full preset from backend
      const response = await fetch(`${API_BASE}/api/presets/${presetId}`);
      if (!response.ok) {
        console.error(`Failed to load preset ${presetId}`);
        return;
      }

      const fullPreset = await response.json();
      console.log('Loaded backend preset:', fullPreset);

      // Clear existing graph
      setNodes([]);
      setConnections([]);

      // Map backend nodes to frontend node format
      // Note: This is a simplified mapping - would need to match backend node types to frontend NODE_PRESETS
      const newNodes: Node[] = fullPreset.nodes.map((backendNode: any, index: number) => {
        // Try to find matching preset in NODE_PRESETS based on node type
        // For now, use a default preset - this needs proper mapping logic
        const defaultPreset = NODE_PRESETS.oscillators[0]; // Fallback

        return {
          id: backendNode.id || `node-${Date.now()}-${index}`,
          preset: defaultPreset,
          x: backendNode.position?.x || 100 + index * 200,
          y: backendNode.position?.y || 100,
          config: { ...defaultPreset.config, ...backendNode.config },
        };
      });

      setNodes(newNodes);

      // Note: Backend connections use different format - would need proper parsing
      // For now, skip connections - needs implementation

      setShowGraphPresets(false);
    } catch (error) {
      console.error('Failed to load backend preset:', error);
    }
  }, []);

  const handleDeleteNode = useCallback(async (nodeId: string) => {
    const node = nodesRef.current.find((n) => n.id === nodeId);
    const backendId = node?.backendId;

    if (typeof backendId === 'number') {
      pendingBackendIdsRef.current.delete(backendId);
    }

    setNodes((prev) => prev.filter((n) => n.id !== nodeId));
    setConnections((prev) => prev.filter((c) => c.fromNodeId !== nodeId && c.toNodeId !== nodeId));
    if (selectedNode === nodeId) {
      setSelectedNode(null);
    }

    if (typeof backendId === 'number') {
      try {
        const response = await fetch(`${API_BASE}/node/${backendId}`, {
          method: 'DELETE',
        });

        if (!response.ok) {
          console.error('Failed to delete backend node:', await response.text());
        }
      } catch (error) {
        console.error('Error deleting backend node:', error);
      }
    }

    if (node && (node.preset.category === 'THRML' || node.preset.type === 'thrml')) {
      const processorId = typeof backendId === 'number' ? String(backendId) : nodeId;
      try {
        const response = await fetch(`${API_BASE}/processor/${encodeURIComponent(processorId)}`, {
          method: 'DELETE',
        });

        if (response.ok) {
          const data = await response.json();
          console.log('Processor instance deleted:', data);
        }
      } catch (error) {
        console.error('Error deleting processor instance:', error);
      }
    }
  }, [selectedNode, setSelectedNode]);

  const startWireDrag = useCallback((
    nodeId: string,
    port: string,
    portIndex: number,
    x: number,
    y: number,
    isOutput: boolean,
    event: React.MouseEvent
  ) => {
    event.stopPropagation();
    setDraggingWire({
      fromNodeId: nodeId,
      fromPort: port,
      fromPortIndex: portIndex,
      fromX: x,
      fromY: y,
      currentX: x,
      currentY: y,
      isOutput
    });
  }, []);

  const updateWireDrag = useCallback((event: React.MouseEvent) => {
    if (draggingWire) {
      const rect = event.currentTarget.getBoundingClientRect();
      setDraggingWire(prev => prev ? {
        ...prev,
        currentX: event.clientX - rect.left,
        currentY: event.clientY - rect.top
      } : null);
    }
  }, [draggingWire]);

  const endWireDrag = useCallback((
    targetNodeId?: string,
    targetPort?: string,
    targetPortIndex?: number,
    isTargetOutput?: boolean
  ) => {
    if (draggingWire && targetNodeId && targetPort !== undefined && targetPortIndex !== undefined) {
      // Validate connection: output must connect to input
      if (draggingWire.isOutput && !isTargetOutput) {
        // Valid connection: output to input
        const newConnection: Connection = {
          id: `conn-${Date.now()}`,
          fromNodeId: draggingWire.fromNodeId,
          fromPort: draggingWire.fromPort,
          fromPortIndex: draggingWire.fromPortIndex,
          toNodeId: targetNodeId,
          toPort: targetPort,
          toPortIndex: targetPortIndex
        };
        setConnections(prev => [...prev, newConnection]);
      } else if (!draggingWire.isOutput && isTargetOutput) {
        // Valid connection: input to output (reverse)
        const newConnection: Connection = {
          id: `conn-${Date.now()}`,
          fromNodeId: targetNodeId,
          fromPort: targetPort,
          fromPortIndex: targetPortIndex,
          toNodeId: draggingWire.fromNodeId,
          toPort: draggingWire.fromPort,
          toPortIndex: draggingWire.fromPortIndex
        };
        setConnections(prev => [...prev, newConnection]);
      }
    }
    setDraggingWire(null);
    setHoveredPort(null);
  }, [draggingWire]);

  const deleteConnection = useCallback((connectionId: string) => {
    setConnections(prev => prev.filter(c => c.id !== connectionId));
  }, []);

  // Parameter range helpers
  const getParamMin = (key: string, currentValue: number): number => {
    const ranges: Record<string, number> = {
      'temperature': 0.1,
      'gibbs_steps': 1,
      'cd_k': 1,
      'nodes': 16,
      'spin_nodes': 8,
      'continuous_nodes': 8,
      'width': 200,
      'height': 150,
    };
    return ranges[key] ?? (currentValue < 1 ? 0 : 1);
  };

  const getParamMax = (key: string, currentValue: number): number => {
    const ranges: Record<string, number> = {
      'temperature': 5.0,
      'gibbs_steps': 100,
      'cd_k': 10,
      'nodes': 256,
      'spin_nodes': 128,
      'continuous_nodes': 128,
      'width': 600,
      'height': 600,
    };
    return ranges[key] ?? (currentValue < 1 ? 1 : Math.max(currentValue * 2, 10));
  };

  const getParamStep = (key: string, currentValue: number): number => {
    const steps: Record<string, number> = {
      'temperature': 0.1,
      'gibbs_steps': 1,
      'cd_k': 1,
      'nodes': 16,
      'spin_nodes': 8,
      'continuous_nodes': 8,
      'width': 50,
      'height': 50,
    };
    return steps[key] ?? (currentValue < 1 ? 0.01 : 1);
  };

  const updateNodeConfig = useCallback(async (nodeId: string, configKey: string, value: any) => {
    // Update local state immediately
    setNodes(prev => prev.map(n =>
      n.id === nodeId
        ? { ...n, config: { ...n.config, [configKey]: value } }
        : n
    ));

    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;

    const backendId = node.backendId;
    if (backendId == null) {
      console.warn(`Skipping backend update for ${nodeId} – no backendId is associated yet.`);
      return;
    }

    // Send update to backend
    try {
      const response = await fetch(`${API_BASE}/node/update`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          node_ids: [backendId],
          config_updates: { [configKey]: value }
        })
      });

      if (!response.ok) {
        console.error('Failed to update node config:', await response.text());
      }
    } catch (error) {
      console.error('Error updating node config:', error);
    }
  }, [nodes]);

  const getPortPosition = useCallback((node: Node, portIndex: number, isOutput: boolean) => {
    const nodeWidth = node.preset.type === 'visualizer' ? (node.config.width || 300) : 180;
    const headerHeight = 60;
    const portSpacing = 25;
    const portY = headerHeight + portSpacing * portIndex + 15;

    return {
      x: node.x + (isOutput ? nodeWidth : 0),
      y: node.y + portY
    };
  }, []);

  const handleMouseDown = (e: React.MouseEvent, nodeId: string) => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node) return;
    setDraggingNode({
      id: nodeId,
      offsetX: e.clientX - node.x,
      offsetY: e.clientY - node.y,
    });
    setSelectedNode(nodeId);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (draggingNode) {
      setNodes(prev => prev.map(node =>
        node.id === draggingNode.id
          ? { ...node, x: e.clientX - draggingNode.offsetX, y: e.clientY - draggingNode.offsetY }
          : node
      ));
    } else if (draggingWire) {
      updateWireDrag(e);
    }
  };

  const handleMouseUp = () => {
    if (draggingWire) {
      // Wire released in empty space - snap back
      endWireDrag();
    }
    setDraggingNode(null);
  };

  const renderNode = (node: Node) => {
    const isVisualizer = node.preset.type === 'visualizer';
    const VisualizerComponent = isVisualizer ? VisualizerComponents[node.preset.name] : null;
    const nodeWidth = isVisualizer ? (node.config.width || 300) : 180;
    const isSelected = selectedNode === node.id;
    const isActive = activeNodes.has(node.id);
    const isStubNode = STUB_NODE_TYPES.has(node.preset.type);
    const hasBackendId = typeof node.backendId === 'number';

    return (
      <div
        key={node.id}
        className={`absolute bg-black/60 border rounded transition ${isSelected
          ? 'border-[#00ff99] shadow-[0_0_10px_rgba(88,166,255,0.3)]'
          : isActive
            ? 'border-[#00ff99] shadow-[0_0_15px_rgba(0,255,153,0.4)] animate-pulse'
            : 'border-[#00cc77] hover:border-[#00ff99]'
          }`}
        style={{ left: node.x, top: node.y, width: nodeWidth, cursor: 'move' }}
        onMouseDown={(e) => handleMouseDown(e, node.id)}
      >
        {/* Header */}
        <div
          className={`px-3 py-2 border-b border-[#00cc77] flex items-center justify-between ${node.preset.type === 'oscillator' ? 'bg-[#00ff99]/10' :
            node.preset.type === 'algorithm' ? 'bg-[#00ff99]/10' :
              node.preset.type === 'thrml' ? 'bg-[#f85149]/10' :
                node.preset.type === 'visualizer' ? 'bg-[#d29922]/10' :
                  'bg-[#a371f7]/10'
            }`}
        >
          <div className="flex-1 flex items-center gap-2">
            <div>
              <div className="text-[9px] font-semibold text-[#00cc77] uppercase tracking-wide">
                {node.preset.type}
              </div>
              <div className="text-xs text-[#00ff99] font-semibold">
                {node.preset.name}
              </div>
              {isStubNode && (
                <div className="text-[8px] text-[#f85149] uppercase tracking-wide">Stub – backend execution pending</div>
              )}
              {hasBackendId && (
                <div className="text-[8px] text-[#00ff99]/70 uppercase tracking-wide">Backend #{node.backendId}</div>
              )}
            </div>
            {/* Activity Indicator - only shows when node is actively processing data */}
            {isActive && (
              <div className="flex items-center gap-1">
                <div
                  className="w-2 h-2 rounded-full bg-[#00ff99] animate-pulse"
                  title="Node actively processing data"
                />
                <span className="text-[8px] text-[#00ff99] font-bold">LIVE</span>
              </div>
            )}
          </div>

          {/* Visualizer Monitor Toggle in Header */}
          {isVisualizer && (
            <div className="flex items-center gap-1 mr-2">
              <span className="text-[8px] text-[#00cc77]">MON:</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  updateNodeConfig(node.id, 'active', !(node.config.active ?? true));
                }}
                className={`px-2 py-0.5 text-[9px] rounded font-semibold transition ${(node.config.active ?? true)
                  ? 'bg-[#00ff99] text-black'
                  : 'bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77]'
                  }`}
                title={`Monitor: ${(node.config.active ?? true) ? 'ON' : 'OFF'}`}
              >
                {(node.config.active ?? true) ? 'ON' : 'OFF'}
              </button>
            </div>
          )}

          <button
            onClick={(e) => {
              e.stopPropagation();
              handleDeleteNode(node.id);
            }}
            className="text-[#f85149] hover:text-[#ff7b72] text-lg leading-none"
          >
            ×
          </button>
        </div>

        {/* Body */}
        <div className="p-3">
          {isVisualizer && VisualizerComponent ? (
            <>
              {/* Input Ports */}
              <div className="space-y-1 mb-3">
                {node.preset.inputs.map((input, idx) => {
                  const portPos = getPortPosition(node, idx, false);
                  const isHovered = hoveredPort?.nodeId === node.id && hoveredPort?.portIndex === idx && !hoveredPort?.isOutput;
                  return (
                    <div
                      key={`in-${idx}`}
                      className="flex items-center gap-2 text-[10px] text-[#00cc77] relative"
                      onMouseEnter={() => setHoveredPort({ nodeId: node.id, portIndex: idx, isOutput: false })}
                      onMouseLeave={() => setHoveredPort(null)}
                    >
                      <div
                        className={`w-3 h-3 rounded-full cursor-pointer transition-all ${isHovered ? 'bg-[#00ff99] scale-125 shadow-glow' : 'bg-[#00cc77] hover:bg-[#00ff99]'
                          }`}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          startWireDrag(node.id, input.name, idx, portPos.x, portPos.y, false, e);
                        }}
                        onMouseUp={(e) => {
                          e.stopPropagation();
                          if (draggingWire) {
                            endWireDrag(node.id, input.name, idx, false);
                          }
                        }}
                      />
                      <span>{input.name}</span>
                    </div>
                  );
                })}
              </div>

              {/* Visualizer Controls - Above Monitor */}
              <div className="flex items-center justify-between px-2 py-1 mb-2 bg-black rounded border border-[#00cc77]">
                <span className="text-xs font-semibold text-[#00ff99]">{node.preset.name.toUpperCase()}</span>
                <div className="flex gap-2 text-xs">
                  {node.preset.name === 'Oscilloscope' && (
                    <>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          updateNodeConfig(node.id, 'showMeasurements', !(node.config.showMeasurements ?? true));
                        }}
                        className="px-2 py-0.5 bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
                      >
                        Measure {node.config.showMeasurements ?? true ? 'ON' : 'OFF'}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          const modes: ('auto' | 'normal' | 'single')[] = ['auto', 'normal', 'single'];
                          const currentMode = node.config.triggerMode ?? 'auto';
                          const nextMode = modes[(modes.indexOf(currentMode) + 1) % modes.length];
                          updateNodeConfig(node.id, 'triggerMode', nextMode);
                        }}
                        className="px-2 py-0.5 bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
                      >
                        Trig: {(node.config.triggerMode ?? 'auto').toUpperCase()}
                      </button>
                    </>
                  )}
                  {node.preset.name === 'Spectrogram' && (
                    <>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          updateNodeConfig(node.id, 'showPeaks', !(node.config.showPeaks ?? true));
                        }}
                        className="px-2 py-0.5 bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
                      >
                        Peaks {node.config.showPeaks ?? true ? 'ON' : 'OFF'}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          const maps = ['viridis', 'plasma', 'inferno', 'magma'];
                          const currentMap = node.config.colorMap ?? 'viridis';
                          const nextMap = maps[(maps.indexOf(currentMap) + 1) % maps.length];
                          updateNodeConfig(node.id, 'colorMap', nextMap);
                        }}
                        className="px-2 py-0.5 bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
                      >
                        Color: {(node.config.colorMap ?? 'viridis').toUpperCase()}
                      </button>
                    </>
                  )}
                  {node.preset.name === 'Phase Space 3D' && (
                    <>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          updateNodeConfig(node.id, 'showPoincare', !(node.config.showPoincare ?? false));
                        }}
                        className="px-2 py-0.5 bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
                      >
                        Poincaré {node.config.showPoincare ? 'ON' : 'OFF'}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          updateNodeConfig(node.id, 'showTrail', !(node.config.showTrail ?? true));
                        }}
                        className="px-2 py-0.5 bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
                      >
                        Trail {node.config.showTrail ?? true ? 'ON' : 'OFF'}
                      </button>
                    </>
                  )}
                  {node.preset.name === 'Energy Graph' && (
                    <>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          updateNodeConfig(node.id, 'showHistogram', !(node.config.showHistogram ?? false));
                        }}
                        className="px-2 py-0.5 bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
                      >
                        {node.config.showHistogram ? 'Time Series' : 'Histogram'}
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          updateNodeConfig(node.id, 'showMovingAvg', !(node.config.showMovingAvg ?? true));
                        }}
                        className="px-2 py-0.5 bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
                      >
                        Avg {node.config.showMovingAvg ?? true ? 'ON' : 'OFF'}
                      </button>
                    </>
                  )}
                  {node.preset.name === 'Spin State Matrix' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        updateNodeConfig(node.id, 'showValues', !(node.config.showValues ?? false));
                      }}
                      className="px-2 py-0.5 bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
                    >
                      {node.config.showValues ? 'Hide' : 'Show'} Values
                    </button>
                  )}
                  {node.preset.name === 'Correlation Matrix' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        updateNodeConfig(node.id, 'showColorBar', !(node.config.showColorBar ?? true));
                      }}
                      className="px-2 py-0.5 bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
                    >
                      Color Bar {node.config.showColorBar ?? true ? 'ON' : 'OFF'}
                    </button>
                  )}
                  {node.preset.name === 'XY Plot' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        // Clear points functionality would need backend support
                        updateNodeConfig(node.id, 'clearPlot', Date.now());
                      }}
                      className="px-2 py-0.5 bg-[#1a1a1a] text-[#00ff99] rounded hover:bg-[#00cc77]/20"
                    >
                      Clear
                    </button>
                  )}
                  {/* Waveform Monitor has no additional controls */}
                </div>
              </div>

              {/* Monitor - Minimized when OFF */}
              {(node.config.active ?? true) ? (
                <div className="bg-black rounded border border-[#00cc77] overflow-hidden mb-3">
                  <VisualizerComponent
                    width={node.config.width || 500}
                    height={node.config.height || 350}
                    nodeId={node.id}
                    connections={connections.filter(c => c.toNodeId === node.id)}
                    showHistogram={node.config.showHistogram}
                    showMovingAvg={node.config.showMovingAvg}
                    showValues={node.config.showValues}
                    showColorBar={node.config.showColorBar}
                    showMeasurements={node.config.showMeasurements}
                    triggerMode={node.config.triggerMode}
                    showPeaks={node.config.showPeaks}
                    colorMap={node.config.colorMap}
                    showPoincare={node.config.showPoincare}
                    showTrail={node.config.showTrail}
                    gridSize={node.config.grid_size}
                    colorScheme={node.config.color_scheme}
                    updateRate={node.config.update_rate}
                  />
                </div>
              ) : (
                <div className="bg-black/30 rounded border border-[#00cc77]/50 py-2 text-center mb-3">
                  <span className="text-[#00cc77] text-xs">Monitor Minimized</span>
                </div>
              )}

              {/* Output Ports */}
              <div className="space-y-1">
                {node.preset.outputs.map((output, idx) => {
                  const portPos = getPortPosition(node, idx, true);
                  const isHovered = hoveredPort?.nodeId === node.id && hoveredPort?.portIndex === idx && hoveredPort?.isOutput;
                  return (
                    <div
                      key={`out-${idx}`}
                      className="flex items-center justify-end gap-2 text-[10px] text-[#00cc77] relative"
                      onMouseEnter={() => setHoveredPort({ nodeId: node.id, portIndex: idx, isOutput: true })}
                      onMouseLeave={() => setHoveredPort(null)}
                    >
                      <span>{output.name}</span>
                      <div
                        className={`w-3 h-3 rounded-full cursor-pointer transition-all ${isHovered ? 'bg-[#00ff99] scale-125 shadow-glow' : 'bg-[#00cc77] hover:bg-[#00ff99]'
                          }`}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          startWireDrag(node.id, output.name, idx, portPos.x, portPos.y, true, e);
                        }}
                      />
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <div className="space-y-1">
              {/* Input Ports */}
              {node.preset.inputs.map((input, idx) => {
                const portPos = getPortPosition(node, idx, false);
                const isHovered = hoveredPort?.nodeId === node.id && hoveredPort?.portIndex === idx && !hoveredPort?.isOutput;
                return (
                  <div
                    key={`in-${idx}`}
                    className="flex items-center gap-2 text-[10px] text-[#00cc77] relative"
                    onMouseEnter={() => setHoveredPort({ nodeId: node.id, portIndex: idx, isOutput: false })}
                    onMouseLeave={() => setHoveredPort(null)}
                  >
                    <div
                      className={`w-3 h-3 rounded-full cursor-pointer transition-all ${isHovered ? 'bg-[#00ff99] scale-125 shadow-glow' : 'bg-[#00cc77] hover:bg-[#00ff99]'
                        }`}
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        startWireDrag(node.id, input.name, idx, portPos.x, portPos.y, false, e);
                      }}
                      onMouseUp={(e) => {
                        e.stopPropagation();
                        if (draggingWire) {
                          endWireDrag(node.id, input.name, idx, false);
                        }
                      }}
                    />
                    <span>{input.name}</span>
                  </div>
                );
              })}

              {/* Output Ports */}
              {node.preset.outputs.map((output, idx) => {
                const portPos = getPortPosition(node, idx, true);
                const isHovered = hoveredPort?.nodeId === node.id && hoveredPort?.portIndex === idx && hoveredPort?.isOutput;
                return (
                  <div
                    key={`out-${idx}`}
                    className="flex items-center justify-end gap-2 text-[10px] text-[#00cc77] relative"
                    onMouseEnter={() => setHoveredPort({ nodeId: node.id, portIndex: idx, isOutput: true })}
                    onMouseLeave={() => setHoveredPort(null)}
                  >
                    <span>{output.name}</span>
                    <div
                      className={`w-3 h-3 rounded-full cursor-pointer transition-all ${isHovered ? 'bg-[#00ff99] scale-125 shadow-glow' : 'bg-[#00cc77] hover:bg-[#00ff99]'
                        }`}
                      onMouseDown={(e) => {
                        e.stopPropagation();
                        startWireDrag(node.id, output.name, idx, portPos.x, portPos.y, true, e);
                      }}
                      onMouseUp={(e) => {
                        e.stopPropagation();
                        if (draggingWire) {
                          endWireDrag(node.id, output.name, idx, true);
                        }
                      }}
                    />
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer - Config & Settings */}
        {!isVisualizer && (
          <div className="px-3 py-2 border-t border-[#00cc77]">
            {Object.keys(node.config).length > 0 && (
              <div className="text-[10px] text-[#00cc77] mb-2">
                {Object.entries(node.config).slice(0, 2).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="capitalize">{key.replace(/_/g, ' ')}</span>
                    <span className="text-[#00ff99] font-mono">{String(value)}</span>
                  </div>
                ))}
              </div>
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowNodeInfo(node.id);
              }}
              className="w-full px-2 py-1 text-[10px] bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition flex items-center justify-center gap-1"
            >
              <span>⚙</span>
              <span>Settings</span>
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77] bg-black/60 flex-shrink-0">
        <h2 className="text-sm font-semibold text-[#00ff99] uppercase tracking-wide">Node Graph</h2>
        <div className="flex gap-2">
          <button
            onClick={() => setShowGraphPresets(!showGraphPresets)}
            className="px-3 py-1.5 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition font-semibold"
          >
            Load Preset
          </button>
          <button
            onClick={() => setShowNodeModal(!showNodeModal)}
            className="px-3 py-1.5 text-xs bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition font-semibold"
          >
            + Add Node
          </button>
          <button
            onClick={() => {
              setNodes([]);
              setConnections([]);
            }}
            className="px-3 py-1.5 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#f85149] rounded hover:bg-[#00cc77]/20 transition"
          >
            Clear All
          </button>
        </div>
      </div>

      {/* Add Node Modal */}
      {showNodeModal && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/35 backdrop-blur-sm">
          <div className="w-[700px] max-h-[70vh] bg-black/60 border border-[#00cc77] rounded-lg shadow-2xl overflow-hidden flex flex-col">
            {/* Fixed Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77] flex-shrink-0">
              <h3 className="text-sm font-semibold text-[#00ff99]">Add Node</h3>
              <button
                onClick={() => setShowNodeModal(false)}
                className="text-[#00cc77] hover:text-[#00ff99] text-xl"
              >
                ×
              </button>
            </div>

            {/* Fixed Category Tabs */}
            <div
              className="overflow-x-auto border-b border-[#00cc77] bg-black custom-scrollbar flex-shrink-0"
              onWheel={(e) => {
                // Enable horizontal scrolling with mouse wheel
                if (e.currentTarget.scrollWidth > e.currentTarget.clientWidth) {
                  e.preventDefault();
                  e.currentTarget.scrollLeft += e.deltaY;
                }
              }}
            >
              <div className="flex min-w-max">
                {Object.keys(NODE_PRESETS).map(category => (
                  <button
                    key={category}
                    onClick={() => setPresetCategory(category)}
                    className={`px-4 py-2 text-xs font-semibold uppercase transition whitespace-nowrap ${presetCategory === category
                      ? 'text-[#00ff99] border-b-2 border-[#00ff99]'
                      : 'text-[#00cc77] hover:text-[#00ff99]'
                      }`}
                  >
                    {category}
                  </button>
                ))}
              </div>
            </div>

            {/* Scrollable Node Grid - Fixed height */}
            <div className="overflow-y-auto custom-scrollbar p-4" style={{ height: '500px' }}>
              <div className="grid grid-cols-2 gap-3">
                {NODE_PRESETS[presetCategory]?.map((preset, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleAddNode(preset)}
                    className="text-left p-3 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded hover:border-[#00ff99] transition group"
                  >
                    <div className="flex items-start justify-between mb-1">
                      <div className="text-sm font-semibold text-[#00ff99] group-hover:text-[#00ff99]">
                        {preset.name}
                      </div>
                      <div className={`text-[10px] px-2 py-0.5 rounded ${preset.type === 'oscillator' ? 'bg-[#00ff99]/20 text-[#00ff99]' :
                        preset.type === 'algorithm' ? 'bg-[#00ff99]/20 text-[#00ff99]' :
                          preset.type === 'thrml' ? 'bg-[#f85149]/20 text-[#f85149]' :
                            preset.type === 'visualizer' ? 'bg-[#d29922]/20 text-[#d29922]' :
                              'bg-[#a371f7]/20 text-[#a371f7]'
                        }`}>
                        {preset.category}
                      </div>
                    </div>
                    <div className="text-xs text-[#00cc77] mb-2">{preset.description}</div>
                    <div className="flex items-center justify-between text-[10px]">
                      <div className="text-[#00ff99]">
                        {preset.inputs.length} in
                      </div>
                      <div className="text-[#00ff99]">
                        {preset.outputs.length} out
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Load Graph Preset Modal */}
      {showGraphPresets && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/35 backdrop-blur-sm">
          <div className="w-[700px] h-[85vh] bg-black/60 border border-[#00cc77] rounded-lg shadow-2xl overflow-hidden flex flex-col">
            {/* Fixed Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77] flex-shrink-0">
              <h3 className="text-sm font-semibold text-[#00ff99]">Load Preset Configuration</h3>
              <button
                onClick={() => setShowGraphPresets(false)}
                className="text-[#00cc77] hover:text-[#00ff99] text-xl"
              >
                ×
              </button>
            </div>

            {/* Scrollable Preset Grid */}
            <div className="flex-1 overflow-y-auto custom-scrollbar p-4">
              {loadingPresets ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-[#00ff99] text-sm">Loading presets...</div>
                </div>
              ) : backendPresets.length === 0 ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-[#00cc77] text-sm">No presets available</div>
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-3">
                  {backendPresets.map((preset, idx) => (
                    <button
                      key={preset.id || idx}
                      onClick={() => {
                        if (preset._frontendPreset) {
                          handleLoadGraphPreset(preset._frontendPreset);
                        } else if (preset._jsonPreset) {
                          handleLoadGraphPreset(preset._jsonPreset);
                        } else {
                          handleLoadBackendPreset(preset.id);
                        }
                      }}
                      className="text-left p-4 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded hover:border-[#00ff99] transition group"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="text-base font-semibold text-[#00ff99] group-hover:text-[#00ff99]">
                          {preset.name}
                        </div>
                        <div className="text-[10px] px-2 py-0.5 rounded bg-[#a371f7]/20 text-[#a371f7]">
                          {preset.category}
                        </div>
                      </div>
                      <div className="text-xs text-[#00cc77] mb-2">{preset.description}</div>
                      <div className="flex items-center gap-4 text-[10px]">
                        <div className="text-[#00ff99]">
                          {preset.node_count} nodes
                        </div>
                        <div className="text-[#00ff99]">
                          {preset.connection_count} connections
                        </div>
                        {preset.tags && (
                          <div className="flex gap-1 ml-auto">
                            {preset.tags.slice(0, 3).map((tag: string) => (
                              <span key={tag} className="px-1.5 py-0.5 bg-[#00cc77]/20 text-[#00cc77] rounded text-[9px]">
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Canvas */}
      <div
        className="flex-1 relative overflow-auto custom-scrollbar"
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{
          backgroundImage: 'radial-gradient(circle, rgba(0, 204, 119, 0.3) 1px, transparent 1px)',
          backgroundSize: '20px 20px',
          backgroundPosition: '10px -4px'
        }}
      >
        {/* SVG Overlay for Wires */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 1 }}>
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            {/* Animated dashed pattern for signal flow */}
            <pattern id="signalFlow" patternUnits="userSpaceOnUse" width="20" height="2">
              <line x1="0" y1="1" x2="10" y2="1" stroke="#00ff99" strokeWidth="2" opacity="0.8">
                <animate attributeName="x1" from="0" to="20" dur="1s" repeatCount="indefinite" />
                <animate attributeName="x2" from="10" to="30" dur="1s" repeatCount="indefinite" />
              </line>
            </pattern>
          </defs>

          {/* Render existing connections */}
          {connections.map(conn => {
            const fromNode = nodes.find(n => n.id === conn.fromNodeId);
            const toNode = nodes.find(n => n.id === conn.toNodeId);
            if (!fromNode || !toNode) return null;

            const fromPos = getPortPosition(fromNode, conn.fromPortIndex, true);
            const toPos = getPortPosition(toNode, conn.toPortIndex, false);

            const dx = toPos.x - fromPos.x;
            const controlOffset = Math.abs(dx) * 0.5;
            const cp1x = fromPos.x + controlOffset;
            const cp1y = fromPos.y;
            const cp2x = toPos.x - controlOffset;
            const cp2y = toPos.y;

            const isHovered = hoveredConnection === conn.id;

            const pathId = `path-${conn.id}`;
            const pathD = `M ${fromPos.x} ${fromPos.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${toPos.x} ${toPos.y}`;

            return (
              <g key={conn.id}>
                {/* Main connection path */}
                <path
                  id={pathId}
                  d={pathD}
                  stroke={isHovered ? "#f85149" : "#00ff99"}
                  strokeWidth={isHovered ? "3" : "2"}
                  fill="none"
                  filter="url(#glow)"
                  className="pointer-events-auto cursor-pointer transition-all"
                  onMouseEnter={() => setHoveredConnection(conn.id)}
                  onMouseLeave={() => setHoveredConnection(null)}
                  onClick={(e) => {
                    e.stopPropagation();
                    setDeleteConfirmation({
                      connectionId: conn.id,
                      x: (fromPos.x + toPos.x) / 2,
                      y: (fromPos.y + toPos.y) / 2
                    });
                  }}
                />
                {/* Animated signal dots - ONLY when THIS specific connection has active data flow */}
                {activeConnections.has(conn.id) && [0, 0.33, 0.66].map((offset, i) => (
                  <circle
                    key={`signal-${i}`}
                    r="3"
                    fill="#00ff99"
                    opacity="0.9"
                    filter="url(#glow)"
                  >
                    <animateMotion
                      dur="2s"
                      repeatCount="indefinite"
                      begin={`${offset * 2}s`}
                      path={pathD}
                    />
                  </circle>
                ))}
              </g>
            );
          })}

          {/* Render dragging wire */}
          {draggingWire && (
            <g>
              <path
                d={`M ${draggingWire.fromX} ${draggingWire.fromY} C ${draggingWire.fromX + 100} ${draggingWire.fromY}, ${draggingWire.currentX - 100} ${draggingWire.currentY}, ${draggingWire.currentX} ${draggingWire.currentY}`}
                stroke="#00ff99"
                strokeWidth="2"
                fill="none"
                strokeDasharray="5,5"
                filter="url(#glow)"
                className="animate-pulse"
              />
            </g>
          )}
        </svg>

        {nodes.map(renderNode)}
      </div>

      {/* Node Info Panel */}
      {showNodeInfo && (() => {
        const node = nodes.find(n => n.id === showNodeInfo);
        if (!node) return null;

        return (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
            <div className="w-[600px] max-h-[85vh] bg-black/60 border border-[#00cc77] rounded-lg shadow-2xl overflow-hidden">
              {/* Header */}
              <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77]">
                <h2 className="text-sm font-semibold text-[#00ff99]">Node Information</h2>
                <button
                  onClick={() => setShowNodeInfo(null)}
                  className="text-[#00cc77] hover:text-[#00ff99] text-xl"
                >
                  ×
                </button>
              </div>

              {/* Content */}
              <div className="p-4 overflow-y-auto max-h-[calc(85vh-120px)] custom-scrollbar space-y-4">
                {/* Basic Info */}
                <div className="p-3 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <div className="text-sm font-semibold text-[#00ff99]">{node.preset.name}</div>
                      <div className="text-xs text-[#00cc77] mt-1">{node.preset.description}</div>
                    </div>
                    <div className={`text-[10px] px-2 py-0.5 rounded ${node.preset.type === 'oscillator' ? 'bg-[#00ff99]/20 text-[#00ff99]' :
                      node.preset.type === 'algorithm' ? 'bg-[#00ff99]/20 text-[#00ff99]' :
                        node.preset.type === 'thrml' ? 'bg-[#f85149]/20 text-[#f85149]' :
                          node.preset.type === 'visualizer' ? 'bg-[#d29922]/20 text-[#d29922]' :
                            'bg-[#a371f7]/20 text-[#a371f7]'
                      }`}>
                      {node.preset.category}
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-[#00cc77] mt-3">
                    <span>{node.preset.inputs.length} inputs</span>
                    <span>•</span>
                    <span>{node.preset.outputs.length} outputs</span>
                  </div>
                </div>

                {/* Inputs */}
                {node.preset.inputs.length > 0 && (
                  <div>
                    <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-2">
                      Inputs
                    </h3>
                    <div className="space-y-1">
                      {node.preset.inputs.map((input, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-xs bg-black/60 backdrop-blur-md border border-[#00cc77] rounded px-3 py-2">
                          <div className="w-2 h-2 rounded-full bg-[#00ff99]" />
                          <span className="text-[#00ff99] font-semibold">{input.name}</span>
                          <span className="text-[#00cc77]">({input.type})</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Outputs */}
                {node.preset.outputs.length > 0 && (
                  <div>
                    <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-2">
                      Outputs
                    </h3>
                    <div className="space-y-1">
                      {node.preset.outputs.map((output, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-xs bg-black/60 backdrop-blur-md border border-[#00cc77] rounded px-3 py-2">
                          <div className="w-2 h-2 rounded-full bg-[#00ff99]" />
                          <span className="text-[#00ff99] font-semibold">{output.name}</span>
                          <span className="text-[#00cc77]">({output.type})</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Parameters */}
                {Object.keys(node.config).length > 0 && (
                  <div>
                    <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-2">
                      Parameters
                    </h3>
                    <div className="space-y-2">
                      {Object.entries(node.config).map(([key, value]) => (
                        <div key={key} className="bg-black/60 backdrop-blur-md border border-[#00cc77] rounded px-3 py-2">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs text-[#00ff99] capitalize">{key.replace(/_/g, ' ')}</span>
                            <span className="text-xs text-[#00ff99] font-mono">{String(value)}</span>
                          </div>

                          {/* Number slider */}
                          {typeof value === 'number' && (
                            <input
                              type="range"
                              min={getParamMin(key, value)}
                              max={getParamMax(key, value)}
                              step={getParamStep(key, value)}
                              value={value}
                              onChange={(e) => {
                                const newValue = parseFloat(e.target.value);
                                updateNodeConfig(node.id, key, newValue);
                              }}
                              className="w-full h-1 bg-[#00cc77]/30 rounded-lg appearance-none cursor-pointer slider-thumb"
                            />
                          )}

                          {/* Boolean toggle */}
                          {typeof value === 'boolean' && (
                            <button
                              onClick={() => {
                                updateNodeConfig(node.id, key, !value);
                              }}
                              className={`relative w-12 h-6 rounded-full transition-colors ${value ? 'bg-[#00ff99]' : 'bg-[#1a1a1a] border border-[#00cc77]'
                                }`}
                            >
                              <div
                                className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-black transition-transform ${value ? 'translate-x-6' : 'translate-x-0'
                                  }`}
                              />
                            </button>
                          )}

                          {/* String input */}
                          {typeof value === 'string' && (
                            <input
                              type="text"
                              value={value}
                              onChange={(e) => {
                                updateNodeConfig(node.id, key, e.target.value);
                              }}
                              className="w-full px-2 py-1 text-xs bg-black border border-[#00cc77] text-[#00ff99] rounded focus:border-[#00ff99] outline-none"
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Equations */}
                {node.preset.equations && node.preset.equations.length > 0 && (
                  <div>
                    <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-2">
                      Equations
                    </h3>
                    <div className="p-3 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded">
                      <div className="space-y-2">
                        {node.preset.equations.map((eq, idx) => (
                          <div key={idx} className="text-xs font-mono text-[#00ff99] bg-black/60 px-3 py-2 rounded border border-[#00cc77]">
                            {eq}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* How It Works */}
                {node.preset.howItWorks && (
                  <div>
                    <h3 className="text-xs font-semibold text-[#00cc77] uppercase tracking-wide mb-2">
                      How It Works
                    </h3>
                    <div className="p-3 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded">
                      <p className="text-xs text-[#00ff99] leading-relaxed">
                        {node.preset.howItWorks}
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-[#00cc77] bg-black">
                <button
                  onClick={() => setShowNodeInfo(null)}
                  className="px-3 py-1.5 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        );
      })()}

      {/* Delete Connection Confirmation Modal */}
      {deleteConfirmation && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-black/95 border-2 border-[#f85149] rounded-lg shadow-glow-red p-6 max-w-md w-full mx-4">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-[#f85149]/20 border border-[#f85149] flex items-center justify-center">
                <svg className="w-6 h-6 text-[#f85149]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-[#f85149]">Delete Connection</h3>
            </div>

            <p className="text-sm text-[#00ff99] mb-6">
              Are you sure you want to delete this connection? This action cannot be undone.
            </p>

            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setDeleteConfirmation(null)}
                className="px-4 py-2 text-sm bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  deleteConnection(deleteConfirmation.connectionId);
                  setDeleteConfirmation(null);
                  setHoveredConnection(null);
                }}
                className="px-4 py-2 text-sm bg-[#f85149] text-white rounded hover:bg-[#ff6b6b] transition font-semibold"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

