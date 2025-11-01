"use client";

import { useState } from "react";
import clsx from "clsx";
import { useControlsStore } from "@/lib/stores/controls";
import { useSimulationStore, MAX_SIMULATION_NODES } from "@/lib/stores/simulation";
import { notify } from "@/components/UI/Notification";
import { THRMLControls } from "./THRMLControls";

const sliderClass =
  "w-full appearance-none bg-bg-secondary h-1 rounded-full outline-none transition focus:ring-1 focus:ring-cyber-cyan-glow";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface NodeResponsePayload {
  status: string;
  node_id?: number;
  message?: string;
  data?: Record<string, unknown>;
}

interface NodeListItemPayload {
  node_id: number;
}

interface NodeListResponsePayload {
  nodes: NodeListItemPayload[];
}

async function requestJSON<T>(path: string, init: RequestInit = {}): Promise<T | null> {
  try {
    const response = await fetch(`${API_BASE}${path}`, init);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed with status ${response.status}`);
    }
    return (await response.json()) as T;
  } catch (error) {
    console.warn("Control panel request failed", error);
    return null;
  }
}

async function postJSON<T>(path: string, body: unknown): Promise<T | null> {
  return requestJSON<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function deleteJSON<T>(path: string): Promise<T | null> {
  return requestJSON<T>(path, {
    method: "DELETE",
  });
}

async function getJSON<T>(path: string): Promise<T | null> {
  return requestJSON<T>(path);
}

export default function ControlPanel() {
  const {
    displacementGain,
    setDisplacementGain,
    particleScale,
    setParticleScale,
    chromaticAberration,
    setChromaticAberration,
    showGrid,
    toggleGrid,
    particleLimit,
    setParticleLimit,
  } = useControlsStore();
  const { activeCount } = useSimulationStore((state) => ({
    activeCount: state.activeCount,
  }));

  const [pending, setPending] = useState(false);

  const handleAddNode = async () => {
    setPending(true);
    try {
      const payload = {
        position: [128, 128],
        config: {},
        chain: [],
        initial_perturbation: 0.1,
      };
      const result = await postJSON<NodeResponsePayload>("/node/add", payload);
      if (!result || result.status !== "success") {
        notify.error(result?.message ?? "Failed to add node");
        return;
      }
      notify.success(result.message ?? "Node added");
    } finally {
      setPending(false);
    }
  };

  const handleRemoveNode = async () => {
    setPending(true);
    try {
      const list = await getJSON<NodeListResponsePayload>("/nodes");
      if (!list || list.nodes.length === 0) {
        notify.warning("No active nodes to remove");
        return;
      }
      const targetId = list.nodes[list.nodes.length - 1].node_id;
      const result = await deleteJSON<NodeResponsePayload>(`/node/${targetId}`);
      if (!result || result.status !== "success") {
        notify.error(result?.message ?? "Failed to remove node");
        return;
      }
      notify.success(result.message ?? `Node ${targetId} removed`);
    } finally {
      setPending(false);
    }
  };

  return (
    <aside className="pointer-events-auto fixed right-2 md:right-6 top-16 md:top-24 w-64 md:w-72 max-h-[calc(100vh-5rem)] md:max-h-[calc(100vh-7rem)] overflow-y-auto custom-scrollbar rounded-lg border border-cyber-cyan-border bg-black/60 p-3 md:p-4 shadow-glow-cyber backdrop-blur-xl">
      <h2 className="mb-3 md:mb-4 text-[10px] md:text-xs tracking-[0.25em] md:tracking-[0.35em] text-cyber-cyan-glow">CONTROL</h2>

      <section className="space-y-2 md:space-y-3">
        <header className="text-[9px] md:text-[10px] uppercase text-cyber-cyan-code">VISUAL</header>
        <label className="grid gap-1 text-[9px] md:text-[10px]">
          <span className="whitespace-nowrap overflow-hidden text-ellipsis">DISPLACEMENT [{displacementGain.toFixed(2)}]</span>
          <input
            type="range"
            min={0.5}
            max={4.0}
            step={0.05}
            className={sliderClass}
            value={displacementGain}
            onChange={(event) => setDisplacementGain(Number(event.target.value))}
          />
        </label>
        <label className="grid gap-1 text-[9px] md:text-[10px]">
          <span className="whitespace-nowrap overflow-hidden text-ellipsis">PARTICLE MASS [{particleScale.toFixed(2)}]</span>
          <input
            type="range"
            min={0.4}
            max={2.5}
            step={0.05}
            className={sliderClass}
            value={particleScale}
            onChange={(event) => setParticleScale(Number(event.target.value))}
          />
        </label>
        <label className="grid gap-1 text-[9px] md:text-[10px]">
          <span className="whitespace-nowrap overflow-hidden text-ellipsis">ABERRATION [{chromaticAberration.toFixed(2)}]</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            className={sliderClass}
            value={chromaticAberration}
            onChange={(event) => setChromaticAberration(Number(event.target.value))}
          />
        </label>
        <label className="grid gap-1 text-[9px] md:text-[10px]">
          <span className="whitespace-nowrap overflow-hidden text-ellipsis">PARTICLE CAP [{particleLimit}]</span>
          <input
            type="range"
            min={64}
            max={MAX_SIMULATION_NODES}
            step={32}
            className={sliderClass}
            value={particleLimit}
            onChange={(event) => setParticleLimit(Number(event.target.value))}
          />
        </label>
        <div className="flex items-center justify-between text-[9px] md:text-[10px]">
          <span>GRID OVERLAY</span>
          <button
            type="button"
            onClick={toggleGrid}
            className={clsx(
              "rounded border px-2 py-0.5 md:py-1 text-[9px] md:text-[10px] transition",
              showGrid
                ? "border-cyber-cyan-glow text-cyber-cyan-glow"
                : "border-bg-secondary text-bg-secondary"
            )}
          >
            {showGrid ? "ON" : "OFF"}
          </button>
        </div>
      </section>

      <section className="mt-4 md:mt-5 space-y-2 md:space-y-3">
        <header className="text-[9px] md:text-[10px] uppercase text-cyber-cyan-code">NODES</header>
        <div className="flex items-center justify-between text-[9px] md:text-[10px]">
          <span>ACTIVE</span>
          <span className="font-mono text-cyber-cyan-glow">{activeCount}</span>
        </div>
        <div className="flex gap-1.5 md:gap-2 text-[9px] md:text-[10px]">
          <button
            type="button"
            onClick={handleAddNode}
            disabled={pending}
            className="flex-1 rounded border border-cyber-cyan-glow py-1 text-cyber-cyan-glow transition hover:bg-cyber-cyan-glow/10 disabled:cursor-wait"
          >
            ADD NODE
          </button>
          <button
            type="button"
            onClick={handleRemoveNode}
            disabled={pending || activeCount === 0}
            className="flex-1 rounded border border-red-500 py-1 text-red-400 transition hover:bg-red-500/10 disabled:cursor-not-allowed"
          >
            REMOVE
          </button>
        </div>
      </section>

      <section className="mt-4 md:mt-5">
        <THRMLControls />
      </section>
    </aside>
  );
}
