"use client";

import clsx from 'clsx';
import { useColorSchemeStore } from '@/lib/stores/colorScheme';
import { useSimulationStore } from '@/lib/stores/simulation';

export default function HUD() {
  const { scheme, definition } = useColorSchemeStore((state) => ({
    scheme: state.scheme,
    definition: state.definition,
  }));
  const { connectionState, timestamp, activeCount, simulationRunning } = useSimulationStore((state) => ({
    connectionState: state.connectionState,
    timestamp: state.timestamp,
    activeCount: state.activeCount,
    simulationRunning: state.simulationRunning,
  }));

  const wsStatus = (() => {
    switch (connectionState) {
      case 'connected':
        return { label: 'WS: LINKED', dotClass: 'bg-cyber-cyan-glow animate-pulse' };
      case 'reconnecting':
        return { label: 'WS: RECONNECTING', dotClass: 'bg-yellow-400 animate-pulse' };
      case 'connecting':
        return { label: 'WS: CONNECTING', dotClass: 'bg-yellow-300 animate-pulse' };
      case 'stale':
        return { label: 'WS: STALE', dotClass: 'bg-red-500 animate-pulse' };
      default:
        return { label: 'WS: DISCONNECTED', dotClass: 'bg-red-700 animate-pulse' };
    }
  })();

  const simStatus = (() => {
    if (connectionState === 'connected') {
      if (simulationRunning) {
        return { label: 'SIM: RUNNING', dotClass: 'bg-emerald-400 animate-pulse' };
      }
      return { label: 'SIM: PAUSED', dotClass: 'bg-yellow-400' };
    }

    if (connectionState === 'stale') {
      return { label: 'SIM: STALE', dotClass: 'bg-red-500 animate-pulse' };
    }

    if (connectionState === 'reconnecting' || connectionState === 'connecting') {
      return { label: 'SIM: SYNCING', dotClass: 'bg-yellow-300 animate-pulse' };
    }

    return { label: 'SIM: OFFLINE', dotClass: 'bg-red-700' };
  })();

  return (
    <div className="pointer-events-none fixed inset-0 flex flex-col justify-between p-4 md:p-6">
      <div className="flex flex-col md:flex-row justify-between text-[10px] md:text-xs tracking-[0.2em] md:tracking-[0.4em] text-glitch gap-2">
        <span className="whitespace-nowrap overflow-hidden text-ellipsis">GMCS (GENERALIZED MODULAR CONTROL SYSTEM) V0.1</span>
        <span className="whitespace-nowrap">{definition.name}</span>
      </div>

      <div className="pointer-events-auto mt-auto grid gap-1 md:gap-2 text-[9px] md:text-[10px] opacity-90">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <span className="whitespace-nowrap">SCHEME: {scheme.toUpperCase()}</span>
          <span className="text-[8px] md:text-[9px] opacity-70">PRESS &quot;C&quot;</span>
        </div>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <span className="whitespace-nowrap">
            T&nbsp;=
            {timestamp.toFixed(2)}
          </span>
          <span className="whitespace-nowrap">NODES&nbsp;=&nbsp;{activeCount}</span>
        </div>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3 md:gap-4">
            <span className="flex items-center gap-1 md:gap-2 whitespace-nowrap">
              <span className={clsx('status-dot w-2 h-2 md:w-3 md:h-3 rounded-full', wsStatus.dotClass)} />
              {wsStatus.label}
            </span>
            <span className="flex items-center gap-1 md:gap-2 whitespace-nowrap">
              <span className={clsx('status-dot w-2 h-2 md:w-3 md:h-3 rounded-full', simStatus.dotClass)} />
              {simStatus.label}
            </span>
          </div>
          <span className="whitespace-nowrap">HUD ONLINE</span>
        </div>
      </div>
    </div>
  );
}
