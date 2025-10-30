"use client";

import clsx from 'clsx';
import { useColorSchemeStore } from '@/lib/stores/colorScheme';
import { useSimulationStore } from '@/lib/stores/simulation';

export default function HUD() {
  const { scheme, definition } = useColorSchemeStore((state) => ({
    scheme: state.scheme,
    definition: state.definition,
  }));
  const { connected, timestamp, activeCount } = useSimulationStore((state) => ({
    connected: state.connected,
    timestamp: state.timestamp,
    activeCount: state.activeCount,
  }));

  return (
    <div className="pointer-events-none fixed inset-0 flex flex-col justify-between p-4 md:p-6">
      <div className="flex flex-col md:flex-row justify-between text-[10px] md:text-xs tracking-[0.2em] md:tracking-[0.4em] text-glitch gap-2">
        <span className="whitespace-nowrap overflow-hidden text-ellipsis">GMCS (GENERALIZED MODULAR CONTROL SYSTEM) V0.1</span>
        <span className="whitespace-nowrap">{definition.name}</span>
      </div>

      <div className="pointer-events-auto mt-auto grid gap-1 md:gap-2 text-[9px] md:text-[10px] opacity-90">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <span className="whitespace-nowrap">SCHEME: {scheme.toUpperCase()}</span>
          <span className="text-[8px] md:text-[9px] opacity-70">PRESS "C"</span>
        </div>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <span className="whitespace-nowrap">
            T&nbsp;=
            {timestamp.toFixed(2)}
          </span>
          <span className="whitespace-nowrap">NODES&nbsp;=&nbsp;{activeCount}</span>
        </div>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <span className="flex items-center gap-1 md:gap-2">
            <span
              className={clsx('status-dot w-2 h-2 md:w-3 md:h-3', connected ? 'bg-cyber-cyan-glow' : 'bg-red-500')}
            />
            {connected ? 'STREAM: LINKED' : 'SEARCHING'}
          </span>
          <span className="whitespace-nowrap">HUD ONLINE</span>
        </div>
      </div>
    </div>
  );
}
