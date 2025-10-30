'use client';

import React, { useState } from 'react';

interface Session {
  id: string;
  name: string;
  timestamp: string;
  nodes: number;
  connections: number;
}

export const SessionManagerPanel: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [sessions, setSessions] = useState<Session[]>([
    {
      id: 'session-1',
      name: 'Audio Reactive Setup',
      timestamp: '2025-10-30 01:15:00',
      nodes: 12,
      connections: 8
    },
    {
      id: 'session-2',
      name: 'THRML Experiment',
      timestamp: '2025-10-29 18:30:00',
      nodes: 8,
      connections: 5
    }
  ]);
  const [sessionName, setSessionName] = useState('');

  const saveSession = () => {
    if (!sessionName) return;
    const newSession: Session = {
      id: `session-${Date.now()}`,
      name: sessionName,
      timestamp: new Date().toLocaleString(),
      nodes: 0,
      connections: 0
    };
    setSessions([newSession, ...sessions]);
    setSessionName('');
  };

  const deleteSession = (id: string) => {
    setSessions(sessions.filter(s => s.id !== id));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="w-[700px] max-h-[85vh] bg-black/60 border border-[#00cc77] rounded-lg shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#00cc77]">
          <h2 className="text-sm font-semibold text-[#00ff99]">Session Manager</h2>
          <button onClick={onClose} className="text-[#00cc77] hover:text-[#00ff99] text-xl">×</button>
        </div>

        <div className="p-4 overflow-y-auto max-h-[calc(85vh-120px)] custom-scrollbar">
          {/* Save Current Session */}
          <div className="mb-4 p-4 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded">
            <h3 className="text-xs font-semibold text-[#00ff99] mb-3">Save Current Session</h3>
            <div className="flex gap-2">
              <input
                type="text"
                value={sessionName}
                onChange={(e) => setSessionName(e.target.value)}
                placeholder="Session name..."
                className="flex-1 bg-black/60 backdrop-blur-md border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 text-xs focus:border-[#00ff99] focus:outline-none"
              />
              <button
                onClick={saveSession}
                disabled={!sessionName}
                className="px-4 py-2 text-xs bg-[#00cc77] text-white rounded hover:bg-[#2ea043] transition font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Save
              </button>
            </div>
          </div>

          {/* Saved Sessions */}
          <div className="space-y-2">
            {sessions.length === 0 ? (
              <div className="text-center py-8 text-[#00cc77] text-sm">
                No saved sessions
              </div>
            ) : (
              sessions.map(session => (
                <div key={session.id} className="p-3 bg-black/60 backdrop-blur-md border border-[#00cc77] rounded hover:border-[#00ff99] transition">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <div className="text-sm font-semibold text-[#00ff99]">{session.name}</div>
                      <div className="text-xs text-[#00cc77] mt-1">{session.timestamp}</div>
                    </div>
                    <button
                      onClick={() => deleteSession(session.id)}
                      className="text-[#f85149] hover:text-[#ff7b72] text-sm"
                    >
                      ×
                    </button>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-[#00cc77] mb-3">
                    <span>{session.nodes} nodes</span>
                    <span>•</span>
                    <span>{session.connections} connections</span>
                  </div>
                  <div className="flex gap-2">
                    <button className="flex-1 px-3 py-1.5 text-xs bg-[#00cc77] text-white rounded hover:bg-[#2ea043] transition font-semibold">
                      Load
                    </button>
                    <button className="px-3 py-1.5 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition">
                      Export
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Import */}
          <div className="mt-4 pt-4 border-t border-[#00cc77]">
            <button className="w-full px-3 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition">
              Import Session from File
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between px-4 py-3 border-t border-[#00cc77] bg-black/60">
          <div className="text-xs text-[#00cc77]">
            {sessions.length} saved session{sessions.length !== 1 ? 's' : ''}
          </div>
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

