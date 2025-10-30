/**
 * Session Manager Component
 * 
 * Save, load, and manage simulation sessions.
 */

'use client';

import { useState, useEffect } from 'react';

interface Session {
  session_id: string;
  name: string;
  created_at: string;
}

export default function SessionManager() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [sessionName, setSessionName] = useState('');

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const response = await fetch('http://localhost:8000/session/list');
      const data = await response.json();
      setSessions(data.sessions);
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    }
  };

  const saveSession = async () => {
    if (!sessionName.trim()) return;
    
    try {
      await fetch(`http://localhost:8000/session/save?name=${encodeURIComponent(sessionName)}`, {
        method: 'POST'
      });
      await fetchSessions();
      setShowSaveDialog(false);
      setSessionName('');
    } catch (error) {
      console.error('Failed to save session:', error);
    }
  };

  const loadSession = async (sessionId: string) => {
    try {
      await fetch(`http://localhost:8000/session/${sessionId}/load`, {
        method: 'POST'
      });
      alert('Session loaded successfully!');
    } catch (error) {
      console.error('Failed to load session:', error);
    }
  };

  const deleteSession = async (sessionId: string) => {
    if (!confirm('Delete this session?')) return;
    
    try {
      await fetch(`http://localhost:8000/session/${sessionId}`, {
        method: 'DELETE'
      });
      await fetchSessions();
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  };

  return (
    <div className="session-manager bg-gray-900 border border-green-500/30 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-green-400 text-xl font-bold">Session Manager</h3>
        <button
          onClick={() => setShowSaveDialog(true)}
          className="px-4 py-2 bg-[#00ff99] rounded text-gray-900 font-medium hover:bg-green-400 transition-all"
        >
          💾 Save Session
        </button>
      </div>

      {/* Sessions List */}
      <div className="space-y-2">
        {sessions.length === 0 ? (
          <div className="text-green-400/60 text-center py-8">
            No saved sessions
          </div>
        ) : (
          sessions.map(session => (
            <div
              key={session.session_id}
              className="bg-gray-800 border border-green-500/20 rounded p-4 hover:border-green-400 transition-all"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-green-300 font-medium">{session.name}</h4>
                  <p className="text-green-400/60 text-sm">
                    {new Date(session.created_at).toLocaleString()}
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => loadSession(session.session_id)}
                    className="px-3 py-1 bg-[#00ff99]/20 border border-green-500/30 rounded text-green-400 text-sm hover:border-green-400 transition-all"
                  >
                    Load
                  </button>
                  <button
                    onClick={() => deleteSession(session.session_id)}
                    className="px-3 py-1 bg-red-500/20 border border-red-500/30 rounded text-red-400 text-sm hover:border-red-400 transition-all"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Save Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-green-500/50 rounded-lg p-6 w-full max-w-md">
            <h4 className="text-green-400 text-lg font-bold mb-4">Save Session</h4>
            
            <input
              type="text"
              placeholder="Session name..."
              value={sessionName}
              onChange={(e) => setSessionName(e.target.value)}
              className="w-full bg-gray-800 border border-green-500/30 rounded px-3 py-2 text-green-100 mb-4"
              onKeyPress={(e) => e.key === 'Enter' && saveSession()}
            />

            <div className="flex gap-2">
              <button
                onClick={saveSession}
                className="flex-1 bg-[#00ff99] rounded py-2 text-gray-900 font-medium hover:bg-green-400 transition-all"
              >
                Save
              </button>
              <button
                onClick={() => setShowSaveDialog(false)}
                className="flex-1 bg-gray-800 border border-green-500/30 rounded py-2 text-green-400 hover:border-green-400 transition-all"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

