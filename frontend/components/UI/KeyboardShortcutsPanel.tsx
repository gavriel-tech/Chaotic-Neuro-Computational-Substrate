'use client';

import React, { useState } from 'react';
import { useShortcutStore, formatShortcut, getShortcutsByCategory } from '@/lib/hooks/useKeyboardShortcuts';

interface KeyboardShortcutsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const KeyboardShortcutsPanel: React.FC<KeyboardShortcutsPanelProps> = ({
  isOpen,
  onClose
}) => {
  const { shortcuts } = useShortcutStore();
  const [searchTerm, setSearchTerm] = useState('');

  if (!isOpen) return null;

  const groupedShortcuts = getShortcutsByCategory(shortcuts);
  
  // Filter shortcuts by search term
  const filteredGroups = Object.entries(groupedShortcuts).reduce((acc, [category, items]) => {
    const filtered = items.filter(
      shortcut =>
        shortcut.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        formatShortcut(shortcut).toLowerCase().includes(searchTerm.toLowerCase())
    );
    if (filtered.length > 0) {
      acc[category] = filtered;
    }
    return acc;
  }, {} as Record<string, typeof items>);

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[9998] animate-fade-in"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="fixed inset-0 flex items-center justify-center z-[9999] pointer-events-none">
        <div className="panel bg-black/60 backdrop-blur-md border-2 border-cyber-cyan-core rounded-lg w-full max-w-3xl max-h-[80vh] overflow-hidden shadow-[0_0_40px_rgba(0,247,255,0.6)] pointer-events-auto animate-scale-in">
          {/* Header */}
          <div className="px-6 py-4 border-b border-cyber-cyan-core/30">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-cyber-cyan-core">
                  Keyboard Shortcuts
                </h2>
                <p className="text-sm text-cyber-cyan-muted mt-1">
                  Master GMCS with these keyboard commands
                </p>
              </div>
              <button
                onClick={onClose}
                className="text-cyber-cyan-core hover:text-cyber-cyan-glow text-3xl leading-none transition-colors"
                aria-label="Close"
              >
                ×
              </button>
            </div>

            {/* Search */}
            <div className="mt-4">
              <input
                type="text"
                placeholder="Search shortcuts..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full bg-gray-900 border border-cyber-cyan-muted rounded px-3 py-2 text-cyber-cyan-glow font-mono text-sm focus:outline-none focus:border-cyber-cyan-core transition-colors"
              />
            </div>
          </div>

          {/* Shortcuts list */}
          <div className="px-6 py-4 overflow-y-auto max-h-[calc(80vh-200px)] custom-scrollbar">
            {Object.entries(filteredGroups).length === 0 ? (
              <div className="text-center py-8 text-cyber-cyan-muted">
                No shortcuts found matching "{searchTerm}"
              </div>
            ) : (
              <div className="space-y-6">
                {Object.entries(filteredGroups).map(([category, items]) => (
                  <div key={category}>
                    {/* Category title */}
                    <h3 className="text-lg font-semibold text-cyber-magenta-glow mb-3 flex items-center gap-2">
                      <div className="h-0.5 w-8 bg-cyber-magenta-glow" />
                      {category}
                    </h3>

                    {/* Shortcuts in category */}
                    <div className="space-y-2">
                      {items.map((shortcut, index) => (
                        <div
                          key={`${category}-${index}`}
                          className="flex items-center justify-between py-2 px-3 rounded bg-gray-900/50 hover:bg-gray-900/80 transition-colors"
                        >
                          <span className="text-cyber-cyan-glow font-mono text-sm">
                            {shortcut.description}
                          </span>
                          <kbd className="flex items-center gap-1 px-3 py-1.5 bg-gray-800 border border-cyber-cyan-muted rounded text-cyber-cyan-core font-mono text-sm shadow-[0_0_10px_rgba(0,247,255,0.3)]">
                            {formatShortcut(shortcut)}
                          </kbd>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-6 py-3 border-t border-cyber-cyan-core/30 bg-gray-900/50">
            <p className="text-xs text-cyber-cyan-muted text-center font-mono">
              Press <kbd className="px-2 py-0.5 bg-gray-800 border border-cyber-cyan-muted rounded">Shift + ?</kbd> to toggle this panel
            </p>
          </div>
        </div>
      </div>
    </>
  );
};

// Hook to toggle the panel
export const useShortcutsPanel = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggle = () => setIsOpen(prev => !prev);
  const open = () => setIsOpen(true);
  const close = () => setIsOpen(false);

  return { isOpen, toggle, open, close };
};

export default KeyboardShortcutsPanel;

