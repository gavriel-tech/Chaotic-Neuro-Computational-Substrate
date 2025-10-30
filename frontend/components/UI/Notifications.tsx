'use client';

import React, { createContext, useContext, useState, useCallback } from 'react';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
}

interface NotificationContextType {
  showNotification: (message: string, type?: 'success' | 'error' | 'warning' | 'info') => void;
  showConfirm: (message: string) => Promise<boolean>;
  showPrompt: (message: string, defaultValue?: string) => Promise<string | null>;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within NotificationProvider');
  }
  return context;
};

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [confirmDialog, setConfirmDialog] = useState<{ message: string; resolve: (value: boolean) => void } | null>(null);
  const [promptDialog, setPromptDialog] = useState<{ message: string; defaultValue: string; resolve: (value: string | null) => void } | null>(null);
  const [promptValue, setPromptValue] = useState('');

  const showNotification = useCallback((message: string, type: 'success' | 'error' | 'warning' | 'info' = 'info') => {
    const id = Date.now().toString();
    setNotifications(prev => [...prev, { id, type, message }]);
    
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 3000);
  }, []);

  const showConfirm = useCallback((message: string): Promise<boolean> => {
    return new Promise((resolve) => {
      setConfirmDialog({ message, resolve });
    });
  }, []);

  const showPrompt = useCallback((message: string, defaultValue: string = ''): Promise<string | null> => {
    return new Promise((resolve) => {
      setPromptValue(defaultValue);
      setPromptDialog({ message, defaultValue, resolve });
    });
  }, []);

  const handleConfirm = (value: boolean) => {
    if (confirmDialog) {
      confirmDialog.resolve(value);
      setConfirmDialog(null);
    }
  };

  const handlePromptSubmit = () => {
    if (promptDialog) {
      promptDialog.resolve(promptValue || null);
      setPromptDialog(null);
      setPromptValue('');
    }
  };

  const handlePromptCancel = () => {
    if (promptDialog) {
      promptDialog.resolve(null);
      setPromptDialog(null);
      setPromptValue('');
    }
  };

  return (
    <NotificationContext.Provider value={{ showNotification, showConfirm, showPrompt }}>
      {children}
      
      {/* Toast Notifications */}
      <div className="fixed top-4 right-4 z-[10000] flex flex-col gap-2">
        {notifications.map(notification => (
          <div
            key={notification.id}
            className={`px-4 py-3 rounded border backdrop-blur-md shadow-glow animate-slide-in min-w-[300px] ${
              notification.type === 'success' ? 'bg-[#00ff99]/20 border-[#00ff99] text-[#00ff99]' :
              notification.type === 'error' ? 'bg-[#f85149]/20 border-[#f85149] text-[#f85149]' :
              notification.type === 'warning' ? 'bg-[#d29922]/20 border-[#d29922] text-[#d29922]' :
              'bg-[#00cc77]/20 border-[#00cc77] text-[#00ff99]'
            }`}
          >
            <p className="text-sm font-mono">{notification.message}</p>
          </div>
        ))}
      </div>

      {/* Confirm Dialog */}
      {confirmDialog && (
        <>
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[10001]" onClick={() => handleConfirm(false)} />
          <div className="fixed inset-0 flex items-center justify-center z-[10002] pointer-events-none">
            <div className="bg-black/95 border border-[#00cc77] rounded-lg p-6 shadow-glow max-w-md pointer-events-auto">
              <p className="text-[#00ff99] text-base mb-6 font-mono">{confirmDialog.message}</p>
              <div className="flex gap-3 justify-end">
                <button
                  onClick={() => handleConfirm(false)}
                  className="px-4 py-2 bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition font-mono"
                >
                  Cancel
                </button>
                <button
                  onClick={() => handleConfirm(true)}
                  className="px-4 py-2 bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition font-semibold font-mono"
                >
                  Confirm
                </button>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Prompt Dialog */}
      {promptDialog && (
        <>
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[10001]" onClick={handlePromptCancel} />
          <div className="fixed inset-0 flex items-center justify-center z-[10002] pointer-events-none">
            <div className="bg-black/95 border border-[#00cc77] rounded-lg p-6 shadow-glow max-w-md w-full mx-4 pointer-events-auto">
              <p className="text-[#00ff99] text-base mb-4 font-mono">{promptDialog.message}</p>
              <input
                type="text"
                value={promptValue}
                onChange={(e) => setPromptValue(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handlePromptSubmit()}
                className="w-full bg-black/60 border border-[#00cc77] text-[#00ff99] rounded px-3 py-2 mb-6 font-mono focus:outline-none focus:border-[#00ff99]"
                autoFocus
              />
              <div className="flex gap-3 justify-end">
                <button
                  onClick={handlePromptCancel}
                  className="px-4 py-2 bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition font-mono"
                >
                  Cancel
                </button>
                <button
                  onClick={handlePromptSubmit}
                  className="px-4 py-2 bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition font-semibold font-mono"
                >
                  OK
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </NotificationContext.Provider>
  );
};

