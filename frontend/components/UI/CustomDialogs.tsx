'use client';

import React, { useState } from 'react';

interface AlertProps {
  message: string;
  onClose: () => void;
}

export const CustomAlert: React.FC<AlertProps> = ({ message, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999]">
      <div className="bg-black/95 border-2 border-[#00cc77] rounded-lg shadow-glow p-6 max-w-md w-full mx-4">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-[#00ff99]/20 border border-[#00ff99] flex items-center justify-center">
            <svg className="w-6 h-6 text-[#00ff99]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-[#00ff99]">Notice</h3>
        </div>
        
        <p className="text-sm text-[#00ff99] mb-6 leading-relaxed">{message}</p>
        
        <div className="flex justify-end">
          <button
            onClick={onClose}
            className="px-6 py-2 text-sm bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition font-semibold"
          >
            OK
          </button>
        </div>
      </div>
    </div>
  );
};

interface ConfirmProps {
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export const CustomConfirm: React.FC<ConfirmProps> = ({ message, onConfirm, onCancel }) => {
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999]">
      <div className="bg-black/95 border-2 border-[#00cc77] rounded-lg shadow-glow p-6 max-w-md w-full mx-4">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-[#58a6ff]/20 border border-[#58a6ff] flex items-center justify-center">
            <svg className="w-6 h-6 text-[#58a6ff]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-[#00ff99]">Confirm Action</h3>
        </div>
        
        <p className="text-sm text-[#00ff99] mb-6 leading-relaxed">{message}</p>
        
        <div className="flex items-center justify-end gap-3">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="px-6 py-2 text-sm bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition font-semibold"
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
};

interface PromptProps {
  message: string;
  defaultValue?: string;
  placeholder?: string;
  onSubmit: (value: string) => void;
  onCancel: () => void;
}

export const CustomPrompt: React.FC<PromptProps> = ({ message, defaultValue = '', placeholder, onSubmit, onCancel }) => {
  const [value, setValue] = useState(defaultValue);

  const handleSubmit = () => {
    if (value.trim()) {
      onSubmit(value);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[9999]">
      <div className="bg-black/95 border-2 border-[#00cc77] rounded-lg shadow-glow p-6 max-w-md w-full mx-4">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-[#00ff99]/20 border border-[#00ff99] flex items-center justify-center">
            <svg className="w-6 h-6 text-[#00ff99]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-[#00ff99]">Input Required</h3>
        </div>
        
        <p className="text-sm text-[#00ff99] mb-4 leading-relaxed">{message}</p>
        
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSubmit();
            if (e.key === 'Escape') onCancel();
          }}
          placeholder={placeholder}
          className="w-full px-3 py-2 bg-black border border-[#00cc77] text-[#00ff99] rounded focus:border-[#00ff99] outline-none mb-6"
          autoFocus
        />
        
        <div className="flex items-center justify-end gap-3">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm bg-[#1a1a1a] border border-[#00cc77] text-[#00ff99] rounded hover:bg-[#00cc77]/20 transition"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            className="px-6 py-2 text-sm bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition font-semibold"
          >
            OK
          </button>
        </div>
      </div>
    </div>
  );
};

// Helper functions to use dialogs
export const useCustomDialog = () => {
  const [alertState, setAlertState] = useState<{ message: string; show: boolean }>({ message: '', show: false });
  const [confirmState, setConfirmState] = useState<{ message: string; show: boolean; resolve?: (value: boolean) => void }>({ 
    message: '', 
    show: false 
  });
  const [promptState, setPromptState] = useState<{ 
    message: string; 
    defaultValue: string;
    placeholder: string;
    show: boolean; 
    resolve?: (value: string | null) => void 
  }>({ 
    message: '', 
    defaultValue: '',
    placeholder: '',
    show: false 
  });

  const customAlert = (message: string): Promise<void> => {
    return new Promise((resolve) => {
      setAlertState({ message, show: true });
      setTimeout(() => resolve(), 100);
    });
  };

  const customConfirm = (message: string): Promise<boolean> => {
    return new Promise((resolve) => {
      setConfirmState({ message, show: true, resolve });
    });
  };

  const customPrompt = (message: string, defaultValue = '', placeholder = ''): Promise<string | null> => {
    return new Promise((resolve) => {
      setPromptState({ message, defaultValue, placeholder, show: true, resolve });
    });
  };

  const AlertDialog = alertState.show ? (
    <CustomAlert
      message={alertState.message}
      onClose={() => {
        setAlertState({ message: '', show: false });
      }}
    />
  ) : null;

  const ConfirmDialog = confirmState.show ? (
    <CustomConfirm
      message={confirmState.message}
      onConfirm={() => {
        confirmState.resolve?.(true);
        setConfirmState({ message: '', show: false });
      }}
      onCancel={() => {
        confirmState.resolve?.(false);
        setConfirmState({ message: '', show: false });
      }}
    />
  ) : null;

  const PromptDialog = promptState.show ? (
    <CustomPrompt
      message={promptState.message}
      defaultValue={promptState.defaultValue}
      placeholder={promptState.placeholder}
      onSubmit={(value) => {
        promptState.resolve?.(value);
        setPromptState({ message: '', defaultValue: '', placeholder: '', show: false });
      }}
      onCancel={() => {
        promptState.resolve?.(null);
        setPromptState({ message: '', defaultValue: '', placeholder: '', show: false });
      }}
    />
  ) : null;

  return {
    customAlert,
    customConfirm,
    customPrompt,
    AlertDialog,
    ConfirmDialog,
    PromptDialog
  };
};

