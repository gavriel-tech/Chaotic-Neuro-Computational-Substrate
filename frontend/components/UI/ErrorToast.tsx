'use client';

import React, { useEffect, useState } from 'react';
import { create } from 'zustand';

// Toast store for managing notifications
interface Toast {
  id: string;
  type: 'error' | 'warning' | 'success' | 'info';
  message: string;
  duration?: number;
}

interface ToastStore {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  clearAll: () => void;
}

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  addToast: (toast) => {
    const id = `${Date.now()}-${Math.random()}`;
    set((state) => ({
      toasts: [...state.toasts, { ...toast, id }]
    }));

    // Auto-remove after duration
    if (toast.duration !== 0) {
      setTimeout(() => {
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id)
        }));
      }, toast.duration || 5000);
    }
  },
  removeToast: (id) => set((state) => ({
    toasts: state.toasts.filter((t) => t.id !== id)
  })),
  clearAll: () => set({ toasts: [] })
}));

// Toast component
interface ToastItemProps {
  toast: Toast;
  onClose: (id: string) => void;
}

const ToastItem: React.FC<ToastItemProps> = ({ toast, onClose }) => {
  const [isExiting, setIsExiting] = useState(false);

  const handleClose = () => {
    setIsExiting(true);
    setTimeout(() => onClose(toast.id), 300);
  };

  const getStyles = () => {
    switch (toast.type) {
      case 'error':
        return {
          bg: 'bg-red-900/90',
          border: 'border-red-500',
          text: 'text-red-200',
          icon: '⚠️',
          glow: 'shadow-[0_0_20px_rgba(239,68,68,0.5)]'
        };
      case 'warning':
        return {
          bg: 'bg-yellow-900/90',
          border: 'border-yellow-500',
          text: 'text-yellow-200',
          icon: '⚡',
          glow: 'shadow-[0_0_20px_rgba(234,179,8,0.5)]'
        };
      case 'success':
        return {
          bg: 'bg-cyber-cyan-core/20',
          border: 'border-cyber-cyan-core',
          text: 'text-cyber-cyan-glow',
          icon: '✓',
          glow: 'shadow-[0_0_20px_rgba(0,247,255,0.5)]'
        };
      case 'info':
        return {
          bg: 'bg-cyber-magenta-glow/20',
          border: 'border-cyber-magenta-glow',
          text: 'text-cyber-magenta-accent',
          icon: 'ℹ',
          glow: 'shadow-[0_0_20px_rgba(255,0,255,0.5)]'
        };
    }
  };

  const styles = getStyles();

  return (
    <div
      className={`
        ${styles.bg} ${styles.border} ${styles.glow}
        border backdrop-blur-md rounded-lg p-4 min-w-[300px] max-w-[500px]
        transition-all duration-300 ease-out
        ${isExiting ? 'opacity-0 translate-x-full' : 'opacity-100 translate-x-0'}
      `}
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className="text-2xl">{styles.icon}</div>

        {/* Message */}
        <div className="flex-1">
          <p className={`${styles.text} font-mono text-sm leading-relaxed`}>
            {toast.message}
          </p>
        </div>

        {/* Close button */}
        <button
          onClick={handleClose}
          className={`${styles.text} hover:opacity-100 opacity-60 transition-opacity text-xl leading-none`}
          aria-label="Close notification"
        >
          ×
        </button>
      </div>

      {/* Progress bar */}
      {toast.duration && toast.duration > 0 && (
        <div className="mt-2 h-0.5 w-full bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full ${styles.border.replace('border-', 'bg-')}`}
            style={{
              animation: `shrink ${toast.duration}ms linear`
            }}
          />
        </div>
      )}
    </div>
  );
};

// Toast container
export const ToastContainer: React.FC = () => {
  const { toasts, removeToast } = useToastStore();

  return (
    <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-3 pointer-events-none">
      {toasts.map((toast) => (
        <div key={toast.id} className="pointer-events-auto">
          <ToastItem toast={toast} onClose={removeToast} />
        </div>
      ))}
    </div>
  );
};

// Utility functions for easy toast creation
export const toast = {
  error: (message: string, duration?: number) => {
    useToastStore.getState().addToast({ type: 'error', message, duration });
  },
  warning: (message: string, duration?: number) => {
    useToastStore.getState().addToast({ type: 'warning', message, duration });
  },
  success: (message: string, duration?: number) => {
    useToastStore.getState().addToast({ type: 'success', message, duration });
  },
  info: (message: string, duration?: number) => {
    useToastStore.getState().addToast({ type: 'info', message, duration });
  }
};

// CSS for progress bar animation (add to globals.css)
const progressBarCSS = `
@keyframes shrink {
  from {
    width: 100%;
  }
  to {
    width: 0%;
  }
}
`;

export default ToastContainer;

