'use client';

import React from 'react';

interface ConfirmDialogProps {
  message: string;
  title?: string;
  onConfirm: () => void;
  onCancel: () => void;
  confirmText?: string;
  cancelText?: string;
}

export const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  message,
  title = 'CONFIRM',
  onConfirm,
  onCancel,
  confirmText = 'CONFIRM',
  cancelText = 'CANCEL'
}) => {
  return (
    <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-black/95 border-2 border-[#00cc77] shadow-[0_0_30px_rgba(0,204,119,0.4)] rounded p-6 min-w-[400px] max-w-[600px] font-mono">
        {/* Title */}
        <div className="text-[#00ff99] font-bold text-sm mb-4 uppercase tracking-wider">
          [{title}]
        </div>
        
        {/* Message */}
        <div className="text-[#00cc77] text-sm leading-relaxed mb-6">
          {message}
        </div>
        
        {/* Buttons */}
        <div className="flex gap-3 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-xs bg-[#1a1a1a] border border-[#00cc77] text-[#00cc77] rounded hover:bg-[#00cc77]/20 transition font-bold uppercase tracking-wide"
          >
            [{cancelText}]
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 text-xs bg-[#00cc77] text-black rounded hover:bg-[#00ff99] transition font-bold uppercase tracking-wide"
          >
            [{confirmText}]
          </button>
        </div>
      </div>
    </div>
  );
};

// Confirmation manager
let confirmCallback: ((dialog: ConfirmDialogProps) => void) | null = null;

export const ConfirmContainer: React.FC = () => {
  const [dialog, setDialog] = React.useState<ConfirmDialogProps | null>(null);

  React.useEffect(() => {
    confirmCallback = (dialogProps: ConfirmDialogProps) => {
      setDialog(dialogProps);
    };

    return () => {
      confirmCallback = null;
    };
  }, []);

  if (!dialog) return null;

  return (
    <ConfirmDialog
      {...dialog}
      onConfirm={() => {
        dialog.onConfirm();
        setDialog(null);
      }}
      onCancel={() => {
        dialog.onCancel();
        setDialog(null);
      }}
    />
  );
};

// Global confirm function
export const confirm = (
  message: string,
  options?: {
    title?: string;
    confirmText?: string;
    cancelText?: string;
  }
): Promise<boolean> => {
  return new Promise((resolve) => {
    if (confirmCallback) {
      confirmCallback({
        message,
        title: options?.title,
        confirmText: options?.confirmText,
        cancelText: options?.cancelText,
        onConfirm: () => resolve(true),
        onCancel: () => resolve(false),
      });
    } else {
      // Fallback to browser confirm if container not mounted
      resolve(window.confirm(message));
    }
  });
};

