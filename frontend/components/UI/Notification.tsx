'use client';

import React, { useEffect } from 'react';

interface NotificationProps {
  message: string;
  type: 'error' | 'success' | 'warning' | 'info';
  onClose: () => void;
  duration?: number;
}

export const Notification: React.FC<NotificationProps> = ({
  message,
  type,
  onClose,
  duration = 5000,
}) => {
  const [isClosing, setIsClosing] = React.useState(false);

  useEffect(() => {
    // Start fade out animation 500ms before actual close
    const fadeTimer = setTimeout(() => {
      setIsClosing(true);
    }, duration - 500);

    // Actually remove the notification
    const closeTimer = setTimeout(onClose, duration);

    return () => {
      clearTimeout(fadeTimer);
      clearTimeout(closeTimer);
    };
  }, [duration, onClose]);

  const styles = {
    error: {
      border: 'border-[#f85149]',
      text: 'text-[#f85149]',
      label: '[ERR]',
      glow: 'shadow-[0_0_20px_rgba(248,81,73,0.3)]'
    },
    success: {
      border: 'border-[#00ff99]',
      text: 'text-[#00ff99]',
      label: '[OK]',
      glow: 'shadow-[0_0_20px_rgba(0,255,153,0.3)]'
    },
    warning: {
      border: 'border-[#ffa500]',
      text: 'text-[#ffa500]',
      label: '[WARN]',
      glow: 'shadow-[0_0_20px_rgba(255,165,0,0.3)]'
    },
    info: {
      border: 'border-[#00cc77]',
      text: 'text-[#00cc77]',
      label: '[INFO]',
      glow: 'shadow-[0_0_20px_rgba(0,204,119,0.3)]'
    },
  };

  const style = styles[type];

  return (
    <div className={`fixed top-4 right-4 z-[9999] transition-all duration-500 ${isClosing ? 'opacity-0 translate-x-8' : 'opacity-100 translate-x-0 animate-slide-in'}`}>
      <div
        className={`bg-black/95 ${style.border} ${style.glow} border-2 backdrop-blur-md rounded p-4 min-w-[350px] max-w-[500px] font-mono`}
      >
        <div className="flex items-start gap-3">
          <div className={`${style.text} font-bold text-xs flex-shrink-0 mt-0.5`}>
            {style.label}
          </div>
          <div className="flex-1">
            <div className="text-xs text-[#00cc77] leading-relaxed break-words">
              {message}
            </div>
          </div>
          <button
            onClick={onClose}
            className={`${style.text} hover:opacity-70 text-lg leading-none flex-shrink-0 font-bold`}
          >
            [X]
          </button>
        </div>
      </div>
    </div>
  );
};

// Notification Manager Component
interface NotificationItem {
  id: string;
  message: string;
  type: 'error' | 'success' | 'warning' | 'info';
}

let notificationCallback: ((notification: NotificationItem) => void) | null = null;

export const NotificationContainer: React.FC = () => {
  const [notifications, setNotifications] = React.useState<NotificationItem[]>([]);

  React.useEffect(() => {
    notificationCallback = (notification: NotificationItem) => {
      setNotifications((prev) => [...prev, notification]);
    };

    return () => {
      notificationCallback = null;
    };
  }, []);

  const removeNotification = (id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  };

  return (
    <>
      {notifications.map((notification, index) => (
        <div
          key={notification.id}
          style={{ top: `${16 + index * 110}px` }}
          className="fixed right-4 z-[9999]"
        >
          <Notification
            message={notification.message}
            type={notification.type}
            onClose={() => removeNotification(notification.id)}
          />
        </div>
      ))}
    </>
  );
};

// Global notification function
export const notify = {
  error: (message: string) => {
    if (notificationCallback) {
      notificationCallback({
        id: `${Date.now()}-${Math.random()}`,
        message,
        type: 'error',
      });
    }
  },
  success: (message: string) => {
    if (notificationCallback) {
      notificationCallback({
        id: `${Date.now()}-${Math.random()}`,
        message,
        type: 'success',
      });
    }
  },
  warning: (message: string) => {
    if (notificationCallback) {
      notificationCallback({
        id: `${Date.now()}-${Math.random()}`,
        message,
        type: 'warning',
      });
    }
  },
  info: (message: string) => {
    if (notificationCallback) {
      notificationCallback({
        id: `${Date.now()}-${Math.random()}`,
        message,
        type: 'info',
      });
    }
  },
};
