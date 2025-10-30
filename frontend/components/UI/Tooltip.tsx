'use client';

import React, { useState, useRef, useEffect } from 'react';

interface TooltipProps {
  content: string | React.ReactNode;
  children: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
  delay?: number;
  className?: string;
}

export const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  position = 'top',
  delay = 500,
  className = ''
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const showTooltip = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = setTimeout(() => {
      updatePosition();
      setIsVisible(true);
    }, delay);
  };

  const hideTooltip = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  const updatePosition = () => {
    if (!containerRef.current || !tooltipRef.current) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const tooltipRect = tooltipRef.current.getBoundingClientRect();

    let x = 0;
    let y = 0;

    switch (position) {
      case 'top':
        x = containerRect.left + containerRect.width / 2 - tooltipRect.width / 2;
        y = containerRect.top - tooltipRect.height - 8;
        break;
      case 'bottom':
        x = containerRect.left + containerRect.width / 2 - tooltipRect.width / 2;
        y = containerRect.bottom + 8;
        break;
      case 'left':
        x = containerRect.left - tooltipRect.width - 8;
        y = containerRect.top + containerRect.height / 2 - tooltipRect.height / 2;
        break;
      case 'right':
        x = containerRect.right + 8;
        y = containerRect.top + containerRect.height / 2 - tooltipRect.height / 2;
        break;
    }

    // Clamp to viewport
    x = Math.max(8, Math.min(x, window.innerWidth - tooltipRect.width - 8));
    y = Math.max(8, Math.min(y, window.innerHeight - tooltipRect.height - 8));

    setCoords({ x, y });
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const getArrowStyles = () => {
    switch (position) {
      case 'top':
        return 'bottom-[-6px] left-1/2 -translate-x-1/2 border-t-[#00cc77] border-l-transparent border-r-transparent border-b-transparent';
      case 'bottom':
        return 'top-[-6px] left-1/2 -translate-x-1/2 border-b-[#00cc77] border-l-transparent border-r-transparent border-t-transparent';
      case 'left':
        return 'right-[-6px] top-1/2 -translate-y-1/2 border-l-[#00cc77] border-t-transparent border-b-transparent border-r-transparent';
      case 'right':
        return 'left-[-6px] top-1/2 -translate-y-1/2 border-r-[#00cc77] border-t-transparent border-b-transparent border-l-transparent';
    }
  };

  return (
    <>
      <div
        ref={containerRef}
        onMouseEnter={showTooltip}
        onMouseLeave={hideTooltip}
        onFocus={showTooltip}
        onBlur={hideTooltip}
        className={`inline-block ${className}`}
      >
        {children}
      </div>

      {isVisible && (
        <div
          ref={tooltipRef}
          className="fixed z-[10000] pointer-events-none"
          style={{
            left: `${coords.x}px`,
            top: `${coords.y}px`
          }}
        >
          <div className="relative animate-fade-in">
            {/* Tooltip body */}
            <div className="bg-black/95 backdrop-blur-md border border-[#00cc77] rounded px-3 py-2 max-w-xs shadow-glow">
              <div className="text-sm text-[#00ff99] font-mono leading-relaxed whitespace-pre-wrap">
                {content}
              </div>
            </div>

            {/* Arrow */}
            <div
              className={`absolute w-0 h-0 border-[6px] ${getArrowStyles()}`}
            />
          </div>
        </div>
      )}
    </>
  );
};

// Keyboard shortcut tooltip variant
interface ShortcutTooltipProps {
  shortcut: string;
  description: string;
  children: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
}

export const ShortcutTooltip: React.FC<ShortcutTooltipProps> = ({
  shortcut,
  description,
  children,
  position = 'top'
}) => {
  const content = (
    <div>
      <div className="font-semibold text-[#00ff99] mb-1">{description}</div>
      <div className="flex items-center gap-1">
        <kbd className="px-2 py-0.5 bg-[#1a1a1a] border border-[#00cc77] rounded text-xs text-[#00ff99]">
          {shortcut}
        </kbd>
      </div>
    </div>
  );

  return (
    <Tooltip content={content} position={position}>
      {children}
    </Tooltip>
  );
};

// Parameter tooltip variant with value display
interface ParameterTooltipProps {
  name: string;
  value: number | string;
  range?: [number, number];
  unit?: string;
  description: string;
  children: React.ReactNode;
}

export const ParameterTooltip: React.FC<ParameterTooltipProps> = ({
  name,
  value,
  range,
  unit,
  description,
  children
}) => {
  const content = (
    <div className="space-y-2">
      <div className="font-semibold text-[#00ff99]">{name}</div>
      <div className="text-[#00ff99]">
        Current: <span className="text-[#58a6ff]">{value}{unit}</span>
      </div>
      {range && (
        <div className="text-[#00cc77] text-xs">
          Range: {range[0]} - {range[1]} {unit}
        </div>
      )}
      <div className="text-[#00cc77] text-xs border-t border-[#00cc77]/30 pt-2">
        {description}
      </div>
    </div>
  );

  return (
    <Tooltip content={content} position="right" delay={300}>
      {children}
    </Tooltip>
  );
};

export default Tooltip;

