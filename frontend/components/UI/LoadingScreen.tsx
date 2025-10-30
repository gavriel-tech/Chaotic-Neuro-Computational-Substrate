'use client';

import React, { useEffect, useState } from 'react';

interface LoadingScreenProps {
  isLoading: boolean;
  progress?: number;
  status?: string;
}

export const LoadingScreen: React.FC<LoadingScreenProps> = ({
  isLoading,
  progress = 0,
  status = 'Initializing GMCS...'
}) => {
  const [dots, setDots] = useState('');

  useEffect(() => {
    if (!isLoading) return;

    const interval = setInterval(() => {
      setDots(prev => (prev.length >= 3 ? '' : prev + '.'));
    }, 500);

    return () => clearInterval(interval);
  }, [isLoading]);

  if (!isLoading) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      {/* Animated background grid */}
      <div className="absolute inset-0 overflow-hidden opacity-20">
        <div className="absolute inset-0 bg-gradient-to-br from-cyber-cyan-core via-transparent to-cyber-magenta-glow animate-pulse" />
        <div className="scanlines" />
      </div>

      {/* Loading content */}
      <div className="relative z-10 flex flex-col items-center gap-8 p-8">
        {/* Logo/Title */}
        <div className="text-center">
          <h1 className="text-6xl font-bold tracking-wider text-cyber-cyan-core drop-shadow-[0_0_20px_rgba(0,247,255,0.8)]">
            GMCS
          </h1>
          <p className="mt-2 text-sm tracking-widest text-cyber-cyan-muted">
            UNIVERSAL PLATFORM v2.0
          </p>
        </div>

        {/* Animated spinner */}
        <div className="relative h-32 w-32">
          {/* Outer ring */}
          <div className="absolute inset-0 animate-spin">
            <div className="h-full w-full rounded-full border-4 border-transparent border-t-cyber-cyan-core border-r-cyber-cyan-glow" />
          </div>
          
          {/* Middle ring */}
          <div className="absolute inset-2 animate-spin-slow">
            <div className="h-full w-full rounded-full border-4 border-transparent border-b-cyber-magenta-glow border-l-cyber-magenta-accent" />
          </div>
          
          {/* Inner ring */}
          <div className="absolute inset-4 animate-spin-reverse">
            <div className="h-full w-full rounded-full border-4 border-transparent border-t-cyber-chromatic-1 border-r-cyber-chromatic-2" />
          </div>

          {/* Center pulse */}
          <div className="absolute inset-8 flex items-center justify-center">
            <div className="h-full w-full rounded-full bg-cyber-cyan-core opacity-50 animate-pulse-glow" />
          </div>
        </div>

        {/* Progress bar */}
        {progress > 0 && (
          <div className="w-80">
            <div className="h-1 overflow-hidden rounded-full bg-gray-800">
              <div
                className="h-full bg-gradient-to-r from-cyber-cyan-core via-cyber-magenta-glow to-cyber-chromatic-3 transition-all duration-300 shadow-[0_0_10px_rgba(0,247,255,0.6)]"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Status text */}
        <div className="text-center">
          <p className="text-lg font-mono text-cyber-cyan-glow">
            {status}{dots}
          </p>
          {progress > 0 && progress < 100 && (
            <p className="mt-2 text-sm text-cyber-cyan-muted font-mono">
              {Math.round(progress)}% complete
            </p>
          )}
        </div>

        {/* Initialization steps */}
        <div className="w-96 space-y-2 font-mono text-xs text-cyber-cyan-muted">
          <LoadingStep completed={progress > 20} label="Initializing JAX runtime" />
          <LoadingStep completed={progress > 40} label="Compiling simulation kernels" />
          <LoadingStep completed={progress > 60} label="Allocating GPU memory" />
          <LoadingStep completed={progress > 80} label="Establishing WebSocket connection" />
          <LoadingStep completed={progress >= 100} label="System ready" />
        </div>
      </div>
    </div>
  );
};

interface LoadingStepProps {
  completed: boolean;
  label: string;
}

const LoadingStep: React.FC<LoadingStepProps> = ({ completed, label }) => {
  return (
    <div className="flex items-center gap-3">
      <div className={`h-2 w-2 rounded-full transition-colors ${
        completed 
          ? 'bg-cyber-cyan-core shadow-[0_0_8px_rgba(0,247,255,0.8)]' 
          : 'bg-gray-700'
      }`} />
      <span className={completed ? 'text-cyber-cyan-core' : ''}>
        {label}
      </span>
    </div>
  );
};

export default LoadingScreen;

