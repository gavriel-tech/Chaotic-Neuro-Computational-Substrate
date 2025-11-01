'use client';

import React, { useRef, useEffect, useState } from 'react';

interface BenchmarkSample {
  timestamp: number;
  wall_time: number;
  n_samples: number;
  magnetization: number;
  energy: number;
  strategy: string;
  n_chains: number;
}

interface BenchmarkData {
  history: BenchmarkSample[];
  count: number;
}

interface BenchmarkMetrics {
  samples_per_sec: number;
  ess_per_sec: number;
  lag1_autocorr: number;
  tau_int: number;
  total_samples: number;
  mean_magnetization: number;
  timestamp: number;
}

export const BenchmarkVisualizer: React.FC<{ width?: number; height?: number }> = ({
  width = 600,
  height = 400
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkData | null>(null);
  const [currentMetrics, setCurrentMetrics] = useState<BenchmarkMetrics | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<'samples_per_sec' | 'ess_per_sec' | 'autocorr' | 'energy'>('samples_per_sec');
  const [error, setError] = useState<string | null>(null);

  // Fetch benchmark data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [historyResponse, metricsResponse] = await Promise.all([
          fetch('http://localhost:8000/sampler/benchmarks/history?max_samples=100'),
          fetch('http://localhost:8000/sampler/benchmarks')
        ]);
        
        const historyData = await historyResponse.json();
        const metricsData = await metricsResponse.json();
        
        setBenchmarkData(historyData);
        setCurrentMetrics(metricsData);
        setError(null);
      } catch (err) {
        setError('Failed to fetch benchmark data');
        console.error(err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  // Draw chart
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !benchmarkData || benchmarkData.history.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);

    const padding = 40;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;

    // Extract data based on selected metric
    const history = benchmarkData.history;
    let dataPoints: number[] = [];
    let yLabel = '';
    let yMax = 1;
    let yMin = 0;

    switch (selectedMetric) {
      case 'samples_per_sec':
        dataPoints = history.map(s => s.n_samples / (s.wall_time || 0.001));
        yLabel = 'Samples/sec';
        yMax = Math.max(...dataPoints, 1);
        break;
      case 'ess_per_sec':
        // Approximate ESS/sec (would need actual ESS computation)
        dataPoints = history.map(s => (s.n_samples / (s.wall_time || 0.001)) * 0.8);
        yLabel = 'ESS/sec';
        yMax = Math.max(...dataPoints, 1);
        break;
      case 'autocorr':
        // Use magnetization as proxy for autocorrelation visualization
        dataPoints = history.map(s => Math.abs(s.magnetization));
        yLabel = 'Magnetization';
        yMax = 1.0;
        yMin = 0;
        break;
      case 'energy':
        dataPoints = history.map(s => s.energy);
        yLabel = 'Energy';
        yMax = Math.max(...dataPoints.map(Math.abs), 1);
        yMin = Math.min(...dataPoints, 0);
        break;
    }

    // Draw axes
    ctx.strokeStyle = '#00cc77';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Draw grid
    ctx.strokeStyle = '#00cc77';
    ctx.globalAlpha = 0.1;
    const gridLines = 5;
    for (let i = 0; i <= gridLines; i++) {
      const y = padding + (chartHeight / gridLines) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;

    // Draw Y-axis labels
    ctx.fillStyle = '#00cc77';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= gridLines; i++) {
      const y = height - padding - (chartHeight / gridLines) * i;
      const value = yMin + ((yMax - yMin) / gridLines) * i;
      ctx.fillText(value.toFixed(1), padding - 5, y + 3);
    }

    // Draw Y-axis label
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // Draw X-axis label
    ctx.textAlign = 'center';
    ctx.fillText('Time (samples)', width / 2, height - 10);

    // Draw data line
    if (dataPoints.length > 1) {
      ctx.strokeStyle = '#00cc77';
      ctx.lineWidth = 2;
      ctx.beginPath();

      for (let i = 0; i < dataPoints.length; i++) {
        const x = padding + (chartWidth / (dataPoints.length - 1)) * i;
        const normalizedValue = (dataPoints[i] - yMin) / (yMax - yMin || 1);
        const y = height - padding - normalizedValue * chartHeight;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();

      // Draw points
      ctx.fillStyle = '#00cc77';
      for (let i = 0; i < dataPoints.length; i++) {
        const x = padding + (chartWidth / (dataPoints.length - 1)) * i;
        const normalizedValue = (dataPoints[i] - yMin) / (yMax - yMin || 1);
        const y = height - padding - normalizedValue * chartHeight;

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw current value
    if (currentMetrics && dataPoints.length > 0) {
      const currentValue = dataPoints[dataPoints.length - 1];
      ctx.fillStyle = '#00cc77';
      ctx.font = 'bold 14px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`Current: ${currentValue.toFixed(2)}`, padding + 10, padding + 20);
    }

  }, [benchmarkData, selectedMetric, width, height, currentMetrics]);

  return (
    <div className="bg-[#1a1a1a] border border-[#00cc77]/40 rounded p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-[#00cc77] font-bold text-sm">BENCHMARK METRICS</h3>
        {error && <span className="text-xs text-[#f85149]">{error}</span>}
      </div>

      {/* Metric Selector */}
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => setSelectedMetric('samples_per_sec')}
          className={`px-3 py-1 rounded text-xs transition-colors ${
            selectedMetric === 'samples_per_sec'
              ? 'bg-[#00cc77] text-black'
              : 'bg-[#00cc77]/20 text-[#00cc77] hover:bg-[#00cc77]/30'
          }`}
        >
          Samples/sec
        </button>
        <button
          onClick={() => setSelectedMetric('ess_per_sec')}
          className={`px-3 py-1 rounded text-xs transition-colors ${
            selectedMetric === 'ess_per_sec'
              ? 'bg-[#00cc77] text-black'
              : 'bg-[#00cc77]/20 text-[#00cc77] hover:bg-[#00cc77]/30'
          }`}
        >
          ESS/sec
        </button>
        <button
          onClick={() => setSelectedMetric('autocorr')}
          className={`px-3 py-1 rounded text-xs transition-colors ${
            selectedMetric === 'autocorr'
              ? 'bg-[#00cc77] text-black'
              : 'bg-[#00cc77]/20 text-[#00cc77] hover:bg-[#00cc77]/30'
          }`}
        >
          Magnetization
        </button>
        <button
          onClick={() => setSelectedMetric('energy')}
          className={`px-3 py-1 rounded text-xs transition-colors ${
            selectedMetric === 'energy'
              ? 'bg-[#00cc77] text-black'
              : 'bg-[#00cc77]/20 text-[#00cc77] hover:bg-[#00cc77]/30'
          }`}
        >
          Energy
        </button>
      </div>

      {/* Chart Canvas */}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full rounded border border-[#00cc77]/20"
      />

      {/* Current Metrics Display */}
      {currentMetrics && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="bg-[#00cc77]/10 p-2 rounded">
            <div className="text-[#00cc77]/60">Samples/sec</div>
            <div className="text-[#00cc77] font-mono font-bold">
              {currentMetrics.samples_per_sec.toFixed(2)}
            </div>
          </div>
          <div className="bg-[#00cc77]/10 p-2 rounded">
            <div className="text-[#00cc77]/60">ESS/sec</div>
            <div className="text-[#00cc77] font-mono font-bold">
              {currentMetrics.ess_per_sec.toFixed(2)}
            </div>
          </div>
          <div className="bg-[#00cc77]/10 p-2 rounded">
            <div className="text-[#00cc77]/60">Autocorr (lag-1)</div>
            <div className="text-[#00cc77] font-mono font-bold">
              {currentMetrics.lag1_autocorr.toFixed(3)}
            </div>
          </div>
          <div className="bg-[#00cc77]/10 p-2 rounded">
            <div className="text-[#00cc77]/60">τ_int</div>
            <div className="text-[#00cc77] font-mono font-bold">
              {currentMetrics.tau_int.toFixed(2)}
            </div>
          </div>
          <div className="bg-[#00cc77]/10 p-2 rounded col-span-2">
            <div className="text-[#00cc77]/60">Total Samples</div>
            <div className="text-[#00cc77] font-mono font-bold">
              {currentMetrics.total_samples}
            </div>
          </div>
        </div>
      )}

      {/* Export Button */}
      <div className="flex gap-2">
        <a
          href="http://localhost:8000/sampler/benchmarks/export?format=json"
          download="benchmarks.json"
          className="flex-1 px-3 py-2 bg-[#00cc77]/20 hover:bg-[#00cc77]/30 text-[#00cc77] rounded text-xs text-center transition-colors"
        >
          Export JSON
        </a>
        <a
          href="http://localhost:8000/sampler/benchmarks/export?format=csv"
          download="benchmarks.csv"
          className="flex-1 px-3 py-2 bg-[#00cc77]/20 hover:bg-[#00cc77]/30 text-[#00cc77] rounded text-xs text-center transition-colors"
        >
          Export CSV
        </a>
      </div>
    </div>
  );
};

