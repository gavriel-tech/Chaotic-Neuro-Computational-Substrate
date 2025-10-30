/**
 * Observer Dashboard Component
 * 
 * Real-time monitoring of THRML observers and system metrics.
 */

'use client';

import { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ObserverData {
  timestamp: number;
  energy: number;
  acceptance_rate: number;
  magnetization: number;
}

export default function ObserverDashboard() {
  const [observerData, setObserverData] = useState<ObserverData[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isMonitoring) {
      interval = setInterval(fetchObserverData, 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isMonitoring]);

  const fetchObserverData = async () => {
    try {
      const response = await fetch('http://localhost:8000/thrml/diagnostics');
      const data = await response.json();
      
      setObserverData(prev => {
        const newData = [...prev, data];
        return newData.slice(-50); // Keep last 50 points
      });
    } catch (error) {
      console.error('Failed to fetch observer data:', error);
    }
  };

  const chartData = {
    labels: observerData.map((_, i) => i.toString()),
    datasets: [
      {
        label: 'Energy',
        data: observerData.map(d => d.energy),
        borderColor: 'rgb(6, 182, 212)',
        backgroundColor: 'rgba(6, 182, 212, 0.1)',
        tension: 0.4
      },
      {
        label: 'Acceptance Rate',
        data: observerData.map(d => d.acceptance_rate * 100),
        borderColor: 'rgb(236, 72, 153)',
        backgroundColor: 'rgba(236, 72, 153, 0.1)',
        tension: 0.4
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: 'rgb(6, 182, 212)'
        }
      }
    },
    scales: {
      x: {
        ticks: { color: 'rgb(6, 182, 212, 0.6)' },
        grid: { color: 'rgba(6, 182, 212, 0.1)' }
      },
      y: {
        ticks: { color: 'rgb(6, 182, 212, 0.6)' },
        grid: { color: 'rgba(6, 182, 212, 0.1)' }
      }
    }
  };

  return (
    <div className="observer-dashboard bg-gray-900 border border-[#00ff99] 500/30 rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-[#00ff99] 400 text-xl font-bold">Observer Dashboard</h3>
        <button
          onClick={() => setIsMonitoring(!isMonitoring)}
          className={`px-4 py-2 rounded font-medium transition-all ${
            isMonitoring
              ? 'bg-red-500 text-white hover:bg-red-400'
              : 'bg-[#00ff99] 500 text-gray-900 hover:bg-[#00ff99] 400'
          }`}
        >
          {isMonitoring ? '⏸ Stop' : '▶ Start'} Monitoring
        </button>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800 border border-[#00ff99] 500/20 rounded p-4">
          <div className="text-[#00ff99] 400/60 text-sm mb-1">Current Energy</div>
          <div className="text-[#00ff99] 300 text-2xl font-bold">
            {observerData.length > 0 ? observerData[observerData.length - 1].energy.toFixed(2) : '--'}
          </div>
        </div>
        
        <div className="bg-gray-800 border border-magenta-500/20 rounded p-4">
          <div className="text-magenta-400/60 text-sm mb-1">Acceptance Rate</div>
          <div className="text-magenta-300 text-2xl font-bold">
            {observerData.length > 0 
              ? (observerData[observerData.length - 1].acceptance_rate * 100).toFixed(1) + '%'
              : '--'}
          </div>
        </div>
        
        <div className="bg-gray-800 border border-purple-500/20 rounded p-4">
          <div className="text-purple-400/60 text-sm mb-1">Magnetization</div>
          <div className="text-purple-300 text-2xl font-bold">
            {observerData.length > 0 
              ? observerData[observerData.length - 1].magnetization.toFixed(3)
              : '--'}
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="bg-gray-800 border border-[#00ff99] 500/20 rounded p-4" style={{ height: '300px' }}>
        {observerData.length > 0 ? (
          <Line data={chartData} options={chartOptions} />
        ) : (
          <div className="flex items-center justify-center h-full text-[#00ff99] 400/60">
            Start monitoring to see real-time data
          </div>
        )}
      </div>
    </div>
  );
}

