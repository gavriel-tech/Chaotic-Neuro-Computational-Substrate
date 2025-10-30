'use client';

import React, { useEffect, useRef, useState } from 'react';

interface PhaseSpaceProps {
  width: number;
  height: number;
  nodeId?: number;
  // Props for external control
  showPoincare?: boolean;
  showTrail?: boolean;
}

// ============================================================================
// 3D Phase Space with Attractor Visualization
// ============================================================================

interface Point3D {
  x: number;
  y: number;
  z: number;
}

export const PhaseSpace3D: React.FC<PhaseSpaceProps> = ({ 
  width, 
  height, 
  nodeId = 0,
  showPoincare = false,
  showTrail = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<Point3D[]>([]);
  const [rotation, setRotation] = useState({ x: 0.5, y: 0.5 });
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [trailLength, setTrailLength] = useState(500);
  const maxPoints = 2000;

  // Fetch oscillator data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/visualizer/oscilloscope/${nodeId}`);
        const result = await response.json();
        
        if (result.current) {
          const newPoint: Point3D = {
            x: result.current.x,
            y: result.current.y,
            z: result.current.z
          };
          
          setPoints(prev => {
            const updated = [...prev, newPoint];
            return updated.slice(-maxPoints);
          });
        }
      } catch (err) {
        console.error('Failed to fetch phase space data:', err);
      }
    };

    const interval = setInterval(fetchData, 33); // ~30 Hz
    return () => clearInterval(interval);
  }, [nodeId]);

  // Render 3D phase space
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points.length < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    // 3D projection parameters
    const centerX = width / 2;
    const centerY = height / 2;
    const scale = Math.min(width, height) / 8;

    // Rotation matrices
    const cosX = Math.cos(rotation.x);
    const sinX = Math.sin(rotation.x);
    const cosY = Math.cos(rotation.y);
    const sinY = Math.sin(rotation.y);

    const project = (p: Point3D): { x: number; y: number; z: number } => {
      // Rotate around Y axis
      let x = p.x * cosY - p.z * sinY;
      let y = p.y;
      let z = p.x * sinY + p.z * cosY;
      
      // Rotate around X axis
      const y2 = y * cosX - z * sinX;
      const z2 = y * sinX + z * cosX;
      
      // Project to 2D
      const perspective = 1 / (1 + z2 / 10);
      return {
        x: centerX + x * scale * perspective,
        y: centerY - y2 * scale * perspective,
        z: z2
      };
    };

    // Draw axes
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 1;
    const axisLength = 3;
    
    // X axis (red)
    const xAxis = [
      project({ x: 0, y: 0, z: 0 }),
      project({ x: axisLength, y: 0, z: 0 })
    ];
    ctx.strokeStyle = '#f85149';
    ctx.beginPath();
    ctx.moveTo(xAxis[0].x, xAxis[0].y);
    ctx.lineTo(xAxis[1].x, xAxis[1].y);
    ctx.stroke();
    
    // Y axis (green)
    const yAxis = [
      project({ x: 0, y: 0, z: 0 }),
      project({ x: 0, y: axisLength, z: 0 })
    ];
    ctx.strokeStyle = '#00ff99';
    ctx.beginPath();
    ctx.moveTo(yAxis[0].x, yAxis[0].y);
    ctx.lineTo(yAxis[1].x, yAxis[1].y);
    ctx.stroke();
    
    // Z axis (blue)
    const zAxis = [
      project({ x: 0, y: 0, z: 0 }),
      project({ x: 0, y: 0, z: axisLength })
    ];
    ctx.strokeStyle = '#58a6ff';
    ctx.beginPath();
    ctx.moveTo(zAxis[0].x, zAxis[0].y);
    ctx.lineTo(zAxis[1].x, zAxis[1].y);
    ctx.stroke();

    // Draw attractor trail
    const displayPoints = showTrail ? points.slice(-trailLength) : [points[points.length - 1]];
    
    if (displayPoints.length > 1) {
      ctx.lineWidth = 2;
      
      displayPoints.forEach((point, i) => {
        if (i === 0) return;
        
        const prev = project(displayPoints[i - 1]);
        const curr = project(point);
        
        // Color based on position in trail
        const alpha = showTrail ? (i / displayPoints.length) : 1;
        const hue = 200 + (i / displayPoints.length) * 60;
        ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${alpha})`;
        
        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(curr.x, curr.y);
        ctx.stroke();
      });
    }

    // Draw Poincaré section (intersection with z=0 plane)
    if (showPoincare && points.length > 1) {
      ctx.fillStyle = '#f85149';
      for (let i = 1; i < points.length; i++) {
        const prev = points[i - 1];
        const curr = points[i];
        
        // Check if trajectory crosses z=0 plane
        if ((prev.z < 0 && curr.z >= 0) || (prev.z >= 0 && curr.z < 0)) {
          // Interpolate to find exact crossing point
          const t = -prev.z / (curr.z - prev.z);
          const crossPoint = {
            x: prev.x + t * (curr.x - prev.x),
            y: prev.y + t * (curr.y - prev.y),
            z: 0
          };
          
          const projected = project(crossPoint);
          ctx.beginPath();
          ctx.arc(projected.x, projected.y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // Draw current point
    if (points.length > 0) {
      const current = project(points[points.length - 1]);
      ctx.fillStyle = '#00ff99';
      ctx.beginPath();
      ctx.arc(current.x, current.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw labels
    ctx.fillStyle = '#c9d1d9';
    ctx.font = '10px monospace';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'start';
    ctx.fillText('X', xAxis[1].x + 5, xAxis[1].y);
    ctx.fillText('Y', yAxis[1].x + 5, yAxis[1].y);
    ctx.fillText('Z', zAxis[1].x + 5, zAxis[1].y);

    // Draw stats
    if (points.length > 0) {
      const current = points[points.length - 1];
      ctx.textBaseline = 'top';
      ctx.fillText(`X: ${current.x.toFixed(3)}`, 5, 5);
      ctx.fillText(`Y: ${current.y.toFixed(3)}`, 5, 17);
      ctx.fillText(`Z: ${current.z.toFixed(3)}`, 5, 29);
      ctx.fillText(`Points: ${points.length}`, 5, 41);
    }

  }, [points, width, height, rotation, showPoincare, showTrail, trailLength]);

  // Mouse interaction
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    
    setRotation(prev => ({
      x: prev.x + dy * 0.01,
      y: prev.y + dx * 0.01
    }));
    
    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <canvas 
      ref={canvasRef} 
      width={width} 
      height={height} 
      className="cursor-move"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    />
  );
};


// ============================================================================
// XY Plot (Lissajous Figures)
// ============================================================================

export const XYPlot: React.FC<PhaseSpaceProps> = ({ 
  width, 
  height, 
  nodeId = 0
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<{ x: number; y: number }[]>([]);
  const [persistence, setPersistence] = useState(80);
  const maxPoints = 1000;

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`http://localhost:8000/visualizer/oscilloscope/${nodeId}`);
        const result = await response.json();
        
        if (result.current) {
          const newPoint = {
            x: result.current.x,
            y: result.current.y
          };
          
          setPoints(prev => {
            const updated = [...prev, newPoint];
            return updated.slice(-maxPoints);
          });
        }
      } catch (err) {
        console.error('Failed to fetch XY data:', err);
      }
    };

    const interval = setInterval(fetchData, 33); // ~30 Hz
    return () => clearInterval(interval);
  }, [nodeId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || points.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Fade effect for persistence
    if (persistence > 0) {
      ctx.fillStyle = `rgba(13, 17, 23, ${1 - persistence / 100})`;
      ctx.fillRect(0, 0, width, height);
    } else {
      ctx.fillStyle = '#0d1117';
      ctx.fillRect(0, 0, width, height);
    }

    // Draw axes
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw grid
    ctx.strokeStyle = '#21262d';
    for (let i = 0; i < width; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, height);
      ctx.stroke();
    }
    for (let i = 0; i < height; i += 40) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(width, i);
      ctx.stroke();
    }

    // Draw Lissajous figure
    const scale = Math.min(width, height) / 6;
    const centerX = width / 2;
    const centerY = height / 2;

    ctx.strokeStyle = '#a371f7';
    ctx.lineWidth = 2;
    ctx.beginPath();

    points.forEach((point, i) => {
      const x = centerX + point.x * scale;
      const y = centerY - point.y * scale;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw current point
    if (points.length > 0) {
      const current = points[points.length - 1];
      const x = centerX + current.x * scale;
      const y = centerY - current.y * scale;
      
      ctx.fillStyle = '#00ff99';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }

  }, [points, width, height, persistence]);

  return <canvas ref={canvasRef} width={width} height={height} />;
};
