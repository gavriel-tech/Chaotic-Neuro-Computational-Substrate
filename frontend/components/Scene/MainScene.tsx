"use client";

import dynamic from "next/dynamic";
import React, { Suspense } from "react";
import FieldVisualization from "@/components/Scene/FieldVisualization";
import OscillatorParticles from "@/components/Scene/OscillatorParticles";
import PostProcessing from "@/components/Scene/PostProcessing";
import { useControlsStore } from "@/lib/stores/controls";

const Canvas = dynamic(
  () => import("@react-three/fiber").then((mod) => mod.Canvas),
  { ssr: false }
);

export default function MainScene() {
  const showGrid = useControlsStore((state) => state.showGrid);

  return (
    <div className="absolute inset-0">
      <Suspense fallback={null}>
        <Canvas
          camera={{ position: [0, 6, 10], fov: 55 }}
          gl={{ antialias: true, alpha: true }}
        >
          <color attach="background" args={[0x000000]} />
          <ambientLight intensity={0.2} />
          <directionalLight position={[5, 10, 5]} intensity={0.6} />
          {showGrid && <gridHelper args={[14, 20, 0x00fff0, 0x001111]} position={[0, 0.01, 0]} />}
          <FieldVisualization />
          <OscillatorParticles />
          <PostProcessing />
        </Canvas>
      </Suspense>
    </div>
  );
}
