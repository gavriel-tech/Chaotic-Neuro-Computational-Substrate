"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { useSimulationStore } from "@/lib/stores/simulation";
import { useColorSchemeStore } from "@/lib/stores/colorScheme";
import { useControlsStore } from "@/lib/stores/controls";

const NODE_SCALE = 0.12;
const PLANE_SIZE = 14;

export default function OscillatorParticles() {
  const meshRef = useRef<THREE.InstancedMesh>(null!);
  const tempObject = useRef(new THREE.Object3D());
  const tempColor = useRef(new THREE.Color());

  const { positions, amplitudes, activeCount, gridWidth, gridHeight } = useSimulationStore((state) => ({
    positions: state.oscillatorPositions,
    amplitudes: state.oscillatorAmplitudes,
    activeCount: state.activeCount,
    gridWidth: state.gridWidth,
    gridHeight: state.gridHeight,
  }));
  const colorScheme = useColorSchemeStore((state) => state.definition);
  const particleScale = useControlsStore((state) => state.particleScale);
  const particleLimit = useControlsStore((state) => state.particleLimit);

  const renderCount = Math.min(activeCount, particleLimit);
  const stride = renderCount > 0 ? Math.max(1, Math.floor(activeCount / renderCount)) : 1;

  const scaleFactorX = gridWidth > 0 ? PLANE_SIZE / gridWidth : 1;
  const scaleFactorZ = gridHeight > 0 ? PLANE_SIZE / gridHeight : 1;

  useEffect(() => {
    if (!meshRef.current) return;
    const mesh = meshRef.current;

    const [r, g, b] = colorScheme.accents[1]
      ? [
          ((colorScheme.accents[1] >> 16) & 0xff) / 255,
          ((colorScheme.accents[1] >> 8) & 0xff) / 255,
          (colorScheme.accents[1] & 0xff) / 255,
        ]
      : [0, 1, 1];
    tempColor.current.setRGB(r, g, b);
    (mesh.material as THREE.MeshBasicMaterial).color.copy(tempColor.current);
    (mesh.material as THREE.MeshBasicMaterial).needsUpdate = true;
  }, [colorScheme]);

  useEffect(() => {
    if (!meshRef.current) return;
    const mesh = meshRef.current;
    mesh.count = renderCount;

    for (let renderIndex = 0; renderIndex < renderCount; renderIndex += 1) {
      const sourceIndex = Math.min(renderIndex * stride, activeCount - 1);
      const baseIdx = sourceIndex * 3;

      const gridX = positions[baseIdx] ?? 0;
      const gridZ = positions[baseIdx + 2] ?? 0;
      const height = amplitudes[sourceIndex] ?? 0;

      const worldX = (gridX - gridWidth * 0.5) * scaleFactorX;
      const worldZ = (gridZ - gridHeight * 0.5) * scaleFactorZ;

      tempObject.current.position.set(worldX, height * 2.5, worldZ);
      const scale = (NODE_SCALE + height * 0.3) * particleScale;
      tempObject.current.scale.set(scale, scale, scale);
      tempObject.current.updateMatrix();
      mesh.setMatrixAt(renderIndex, tempObject.current.matrix);
    }

    mesh.instanceMatrix.needsUpdate = true;
  }, [positions, amplitudes, activeCount, renderCount, stride, particleScale, gridWidth, gridHeight, scaleFactorX, scaleFactorZ]);

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const mesh = meshRef.current;
    const t = clock.getElapsedTime();
    for (let renderIndex = 0; renderIndex < mesh.count; renderIndex += 1) {
      mesh.getMatrixAt(renderIndex, tempObject.current.matrix);
      const sourceIndex = Math.min(renderIndex * stride, activeCount - 1);
      const baseIdx = sourceIndex * 3;
      const height = amplitudes[sourceIndex] ?? 0;
      const pulse = Math.sin(t + sourceIndex * 0.13) * 0.02;
      const scale = (NODE_SCALE + height * 0.3 + pulse) * particleScale;
      tempObject.current.scale.set(scale, scale, scale);
      tempObject.current.updateMatrix();
      mesh.setMatrixAt(renderIndex, tempObject.current.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, particleLimit]}>
      <sphereGeometry args={[1, 12, 12]} />
      <meshBasicMaterial color="#00fff0" transparent opacity={0.85} />
    </instancedMesh>
  );
}
