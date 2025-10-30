"use client";

import { useMemo, useRef, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useSimulationStore } from "@/lib/stores/simulation";
import { useColorSchemeStore } from "@/lib/stores/colorScheme";
import { useControlsStore } from "@/lib/stores/controls";
import { fieldVertexShader, fieldFragmentShader } from "@/shaders/field";
import { hexToVec3 } from "@/lib/color-schemes";

const PLANE_SIZE = 14; // World units

export default function FieldVisualization() {
  const meshRef = useRef<THREE.Mesh>(null!);
  const uniformsRef = useRef<{ [key: string]: THREE.IUniform<any> }>();
  const { field, gridWidth, gridHeight } = useSimulationStore((state) => ({
    field: state.field,
    gridWidth: state.gridWidth,
    gridHeight: state.gridHeight,
  }));
  const colorScheme = useColorSchemeStore((state) => state.definition);
  const displacementGain = useControlsStore((state) => state.displacementGain);

  const fieldTexture = useMemo(() => {
    const texture = new THREE.DataTexture(
      field,
      gridWidth,
      gridHeight,
      THREE.RedFormat,
      THREE.FloatType
    );
    texture.needsUpdate = true;
    texture.flipY = true;
    texture.colorSpace = THREE.LinearSRGBColorSpace;
    return texture;
  }, [field, gridWidth, gridHeight]);

  const colorCoreVec = hexToVec3(colorScheme.particleCore);
  const colorGlowVec = hexToVec3(colorScheme.particleGlow);
  const colorAccentVec = hexToVec3(colorScheme.accents[0]);

  const material = useMemo(() => {
    const uniforms = {
      fieldTexture: { value: fieldTexture },
      displacementGain: { value: displacementGain },
      colorCore: { value: new THREE.Vector3(colorCoreVec[0], colorCoreVec[1], colorCoreVec[2]) },
      colorGlow: { value: new THREE.Vector3(colorGlowVec[0], colorGlowVec[1], colorGlowVec[2]) },
      colorAccent: { value: new THREE.Vector3(colorAccentVec[0], colorAccentVec[1], colorAccentVec[2]) },
      time: { value: 0 },
    } satisfies Record<string, THREE.IUniform>;
    uniformsRef.current = uniforms;

    return new THREE.ShaderMaterial({
      uniforms,
      vertexShader: fieldVertexShader,
      fragmentShader: fieldFragmentShader,
      side: THREE.DoubleSide,
      transparent: false,
    });
  }, [colorAccentVec, colorCoreVec, colorGlowVec, fieldTexture, displacementGain]);

  // Update uniform colors when scheme changes
  useEffect(() => {
    if (!uniformsRef.current) return;
    const uniforms = uniformsRef.current;
    const [cr, cg, cb] = hexToVec3(colorScheme.particleCore);
    const [gr, gg, gb] = hexToVec3(colorScheme.particleGlow);
    const [ar, ag, ab] = hexToVec3(colorScheme.accents[0]);
    (uniforms.colorCore.value as THREE.Vector3).set(cr, cg, cb);
    (uniforms.colorGlow.value as THREE.Vector3).set(gr, gg, gb);
    (uniforms.colorAccent.value as THREE.Vector3).set(ar, ag, ab);
  }, [colorScheme]);

  // Update texture when new data arrives
  useEffect(() => {
    if (!uniformsRef.current) return;
    const uniforms = uniformsRef.current;
    const texture: THREE.DataTexture = uniforms.fieldTexture.value;
    texture.image.data = field;
    texture.needsUpdate = true;
  }, [field]);

  useFrame((state, delta) => {
    if (!uniformsRef.current) return;
    uniformsRef.current.time.value += delta;
    uniformsRef.current.displacementGain.value = displacementGain;
  });

  return (
    <group position={[0, 0, 0]}>
      <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]} material={material}>
        <planeGeometry args={[PLANE_SIZE, PLANE_SIZE, gridWidth - 1, gridHeight - 1]} />
      </mesh>
    </group>
  );
}
