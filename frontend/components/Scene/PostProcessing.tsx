"use client";

import { useThree } from "@react-three/fiber";
import { useEffect } from "react";

export default function PostProcessing() {
  const { gl } = useThree();

  useEffect(() => {
    gl.toneMappingExposure = 1.2;
  }, [gl]);

  return null;
}
