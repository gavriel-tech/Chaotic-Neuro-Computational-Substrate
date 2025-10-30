export const fieldVertexShader = /* glsl */ `
  uniform sampler2D fieldTexture;
  uniform float displacementGain;
  varying vec2 vUv;
  varying float vDisplacement;

  void main() {
    vUv = uv;
    vec4 fieldSample = texture2D(fieldTexture, uv);
    vDisplacement = fieldSample.r;
    vec3 displacedPosition = position + normal * vDisplacement * displacementGain;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(displacedPosition, 1.0);
  }
`;

export const fieldFragmentShader = /* glsl */ `
  uniform vec3 colorCore;
  uniform vec3 colorGlow;
  uniform vec3 colorAccent;
  uniform float time;
  varying vec2 vUv;
  varying float vDisplacement;

  void main() {
    float displacement = vDisplacement * 0.5 + 0.5;
    vec3 baseColor = mix(colorGlow, colorCore, displacement);
    float fresnel = pow(clamp(1.0 - length(vUv - vec2(0.5)), 0.0, 1.0), 3.0);
    vec3 holo = colorAccent * sin(time * 0.75 + vUv.x * 10.0) * 0.1;
    vec3 finalColor = baseColor + holo + (colorCore * fresnel * 0.25);
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;
