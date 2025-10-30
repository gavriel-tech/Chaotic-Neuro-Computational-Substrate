export interface NodePreset {
  type: 'oscillator' | 'algorithm' | 'thrml' | 'visualizer' | 'custom';
  name: string;
  description: string;
  config: Record<string, any>;
  inputs: Array<{ name: string; type: string }>;
  outputs: Array<{ name: string; type: string }>;
  category: string;
  equations?: string[];
  howItWorks?: string;
}

export const NODE_PRESETS: Record<string, NodePreset[]> = {
  system: [
    {
      type: 'custom',
      name: 'Audio Settings',
      description: 'Control audio input/output and routing',
      config: {
        inputEnabled: true,
        outputEnabled: true,
        inputGain: 1.0,
        outputVolume: 0.7,
        sampleRate: 48000,
        bufferSize: 256,
        inputDevice: 'default',
        outputDevice: 'default'
      },
      inputs: [
        { name: 'audio_in', type: 'audio' },
        { name: 'gain_mod', type: 'signal' }
      ],
      outputs: [
        { name: 'audio_out', type: 'audio' },
        { name: 'level', type: 'signal' }
      ],
      category: 'System',
      howItWorks: 'Manages audio I/O for the entire system. Connect audio sources to audio_in, and route audio_out to other processing nodes or outputs. The level output provides real-time audio level monitoring. Gain can be modulated via the gain_mod input.'
    }
  ],
  oscillators: [
    {
      type: 'oscillator',
      name: 'Chua Oscillator',
      description: 'Chaotic oscillator with three equilibrium points',
      config: { alpha: 15.6, beta: 28.0, m0: -1.143, m1: -0.714 },
      inputs: [],
      outputs: [{ name: 'x', type: 'signal' }, { name: 'y', type: 'signal' }, { name: 'z', type: 'signal' }],
      category: 'Source',
      equations: [
        'dx/dt = α(y - x - f(x))',
        'dy/dt = x - y + z',
        'dz/dt = -βy',
        'f(x) = m₁x + 0.5(m₀ - m₁)(|x + 1| - |x - 1|)'
      ],
      howItWorks: 'The Chua circuit is a simple electronic circuit that exhibits classic chaotic behavior. It uses a nonlinear resistor (Chua diode) to create a piecewise-linear function f(x), which drives the system into chaotic oscillations. The three state variables (x, y, z) represent voltages across capacitors and current through an inductor.'
    },
    {
      type: 'oscillator',
      name: 'Lorenz Attractor',
      description: 'Classic chaotic system',
      config: { sigma: 10.0, rho: 28.0, beta: 8 / 3 },
      inputs: [],
      outputs: [{ name: 'x', type: 'signal' }, { name: 'y', type: 'signal' }, { name: 'z', type: 'signal' }],
      category: 'Source',
      equations: [
        'dx/dt = σ(y - x)',
        'dy/dt = x(ρ - z) - y',
        'dz/dt = xy - βz'
      ],
      howItWorks: 'Originally derived from atmospheric convection equations, the Lorenz system exhibits sensitive dependence on initial conditions - the "butterfly effect". The attractor has a distinctive double-spiral structure. Parameter σ represents the Prandtl number, ρ the Rayleigh number, and β relates to physical dimensions.'
    },
    {
      type: 'oscillator',
      name: 'Van der Pol',
      description: 'Non-conservative oscillator',
      config: { mu: 1.0 },
      inputs: [],
      outputs: [{ name: 'x', type: 'signal' }, { name: 'dx', type: 'signal' }],
      category: 'Source',
    },
  ],

  algorithms: [
    // Wave Processing & Dynamics
    {
      type: 'algorithm',
      name: 'Waveshaper',
      description: 'Non-linear waveshaping for any signal',
      config: { gain: 2.0, mix: 0.5 },
      inputs: [{ name: 'input', type: 'signal' }],
      outputs: [{ name: 'output', type: 'signal' }],
      category: 'Wave Processing',
      equations: [
        'y = tanh(gain × x)',
        'output = mix × y + (1 - mix) × x'
      ],
      howItWorks: 'Applies hyperbolic tangent saturation to create harmonic distortion on any waveform. The gain parameter controls the amount of nonlinearity, while mix blends between the processed and dry signals. Works on chaotic oscillators, photonic signals, or any time-varying waveform.'
    },
    {
      type: 'algorithm',
      name: 'Resonator',
      description: 'Resonant filter for waveform emphasis',
      config: { frequency: 440.0, Q: 10.0 },
      inputs: [{ name: 'input', type: 'signal' }],
      outputs: [{ name: 'output', type: 'signal' }],
      category: 'Wave Processing',
      equations: [
        'H(ω) = 1 / (1 + jQ(ω/ω₀ - ω₀/ω))',
        'ω₀ = 2πf₀'
      ],
      howItWorks: 'A resonant bandpass filter that emphasizes frequency components near f₀ in any waveform. The Q factor controls the sharpness of the resonance - higher Q creates a narrower, more pronounced peak. At high Q values, the filter can self-oscillate. Works on any oscillator output or signal chain.'
    },
    {
      type: 'algorithm',
      name: 'Hilbert Transform',
      description: 'Phase shift and envelope extraction',
      config: { order: 64 },
      inputs: [{ name: 'input', type: 'signal' }],
      outputs: [{ name: 'output', type: 'signal' }, { name: 'envelope', type: 'signal' }],
      category: 'Wave Processing',
    },
    {
      type: 'algorithm',
      name: 'Compressor',
      description: 'Dynamic range compression for waveforms',
      config: { threshold: -20.0, ratio: 4.0, attack: 10.0, release: 100.0, knee: 6.0, makeupGain: 0.0 },
      inputs: [{ name: 'input', type: 'signal' }],
      outputs: [{ name: 'output', type: 'signal' }, { name: 'gain_reduction', type: 'signal' }],
      category: 'Wave Processing',
      equations: [
        'if input_dB > threshold:',
        '  gain_reduction = (input_dB - threshold) × (1 - 1/ratio)',
        'output_dB = input_dB - gain_reduction + makeup_gain'
      ],
      howItWorks: 'Reduces the dynamic range of any waveform by attenuating amplitudes above the threshold. The ratio determines how much compression is applied. Attack and release control response time. Can be used to tame chaotic oscillator peaks, smooth photonic signals, or control any time-varying amplitude.'
    },
    {
      type: 'algorithm',
      name: 'Limiter',
      description: 'Hard limiting for waveform control',
      config: { threshold: -3.0, attack: 0.5, release: 50.0, ceiling: -0.1 },
      inputs: [{ name: 'input', type: 'signal' }],
      outputs: [{ name: 'output', type: 'signal' }, { name: 'limiting', type: 'signal' }],
      category: 'Wave Processing',
      equations: [
        'ratio = ∞:1 (or very high, e.g., 100:1)',
        'output = min(input, ceiling)',
        'if input > threshold: apply gain reduction'
      ],
      howItWorks: 'A compressor with an extremely high ratio (approaching ∞:1) that prevents any waveform from exceeding a set ceiling. Very fast attack time ensures no peaks get through. Useful for protecting downstream systems from extreme amplitudes in chaotic or nonlinear dynamics.'
    },
    {
      type: 'algorithm',
      name: 'Expander',
      description: 'Dynamic range expansion for waveforms',
      config: { threshold: -40.0, ratio: 2.0, attack: 5.0, release: 50.0, knee: 6.0 },
      inputs: [{ name: 'input', type: 'signal' }],
      outputs: [{ name: 'output', type: 'signal' }, { name: 'gain_change', type: 'signal' }],
      category: 'Wave Processing',
      equations: [
        'if input_dB < threshold:',
        '  gain_increase = (threshold - input_dB) × (ratio - 1)',
        'output_dB = input_dB - gain_increase'
      ],
      howItWorks: 'Opposite of compression - increases dynamic range by attenuating signals below the threshold. Can be used to enhance contrast in waveforms, suppress low-amplitude noise, or add emphasis to transient features in any signal type.'
    },
    {
      type: 'algorithm',
      name: 'Gate',
      description: 'Threshold gate for waveform control',
      config: { threshold: -50.0, attack: 1.0, hold: 10.0, release: 100.0, range: -80.0 },
      inputs: [{ name: 'input', type: 'signal' }, { name: 'sidechain', type: 'signal' }],
      outputs: [{ name: 'output', type: 'signal' }, { name: 'gate_state', type: 'signal' }],
      category: 'Wave Processing',
      equations: [
        'if input_dB > threshold: gate_open = true',
        'if input_dB < threshold - range: gate_open = false',
        'output = input × gate_envelope'
      ],
      howItWorks: 'Silences waveforms below a threshold, effectively removing low-amplitude components. Attack controls gate opening speed, hold keeps it open for minimum time, and release determines closing smoothness. Sidechain input allows external trigger control. Works on any signal type.'
    },

    // Photonic Algorithms
    {
      type: 'algorithm',
      name: 'Optical Kerr Effect',
      description: 'Intensity-dependent refractive index',
      config: { n2: 2.6e-20 },
      inputs: [{ name: 'field', type: 'complex' }],
      outputs: [{ name: 'field', type: 'complex' }],
      category: 'Photonic',
      equations: [
        'n(I) = n₀ + n₂I',
        'φ = n₂I × L × k₀',
        'E_out = E_in × exp(iφ)'
      ],
      howItWorks: 'The Kerr effect causes the refractive index to change with light intensity. This creates a nonlinear phase shift proportional to the optical power. Used in optical switching, self-focusing, and soliton formation. The n₂ coefficient determines the strength of the nonlinearity.'
    },
    {
      type: 'algorithm',
      name: 'Four-Wave Mixing',
      description: 'Non-linear optical interaction',
      config: { chi3: 1e-22 },
      inputs: [{ name: 'pump', type: 'complex' }, { name: 'signal', type: 'complex' }],
      outputs: [{ name: 'idler', type: 'complex' }],
      category: 'Photonic',
    },
  ],

  pbit_dynamics: [
    {
      type: 'algorithm',
      name: 'P-Bit Compressor',
      description: 'Probability distribution compression for p-bits',
      config: { threshold: 0.5, ratio: 4.0, attack: 5.0, release: 50.0, bias: 0.0 },
      inputs: [{ name: 'p_in', type: 'discrete' }, { name: 'modulation', type: 'signal' }],
      outputs: [{ name: 'p_out', type: 'discrete' }, { name: 'compression', type: 'signal' }],
      category: 'P-Bit',
      equations: [
        'if P(1) > threshold:',
        '  compression = (P(1) - threshold) × (1 - 1/ratio)',
        'P_out(1) = P(1) - compression + bias'
      ],
      howItWorks: 'Compresses the probability distribution of p-bits, reducing the dynamic range of flip probabilities. When p-bits are too "hot" (high flip rate), compression cools them down. Useful for stabilizing THRML networks and preventing runaway thermal fluctuations. The compression output tracks how much the distribution is being squeezed.'
    },
    {
      type: 'algorithm',
      name: 'P-Bit Limiter',
      description: 'Hard limit on p-bit flip probability',
      config: { threshold: 0.8, attack: 1.0, release: 20.0, ceiling: 0.95 },
      inputs: [{ name: 'p_in', type: 'discrete' }],
      outputs: [{ name: 'p_out', type: 'discrete' }, { name: 'limiting', type: 'signal' }],
      category: 'P-Bit',
      equations: [
        'P_out(1) = min(P_in(1), ceiling)',
        'P_out(0) = max(P_in(0), 1 - ceiling)',
        'limiting = amount of probability clipped'
      ],
      howItWorks: 'Prevents p-bit probabilities from reaching extreme values (too close to 0 or 1), which can cause numerical instability in THRML systems. Acts as a safety mechanism to maintain healthy thermal noise levels. The ceiling parameter ensures p-bits always retain some uncertainty, preserving their probabilistic nature.'
    },
    {
      type: 'algorithm',
      name: 'P-Bit Expander',
      description: 'Expand probability distribution for enhanced fluctuations',
      config: { threshold: 0.3, ratio: 2.0, attack: 5.0, release: 50.0 },
      inputs: [{ name: 'p_in', type: 'discrete' }],
      outputs: [{ name: 'p_out', type: 'discrete' }, { name: 'expansion', type: 'signal' }],
      category: 'P-Bit',
      equations: [
        'if P(1) < threshold:',
        '  expansion = (threshold - P(1)) × (ratio - 1)',
        'P_out(1) = P(1) + expansion'
      ],
      howItWorks: 'Amplifies thermal fluctuations by expanding the probability distribution of low-activity p-bits. Useful for "heating up" regions of the THRML network that have become too cold or stuck. Helps escape local minima in energy landscapes by increasing exploration. Can enhance the representational capacity of p-bit arrays.'
    },
    {
      type: 'algorithm',
      name: 'P-Bit Gate',
      description: 'Threshold-based p-bit activation gate',
      config: { threshold: 0.4, attack: 2.0, hold: 10.0, release: 30.0, range: 0.1 },
      inputs: [{ name: 'p_in', type: 'discrete' }, { name: 'sidechain', type: 'discrete' }],
      outputs: [{ name: 'p_out', type: 'discrete' }, { name: 'gate_state', type: 'signal' }],
      category: 'P-Bit',
      equations: [
        'if P(1) > threshold: gate_open = true',
        'if P(1) < threshold - range: gate_open = false',
        'P_out = P_in × gate_envelope'
      ],
      howItWorks: 'Selectively activates or deactivates p-bits based on their flip probability. Below threshold, p-bits are forced into a low-entropy state (close to 0 or 1). Above threshold, they operate normally. Sidechain input allows one set of p-bits to control another, enabling conditional computation and attention mechanisms in THRML networks.'
    },
    {
      type: 'algorithm',
      name: 'P-Bit Threshold',
      description: 'Binary thresholding for p-bit states',
      config: { threshold: 0.5, hysteresis: 0.1, smoothing: 0.0 },
      inputs: [{ name: 'p_in', type: 'discrete' }],
      outputs: [{ name: 'binary', type: 'discrete' }, { name: 'analog', type: 'signal' }],
      category: 'P-Bit',
      equations: [
        'if P(1) > threshold + hysteresis: output = 1',
        'if P(1) < threshold - hysteresis: output = 0',
        'else: maintain previous state'
      ],
      howItWorks: 'Converts probabilistic p-bits into deterministic binary values using a threshold with hysteresis (Schmitt trigger behavior). The hysteresis prevents oscillation around the threshold. Useful for reading out final states from THRML computation or creating discrete control signals. The analog output provides a smoothed version for monitoring.'
    },
  ],

  thrml: [
    {
      type: 'thrml',
      name: 'Spin Glass EBM',
      description: 'Ising-like energy-based model',
      config: { nodes: 64, temperature: 1.0, gibbs_steps: 5 },
      inputs: [{ name: 'bias', type: 'signal' }],
      outputs: [{ name: 'spins', type: 'discrete' }, { name: 'energy', type: 'scalar' }],
      category: 'THRML',
      equations: [
        'E(s) = -∑ᵢⱼ Jᵢⱼsᵢsⱼ - ∑ᵢ hᵢsᵢ',
        'P(s) = exp(-E(s)/T) / Z',
        'P(sᵢ = 1) = σ(2∑ⱼ Jᵢⱼsⱼ + hᵢ) / T)'
      ],
      howItWorks: 'An Ising-type energy-based model where binary spins (±1) interact through learned coupling weights J. Uses block Gibbs sampling to generate samples from the Boltzmann distribution. Temperature T controls the stochasticity - low T produces deterministic states, high T produces random exploration. Connect the bias input to modulate p-bit flip probabilities, and use spins/energy outputs to drive visualizers or other nodes.'
    },
    {
      type: 'thrml',
      name: 'Continuous EBM',
      description: 'Continuous-valued probabilistic model',
      config: { nodes: 64, temperature: 1.0, gibbs_steps: 5 },
      inputs: [{ name: 'input', type: 'signal' }],
      outputs: [{ name: 'output', type: 'signal' }, { name: 'energy', type: 'scalar' }],
      category: 'THRML',
      equations: [
        'E(x) = ½xᵀAx + bᵀx',
        'P(x) = exp(-E(x)/T) / Z',
        'xᵢ ~ N(μᵢ, T)'
      ],
      howItWorks: 'A continuous-valued energy-based model where variables can take any real value. The quadratic energy function creates Gaussian-like distributions. Uses Langevin dynamics or continuous Gibbs sampling to generate samples. Useful for modeling continuous signals and real-valued data. Connect input from signal sources, and use output to modulate other parameters or feed visualizers.'
    },
    {
      type: 'thrml',
      name: 'Heterogeneous Model',
      description: 'Mixed discrete and continuous nodes',
      config: { spin_nodes: 32, continuous_nodes: 32, temperature: 1.0 },
      inputs: [{ name: 'input', type: 'signal' }],
      outputs: [{ name: 'spins', type: 'discrete' }, { name: 'continuous', type: 'signal' }],
      category: 'THRML',
      equations: [
        'E(s,x) = Eₛ(s) + Eₓ(x) + Eₛₓ(s,x)',
        'P(s,x) = exp(-E(s,x)/T) / Z',
        'Eₛₓ = ∑ᵢⱼ Jᵢⱼsᵢxⱼ'
      ],
      howItWorks: 'Combines binary spins and continuous variables in a single energy-based model. Allows modeling of hybrid systems where some variables are discrete (spins) and others are continuous (real values). The cross-coupling terms Eₛₓ allow spins to influence continuous variables and vice versa. Use input for analog modulation and read spins/continuous outputs for different signal types.'
    },
  ],

  visualizers: [
    {
      type: 'visualizer',
      name: 'Oscilloscope',
      description: 'Time-domain waveform display',
      config: { buffer_size: 1024, channels: 3, width: 500, height: 350, active: true },
      inputs: [{ name: 'x', type: 'signal' }, { name: 'y', type: 'signal' }, { name: 'z', type: 'signal' }],
      outputs: [],
      category: 'Visualizer',
    },
    {
      type: 'visualizer',
      name: 'Spectrogram',
      description: 'Frequency-time analysis',
      config: { fft_size: 2048, overlap: 0.75, width: 500, height: 350, active: true },
      inputs: [{ name: 'signal', type: 'signal' }],
      outputs: [],
      category: 'Visualizer',
    },
    {
      type: 'visualizer',
      name: 'Phase Space 3D',
      description: '3D attractor visualization',
      config: { width: 500, height: 500, active: true },
      inputs: [{ name: 'x', type: 'signal' }, { name: 'y', type: 'signal' }, { name: 'z', type: 'signal' }],
      outputs: [],
      category: 'Visualizer',
    },
    {
      type: 'visualizer',
      name: 'Energy Graph',
      description: 'THRML energy over time',
      config: { width: 500, height: 300, active: true },
      inputs: [{ name: 'energy', type: 'scalar' }],
      outputs: [],
      category: 'Visualizer',
    },
    {
      type: 'visualizer',
      name: 'Spin State Matrix',
      description: 'THRML spin visualization',
      config: { grid_size: 8, width: 400, height: 400, active: true },
      inputs: [{ name: 'spins', type: 'discrete' }],
      outputs: [],
      category: 'Visualizer',
    },
    {
      type: 'visualizer',
      name: 'Correlation Matrix',
      description: 'Node correlation heatmap',
      config: { grid_size: 8, width: 400, height: 400, active: true },
      inputs: [{ name: 'correlations', type: 'matrix' }],
      outputs: [],
      category: 'Visualizer',
    },
    {
      type: 'visualizer',
      name: 'Waveform Monitor',
      description: 'Single-channel waveform',
      config: { buffer_size: 512, width: 450, height: 250, active: true },
      inputs: [{ name: 'signal', type: 'signal' }],
      outputs: [],
      category: 'Visualizer',
    },
    {
      type: 'visualizer',
      name: 'XY Plot',
      description: 'Lissajous figure display',
      config: { width: 450, height: 450, active: true },
      inputs: [{ name: 'x', type: 'signal' }, { name: 'y', type: 'signal' }],
      outputs: [],
      category: 'Visualizer',
    },
    {
      type: 'visualizer',
      name: 'P-Bit Mapper',
      description: 'P-bit state grid visualization',
      config: { grid_size: 8, color_scheme: 'red-green', update_rate: 100, width: 400, height: 400, active: true },
      inputs: [{ name: 'states', type: 'discrete' }],
      outputs: [],
      category: 'Visualizer',
    },
  ],
};

export const ALGORITHM_IDS = {
  // Audio/Signal Processing (0-6)
  PASSTHROUGH: 0,
  TANH_SATURATE: 1,
  SOFT_CLIP: 2,
  FOLD: 3,
  RING_MOD: 4,
  DELAY: 5,
  ALLPASS: 6,

  // Extended Audio (7-13)
  RESONATOR: 7,
  HILBERT: 8,
  RECTIFIER: 9,
  QUANTIZER: 10,
  SLEW_LIMITER: 11,
  CROSS_MOD: 12,
  BIPOLAR_FOLD: 13,

  // Photonic (14-20)
  OPTICAL_KERR: 14,
  ELECTRO_OPTIC: 15,
  OPTICAL_SWITCH: 16,
  FOUR_WAVE_MIXING: 17,
  RAMAN_AMPLIFIER: 18,
  SATURATION: 19,
  OPTICAL_GAIN: 20,
};

export const ALGORITHM_NAMES: Record<number, string> = {
  [ALGORITHM_IDS.PASSTHROUGH]: 'Passthrough',
  [ALGORITHM_IDS.TANH_SATURATE]: 'Tanh Saturate',
  [ALGORITHM_IDS.SOFT_CLIP]: 'Soft Clip',
  [ALGORITHM_IDS.FOLD]: 'Fold',
  [ALGORITHM_IDS.RING_MOD]: 'Ring Modulator',
  [ALGORITHM_IDS.DELAY]: 'Delay',
  [ALGORITHM_IDS.ALLPASS]: 'Allpass Filter',
  [ALGORITHM_IDS.RESONATOR]: 'Resonator',
  [ALGORITHM_IDS.HILBERT]: 'Hilbert Transform',
  [ALGORITHM_IDS.RECTIFIER]: 'Rectifier',
  [ALGORITHM_IDS.QUANTIZER]: 'Quantizer',
  [ALGORITHM_IDS.SLEW_LIMITER]: 'Slew Limiter',
  [ALGORITHM_IDS.CROSS_MOD]: 'Cross Modulator',
  [ALGORITHM_IDS.BIPOLAR_FOLD]: 'Bipolar Fold',
  [ALGORITHM_IDS.OPTICAL_KERR]: 'Optical Kerr Effect',
  [ALGORITHM_IDS.ELECTRO_OPTIC]: 'Electro-Optic Modulator',
  [ALGORITHM_IDS.OPTICAL_SWITCH]: 'Optical Switch',
  [ALGORITHM_IDS.FOUR_WAVE_MIXING]: 'Four-Wave Mixing',
  [ALGORITHM_IDS.RAMAN_AMPLIFIER]: 'Raman Amplifier',
  [ALGORITHM_IDS.SATURATION]: 'Saturation',
  [ALGORITHM_IDS.OPTICAL_GAIN]: 'Optical Gain',
};

