export interface NodePreset {
  type: 'oscillator' | 'algorithm' | 'thrml' | 'visualizer' | 'custom' | 'ml' | 'analysis' | 'control' | 'generator';
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
    },
    {
      type: 'custom',
      name: 'Sampler Config',
      description: 'Configure sampler backend, blocking strategy, and multi-chain parallelism',
      config: {
        backend: 'thrml',
        num_chains: -1,
        blocking_strategy: 'checkerboard',
        auto_adapt_strategy: false,
        clamp_mode: false,
        export_benchmarks: false
      },
      inputs: [
        { name: 'enable_clamp', type: 'discrete' },
        { name: 'chain_count', type: 'signal' }
      ],
      outputs: [
        { name: 'samples_per_sec', type: 'signal' },
        { name: 'ess_per_sec', type: 'signal' },
        { name: 'autocorr', type: 'signal' }
      ],
      category: 'System',
      howItWorks: 'Controls the sampler backend configuration for THRML and other computational substrates. Set the backend type (thrml, photonic, neuromorphic, quantum), configure multi-chain parallelism (-1 for auto-detect, 1 for single, >1 for specific count), and choose a blocking strategy (checkerboard, random, stripes, supercell, graph-coloring). Outputs provide real-time performance metrics. Enable clamp_mode for conditional sampling (inpainting, constrained synthesis).'
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
    {
      type: 'thrml',
      name: 'Categorical EBM',
      description: 'Multi-state categorical variables (beyond binary)',
      config: {
        nodes: 64,
        num_states: 5,
        temperature: 1.0,
        gibbs_steps: 5
      },
      inputs: [
        { name: 'bias', type: 'signal_array' },
        { name: 'observations', type: 'discrete' }
      ],
      outputs: [
        { name: 'states', type: 'discrete' },
        { name: 'class_probs', type: 'matrix' },
        { name: 'energy', type: 'scalar' }
      ],
      category: 'THRML',
      howItWorks: 'Like Spin Glass but each variable can take multiple discrete values (e.g., 0,1,2,3,4 instead of just ±1). Uses CategoricalGibbsConditional for sampling. Perfect for classification tasks, symbolic reasoning, or modeling discrete multi-state systems like nucleotide sequences or musical notes.'
    },
    {
      type: 'thrml',
      name: 'Block Sampling Config',
      description: 'Configure blocking strategies for THRML sampling',
      config: {
        strategy: 'checkerboard',
        block_size: 8,
        n_colors: 2
      },
      inputs: [{ name: 'model_state', type: 'discrete' }],
      outputs: [
        { name: 'blocked_state', type: 'discrete' },
        { name: 'effective_sample_rate', type: 'scalar' }
      ],
      category: 'THRML',
      howItWorks: 'Configures block partitioning strategies for parallel Gibbs sampling. Checkerboard creates 2-color alternating blocks. Random partitions nodes randomly. Stripes creates linear partitions. Supercell uses grid-based blocks. Graph-coloring optimally colors interaction graph. Different strategies trade off parallelism vs. mixing speed.'
    },
    {
      type: 'thrml',
      name: 'Conditional Sampler',
      description: 'Sample with clamped/observed variables',
      config: {
        clamp_mask: [],
        clamp_values: []
      },
      inputs: [
        { name: 'base_state', type: 'discrete' },
        { name: 'observations', type: 'discrete' }
      ],
      outputs: [
        { name: 'conditioned_state', type: 'discrete' },
        { name: 'log_prob', type: 'scalar' }
      ],
      category: 'THRML',
      howItWorks: 'Samples from conditional distributions by clamping observed variables. Essential for inpainting (fill missing data), constrained generation (force certain outputs), or Bayesian inference (condition on evidence). Clamp_mask specifies which variables to fix, clamp_values sets their values. Log_prob tracks conditional likelihood.'
    },
    {
      type: 'thrml',
      name: 'Multi-Chain Sampler',
      description: 'Run multiple independent sampling chains',
      config: {
        n_chains: 8,
        chain_length: 100,
        parallel: true
      },
      inputs: [{ name: 'model_state', type: 'discrete' }],
      outputs: [
        { name: 'chain_samples', type: 'discrete' },
        { name: 'chain_energies', type: 'signal_array' },
        { name: 'r_hat', type: 'scalar' }
      ],
      category: 'THRML',
      howItWorks: 'Runs multiple MCMC chains in parallel for better exploration and convergence diagnostics. R_hat output measures chain convergence (< 1.1 = good). Exploits GPU parallelism for massive speedup. Essential for reliable inference and detecting multimodal distributions. Each chain starts from different initialization.'
    },
    {
      type: 'thrml',
      name: 'Annealing Scheduler',
      description: 'Dynamic temperature scheduling for sampling',
      config: {
        schedule_type: 'linear',
        T_start: 10.0,
        T_end: 0.1,
        steps: 1000
      },
      inputs: [
        { name: 'model_state', type: 'discrete' },
        { name: 'progress', type: 'scalar' }
      ],
      outputs: [
        { name: 'annealed_state', type: 'discrete' },
        { name: 'current_temp', type: 'scalar' },
        { name: 'acceptance_rate', type: 'scalar' }
      ],
      category: 'THRML',
      howItWorks: 'Implements simulated annealing or parallel tempering. Start hot (high T) for exploration, cool down (low T) for exploitation. Linear/exponential schedules gradually reduce temperature. Cyclic schedules can escape local minima. Parallel tempering swaps states between temperatures for better mixing and mode discovery.'
    },
    {
      type: 'thrml',
      name: 'Moment Accumulator',
      description: 'Track mean, variance, correlations over sampling',
      config: { window_size: 1000 },
      inputs: [{ name: 'samples', type: 'discrete' }],
      outputs: [
        { name: 'mean', type: 'signal_array' },
        { name: 'variance', type: 'signal_array' },
        { name: 'correlations', type: 'matrix' }
      ],
      category: 'THRML',
      howItWorks: 'Accumulates first and second moments (mean, variance) and cross-correlations from THRML samples. Essential for moment matching training, convergence monitoring, and understanding learned distributions. Correlations reveal interaction structure. Window_size controls temporal averaging - larger = more stable, smaller = more responsive.'
    },
    {
      type: 'thrml',
      name: 'Weighted Factor',
      description: 'Define weighted interaction factors',
      config: {
        factor_type: 'pairwise',
        weight: 1.0
      },
      inputs: [
        { name: 'variables', type: 'discrete' },
        { name: 'weight_mod', type: 'signal' }
      ],
      outputs: [
        { name: 'energy_contribution', type: 'scalar' },
        { name: 'factor_graph', type: 'custom' }
      ],
      category: 'THRML',
      howItWorks: 'Creates weighted factor graph edges between variables. Pairwise factors couple pairs of spins. Higher-order factors create multi-body interactions. Weight_mod input allows dynamic modulation of interaction strength. Energy_contribution shows factor contribution to total energy. Essential for building structured PGMs.'
    },
    {
      type: 'thrml',
      name: 'Higher-Order Interactions',
      description: 'Multi-body spin interactions beyond pairwise',
      config: {
        order: 3,
        coupling_strength: 0.5
      },
      inputs: [{ name: 'spin_groups', type: 'discrete' }],
      outputs: [
        { name: 'interaction_energy', type: 'scalar' },
        { name: 'modified_spins', type: 'discrete' }
      ],
      category: 'THRML',
      howItWorks: 'Implements multi-body interactions (3-spin, 4-spin, etc.) beyond standard pairwise Ising. Required for modeling complex systems like frustrated magnets, error correction codes, or constraint satisfaction problems. Order parameter sets interaction size. More expressive than pairwise but computationally heavier.'
    },
    {
      type: 'thrml',
      name: 'Ising Trainer',
      description: 'Train Ising model with moment matching',
      config: {
        learning_rate: 0.01,
        n_chains: 4,
        cd_steps: 10
      },
      inputs: [
        { name: 'data_samples', type: 'discrete' },
        { name: 'trigger', type: 'discrete' }
      ],
      outputs: [
        { name: 'trained_weights', type: 'matrix' },
        { name: 'kl_divergence', type: 'scalar' },
        { name: 'grad_norm', type: 'scalar' }
      ],
      category: 'THRML',
      howItWorks: 'Trains Ising model weights using contrastive divergence to match data moments. N_chains controls parallel chain count for gradient estimation. CD_steps sets number of Gibbs steps per update. KL_divergence tracks training progress. Grad_norm monitors gradient magnitude for learning rate tuning.'
    },
    {
      type: 'thrml',
      name: 'Persistent CD Trainer',
      description: 'PCD training with persistent chains',
      config: {
        learning_rate: 0.01,
        n_persistent_chains: 10,
        k_steps: 1
      },
      inputs: [
        { name: 'data_batch', type: 'discrete' },
        { name: 'train_trigger', type: 'discrete' }
      ],
      outputs: [
        { name: 'updated_weights', type: 'matrix' },
        { name: 'fantasy_particles', type: 'discrete' },
        { name: 'reconstruction_error', type: 'scalar' }
      ],
      category: 'THRML',
      howItWorks: 'Like CD-k but maintains persistent fantasy particles that continuously explore model distribution. More stable gradients than CD. Fantasy particles show what the model generates without data influence. Better for deep energy-based models. Each persistent chain maintains its own trajectory across updates.'
    },
    {
      type: 'thrml',
      name: 'Sampling Profiler',
      description: 'Measure sampling efficiency and performance',
      config: {
        window_size: 1000,
        track_autocorr: true,
        track_ess: true
      },
      inputs: [{ name: 'samples', type: 'discrete' }],
      outputs: [
        { name: 'samples_per_sec', type: 'scalar' },
        { name: 'ess_per_sec', type: 'scalar' },
        { name: 'autocorr_time', type: 'scalar' },
        { name: 'acceptance_rate', type: 'scalar' }
      ],
      category: 'THRML',
      howItWorks: 'Tracks MCMC performance metrics. ESS (Effective Sample Size) measures effective independent samples accounting for autocorrelation. Samples_per_sec shows raw throughput. Autocorr_time indicates mixing speed - lower is better. Essential for tuning sampler parameters, comparing backends, and production monitoring.'
    },
    {
      type: 'thrml',
      name: 'Graph Coloring Optimizer',
      description: 'Find optimal block partitions for sampling',
      config: {
        method: 'graph_coloring',
        min_colors: 2,
        max_colors: 8,
        recompute_interval: 100
      },
      inputs: [
        { name: 'interaction_graph', type: 'matrix' },
        { name: 'trigger', type: 'discrete' }
      ],
      outputs: [
        { name: 'block_assignment', type: 'discrete' },
        { name: 'n_colors', type: 'scalar' },
        { name: 'parallelism_factor', type: 'scalar' }
      ],
      category: 'THRML',
      howItWorks: 'Automatically finds optimal block partitions for parallel Gibbs sampling. Graph coloring ensures no connected variables share a block (independence). Metis uses graph partitioning. Spectral uses eigendecomposition. Fewer colors = more parallelism. Dynamically adapts to changing interaction structures via trigger input.'
    },
    {
      type: 'thrml',
      name: 'State Validator',
      description: 'Verify and repair THRML block states',
      config: {
        repair_invalid: true,
        check_interval: 10
      },
      inputs: [{ name: 'block_state', type: 'discrete' }],
      outputs: [
        { name: 'valid_state', type: 'discrete' },
        { name: 'is_valid', type: 'discrete' },
        { name: 'n_violations', type: 'scalar' }
      ],
      category: 'THRML',
      howItWorks: 'Checks block state validity: correct dimensions, value ranges, block structure, NaN/Inf detection. Repairs invalid states if repair_invalid is enabled. Prevents downstream errors from corrupted states. Essential for debugging complex THRML pipelines. Check_interval controls validation frequency for performance.'
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

  ml_nodes: [
    {
      type: 'ml',
      name: 'MLP Predictor',
      description: 'Multi-layer perceptron for prediction',
      config: {
        input_dim: 10,
        hidden_dims: [64, 32],
        output_dim: 3,
        model_path: null,
        learning_rate: 0.001
      },
      inputs: [
        { name: 'features', type: 'signal_array' },
        { name: 'train_trigger', type: 'discrete' }
      ],
      outputs: [
        { name: 'prediction', type: 'signal_array' },
        { name: 'loss', type: 'scalar' }
      ],
      category: 'ML',
      howItWorks: 'Multi-layer perceptron neural network for general-purpose prediction. Feed chaotic time series or any signal array into features input. Predictions flow from oscillator states through hidden layers to output. Use train_trigger to enable/disable online learning. Loss output tracks prediction error for monitoring.'
    },
    {
      type: 'ml',
      name: 'CNN Classifier',
      description: 'Convolutional neural network for pattern recognition',
      config: {
        channels: [16, 32],
        kernel_size: 3,
        output_classes: 5,
        model_path: null
      },
      inputs: [{ name: 'time_series', type: 'signal_array' }],
      outputs: [
        { name: 'class_probs', type: 'signal_array' },
        { name: 'predicted_class', type: 'discrete' }
      ],
      category: 'ML',
      howItWorks: 'Convolutional neural network designed for time-series pattern recognition. Automatically learns hierarchical features from chaotic oscillator trajectories. Useful for classifying attractor regimes, detecting bifurcations, or recognizing emergent patterns in THRML networks. Outputs class probabilities and most likely class.'
    },
    {
      type: 'ml',
      name: 'Transformer Encoder',
      description: 'BERT-style transformer for sequence processing',
      config: {
        model_name: 'bert-base',
        hidden_size: 768,
        num_heads: 12,
        num_layers: 6
      },
      inputs: [{ name: 'sequence', type: 'signal_array' }],
      outputs: [
        { name: 'embeddings', type: 'signal_array' },
        { name: 'attention_weights', type: 'matrix' }
      ],
      category: 'ML',
      howItWorks: 'Transformer encoder with multi-head self-attention for processing sequential data from oscillators or THRML samples. Captures long-range dependencies in chaotic dynamics. Can learn temporal structure in music, detect patterns in neural activity, or process any time-series data. Attention weights reveal which time steps influence predictions.'
    },
    {
      type: 'ml',
      name: 'Diffusion Generator',
      description: 'DDPM/DDIM diffusion model for generation',
      config: {
        timesteps: 1000,
        data_shape: [1, 256],
        beta_schedule: 'cosine',
        num_inference_steps: 50
      },
      inputs: [
        { name: 'noise', type: 'signal_array' },
        { name: 'guidance', type: 'signal' }
      ],
      outputs: [{ name: 'generated', type: 'signal_array' }],
      category: 'ML',
      howItWorks: 'Diffusion probabilistic model for generating novel patterns. Starts from random noise and gradually denoises to create structured outputs. Guidance input allows chaos or THRML energy to steer generation toward specific regions of the learned distribution. Useful for creative synthesis, pattern discovery, or data augmentation.'
    },
    {
      type: 'ml',
      name: 'GAN Generator',
      description: 'Generative adversarial network',
      config: {
        latent_dim: 100,
        output_length: 256,
        architecture: 'dcgan'
      },
      inputs: [
        { name: 'latent', type: 'signal_array' },
        { name: 'chaos_noise', type: 'signal' }
      ],
      outputs: [{ name: 'generated', type: 'signal_array' }],
      category: 'ML',
      howItWorks: 'Generative adversarial network trained to produce realistic patterns. Latent input controls generation style. Chaos_noise input can use chaotic oscillators as a unique noise source, creating diverse outputs that traditional GANs cannot produce. Excellent for sprite generation, world creation, or novel pattern synthesis.'
    },
    {
      type: 'ml',
      name: 'RL Controller',
      description: 'PPO/SAC reinforcement learning agent',
      config: {
        state_dim: 3,
        action_dim: 1,
        algorithm: 'ppo',
        learning_rate: 0.0003,
        gamma: 0.99
      },
      inputs: [
        { name: 'state', type: 'signal_array' },
        { name: 'reward', type: 'scalar' }
      ],
      outputs: [
        { name: 'action', type: 'signal' },
        { name: 'value', type: 'scalar' }
      ],
      category: 'ML',
      howItWorks: 'Reinforcement learning agent that learns to control systems through trial and error. Feed oscillator states as observations, define rewards (e.g., energy minimization, stability), and the agent learns optimal control policies. Output actions modulate oscillator parameters. Value estimates track expected future rewards.'
    },
    {
      type: 'ml',
      name: 'Autoencoder',
      description: 'Dimensionality reduction and compression',
      config: {
        input_dim: 100,
        latent_dim: 10,
        architecture: 'vanilla'
      },
      inputs: [{ name: 'input', type: 'signal_array' }],
      outputs: [
        { name: 'latent', type: 'signal_array' },
        { name: 'reconstructed', type: 'signal_array' }
      ],
      category: 'ML',
      howItWorks: 'Autoencoder compresses high-dimensional data into low-dimensional latent representations. Useful for visualizing attractor structures, finding compact representations of THRML states, or preprocessing for other ML models. Latent output reveals intrinsic dimensionality of chaotic dynamics.'
    }
  ],

  analysis_nodes: [
    {
      type: 'analysis',
      name: 'FFT Analyzer',
      description: 'Fast Fourier Transform spectral analysis',
      config: {
        size: 2048,
        window: 'hann',
        overlap: 0.5
      },
      inputs: [{ name: 'signal', type: 'signal' }],
      outputs: [
        { name: 'magnitude', type: 'signal_array' },
        { name: 'phase', type: 'signal_array' },
        { name: 'peak_freq', type: 'scalar' },
        { name: 'spectral_centroid', type: 'scalar' }
      ],
      category: 'Analysis',
      howItWorks: 'Transforms time-domain signals to frequency domain using Fast Fourier Transform. Reveals harmonic structure, dominant frequencies, and spectral content of oscillator outputs or audio signals. Peak frequency tracks the strongest spectral component. Spectral centroid indicates brightness or center of mass of the spectrum.'
    },
    {
      type: 'analysis',
      name: 'Pattern Recognizer',
      description: 'Detect repeating patterns in time series',
      config: {
        window_size: 100,
        threshold: 0.8,
        num_patterns: 10
      },
      inputs: [{ name: 'signal', type: 'signal' }],
      outputs: [
        { name: 'patterns', type: 'discrete' },
        { name: 'confidence', type: 'scalar' },
        { name: 'period', type: 'scalar' }
      ],
      category: 'Analysis',
      howItWorks: 'Uses autocorrelation and template matching to identify repeating patterns in chaotic or periodic signals. Outputs detected pattern IDs, confidence scores, and estimated periods. Useful for finding quasi-periodic orbits, detecting bifurcations, or identifying musical motifs in generated audio.'
    },
    {
      type: 'analysis',
      name: 'Lyapunov Calculator',
      description: 'Compute largest Lyapunov exponent',
      config: {
        window: 1000,
        neighbors: 10,
        embedding_dim: 3
      },
      inputs: [{ name: 'trajectory', type: 'signal_array' }],
      outputs: [
        { name: 'lyapunov', type: 'scalar' },
        { name: 'is_chaotic', type: 'discrete' }
      ],
      category: 'Analysis',
      howItWorks: 'Calculates the largest Lyapunov exponent to quantify chaos. Positive exponents indicate chaotic dynamics (sensitive dependence on initial conditions), zero indicates marginal stability, negative indicates convergence. Is_chaotic output provides binary classification. Essential for analyzing dynamical regime transitions.'
    },
    {
      type: 'analysis',
      name: 'Attractor Analyzer',
      description: 'Characterize strange attractors',
      config: {
        embedding_dim: 3,
        time_delay: 10
      },
      inputs: [{ name: 'time_series', type: 'signal_array' }],
      outputs: [
        { name: 'correlation_dim', type: 'scalar' },
        { name: 'entropy', type: 'scalar' }
      ],
      category: 'Analysis',
      howItWorks: 'Reconstructs phase space using time-delay embedding and estimates attractor properties. Correlation dimension reveals fractal structure. Entropy quantifies unpredictability. Helps classify oscillator regimes and detect transitions between periodic, quasi-periodic, and chaotic behavior.'
    },
    {
      type: 'analysis',
      name: 'Energy Surface Scanner',
      description: 'Map and analyze THRML energy landscape',
      config: {
        scan_resolution: 32,
        num_samples: 1000,
        compute_barriers: true
      },
      inputs: [
        { name: 'model_state', type: 'discrete' },
        { name: 'weights', type: 'matrix' }
      ],
      outputs: [
        { name: 'energy_histogram', type: 'signal_array' },
        { name: 'min_energy', type: 'scalar' },
        { name: 'max_energy', type: 'scalar' },
        { name: 'energy_barriers', type: 'signal_array' }
      ],
      category: 'Analysis',
      howItWorks: 'Samples the energy landscape to find minima, barriers, and metastable states. Useful for understanding model capacity, detecting mode collapse, or finding ground states. Energy barriers indicate difficulty of mixing between modes. Scan_resolution controls granularity of landscape mapping.'
    },
    {
      type: 'analysis',
      name: 'Convergence Detector',
      description: 'Estimate MCMC mixing and convergence time',
      config: {
        check_interval: 100,
        max_lag: 50,
        threshold: 0.01
      },
      inputs: [{ name: 'sample_stream', type: 'discrete' }],
      outputs: [
        { name: 'mixing_time', type: 'scalar' },
        { name: 'converged', type: 'discrete' },
        { name: 'autocorr_func', type: 'signal_array' }
      ],
      category: 'Analysis',
      howItWorks: 'Estimates how long it takes for the chain to forget initial conditions. Tracks autocorrelation function decay. Signals convergence when autocorrelation drops below threshold. Critical for knowing when to trust your samples. Max_lag sets how far back to look for correlations.'
    }
  ],

  control_nodes: [
    {
      type: 'control',
      name: 'Parameter Optimizer',
      description: 'Gradient-based parameter optimization',
      config: {
        learning_rate: 0.01,
        target_metric: 'energy',
        optimizer: 'adam',
        momentum: 0.9
      },
      inputs: [
        { name: 'current_value', type: 'scalar' },
        { name: 'target_value', type: 'scalar' }
      ],
      outputs: [
        { name: 'control_signal', type: 'signal' },
        { name: 'error', type: 'scalar' }
      ],
      category: 'Control',
      howItWorks: 'Gradient descent controller that adjusts parameters to minimize error between current and target values. Can optimize THRML energy, stabilize oscillators, or tune any continuous parameter. Supports multiple optimizers (SGD, Adam, RMSprop). Error output tracks convergence progress.'
    },
    {
      type: 'control',
      name: 'Chaos Controller',
      description: 'Control and stabilize chaotic dynamics',
      config: {
        method: 'pyragas',
        strength: 0.1,
        delay: 100
      },
      inputs: [
        { name: 'state', type: 'signal_array' },
        { name: 'target', type: 'signal_array' }
      ],
      outputs: [{ name: 'control', type: 'signal_array' }],
      category: 'Control',
      howItWorks: 'Implements Pyragas or OGY chaos control methods. Stabilizes unstable periodic orbits embedded in chaotic attractors using small perturbations. Delay parameter sets the feedback time scale. Useful for taming chaos when needed or navigating between different dynamical regimes.'
    },
    {
      type: 'control',
      name: 'PID Controller',
      description: 'Proportional-Integral-Derivative feedback control',
      config: {
        Kp: 1.0,
        Ki: 0.1,
        Kd: 0.01,
        setpoint: 0.0
      },
      inputs: [{ name: 'measurement', type: 'signal' }],
      outputs: [
        { name: 'control', type: 'signal' },
        { name: 'error', type: 'scalar' }
      ],
      category: 'Control',
      howItWorks: 'Classic PID controller for maintaining a desired setpoint. Proportional term responds to current error, integral term eliminates steady-state error, derivative term reduces overshoot. Tune Kp, Ki, Kd for stability. Applicable to oscillator control, THRML temperature regulation, or any feedback system.'
    },
    {
      type: 'control',
      name: 'Adaptive Gibbs Steps',
      description: 'Dynamically adjust number of sampling steps',
      config: {
        min_steps: 1,
        max_steps: 50,
        target_acceptance: 0.5,
        adaptation_rate: 0.1
      },
      inputs: [
        { name: 'current_acceptance', type: 'scalar' },
        { name: 'error_signal', type: 'scalar' }
      ],
      outputs: [
        { name: 'n_steps', type: 'scalar' },
        { name: 'step_size', type: 'scalar' }
      ],
      category: 'Control',
      howItWorks: 'Automatically tunes sampling steps based on acceptance rate and error metrics. More steps when acceptance is low (poor mixing). Fewer steps when acceptance is good (efficient sampling). Saves computation while maintaining quality. Adaptation_rate controls how quickly the controller responds to changes.'
    }
  ],

  generator_nodes: [
    {
      type: 'generator',
      name: 'Noise Generator',
      description: 'Generate various types of noise',
      config: {
        type: 'white',
        amplitude: 1.0,
        seed: 42
      },
      inputs: [{ name: 'amplitude_mod', type: 'signal' }],
      outputs: [{ name: 'noise', type: 'signal' }],
      category: 'Generator',
      howItWorks: 'Generates noise signals: white (flat spectrum), pink (1/f), brown (1/f²), or perlin (smooth). Amplitude modulation input allows dynamic control. Useful for testing systems, adding stochasticity, or as input to generative models. Seed ensures reproducibility.'
    },
    {
      type: 'generator',
      name: 'Pattern Generator',
      description: 'Generate test signals and patterns',
      config: {
        pattern: 'sine',
        frequency: 440,
        amplitude: 1.0,
        phase: 0.0
      },
      inputs: [
        { name: 'freq_mod', type: 'signal' },
        { name: 'amp_mod', type: 'signal' }
      ],
      outputs: [{ name: 'pattern', type: 'signal' }],
      category: 'Generator',
      howItWorks: 'Generates standard test signals: sine, square, triangle, sawtooth, or custom patterns. Frequency and amplitude can be modulated in real-time. Essential for testing, calibration, or as driving forces for oscillators. Phase parameter sets initial offset.'
    },
    {
      type: 'generator',
      name: 'Sequence Generator',
      description: 'Generate structured sequences',
      config: {
        length: 100,
        pattern_type: 'arithmetic',
        start: 0,
        step: 1
      },
      inputs: [{ name: 'trigger', type: 'discrete' }],
      outputs: [
        { name: 'sequence', type: 'signal_array' },
        { name: 'position', type: 'scalar' }
      ],
      category: 'Generator',
      howItWorks: 'Creates structured number sequences: arithmetic progressions, geometric series, fibonacci, or custom patterns. Trigger input restarts sequence. Useful for generating test data, creating rhythmic patterns, or providing controlled inputs to ML models or oscillators.'
    }
  ],

  audio: [
    {
      type: 'custom',
      name: 'Audio File Upload',
      description: 'Load and play audio files in the node graph',
      config: {
        file_path: null,
        sample_rate: 48000,
        loop: true,
        speed: 1.0,
        volume: 1.0
      },
      inputs: [
        { name: 'speed_mod', type: 'signal' },
        { name: 'volume_mod', type: 'signal' },
        { name: 'trigger', type: 'discrete' }
      ],
      outputs: [
        { name: 'audio_out', type: 'signal' },
        { name: 'position', type: 'scalar' },
        { name: 'duration', type: 'scalar' },
        { name: 'playing', type: 'discrete' }
      ],
      category: 'Audio',
      howItWorks: 'Upload and play audio files (MP3, WAV, FLAC, OGG, M4A, AAC) for use in the node graph. Supports looping, variable playback speed, and real-time volume control. Speed_mod and volume_mod inputs allow dynamic modulation. Trigger input resets playback to start. Position output tracks current playback time. Playing output indicates active playback state. Audio_out connects to oscillators, algorithms, or output nodes.'
    },
    {
      type: 'custom',
      name: 'Audio Input',
      description: 'Capture live audio from microphone or input device',
      config: {
        input_device: 'default',
        sample_rate: 48000,
        buffer_size: 256,
        gain: 1.0,
        enable_monitoring: true
      },
      inputs: [
        { name: 'gain_mod', type: 'signal' },
        { name: 'enable', type: 'discrete' }
      ],
      outputs: [
        { name: 'audio_out', type: 'signal' },
        { name: 'level', type: 'scalar' },
        { name: 'peak', type: 'scalar' },
        { name: 'clipping', type: 'discrete' }
      ],
      category: 'Audio',
      howItWorks: 'Captures live audio from microphone or audio interface. Enable input activates/deactivates capture. Gain_mod modulates input gain in real-time. Level output provides RMS audio level. Peak tracks maximum amplitude. Clipping signals when input exceeds 0dB. Use audio_out to feed oscillators, THRML models, or ML networks for real-time audio analysis and synthesis.'
    },
    {
      type: 'custom',
      name: 'Audio Output',
      description: 'Route audio to speakers or output device',
      config: {
        output_device: 'default',
        volume: 0.7,
        mute: false,
        limiter_enabled: true,
        limiter_threshold: -3.0
      },
      inputs: [
        { name: 'audio_in', type: 'signal' },
        { name: 'volume_mod', type: 'signal' },
        { name: 'mute', type: 'discrete' }
      ],
      outputs: [
        { name: 'level_out', type: 'scalar' },
        { name: 'limiting', type: 'discrete' }
      ],
      category: 'Audio',
      howItWorks: 'Routes audio to speakers or audio interface. Limiter prevents output from exceeding safe levels. Volume_mod allows dynamic volume control. Mute input silences output. Level_out monitors output amplitude. Limiting signals when limiter is active. Connect audio from oscillators, algorithms, or file players to this node for playback.'
    }
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

// Demo Configuration Presets
export interface GraphPreset {
  name: string;
  description: string;
  category: string;
  nodes: Array<{
    presetKey: string; // Key into NODE_PRESETS
    presetIndex: number; // Index within that category
    x: number;
    y: number;
    config?: Record<string, any>; // Optional config overrides
  }>;
  connections?: Array<{
    fromNode: number; // Index into nodes array
    fromPort: number;
    toNode: number;
    toPort: number;
  }>;
}

export const GRAPH_PRESETS: GraphPreset[] = [
  {
    name: 'Chaotic Attractor Visualization',
    description: 'Lorenz attractor with 3D phase space and waveform visualizers. Perfect for exploring chaotic dynamics.',
    category: 'Visualization',
    nodes: [
      { presetKey: 'oscillators', presetIndex: 1, x: 100, y: 150 }, // Lorenz
      { presetKey: 'visualizers', presetIndex: 2, x: 500, y: 100 }, // Phase Space 3D
      { presetKey: 'visualizers', presetIndex: 0, x: 500, y: 350 }, // Oscilloscope
    ],
    connections: [
      { fromNode: 0, fromPort: 0, toNode: 1, toPort: 0 }, // x -> Phase Space
      { fromNode: 0, fromPort: 1, toNode: 1, toPort: 1 }, // y -> Phase Space
      { fromNode: 0, fromPort: 2, toNode: 1, toPort: 2 }, // z -> Phase Space
      { fromNode: 0, fromPort: 0, toNode: 2, toPort: 0 }, // x -> Oscilloscope
    ]
  },
  {
    name: 'THRML Spin Glass Experiment',
    description: 'Spin Glass EBM with spin state matrix, energy tracking, and correlation analysis. Demonstrates THRML sampling.',
    category: 'THRML',
    nodes: [
      { presetKey: 'thrml', presetIndex: 0, x: 100, y: 200 }, // Spin Glass EBM
      { presetKey: 'visualizers', presetIndex: 4, x: 500, y: 100 }, // Spin State Matrix
      { presetKey: 'visualizers', presetIndex: 3, x: 500, y: 300 }, // Energy Graph
      { presetKey: 'visualizers', presetIndex: 5, x: 500, y: 500 }, // Correlation Matrix
    ],
    connections: [
      { fromNode: 0, fromPort: 0, toNode: 1, toPort: 0 }, // spins -> Spin Matrix
      { fromNode: 0, fromPort: 1, toNode: 2, toPort: 0 }, // energy -> Energy Graph
    ]
  },
  {
    name: 'Audio Processing Chain',
    description: 'Chua oscillator through waveshaper, compressor, and spectrogram. Shows complete audio signal flow.',
    category: 'Audio',
    nodes: [
      { presetKey: 'oscillators', presetIndex: 0, x: 100, y: 200 }, // Chua
      { presetKey: 'algorithms', presetIndex: 0, x: 350, y: 200 }, // Waveshaper
      { presetKey: 'algorithms', presetIndex: 3, x: 600, y: 200 }, // Compressor
      { presetKey: 'visualizers', presetIndex: 1, x: 850, y: 150 }, // Spectrogram
      { presetKey: 'visualizers', presetIndex: 0, x: 850, y: 350 }, // Oscilloscope
    ],
    connections: [
      { fromNode: 0, fromPort: 0, toNode: 1, toPort: 0 }, // Chua x -> Waveshaper
      { fromNode: 1, fromPort: 0, toNode: 2, toPort: 0 }, // Waveshaper -> Compressor
      { fromNode: 2, fromPort: 0, toNode: 3, toPort: 0 }, // Compressor -> Spectrogram
      { fromNode: 2, fromPort: 0, toNode: 4, toPort: 0 }, // Compressor -> Oscilloscope
    ]
  },
  {
    name: 'P-Bit Dynamics Exploration',
    description: 'Spin Glass with P-Bit compressor, gate, and threshold nodes. Demonstrates probabilistic bit manipulation.',
    category: 'THRML',
    nodes: [
      { presetKey: 'thrml', presetIndex: 0, x: 100, y: 250 }, // Spin Glass
      { presetKey: 'pbit_dynamics', presetIndex: 0, x: 400, y: 150 }, // P-Bit Compressor
      { presetKey: 'pbit_dynamics', presetIndex: 3, x: 400, y: 300 }, // P-Bit Gate
      { presetKey: 'pbit_dynamics', presetIndex: 4, x: 700, y: 225 }, // P-Bit Threshold
      { presetKey: 'visualizers', presetIndex: 7, x: 1000, y: 225 }, // P-Bit Mapper
    ],
    connections: [
      { fromNode: 0, fromPort: 0, toNode: 1, toPort: 0 }, // spins -> Compressor
      { fromNode: 1, fromPort: 0, toNode: 2, toPort: 0 }, // Compressor -> Gate
      { fromNode: 2, fromPort: 0, toNode: 3, toPort: 0 }, // Gate -> Threshold
      { fromNode: 3, fromPort: 0, toNode: 4, toPort: 0 }, // Threshold -> P-Bit Mapper
    ]
  },
  {
    name: 'Multi-Oscillator Synthesis',
    description: 'Chua, Lorenz, and Van der Pol oscillators mixed together. Create complex emergent behavior.',
    category: 'Synthesis',
    nodes: [
      { presetKey: 'oscillators', presetIndex: 0, x: 100, y: 100 }, // Chua
      { presetKey: 'oscillators', presetIndex: 1, x: 100, y: 300 }, // Lorenz
      { presetKey: 'oscillators', presetIndex: 2, x: 100, y: 500 }, // Van der Pol
      { presetKey: 'algorithms', presetIndex: 0, x: 450, y: 300 }, // Waveshaper (mixer)
      { presetKey: 'visualizers', presetIndex: 0, x: 750, y: 300 }, // Oscilloscope
    ],
    connections: [
      { fromNode: 0, fromPort: 0, toNode: 3, toPort: 0 }, // Chua -> Waveshaper
      { fromNode: 1, fromPort: 0, toNode: 3, toPort: 0 }, // Lorenz -> Waveshaper (same input)
      { fromNode: 3, fromPort: 0, toNode: 4, toPort: 0 }, // Waveshaper -> Oscilloscope
    ]
  },
  {
    name: 'THRML Training Pipeline',
    description: 'Ising model with training, moment accumulation, and convergence detection. Full THRML learning workflow.',
    category: 'THRML',
    nodes: [
      { presetKey: 'thrml', presetIndex: 10, x: 100, y: 250 }, // Ising Trainer
      { presetKey: 'thrml', presetIndex: 7, x: 450, y: 150 }, // Moment Accumulator
      { presetKey: 'analysis_nodes', presetIndex: 5, x: 450, y: 350 }, // Convergence Detector
      { presetKey: 'visualizers', presetIndex: 3, x: 800, y: 150 }, // Energy Graph
      { presetKey: 'visualizers', presetIndex: 5, x: 800, y: 350 }, // Correlation Matrix
    ],
    connections: [
      { fromNode: 0, fromPort: 0, toNode: 1, toPort: 0 }, // weights -> Moment Accumulator
      { fromNode: 0, fromPort: 1, toNode: 3, toPort: 0 }, // kl_divergence -> Energy Graph
      { fromNode: 1, fromPort: 2, toNode: 4, toPort: 0 }, // correlations -> Correlation Matrix
    ]
  },
  {
    name: 'Photonic Processing',
    description: 'Lorenz oscillator through optical Kerr effect and four-wave mixing. Explore photonic nonlinearities.',
    category: 'Photonic',
    nodes: [
      { presetKey: 'oscillators', presetIndex: 1, x: 100, y: 200 }, // Lorenz
      { presetKey: 'algorithms', presetIndex: 8, x: 400, y: 200 }, // Optical Kerr Effect
      { presetKey: 'algorithms', presetIndex: 10, x: 700, y: 200 }, // Four-Wave Mixing
      { presetKey: 'visualizers', presetIndex: 7, x: 1000, y: 200 }, // XY Plot
    ],
    connections: [
      { fromNode: 0, fromPort: 0, toNode: 1, toPort: 0 }, // Lorenz -> Kerr
      { fromNode: 1, fromPort: 0, toNode: 2, toPort: 0 }, // Kerr -> Four-Wave Mixing
      { fromNode: 2, fromPort: 0, toNode: 3, toPort: 0 }, // Four-Wave -> XY Plot
    ]
  },
  {
    name: 'ML Predictor Training',
    description: 'Lorenz attractor feeding MLP predictor for time series forecasting. Real-time ML training on chaos.',
    category: 'Machine Learning',
    nodes: [
      { presetKey: 'oscillators', presetIndex: 1, x: 100, y: 200 }, // Lorenz
      { presetKey: 'ml_nodes', presetIndex: 0, x: 450, y: 200 }, // MLP Predictor
      { presetKey: 'visualizers', presetIndex: 3, x: 800, y: 200 }, // Energy Graph (shows loss)
    ],
    connections: [
      { fromNode: 0, fromPort: 0, toNode: 1, toPort: 0 }, // Lorenz -> MLP features
      { fromNode: 1, fromPort: 1, toNode: 2, toPort: 0 }, // loss -> Energy Graph
    ]
  },
  {
    name: 'Chaos Analysis Suite',
    description: 'Chua oscillator with Lyapunov calculator, attractor analyzer, and FFT. Complete chaos characterization.',
    category: 'Analysis',
    nodes: [
      { presetKey: 'oscillators', presetIndex: 0, x: 100, y: 300 }, // Chua
      { presetKey: 'analysis_nodes', presetIndex: 2, x: 450, y: 150 }, // Lyapunov Calculator
      { presetKey: 'analysis_nodes', presetIndex: 3, x: 450, y: 300 }, // Attractor Analyzer
      { presetKey: 'analysis_nodes', presetIndex: 0, x: 450, y: 450 }, // FFT Analyzer
      { presetKey: 'visualizers', presetIndex: 0, x: 800, y: 300 }, // Oscilloscope
    ],
    connections: [
      { fromNode: 0, fromPort: 0, toNode: 1, toPort: 0 }, // Chua -> Lyapunov
      { fromNode: 0, fromPort: 0, toNode: 2, toPort: 0 }, // Chua -> Attractor Analyzer
      { fromNode: 0, fromPort: 0, toNode: 3, toPort: 0 }, // Chua -> FFT
      { fromNode: 0, fromPort: 0, toNode: 4, toPort: 0 }, // Chua -> Oscilloscope
    ]
  },
  {
    name: 'Feedback Control System',
    description: 'Van der Pol oscillator with PID controller for stabilization. Demonstrates chaos control.',
    category: 'Control',
    nodes: [
      { presetKey: 'oscillators', presetIndex: 2, x: 100, y: 200 }, // Van der Pol
      { presetKey: 'control_nodes', presetIndex: 2, x: 400, y: 200 }, // PID Controller
      { presetKey: 'visualizers', presetIndex: 0, x: 700, y: 200 }, // Oscilloscope
    ],
    connections: [
      { fromNode: 0, fromPort: 0, toNode: 1, toPort: 0 }, // Van der Pol -> PID
      { fromNode: 1, fromPort: 0, toNode: 2, toPort: 0 }, // PID -> Oscilloscope
    ]
  }
];

