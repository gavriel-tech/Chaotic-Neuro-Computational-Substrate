# GMCS Algorithm Reference

Complete reference for all 21 GMCS algorithms.

## Table of Contents

1. [Basic Algorithms (0-6)](#basic-algorithms)
2. [Audio/Signal Processing (7-13)](#audiosignal-processing)
3. [Photonic Algorithms (14-20)](#photonic-algorithms)
4. [Parameter Ranges](#parameter-ranges)
5. [Usage Examples](#usage-examples)

---

## Basic Algorithms

### 0. No-Op (ALGO_NOP)
**Description**: Pass-through algorithm with no modification.

**Equation**: `output = input`

**Parameters**: None

**Use Cases**: Placeholder, debugging, bypass

---

### 1. Limiter (ALGO_LIMITER)
**Description**: Soft clipping using hyperbolic tangent.

**Equation**: `output = A_max * tanh(input / A_max)`

**Parameters**:
- `A_max` (0.1 - 10.0): Maximum amplitude

**Use Cases**: Prevent overflow, soft saturation

---

### 2. Compressor (ALGO_COMPRESSOR)
**Description**: Dynamic range compression above threshold.

**Equation**:
```
if input > T:
    output = T + (input - T) / R
else:
    output = input
```

**Parameters**:
- `R_comp` (1.0 - 20.0): Compression ratio
- `T_comp` (0.0 - 5.0): Threshold

**Use Cases**: Dynamic range control, leveling

---

### 3. Expander (ALGO_EXPANDER)
**Description**: Dynamic range expansion below threshold.

**Equation**:
```
if input < T:
    output = T - (T - input) / R
else:
    output = input
```

**Parameters**:
- `R_exp` (1.0 - 20.0): Expansion ratio
- `T_exp` (0.0 - 5.0): Threshold

**Use Cases**: Noise gating, dynamic enhancement

---

### 4. Threshold (ALGO_THRESHOLD)
**Description**: Hard threshold gate.

**Equation**:
```
if input >= T:
    output = input
else:
    output = 0
```

**Parameters**:
- `T_comp` (0.0 - 5.0): Threshold value

**Use Cases**: Binary gating, noise removal

---

### 5. Phase Modulation (ALGO_PHASEMOD)
**Description**: Time-varying amplitude modulation.

**Equation**: `output = input * (1 + Φ * sin(ω * t))`

**Parameters**:
- `Phi` (0.0 - 1.0): Modulation depth
- `omega` (0.0 - 100.0): Modulation frequency

**Use Cases**: Tremolo, amplitude modulation, vibrato

---

### 6. Fold (ALGO_FOLD)
**Description**: Wave folding nonlinearity.

**Equation**: `output = γ * arcsin(sin(β * input))`

**Parameters**:
- `gamma` (0.1 - 5.0): Output gain
- `beta` (0.1 - 10.0): Fold factor

**Use Cases**: Harmonic generation, distortion

---

## Audio/Signal Processing

### 7. Resonator (ALGO_RESONATOR)
**Description**: Resonant bandpass filter.

**Equation**: Second-order resonator with center frequency f0 and Q factor

**Parameters**:
- `f0` (20.0 - 20000.0): Center frequency (Hz)
- `Q` (0.5 - 100.0): Quality factor

**Use Cases**: Formant synthesis, resonance effects

---

### 8. Hilbert Transform (ALGO_HILBERT)
**Description**: 90° phase shift (approximation).

**Equation**: Approximated using all-pass filters

**Parameters**: None

**Use Cases**: Single-sideband modulation, analytic signal

---

### 9. Rectifier (ALGO_RECTIFIER)
**Description**: Full-wave rectification.

**Equation**: `output = |input|`

**Parameters**: None

**Use Cases**: Envelope detection, frequency doubling

---

### 10. Quantizer (ALGO_QUANTIZER)
**Description**: Bit-depth reduction.

**Equation**: `output = round(input * levels) / levels`

**Parameters**:
- `levels` (2 - 256): Number of quantization levels

**Use Cases**: Bit crushing, lo-fi effects

---

### 11. Slew Limiter (ALGO_SLEW_LIMITER)
**Description**: Rate-of-change limiter.

**Equation**: Limits `|output[n] - output[n-1]|` to rate_limit

**Parameters**:
- `rate_limit` (0.01 - 10.0): Maximum rate of change

**Use Cases**: Portamento, glide, smoothing

---

### 12. Cross Modulation (ALGO_CROSS_MOD)
**Description**: Ring modulation.

**Equation**: `output = input * (1 + Φ * sin(ω * input))`

**Parameters**:
- `Phi` (0.0 - 1.0): Modulation depth
- `omega` (0.0 - 100.0): Modulation frequency

**Use Cases**: Ring modulation, metallic timbres

---

### 13. Bipolar Fold (ALGO_BIPOLAR_FOLD)
**Description**: Symmetric wave folding.

**Equation**:
```
if |input| > T:
    output = sign(input) * (2*T - |input|)
else:
    output = input
```

**Parameters**:
- `T_comp` (0.1 - 5.0): Fold threshold

**Use Cases**: Symmetric distortion, harmonic generation

---

## Photonic Algorithms

### 14. Optical Kerr Effect (ALGO_OPTICAL_KERR)
**Description**: Intensity-dependent refractive index (χ³ nonlinearity).

**Equation**: `output = input * (1 + n2 * |input|² * β)`

**Parameters**:
- `n2` (0.0 - 1.0): Nonlinear refractive index
- `beta` (0.0 - 10.0): Propagation constant

**Use Cases**: Self-phase modulation, solitons

---

### 15. Electro-Optic Modulation (ALGO_ELECTRO_OPTIC)
**Description**: Pockels effect modulation.

**Equation**: `output = input * cos(π * V / (2 * V_pi))`

**Parameters**:
- `V` (0.0 - 10.0): Applied voltage
- `V_pi` (0.1 - 10.0): Half-wave voltage

**Use Cases**: Optical switching, modulation

---

### 16. Optical Switch (ALGO_OPTICAL_SWITCH)
**Description**: Intensity-dependent switching.

**Equation**:
```
if |input| > T:
    output = input * γ
else:
    output = 0
```

**Parameters**:
- `T_comp` (0.0 - 5.0): Switching threshold
- `gamma` (0.1 - 5.0): Transmission coefficient

**Use Cases**: All-optical switching, thresholding

---

### 17. Four-Wave Mixing (ALGO_FOUR_WAVE_MIXING)
**Description**: Third-order nonlinear interaction.

**Equation**: `output = input + γ * n2 * |input|² * input`

**Parameters**:
- `gamma` (0.0 - 5.0): Nonlinear coefficient
- `n2` (0.0 - 1.0): Nonlinear refractive index

**Use Cases**: Wavelength conversion, parametric amplification

---

### 18. Raman Amplifier (ALGO_RAMAN_AMPLIFIER)
**Description**: Stimulated Raman scattering amplification.

**Equation**: `output = input * exp(γ * n2 * β * |input|²)`

**Parameters**:
- `gamma` (0.0 - 5.0): Raman gain coefficient
- `n2` (0.0 - 1.0): Nonlinear coefficient
- `beta` (0.0 - 10.0): Propagation constant

**Use Cases**: Optical amplification, distributed gain

---

### 19. Saturation (ALGO_SATURATION)
**Description**: Soft saturation nonlinearity.

**Equation**: `output = A_max * tanh(input / A_max)`

**Parameters**:
- `A_max` (0.1 - 10.0): Saturation level

**Use Cases**: Gain saturation, soft limiting

---

### 20. Optical Gain (ALGO_OPTICAL_GAIN)
**Description**: Linear optical amplification.

**Equation**: `output = input * γ`

**Parameters**:
- `gamma` (0.1 - 10.0): Gain coefficient

**Use Cases**: Amplification, signal boosting

---

## Parameter Ranges

| Parameter | Min | Max | Default | Unit |
|-----------|-----|-----|---------|------|
| A_max | 0.1 | 10.0 | 1.0 | - |
| R_comp | 1.0 | 20.0 | 4.0 | ratio |
| T_comp | 0.0 | 5.0 | 1.0 | - |
| R_exp | 1.0 | 20.0 | 2.0 | ratio |
| T_exp | 0.0 | 5.0 | 0.5 | - |
| Phi | 0.0 | 1.0 | 0.5 | - |
| omega | 0.0 | 100.0 | 10.0 | rad/s |
| gamma | 0.1 | 5.0 | 1.0 | - |
| beta | 0.1 | 10.0 | 1.0 | - |
| f0 | 20.0 | 20000.0 | 440.0 | Hz |
| Q | 0.5 | 100.0 | 10.0 | - |
| levels | 2 | 256 | 16 | - |
| rate_limit | 0.01 | 10.0 | 1.0 | units/step |
| n2 | 0.0 | 1.0 | 0.1 | - |
| V | 0.0 | 10.0 | 5.0 | V |
| V_pi | 0.1 | 10.0 | 3.14 | V |

---

## Usage Examples

### Example 1: Basic Limiter Chain
```python
# Node with limiter -> compressor -> fold
chain = [
    (ALGO_LIMITER, [2.0, 0, 0, 0, 0, 0, 0, 0]),
    (ALGO_COMPRESSOR, [4.0, 1.5, 0, 0, 0, 0, 0, 0]),
    (ALGO_FOLD, [1.0, 2.0, 0, 0, 0, 0, 0, 0])
]
```

### Example 2: Resonant Filter
```python
# Resonator at 440 Hz with Q=20
chain = [
    (ALGO_RESONATOR, [440.0, 20.0, 0, 0, 0, 0, 0, 0])
]
```

### Example 3: Photonic Kerr Effect
```python
# Optical Kerr with moderate nonlinearity
chain = [
    (ALGO_OPTICAL_KERR, [0.3, 2.0, 0, 0, 0, 0, 0, 0])
]
```

### Example 4: Complex Audio Chain
```python
# Compressor -> Resonator -> Fold -> Limiter
chain = [
    (ALGO_COMPRESSOR, [4.0, 1.0, 0, 0, 0, 0, 0, 0]),
    (ALGO_RESONATOR, [880.0, 15.0, 0, 0, 0, 0, 0, 0]),
    (ALGO_FOLD, [1.5, 3.0, 0, 0, 0, 0, 0, 0]),
    (ALGO_LIMITER, [2.5, 0, 0, 0, 0, 0, 0, 0])
]
```

---

## API Integration

### REST API
```bash
# List all algorithms
GET /algorithms/list

# Get algorithm details
GET /algorithms/{id}

# Get algorithms by category
GET /algorithms/list?category=photonic
```

### Python API
```python
from src.core.gmcs_pipeline import gmcs_pipeline, ALGO_LIMITER

# Apply algorithm
output = gmcs_pipeline(input_signal, chain_matrix, params_matrix)
```

---

## Performance Notes

- All algorithms are JAX-jitted for GPU acceleration
- Typical latency: < 1ms per node on GPU
- Supports vectorized batch processing
- Real-time capable at 48kHz sample rate

---

## See Also

- [Architecture Documentation](architecture.md)
- [API Reference](API_REFERENCE.md)
- [Plugin Development Guide](PLUGIN_DEVELOPMENT.md)

