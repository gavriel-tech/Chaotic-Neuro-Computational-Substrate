"""
Simulation Bridge Nodes for GMCS Node Graph.

Connects the node graph system to the core simulation components:
- OscillatorNode: Access Chua oscillators
- THRMLNode: Access THRML sampling
- WavePDENode: Access wave field simulation

These nodes act as bridges between the node graph executor
and the unified simulation state.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import core simulation components
try:
    from src.core.state import SystemState, initialize_system_state
    from src.core.thrml_integration import THRMLWrapper, create_thrml_model
    from src.core.wave_pde import compute_field_energy
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False
    print("[WARNING] Core simulation modules not available - simulation bridge will use stubs")


# ============================================================================
# Oscillator Node (Bridge to Chua Oscillators)
# ============================================================================

@dataclass
class OscillatorConfig:
    """Configuration for oscillator node."""
    count: int = 64
    alpha: float = 15.6
    beta: float = 28.0
    m0: float = -1.143
    m1: float = -0.714
    initial_state: str = "random"  # "random", "zero", "harmonic", "lattice"
    enable_coupling: bool = True
    coupling_strength: float = 0.2
    coupling_topology: str = "all_to_all"  # "all_to_all", "nearest", "random", "2d_grid", "small_world"


class OscillatorNode:
    """
    Bridge to Chua oscillator array in core simulation.
    
    Provides access to oscillator states (x, y, z) and allows
    modulation of parameters.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'oscillator')
        else:
            self.node_id = node_id_or_config
            config = kwargs
        
        # Map 'num_oscillators' to 'count' for config
        if 'num_oscillators' in config and 'count' not in config:
            config['count'] = config.pop('num_oscillators')
        
        self.config = OscillatorConfig(**config)
        
        # Expose num_oscillators attribute for test compatibility
        self.num_oscillators = self.config.count
        
        # Local state for when not connected to simulation
        self.states = self._initialize_states()
        self.time = 0.0
        self.dt = 0.01
        
        print(f"[OscillatorNode] Initialized {self.config.count} oscillators")
    
    def _initialize_states(self) -> np.ndarray:
        """Initialize oscillator states."""
        if self.config.initial_state == "random":
            states = np.random.randn(self.config.count, 3) * 0.1
        elif self.config.initial_state == "zero":
            states = np.zeros((self.config.count, 3))
        elif self.config.initial_state == "harmonic":
            # Arrange in phase-shifted sine wave
            phases = np.linspace(0, 2*np.pi, self.config.count)
            states = np.zeros((self.config.count, 3))
            states[:, 0] = np.sin(phases)
            states[:, 1] = np.cos(phases)
        elif self.config.initial_state == "lattice":
            # Arrange in regular grid
            grid_size = int(np.ceil(np.sqrt(self.config.count)))
            states = np.zeros((self.config.count, 3))
            for i in range(self.config.count):
                x_idx = i % grid_size
                y_idx = i // grid_size
                states[i, 0] = (x_idx / grid_size - 0.5) * 2
                states[i, 1] = (y_idx / grid_size - 0.5) * 2
        else:
            states = np.random.randn(self.config.count, 3) * 0.1
        
        return states
    
    def process(
        self,
        modulation: Optional[np.ndarray] = None,
        initial_conditions: Optional[np.ndarray] = None,
        **inputs
    ) -> Dict[str, Any]:
        """
        Access oscillator states and apply modulation.
        
        Args:
            modulation: Modulation signal for coupling or parameters
            initial_conditions: Reset oscillators to these states
            **inputs: Additional parameter modulations (alpha, beta, etc.)
            
        Returns:
            Dictionary with:
            - x: X coordinates of all oscillators
            - y: Y coordinates
            - z: Z coordinates
            - states: Full (N, 3) state array
        """
        # Handle reset
        if initial_conditions is not None:
            if isinstance(initial_conditions, np.ndarray):
                self.states[:len(initial_conditions)] = initial_conditions
        
        # Apply parameter modulations
        if 'alpha' in inputs:
            # Would modulate alpha parameter in real simulation
            pass
        if 'beta' in inputs:
            # Would modulate beta parameter
            pass
        if 'coupling_strength' in inputs:
            # Would modulate coupling
            pass
        
        # Simple integration step (placeholder - real simulation uses core integrators)
        self._simple_step()
        
        # Return with both 'state' and 'states' keys for compatibility
        state_copy = self.states.copy()
        return {
            'state': state_copy,  # singular for test compatibility
            'states': state_copy,  # plural for backward compatibility
            'x': state_copy[:, 0].copy(),
            'y': state_copy[:, 1].copy(),
            'z': state_copy[:, 2].copy()
        }
    
    def _simple_step(self):
        """
        Simple oscillator integration (placeholder).
        
        Real implementation would use src/core/integrators.py
        """
        # Simplified Chua oscillator dynamics
        x, y, z = self.states[:, 0], self.states[:, 1], self.states[:, 2]
        
        # Nonlinear function
        h = self.config.m1 * x + 0.5 * (self.config.m0 - self.config.m1) * (np.abs(x + 1) - np.abs(x - 1))
        
        # Derivatives
        dx = self.config.alpha * (y - x - h)
        dy = x - y + z
        dz = -self.config.beta * y
        
        # Euler integration
        self.states[:, 0] += dx * self.dt
        self.states[:, 1] += dy * self.dt
        self.states[:, 2] += dz * self.dt
        
        self.time += self.dt


# ============================================================================
# THRML Node (Bridge to THRML Sampling)
# ============================================================================

@dataclass
class THRMLConfig:
    """Configuration for THRML node."""
    num_nodes: int = 64
    node_type: str = "continuous"  # "spin", "continuous", "discrete"
    temperature: float = 1.0
    num_samples: int = 10
    num_chains: int = 4
    mode: str = "SPEED"  # "SPEED", "ACCURACY", "RESEARCH"
    custom_energy: Optional[str] = None  # "harmonic", "drug_likeness", etc.


class THRMLNode:
    """
    Bridge to THRML sampling system.
    
    Provides probabilistic sampling and energy-based modeling.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'thrml')
        else:
            self.node_id = node_id_or_config
            config = kwargs
        
        self.config = THRMLConfig(**config)
        self.thrml_wrapper = None
        self.samples = np.zeros(self.config.num_nodes)
        self.energy = 0.0
        self.last_temperature = self.config.temperature
        self._rng_key = jax.random.PRNGKey(0)
        
        if SIMULATION_AVAILABLE:
            self._initialize_thrml()
        else:
            print("[THRMLNode] Using stub THRML")
        
        print(f"[THRMLNode] Initialized {self.config.num_nodes} nodes, type={self.config.node_type}")
    
    def _initialize_thrml(self):
        """Initialize THRML wrapper."""
        try:
            self.thrml_wrapper = create_thrml_model(
                n_nodes=self.config.num_nodes,
                node_type=self.config.node_type,
                beta=1.0 / self.config.temperature,
                initial_weights='random',
                initial_biases='zero'
            )
        except Exception as e:
            print(f"[THRMLNode WARNING] Could not initialize THRML: {e}")
            self.thrml_wrapper = None
    
    def process(
        self,
        bias: Optional[np.ndarray] = None,
        temperature: Optional[float] = None,
        structure: Optional[Any] = None,
        activity: Optional[np.ndarray] = None,
        **inputs
    ) -> Dict[str, Any]:
        """
        Run THRML sampling.
        
        Args:
            bias: External bias for sampling
            temperature: Override default temperature
            structure: Update THRML structure (e.g., from architectures)
            activity: Activity signal for plasticity
            **inputs: Additional inputs
            
        Returns:
            Dictionary with:
            - samples: Sampled states
            - energy: System energy
            - temperature: Current temperature
        """
        # Update temperature
        if temperature is not None:
            self.last_temperature = temperature
            if self.thrml_wrapper:
                self.thrml_wrapper.beta = 1.0 / temperature
        
        # Update biases
        if bias is not None and self.thrml_wrapper:
            bias_array = np.array(bias).flatten()[:self.config.num_nodes]
            # Would update biases in real THRML
        
        # Sample
        if self.thrml_wrapper:
            try:
                if bias is not None:
                    bias_array = np.array(bias).flatten()[:self.config.num_nodes]
                    self.thrml_wrapper.update_biases(bias_array)

                # Advance RNG and sample from THRML wrapper
                self._rng_key, subkey = jax.random.split(self._rng_key)
                gibbs_steps = max(1, int(self.config.num_samples))
                sampled = self.thrml_wrapper.sample_gibbs(
                    n_steps=gibbs_steps,
                    temperature=self.last_temperature,
                    key=subkey,
                )
                self.samples = np.asarray(sampled, dtype=float)
                self.energy = float(self.thrml_wrapper.compute_energy(self.samples))

            except Exception as e:
                print(f"[THRMLNode WARNING] Sampling failed: {e}")
                self.samples = np.zeros(self.config.num_nodes)
                self.energy = 0.0
        else:
            # Stub sampling
            if self.config.node_type == "spin":
                self.samples = np.random.choice([-1, 1], size=self.config.num_nodes)
            elif self.config.node_type == "continuous":
                self.samples = np.random.randn(self.config.num_nodes) * np.sqrt(self.last_temperature)
            else:
                self.samples = np.random.randint(0, 10, size=self.config.num_nodes)
            
            self.energy = np.sum(self.samples ** 2) * 0.5
        
        return {
            'samples': self.samples.copy(),
            'energy': float(self.energy),
            'temperature': self.last_temperature
        }


# ============================================================================
# Wave PDE Node (Bridge to Wave Field Simulation)
# ============================================================================

@dataclass
class WavePDEConfig:
    """Configuration for wave PDE node."""
    grid_size: list = None  # [width, height, depth]
    wave_speed: float = 1.0
    damping: float = 0.01
    boundary: str = "periodic"  # "periodic", "absorbing", "reflecting"


class WavePDENode:
    """
    Bridge to 3D wave field simulation.
    
    Provides access to wave dynamics and allows injection of sources.
    """
    
    def __init__(self, node_id_or_config, **kwargs):
        # Support both calling conventions
        if isinstance(node_id_or_config, dict):
            config = node_id_or_config
            self.node_id = config.get('node_id', 'wave_pde')
        else:
            self.node_id = node_id_or_config
            config = kwargs
        
        # Ensure grid_size has a default if not provided
        if 'grid_size' not in config or config['grid_size'] is None:
            config['grid_size'] = [128, 128, 64]
        
        self.config = WavePDEConfig(**config)
        
        # Initialize field
        grid_shape = tuple(self.config.grid_size[:2])  # Use 2D for now
        self.field = np.zeros(grid_shape)
        self.field_prev = np.zeros(grid_shape)
        self.time = 0.0
        self.dt = 0.01
        
        print(f"[WavePDENode] Initialized wave field: {grid_shape}")
    
    def process(
        self,
        source: Optional[np.ndarray] = None,
        **inputs
    ) -> Dict[str, Any]:
        """
        Step wave field simulation.
        
        Args:
            source: Source term to inject into field
            **inputs: Additional inputs
            
        Returns:
            Dictionary with:
            - field: Current wave field
            - energy: Field energy
        """
        # Apply source
        if source is not None:
            source_array = np.array(source)
            # Add source to field (simplified)
            if source_array.ndim == 1:
                # Point sources from oscillators
                pass  # Would distribute sources in real implementation
            else:
                # Field source
                if source_array.shape == self.field.shape:
                    self.field += source_array * 0.01
        
        # Simple wave propagation (placeholder)
        self._simple_wave_step()
        
        # Compute energy
        energy = float(np.sum(self.field ** 2))
        
        return {
            'field': self.field.copy(),
            'energy': energy
        }
    
    def _simple_wave_step(self):
        """
        Simple 2D wave equation integration.
        
        Real implementation would use src/core/wave_pde.py
        """
        # Compute Laplacian (simplified 5-point stencil)
        laplacian = (
            np.roll(self.field, 1, axis=0) +
            np.roll(self.field, -1, axis=0) +
            np.roll(self.field, 1, axis=1) +
            np.roll(self.field, -1, axis=1) -
            4 * self.field
        )
        
        # Wave equation: d²φ/dt² = c² ∇²φ - γ dφ/dt
        c_sq = self.config.wave_speed ** 2
        velocity = (self.field - self.field_prev) / self.dt
        
        field_new = (
            2 * self.field -
            self.field_prev +
            c_sq * self.dt**2 * laplacian -
            self.config.damping * self.dt * velocity
        )
        
        # Apply boundaries
        if self.config.boundary == "absorbing":
            field_new[:2, :] = 0
            field_new[-2:, :] = 0
            field_new[:, :2] = 0
            field_new[:, -2:] = 0
        
        # Update
        self.field_prev = self.field.copy()
        self.field = field_new
        self.time += self.dt


# ============================================================================
# Node Factory Registration Helpers
# ============================================================================

def create_simulation_bridge_node(node_type: str, config: Dict[str, Any]) -> Any:
    """
    Factory function to create simulation bridge nodes.
    
    Args:
        node_type: Type of bridge node ("oscillator", "thrml", "wave_pde")
        config: Node configuration
        
    Returns:
        Bridge node instance
    """
    if node_type == "oscillator":
        return OscillatorNode(config)
    elif node_type == "thrml":
        return THRMLNode(config)
    elif node_type == "wave_pde":
        return WavePDENode(config)
    else:
        raise ValueError(f"Unknown simulation bridge node type: {node_type}")

