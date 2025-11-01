"""
Generic Sampler Backend Interface for GMCS.

Provides an abstract interface for probabilistic samplers that works across
different hardware backends (THRML/JAX, photonic processors, neuromorphic chips,
quantum annealers, etc.).

Key Design Principles:
- Backend-agnostic: Same interface for all sampler types
- Capability detection: Backends declare what they support
- Hot-reload: Switch backends without restarting simulation
- Extensible: Easy to add new backend types

Architecture:
    SamplerBackend (abstract)
        ├── THRMLSamplerBackend (JAX-based, GPU-accelerated)
        ├── PhotonicBackend (future: optical processors)
        ├── NeuromorphicBackend (future: spiking neural hardware)
        └── QuantumBackend (future: quantum annealers)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np

# Import blocking strategies
from src.core.blocking_strategies import BlockingStrategy, Block, ValidationResult

try:
    import jax  # type: ignore
except ImportError:  # pragma: no cover - fallback when JAX unavailable
    jax = None  # type: ignore


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BackendCapabilities:
    """Declarative metadata describing a sampler backend."""

    supports_multi_chain: bool = False
    supports_conditional: bool = False
    supports_gpu: bool = False
    supports_distributed: bool = False
    supports_hot_reload: bool = False
    max_nodes: int = 1024
    optimal_strategies: List[str] = field(default_factory=lambda: ["random"])  # Universal fallback
    hardware_type: str = "cpu"  # "cpu", "gpu", "photonic", "neuromorphic", "quantum"
    extra: Dict[str, Any] = field(default_factory=dict)


class BackendInfo(NamedTuple):
    """Metadata about a registered backend."""
    name: str
    available: bool
    capabilities: BackendCapabilities
    description: str
    version: str


@dataclass
class BackendDiagnostics:
    """
    Standardized diagnostics from any backend.
    
    All backends must provide these metrics for monitoring and benchmarking.
    """
    samples_per_sec: float = 0.0
    ess_per_sec: float = 0.0  # Effective Sample Size per second
    lag1_autocorr: float = 0.0
    tau_int: float = 1.0  # Integrated autocorrelation time
    energy: float = 0.0
    wall_time: float = 0.0
    n_samples: int = 0
    n_chains: int = 1
    strategy: str = "unknown"
    memory_mb: float = 0.0
    
    # Optional per-chain diagnostics
    per_chain_ess: Optional[List[float]] = None
    per_chain_autocorr: Optional[List[float]] = None


class ClampMode(Enum):
    """Mode for conditional sampling (clamped nodes)."""
    INPAINT = "inpaint"  # Restore corrupted/missing data
    CONSTRAIN = "constrain"  # User-specified fixed nodes
    PATTERN = "pattern"  # ML-style pattern completion


# ============================================================================
# Abstract Base Class
# ============================================================================

class SamplerBackend(ABC):
    """
    Abstract base class for probabilistic samplers.
    
    All sampler backends must implement this interface. This enables
    backend-agnostic code in the simulation loop and API layer.
    
    Key Methods:
    - sample(): Generate samples from current distribution
    - compute_energy(): Evaluate energy of a state
    - update_weights(): Learn from data (for EBMs)
    - get_diagnostics(): Return performance metrics
    - set_blocking_strategy(): Configure parallel sampling
    - set_conditional_nodes(): Enable clamped sampling
    
    Lifecycle:
    1. Create backend instance
    2. Configure (blocking strategy, chains, etc.)
    3. Sample repeatedly in simulation loop
    4. Optionally learn (update weights)
    5. Serialize for checkpointing
    """
    
    def __init__(
        self,
        name: str,
        n_nodes: Optional[int] = None,
        initial_weights: Optional[np.ndarray] = None,
        initial_biases: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize backend base state.

        Args:
            name: Backend identifier.
            n_nodes: Number of nodes represented by backend.
            initial_weights: Optional initial weight matrix.
            initial_biases: Optional initial bias vector.
            seed: Optional RNG seed (defaults to random 32-bit value).
            **kwargs: Backend specific configuration captured for subclasses.
        """
        self.name = name
        inferred_nodes = None
        if n_nodes is not None:
            inferred_nodes = int(n_nodes)
        elif initial_weights is not None:
            inferred_nodes = int(initial_weights.shape[0])
        elif initial_biases is not None:
            inferred_nodes = int(initial_biases.shape[0])
        else:
            inferred_nodes = 0

        self.n_nodes: int = inferred_nodes

        def _normalize_weights(weights: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if weights is None:
                if self.n_nodes:
                    return np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
                return np.zeros((0, 0), dtype=np.float32)
            return np.asarray(weights, dtype=np.float32).copy()

        def _normalize_biases(biases: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if biases is None:
                if self.n_nodes:
                    return np.zeros(self.n_nodes, dtype=np.float32)
                return np.zeros(0, dtype=np.float32)
            return np.asarray(biases, dtype=np.float32).copy()

        self.initial_weights: Optional[np.ndarray] = _normalize_weights(initial_weights)
        self.initial_biases: Optional[np.ndarray] = _normalize_biases(initial_biases)
        self.current_weights: np.ndarray = (
            self.initial_weights.copy()
            if self.initial_weights is not None
            else _normalize_weights(None)
        )
        self.current_biases: np.ndarray = (
            self.initial_biases.copy()
            if self.initial_biases is not None
            else _normalize_biases(None)
        )

        self._blocking_strategy: Optional[BlockingStrategy] = None
        self._n_chains: int = 1
        self._clamped_nodes: List[int] = []
        self._clamped_values: List[float] = []
        self._clamp_mode: ClampMode = ClampMode.CONSTRAIN
        self.clamped_nodes: Dict[int, float] = {}

        self._extra_config: Dict[str, Any] = dict(kwargs)

        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.int32).max))
        self._rng_seed: int = seed
        self._rng_counter: int = 0

        if jax is None:
            self._rng_key = None
        else:
            self._rng_key = jax.random.PRNGKey(self._rng_seed)
    
    # ========================================================================
    # Core Sampling Interface
    # ========================================================================
    
    @abstractmethod
    def sample(
        self,
        n_steps: int,
        temperature: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """
        Generate samples from the current distribution.
        
        Args:
            n_steps: Number of sampling steps
            temperature: Sampling temperature (higher = more random)
            **kwargs: Backend-specific parameters
            
        Returns:
            (n_nodes,) array of sampled states
        """
        pass
    
    @abstractmethod
    def compute_energy(self, states: Optional[np.ndarray] = None) -> float:
        """
        Compute energy of a state.
        
        Args:
            states: (n_nodes,) state vector. If None, use current state.
            
        Returns:
            Energy value (lower = more probable)
        """
        pass
    
    @abstractmethod
    def update_weights(
        self,
        data_states: np.ndarray,
        learning_rate: float,
        **kwargs
    ) -> Dict[str, float]:
        """
        Update model weights from data (for learnable backends).
        
        Args:
            data_states: (n_nodes,) observed data
            learning_rate: Learning rate
            **kwargs: Backend-specific parameters (e.g., CD-k steps)
            
        Returns:
            Dict with learning diagnostics (gradient_norm, energy_diff, etc.)
        """
        pass
    
    # ========================================================================
    # Diagnostics & Monitoring
    # ========================================================================
    
    @abstractmethod
    def get_diagnostics(self) -> BackendDiagnostics:
        """
        Get current performance diagnostics.
        
        Returns:
            BackendDiagnostics with samples/sec, ESS/sec, autocorr, etc.
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """
        Declare backend capabilities.
        
        Returns:
            BackendCapabilities describing supported features
        """
        pass
    
    def get_optimal_strategies(self) -> List[str]:
        """
        Get list of optimal blocking strategies for this backend.
        
        Returns:
            List of strategy names (e.g., ["checkerboard", "random"])
        """
        return self.get_capabilities().optimal_strategies

    def set_current_weights(self, weights: np.ndarray) -> None:
        """Safely update the backend's working weight matrix."""
        self.current_weights = np.asarray(weights, dtype=np.float32).copy()

    def set_current_biases(self, biases: np.ndarray) -> None:
        """Safely update the backend's working bias vector."""
        self.current_biases = np.asarray(biases, dtype=np.float32).copy()

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset RNG seed used for JAX-based randomness."""
        if seed is None:
            seed = self._rng_seed
        else:
            self._rng_seed = seed
        self._rng_counter = 0
        if jax is None:
            self._rng_key = None
        else:
            self._rng_key = jax.random.PRNGKey(seed)

    def _update_rng_key(self):
        """Advance RNG state and return a fresh JAX subkey."""
        if jax is None:
            raise RuntimeError(
                "JAX is required for sampler RNG operations but is not installed."
            )
        self._rng_counter += 1
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return jax.random.fold_in(subkey, self._rng_counter)
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    def set_blocking_strategy(self, strategy: BlockingStrategy):
        """
        Set the blocking strategy for parallel sampling.
        
        Args:
            strategy: BlockingStrategy instance
        """
        self._blocking_strategy = strategy
        self._on_strategy_changed()
    
    def get_blocking_strategy(self) -> Optional[BlockingStrategy]:
        """Get current blocking strategy."""
        return self._blocking_strategy
    
    def set_num_chains(self, n_chains: int):
        """
        Set number of parallel sampling chains.
        
        Args:
            n_chains: Number of chains (-1 for auto-detect)
        """
        if n_chains == -1:
            n_chains = self._auto_detect_chains()
        
        self._n_chains = max(1, n_chains)
        self._on_chains_changed()
    
    def get_num_chains(self) -> int:
        """Get current number of chains."""
        return self._n_chains
    
    def set_conditional_nodes(
        self,
        node_ids: List[int],
        values: List[float],
        mode: ClampMode = ClampMode.CONSTRAIN
    ):
        """
        Set nodes to be clamped (fixed) during sampling.
        
        Args:
            node_ids: List of node indices to clamp
            values: List of values to clamp to
            mode: Clamping mode (inpaint, constrain, pattern)
        """
        if len(node_ids) != len(values):
            raise ValueError("node_ids and values must have same length")
        
        self._clamped_nodes = node_ids
        self._clamped_values = values
        self._clamp_mode = mode
        self._on_clamps_changed()
    
    def get_conditional_nodes(self) -> Tuple[List[int], List[float], ClampMode]:
        """Get current clamped nodes."""
        return self._clamped_nodes, self._clamped_values, self._clamp_mode
    
    def clear_conditional_nodes(self):
        """Remove all clamped nodes."""
        self._clamped_nodes = []
        self._clamped_values = []
        self._on_clamps_changed()
    
    def validate_configuration(self) -> ValidationResult:
        """
        Validate current backend configuration.
        
        Returns:
            ValidationResult indicating if configuration is valid
        """
        # Default validation (subclasses can override)
        if self._blocking_strategy is None:
            return ValidationResult(
                valid=False,
                reason="No blocking strategy set"
            )
        
        return ValidationResult(valid=True, reason="Configuration valid")
    
    # ========================================================================
    # Serialization
    # ========================================================================
    
    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize backend state for checkpointing.
        
        Returns:
            Dict with all state needed to reconstruct backend
        """
        pass
    
    @staticmethod
    @abstractmethod
    def deserialize(data: Dict[str, Any]) -> 'SamplerBackend':
        """
        Reconstruct backend from serialized state.
        
        Args:
            data: Dict from serialize()
            
        Returns:
            Reconstructed backend instance
        """
        pass
    
    # ========================================================================
    # Internal Hooks (for subclasses)
    # ========================================================================
    
    def _on_strategy_changed(self):
        """Called when blocking strategy changes. Subclasses can override."""
        pass
    
    def _on_chains_changed(self):
        """Called when number of chains changes. Subclasses can override."""
        pass
    
    def _on_clamps_changed(self):
        """Called when clamped nodes change. Subclasses can override."""
        pass
    
    def _auto_detect_chains(self) -> int:
        """
        Auto-detect optimal number of chains based on hardware.
        
        Returns:
            Recommended number of chains
        """
        # Default implementation (subclasses should override)
        import os
        return max(1, os.cpu_count() // 4)


# ============================================================================
# Backend Registry
# ============================================================================

class BackendRegistry:
    """
    Global registry for sampler backends.
    
    Manages available backends, provides discovery and instantiation.
    Supports hot-reload for development.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._backends: Dict[str, type] = {}
        self._capabilities: Dict[str, BackendCapabilities] = {}
        self._descriptions: Dict[str, str] = {}
        self._versions: Dict[str, str] = {}
        self._availability: Dict[str, bool] = {}
    
    def register(
        self,
        name: str,
        backend_class: type,
        capabilities: BackendCapabilities,
        description: str = "",
        version: str = "1.0.0",
        available: bool = True,
    ):
        """
        Register a backend.
        
        Args:
            name: Backend identifier (must be unique)
            backend_class: Backend class (must inherit from SamplerBackend)
            capabilities: Backend capabilities
            description: Human-readable description
            version: Backend version
        """
        if not issubclass(backend_class, SamplerBackend):
            raise TypeError(f"Backend class must inherit from SamplerBackend")
        
        if name in self._backends:
            print(f"Warning: Overwriting existing backend '{name}'")
        
        self._backends[name] = backend_class
        self._capabilities[name] = capabilities
        self._descriptions[name] = description
        self._versions[name] = version
        self._availability[name] = available
    
    def get(self, name: str) -> type:
        """
        Get backend class by name.
        
        Args:
            name: Backend identifier
            
        Returns:
            Backend class
            
        Raises:
            KeyError: If backend not found
        """
        if name not in self._backends:
            available = list(self._backends.keys())
            raise KeyError(f"Backend '{name}' not found. Available: {available}")
        
        return self._backends[name]
    
    def list_backends(self) -> List[str]:
        """Return sorted list of registered backend names."""
        return sorted(self._backends.keys())

    def list_available_backends(self) -> List[BackendInfo]:
        """
        List all registered backends.
        
        Returns:
            List of BackendInfo objects
        """
        infos = []
        
        for name in self._backends.keys():
            # Combine declared availability with basic import checks
            available_flag = self._availability.get(name, True)
            available = available_flag
            
            info = BackendInfo(
                name=name,
                available=available,
                capabilities=self._capabilities[name],
                description=self._descriptions.get(name, ""),
                version=self._versions.get(name, "1.0.0")
            )
            infos.append(info)
        
        return infos
    
    def get_available_backends(self) -> List[Dict[str, Any]]:
        """Return available backend metadata as JSON-friendly dicts."""
        available = []
        for info in self.list_available_backends():
            capabilities = asdict(info.capabilities)
            extra = capabilities.get("extra") or {}
            if extra:
                # Flatten extra metadata for backward compatibility while
                # keeping the structured copy under "extra".
                capabilities.update(extra)
                capabilities["extra"] = extra
            available.append(
                {
                    "name": info.name,
                    "available": info.available,
                    "capabilities": capabilities,
                    "description": info.description,
                    "version": info.version,
                }
            )
        return available
    
    def reload_backend(self, name: str):
        """
        Hot-reload a backend (for development).
        
        Args:
            name: Backend to reload
        """
        # TODO: Implement hot-reload logic
        raise NotImplementedError("Hot-reload not yet implemented")


# Global registry instance
_global_backend_registry = BackendRegistry()


class SamplerBackendRegistry:
    """Class-based accessors for the global sampler backend registry."""

    @classmethod
    def register(
        cls,
        name: str,
        backend_class: type,
        capabilities: Optional[BackendCapabilities] = None,
        description: str = "",
        version: str = "1.0.0",
        available: bool = True,
    ) -> None:
        if capabilities is None:
            capabilities = BackendCapabilities()
        _global_backend_registry.register(
            name,
            backend_class,
            capabilities,
            description,
            version,
            available,
        )

    @classmethod
    def get_backend(cls, name: str) -> Optional[type]:
        try:
            return _global_backend_registry.get(name)
        except KeyError:
            return None

    @classmethod
    def list_backends(cls) -> List[str]:
        return _global_backend_registry.list_backends()

    @classmethod
    def get_available_backends(cls) -> List[Dict[str, Any]]:
        return _global_backend_registry.get_available_backends()


def register_backend(
    name: str,
    backend_class: type,
    capabilities: Optional[BackendCapabilities] = None,
    description: str = "",
    version: str = "1.0.0",
    available: bool = True,
):
    """Register a backend in the global registry."""
    SamplerBackendRegistry.register(name, backend_class, capabilities, description, version, available)


def get_backend(name: str) -> type:
    """Get a backend class from the global registry."""
    backend = SamplerBackendRegistry.get_backend(name)
    if backend is None:
        available = SamplerBackendRegistry.list_backends()
        raise KeyError(f"Backend '{name}' not found. Available: {available}")
    return backend


def list_available_backends() -> List[BackendInfo]:
    """List all registered backends."""
    return _global_backend_registry.list_available_backends()


def get_available_backends() -> List[Dict[str, Any]]:
    """Return JSON-friendly metadata for all registered backends."""
    return _global_backend_registry.get_available_backends()


# ============================================================================
# Placeholder Backend Stubs (for future implementation)
# ============================================================================

class PhotonicBackend(SamplerBackend):
    """
    Placeholder for photonic processor backend.
    
    Future implementation will interface with optical computing hardware.
    Photonic processors excel at:
    - Extremely low power consumption
    - Massive parallelism (optical interference)
    - Natural probabilistic sampling (shot noise)
    """
    
    def __init__(self):
        super().__init__("photonic")
    
    def sample(self, n_steps: int, temperature: float = 1.0, **kwargs) -> np.ndarray:
        raise NotImplementedError("Photonic backend not yet implemented")
    
    def compute_energy(self, states: Optional[np.ndarray] = None) -> float:
        raise NotImplementedError("Photonic backend not yet implemented")
    
    def update_weights(self, data_states: np.ndarray, learning_rate: float, **kwargs) -> Dict[str, float]:
        raise NotImplementedError("Photonic backend not yet implemented")
    
    def get_diagnostics(self) -> BackendDiagnostics:
        raise NotImplementedError("Photonic backend not yet implemented")
    
    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_multi_chain=True,
            supports_conditional=True,
            supports_gpu=False,  # Photonic, not GPU
            supports_distributed=False,
            max_nodes=10000,  # Optical arrays can be large
            optimal_strategies=["checkerboard", "random"],
            hardware_type="photonic",
            extra={"experimental": True}
        )
    
    def serialize(self) -> Dict[str, Any]:
        raise NotImplementedError("Photonic backend not yet implemented")
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'PhotonicBackend':
        raise NotImplementedError("Photonic backend not yet implemented")


class NeuromorphicBackend(SamplerBackend):
    """
    Placeholder for neuromorphic chip backend.
    
    Future implementation will interface with spiking neural hardware.
    Neuromorphic chips excel at:
    - Event-driven computation
    - Temporal dynamics
    - Low power consumption
    """
    
    def __init__(self):
        super().__init__("neuromorphic")
    
    def sample(self, n_steps: int, temperature: float = 1.0, **kwargs) -> np.ndarray:
        raise NotImplementedError("Neuromorphic backend not yet implemented")
    
    def compute_energy(self, states: Optional[np.ndarray] = None) -> float:
        raise NotImplementedError("Neuromorphic backend not yet implemented")
    
    def update_weights(self, data_states: np.ndarray, learning_rate: float, **kwargs) -> Dict[str, float]:
        raise NotImplementedError("Neuromorphic backend not yet implemented")
    
    def get_diagnostics(self) -> BackendDiagnostics:
        raise NotImplementedError("Neuromorphic backend not yet implemented")
    
    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_multi_chain=True,
            supports_conditional=True,
            supports_gpu=False,  # Neuromorphic, not GPU
            supports_distributed=True,  # Can distribute across chips
            max_nodes=100000,  # Large spiking networks
            optimal_strategies=["graph-coloring", "random"],  # Arbitrary connectivity
            hardware_type="neuromorphic",
            extra={"experimental": True}
        )
    
    def serialize(self) -> Dict[str, Any]:
        raise NotImplementedError("Neuromorphic backend not yet implemented")
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'NeuromorphicBackend':
        raise NotImplementedError("Neuromorphic backend not yet implemented")


class QuantumBackend(SamplerBackend):
    """
    Placeholder for quantum annealer backend.
    
    Future implementation will interface with quantum computing hardware.
    Quantum annealers excel at:
    - Finding global optima
    - Quantum tunneling through barriers
    - Sampling from complex energy landscapes
    """
    
    def __init__(self):
        super().__init__("quantum")
    
    def sample(self, n_steps: int, temperature: float = 1.0, **kwargs) -> np.ndarray:
        raise NotImplementedError("Quantum backend not yet implemented")
    
    def compute_energy(self, states: Optional[np.ndarray] = None) -> float:
        raise NotImplementedError("Quantum backend not yet implemented")
    
    def update_weights(self, data_states: np.ndarray, learning_rate: float, **kwargs) -> Dict[str, float]:
        raise NotImplementedError("Quantum backend not yet implemented")
    
    def get_diagnostics(self) -> BackendDiagnostics:
        raise NotImplementedError("Quantum backend not yet implemented")
    
    def get_capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_multi_chain=False,  # Quantum parallelism is different
            supports_conditional=True,
            supports_gpu=False,  # Quantum, not GPU
            supports_distributed=False,
            max_nodes=5000,  # Limited qubits
            optimal_strategies=["random"],  # Blocking less critical for quantum
            hardware_type="quantum",
            extra={"experimental": True}
        )
    
    def serialize(self) -> Dict[str, Any]:
        raise NotImplementedError("Quantum backend not yet implemented")
    
    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'QuantumBackend':
        raise NotImplementedError("Quantum backend not yet implemented")


# Register placeholder backends (for discovery, will raise NotImplementedError if used)
register_backend(
    "photonic",
    PhotonicBackend,
    PhotonicBackend().get_capabilities(),
    "Optical computing backend (future)",
    "0.1.0",
    available=False
)

register_backend(
    "neuromorphic",
    NeuromorphicBackend,
    NeuromorphicBackend().get_capabilities(),
    "Spiking neural network backend (future)",
    "0.1.0",
    available=False
)

register_backend(
    "quantum",
    QuantumBackend,
    QuantumBackend().get_capabilities(),
    "Quantum annealer backend (future)",
    "0.1.0",
    available=False
)

