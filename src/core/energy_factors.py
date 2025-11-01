"""
Custom Energy Factors for THRML

Domain-specific energy functions for specialized applications:
- Photonic: Optical coupling factors (wavelength, refractive index)
- Audio: Harmonic/dissonance factors (frequencies, intervals)
- ML: Regularization factors (L1, L2, elastic net)
- Spatial: Locality factors (distance-based penalties)

These factors augment the base Ising energy to incorporate domain knowledge.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """Energy factor types."""
    PHOTONIC_COUPLING = "photonic_coupling"
    AUDIO_HARMONY = "audio_harmony"
    ML_REGULARIZATION = "ml_regularization"
    SPATIAL_LOCALITY = "spatial_locality"
    CUSTOM = "custom"


class EnergyFactor:
    """
    Base class for custom energy factors.
    
    Each factor contributes an additional term to the total energy:
    E_total = E_ising + Σ λᵢ F_i(states)
    
    where λᵢ is the factor strength and F_i is the factor function.
    """
    
    def __init__(
        self,
        factor_id: str,
        factor_type: FactorType,
        node_ids: List[int],
        strength: float,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize energy factor.
        
        Args:
            factor_id: Unique identifier
            factor_type: Type of factor
            node_ids: Nodes this factor applies to
            strength: Factor strength (λ)
            properties: Factor-specific properties
        """
        self.factor_id = factor_id
        self.factor_type = factor_type
        self.node_ids = node_ids
        self.strength = strength
        self.properties = properties or {}
    
    def compute(self, states: np.ndarray) -> float:
        """
        Compute factor energy contribution.
        
        Args:
            states: Full state array (all nodes)
            
        Returns:
            Energy contribution
        """
        raise NotImplementedError("Subclasses must implement compute()")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'factor_id': self.factor_id,
            'factor_type': self.factor_type.value,
            'node_ids': self.node_ids,
            'strength': self.strength,
            'properties': self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnergyFactor':
        """Deserialize from dictionary."""
        factor_type = FactorType(data['factor_type'])
        
        # Route to appropriate subclass
        if factor_type == FactorType.PHOTONIC_COUPLING:
            return PhotonicCouplingFactor.from_dict(data)
        elif factor_type == FactorType.AUDIO_HARMONY:
            return AudioHarmonyFactor.from_dict(data)
        elif factor_type == FactorType.ML_REGULARIZATION:
            return MLRegularizationFactor.from_dict(data)
        elif factor_type == FactorType.SPATIAL_LOCALITY:
            return SpatialLocalityFactor.from_dict(data)
        else:
            return CustomFactor.from_dict(data)


class PhotonicCouplingFactor(EnergyFactor):
    """
    Photonic coupling factor for optical neural networks.
    
    Models wavelength-dependent coupling and phase matching in photonic systems.
    
    Energy contribution:
    E_photo = -λ Σᵢⱼ cos(2π(nᵢ - nⱼ)/λ_wave) sᵢ sⱼ
    
    where n is refractive index and λ_wave is wavelength.
    """
    
    def __init__(
        self,
        factor_id: str,
        node_ids: List[int],
        strength: float,
        wavelength: float = 1550e-9,  # 1550 nm (telecom wavelength)
        refractive_index: Optional[List[float]] = None
    ):
        """
        Initialize photonic coupling factor.
        
        Args:
            factor_id: Unique ID
            node_ids: Nodes representing photonic elements
            strength: Coupling strength
            wavelength: Optical wavelength (meters)
            refractive_index: Per-node refractive index (default: 1.5 for all)
        """
        properties = {
            'wavelength': wavelength,
            'refractive_index': refractive_index or [1.5] * len(node_ids)
        }
        
        super().__init__(
            factor_id=factor_id,
            factor_type=FactorType.PHOTONIC_COUPLING,
            node_ids=node_ids,
            strength=strength,
            properties=properties
        )
    
    def compute(self, states: np.ndarray) -> float:
        """
        Compute photonic coupling energy.
        
        Phase-matched nodes have lower energy (favor coupling).
        """
        node_states = states[self.node_ids]
        n_indices = np.array(self.properties['refractive_index'])
        wavelength = self.properties['wavelength']
        
        energy = 0.0
        n = len(self.node_ids)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Phase matching term
                phase_diff = 2 * np.pi * (n_indices[i] - n_indices[j]) / wavelength
                coupling = np.cos(phase_diff)
                
                # Interaction weighted by phase matching
                energy -= coupling * node_states[i] * node_states[j]
        
        return self.strength * energy
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhotonicCouplingFactor':
        """Deserialize PhotonicCouplingFactor from dictionary."""
        return cls(
            factor_id=data['factor_id'],
            node_ids=data['node_ids'],
            strength=data['strength'],
            wavelength=data['properties']['wavelength']
        )


class AudioHarmonyFactor(EnergyFactor):
    """
    Audio harmony factor for sound synthesis.
    
    Favors harmonically related frequency states (consonance).
    Uses Helmholtz consonance theory.
    
    Energy contribution:
    E_audio = λ Σᵢⱼ D(fᵢ, fⱼ) sᵢ sⱼ
    
    where D is dissonance function (high for dissonant intervals).
    """
    
    def __init__(
        self,
        factor_id: str,
        node_ids: List[int],
        strength: float,
        fundamental_freq: float = 440.0,  # A4
        frequencies: Optional[List[float]] = None
    ):
        """
        Initialize audio harmony factor.
        
        Args:
            factor_id: Unique ID
            node_ids: Nodes representing audio oscillators
            strength: Harmony strength
            fundamental_freq: Fundamental frequency (Hz)
            frequencies: Per-node frequencies (default: harmonic series)
        """
        if frequencies is None:
            # Use harmonic series by default
            frequencies = [fundamental_freq * (i + 1) for i in range(len(node_ids))]
        
        properties = {
            'fundamental_freq': fundamental_freq,
            'frequencies': frequencies
        }
        
        super().__init__(
            factor_id=factor_id,
            factor_type=FactorType.AUDIO_HARMONY,
            node_ids=node_ids,
            strength=strength,
            properties=properties
        )
    
    def _compute_dissonance(self, freq1: float, freq2: float) -> float:
        """
        Compute dissonance between two frequencies using Helmholtz model.
        
        Returns value in [0, 1] where 0 is consonant, 1 is dissonant.
        """
        ratio = max(freq1, freq2) / min(freq1, freq2)
        
        # Simple intervals (octave, fifth, fourth, etc.) have low dissonance
        # Check proximity to simple ratios
        simple_ratios = [
            (2.0, 0.0),    # Octave (most consonant)
            (3/2, 0.1),    # Perfect fifth
            (4/3, 0.15),   # Perfect fourth
            (5/4, 0.2),    # Major third
            (6/5, 0.25),   # Minor third
            (8/5, 0.3),    # Minor sixth
            (5/3, 0.3),    # Major sixth
        ]
        
        min_dissonance = 1.0
        for target_ratio, base_diss in simple_ratios:
            # Distance from this simple ratio
            dist = abs(ratio - target_ratio)
            dissonance = base_diss + dist * 0.5
            min_dissonance = min(min_dissonance, dissonance)
        
        return min_dissonance
    
    def compute(self, states: np.ndarray) -> float:
        """
        Compute harmony energy.
        
        Consonant (harmonic) states have lower energy.
        """
        node_states = states[self.node_ids]
        frequencies = np.array(self.properties['frequencies'])
        
        energy = 0.0
        n = len(self.node_ids)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Only penalize if both nodes active
                if node_states[i] > 0 and node_states[j] > 0:
                    dissonance = self._compute_dissonance(frequencies[i], frequencies[j])
                    energy += dissonance * node_states[i] * node_states[j]
        
        return self.strength * energy
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioHarmonyFactor':
        """Deserialize AudioHarmonyFactor from dictionary."""
        return cls(
            factor_id=data['factor_id'],
            node_ids=data['node_ids'],
            strength=data['strength'],
            fundamental_freq=data['properties']['fundamental_freq'],
            frequencies=data['properties']['frequencies']
        )


class MLRegularizationFactor(EnergyFactor):
    """
    ML regularization factor for learned weights.
    
    Applies standard ML regularization (L1, L2, elastic net) to prevent overfitting.
    
    Energy contribution:
    - L1: E_reg = λ Σᵢ |sᵢ|
    - L2: E_reg = λ Σᵢ sᵢ²
    - Elastic Net: E_reg = λ₁ Σᵢ |sᵢ| + λ₂ Σᵢ sᵢ²
    """
    
    def __init__(
        self,
        factor_id: str,
        node_ids: List[int],
        strength: float,
        regularization_type: str = 'l2',  # 'l1', 'l2', 'elastic'
        l1_ratio: float = 0.5  # For elastic net
    ):
        """
        Initialize ML regularization factor.
        
        Args:
            factor_id: Unique ID
            node_ids: Nodes to regularize
            strength: Regularization strength
            regularization_type: Type ('l1', 'l2', 'elastic')
            l1_ratio: Ratio of L1 in elastic net (0=L2, 1=L1)
        """
        properties = {
            'regularization_type': regularization_type,
            'l1_ratio': l1_ratio
        }
        
        super().__init__(
            factor_id=factor_id,
            factor_type=FactorType.ML_REGULARIZATION,
            node_ids=node_ids,
            strength=strength,
            properties=properties
        )
    
    def compute(self, states: np.ndarray) -> float:
        """Compute regularization energy."""
        node_states = states[self.node_ids]
        reg_type = self.properties['regularization_type']
        
        if reg_type == 'l1':
            energy = np.sum(np.abs(node_states))
        elif reg_type == 'l2':
            energy = np.sum(node_states ** 2)
        elif reg_type == 'elastic':
            l1_ratio = self.properties['l1_ratio']
            l1_energy = np.sum(np.abs(node_states))
            l2_energy = np.sum(node_states ** 2)
            energy = l1_ratio * l1_energy + (1 - l1_ratio) * l2_energy
        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")
        
        return self.strength * energy
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLRegularizationFactor':
        """Deserialize MLRegularizationFactor from dictionary."""
        return cls(
            factor_id=data['factor_id'],
            node_ids=data['node_ids'],
            strength=data['strength'],
            regularization_type=data['properties']['regularization_type'],
            l1_ratio=data['properties']['l1_ratio']
        )


class SpatialLocalityFactor(EnergyFactor):
    """
    Spatial locality factor for 2D/3D systems.
    
    Penalizes long-range interactions, favoring local coupling.
    
    Energy contribution:
    E_spatial = λ Σᵢⱼ d(i,j) sᵢ sⱼ
    
    where d(i,j) is distance between nodes.
    """
    
    def __init__(
        self,
        factor_id: str,
        node_ids: List[int],
        strength: float,
        node_positions: np.ndarray,
        decay_length: float = 1.0
    ):
        """
        Initialize spatial locality factor.
        
        Args:
            factor_id: Unique ID
            node_ids: All nodes with positions
            strength: Locality strength
            node_positions: (n_nodes, 2 or 3) position array
            decay_length: Characteristic length scale
        """
        properties = {
            'node_positions': node_positions.tolist(),
            'decay_length': decay_length
        }
        
        super().__init__(
            factor_id=factor_id,
            factor_type=FactorType.SPATIAL_LOCALITY,
            node_ids=node_ids,
            strength=strength,
            properties=properties
        )
    
    def compute(self, states: np.ndarray) -> float:
        """Compute spatial locality energy."""
        node_states = states[self.node_ids]
        positions = np.array(self.properties['node_positions'])
        decay_length = self.properties['decay_length']
        
        energy = 0.0
        n = len(self.node_ids)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Distance
                dist = np.linalg.norm(positions[i] - positions[j])
                
                # Distance-weighted interaction
                # Favor nearby interactions (low dist → low energy)
                energy += (dist / decay_length) * abs(node_states[i] * node_states[j])
        
        return self.strength * energy


class CustomFactor(EnergyFactor):
    """
    Custom energy factor with user-defined function.
    
    Allows arbitrary energy functions for specialized applications.
    """
    
    def __init__(
        self,
        factor_id: str,
        node_ids: List[int],
        strength: float,
        energy_fn: Callable[[np.ndarray], float],
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize custom factor.
        
        Args:
            factor_id: Unique ID
            node_ids: Nodes this factor applies to
            strength: Factor strength
            energy_fn: Function mapping node states to energy
            properties: Optional metadata
        """
        super().__init__(
            factor_id=factor_id,
            factor_type=FactorType.CUSTOM,
            node_ids=node_ids,
            strength=strength,
            properties=properties
        )
        
        self.energy_fn = energy_fn
    
    def compute(self, states: np.ndarray) -> float:
        """Compute custom energy."""
        node_states = states[self.node_ids]
        return self.strength * self.energy_fn(node_states)


class EnergyFactorSystem:
    """
    Manages collection of energy factors.
    
    Computes total energy as:
    E_total = E_base + Σᵢ λᵢ Fᵢ(states)
    """
    
    def __init__(self):
        self.factors: Dict[str, EnergyFactor] = {}
    
    def add_factor(self, factor: EnergyFactor):
        """Add an energy factor."""
        self.factors[factor.factor_id] = factor
        logger.info(f"[FactorSystem] Added {factor.factor_type.value} factor '{factor.factor_id}'")
    
    def remove_factor(self, factor_id: str):
        """Remove an energy factor."""
        if factor_id in self.factors:
            del self.factors[factor_id]
            logger.info(f"[FactorSystem] Removed factor '{factor_id}'")
    
    def compute_total_energy(
        self,
        states: np.ndarray,
        base_energy: float = 0.0
    ) -> float:
        """
        Compute total energy including all factors.
        
        Args:
            states: Full state array
            base_energy: Base Ising energy
            
        Returns:
            Total energy
        """
        total = base_energy
        
        for factor in self.factors.values():
            try:
                contribution = factor.compute(states)
                total += contribution
            except Exception as e:
                logger.warning(f"[FactorSystem] Failed to compute {factor.factor_id}: {e}")
        
        return total
    
    def compute_factor_contributions(
        self,
        states: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute individual factor contributions.
        
        Args:
            states: Full state array
            
        Returns:
            Dict mapping factor_id -> energy contribution
        """
        contributions = {}
        
        for factor_id, factor in self.factors.items():
            try:
                contributions[factor_id] = factor.compute(states)
            except Exception as e:
                logger.warning(f"[FactorSystem] Failed to compute {factor_id}: {e}")
                contributions[factor_id] = 0.0
        
        return contributions
    
    def get_info(self) -> Dict[str, Any]:
        """Get factor system information."""
        return {
            'n_factors': len(self.factors),
            'factors': {
                fid: {
                    'type': f.factor_type.value,
                    'strength': f.strength,
                    'n_nodes': len(f.node_ids)
                }
                for fid, f in self.factors.items()
            }
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize factor system."""
        return {
            'factors': [f.to_dict() for f in self.factors.values()]
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'EnergyFactorSystem':
        """Deserialize factor system."""
        system = cls()
        
        for factor_data in data['factors']:
            factor = EnergyFactor.from_dict(factor_data)
            system.add_factor(factor)
        
        return system


# ============================================================================
# Convenience Functions
# ============================================================================

def add_photonic_coupling(
    system: EnergyFactorSystem,
    node_ids: List[int],
    strength: float = 0.5,
    wavelength: float = 1550e-9
):
    """Add photonic coupling factor to system."""
    factor = PhotonicCouplingFactor(
        factor_id=f"photonic_{len(system.factors)}",
        node_ids=node_ids,
        strength=strength,
        wavelength=wavelength
    )
    system.add_factor(factor)


def add_audio_harmony(
    system: EnergyFactorSystem,
    node_ids: List[int],
    strength: float = 0.3,
    fundamental_freq: float = 440.0
):
    """Add audio harmony factor to system."""
    factor = AudioHarmonyFactor(
        factor_id=f"audio_{len(system.factors)}",
        node_ids=node_ids,
        strength=strength,
        fundamental_freq=fundamental_freq
    )
    system.add_factor(factor)


def add_ml_regularization(
    system: EnergyFactorSystem,
    node_ids: List[int],
    strength: float = 0.01,
    regularization_type: str = 'l2'
):
    """Add ML regularization factor to system."""
    factor = MLRegularizationFactor(
        factor_id=f"ml_reg_{len(system.factors)}",
        node_ids=node_ids,
        strength=strength,
        regularization_type=regularization_type
    )
    system.add_factor(factor)


def add_spatial_locality(
    system: EnergyFactorSystem,
    node_ids: List[int],
    node_positions: np.ndarray,
    strength: float = 0.1,
    decay_length: float = 1.0
):
    """Add spatial locality factor to system."""
    factor = SpatialLocalityFactor(
        factor_id=f"spatial_{len(system.factors)}",
        node_ids=node_ids,
        strength=strength,
        node_positions=node_positions,
        decay_length=decay_length
    )
    system.add_factor(factor)


if __name__ == '__main__':
    # Demo
    print("Energy Factors Demo\n")
    
    # Create factor system
    system = EnergyFactorSystem()
    
    # Add various factors
    print("Adding factors...")
    add_photonic_coupling(system, node_ids=[0, 1, 2, 3], strength=0.5)
    add_audio_harmony(system, node_ids=[4, 5, 6, 7], strength=0.3)
    add_ml_regularization(system, node_ids=[8, 9, 10, 11], strength=0.01, regularization_type='l2')
    
    positions = np.random.rand(4, 2)
    add_spatial_locality(system, node_ids=[12, 13, 14, 15], node_positions=positions, strength=0.1)
    
    print(f"\nSystem info:\n{system.get_info()}")
    
    # Compute energies
    states = np.random.choice([-1, 1], size=16).astype(float)
    print(f"\nRandom states: {states}")
    
    base_energy = -np.sum(states)  # Simple energy
    total_energy = system.compute_total_energy(states, base_energy)
    
    print(f"\nBase energy: {base_energy:.3f}")
    print(f"Total energy: {total_energy:.3f}")
    
    contributions = system.compute_factor_contributions(states)
    print(f"\nFactor contributions:")
    for fid, contrib in contributions.items():
        print(f"  {fid}: {contrib:.4f}")
    
    print("\nDemo complete!")

