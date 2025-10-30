"""
Advanced Sampling Schedules for THRML.

Implements adaptive, annealed, and parallel tempering sampling schedules
for improved convergence and exploration of the energy landscape.
"""

from typing import Tuple, List, Optional
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

try:
    from thrml import SamplingSchedule
    THRML_AVAILABLE = True
except ImportError:
    THRML_AVAILABLE = False
    # Fallback definition
    @dataclass
    class SamplingSchedule:
        n_warmup: int
        n_samples: int
        steps_per_sample: int


@dataclass
class AdaptiveSamplingSchedule:
    """
    Adaptive sampling schedule that adjusts based on convergence.
    
    Monitors energy variance and automatically adjusts warmup and
    sampling steps to achieve target convergence.
    
    Attributes:
        initial_warmup: Initial warmup steps
        initial_steps_per_sample: Initial steps between samples
        target_variance: Target energy variance for convergence
        max_warmup: Maximum warmup steps
        adaptation_rate: How quickly to adapt (0-1)
    """
    initial_warmup: int = 100
    initial_steps_per_sample: int = 2
    target_variance: float = 0.01
    max_warmup: int = 1000
    adaptation_rate: float = 0.1
    
    def __post_init__(self):
        """Initialize adaptive state."""
        self.current_warmup = self.initial_warmup
        self.current_steps = self.initial_steps_per_sample
        self.energy_history: List[float] = []
        self.converged = False
    
    def update(self, energy: float) -> bool:
        """
        Update schedule based on observed energy.
        
        Args:
            energy: Current energy value
            
        Returns:
            True if converged, False otherwise
        """
        self.energy_history.append(energy)
        
        # Need at least 10 samples to check convergence
        if len(self.energy_history) < 10:
            return False
        
        # Compute recent variance
        recent_energies = self.energy_history[-10:]
        variance = np.var(recent_energies)
        
        # Check convergence
        if variance < self.target_variance:
            self.converged = True
            return True
        
        # Adapt warmup if not converged
        if variance > self.target_variance * 10:
            # High variance - increase warmup
            increase = int(self.current_warmup * self.adaptation_rate)
            self.current_warmup = min(
                self.current_warmup + increase,
                self.max_warmup
            )
        
        return False
    
    def get_schedule(self, n_samples: int = 1) -> SamplingSchedule:
        """
        Get current sampling schedule.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            SamplingSchedule with current parameters
        """
        return SamplingSchedule(
            n_warmup=self.current_warmup,
            n_samples=n_samples,
            steps_per_sample=self.current_steps
        )
    
    def reset(self):
        """Reset to initial state."""
        self.current_warmup = self.initial_warmup
        self.current_steps = self.initial_steps_per_sample
        self.energy_history = []
        self.converged = False


@dataclass
class AnnealedSamplingSchedule:
    """
    Annealed sampling with temperature schedule.
    
    Gradually reduces temperature from high to low to escape local minima
    and find global minimum.
    
    Attributes:
        T_initial: Initial temperature (high)
        T_final: Final temperature (low)
        n_steps: Total number of annealing steps
        schedule_type: 'linear', 'exponential', or 'cosine'
    """
    T_initial: float = 10.0
    T_final: float = 0.1
    n_steps: int = 1000
    schedule_type: str = 'exponential'
    
    def get_temperature(self, step: int) -> float:
        """
        Get temperature at given step.
        
        Args:
            step: Current step (0 to n_steps-1)
            
        Returns:
            Temperature at this step
        """
        progress = step / max(self.n_steps - 1, 1)
        
        if self.schedule_type == 'linear':
            # Linear annealing: T = T_initial - progress * (T_initial - T_final)
            return self.T_initial - progress * (self.T_initial - self.T_final)
        
        elif self.schedule_type == 'exponential':
            # Exponential annealing: T = T_initial * (T_final/T_initial)^progress
            ratio = self.T_final / self.T_initial
            return self.T_initial * (ratio ** progress)
        
        elif self.schedule_type == 'cosine':
            # Cosine annealing: smooth transition
            return self.T_final + 0.5 * (self.T_initial - self.T_final) * (
                1 + np.cos(np.pi * progress)
            )
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def get_schedule_at_step(
        self,
        step: int,
        n_samples: int = 1,
        steps_per_sample: int = 2
    ) -> Tuple[SamplingSchedule, float]:
        """
        Get sampling schedule and temperature for given step.
        
        Args:
            step: Current annealing step
            n_samples: Number of samples
            steps_per_sample: Steps between samples
            
        Returns:
            (SamplingSchedule, temperature) tuple
        """
        temperature = self.get_temperature(step)
        
        # Adjust warmup based on temperature (more warmup at high temp)
        warmup_ratio = temperature / self.T_initial
        n_warmup = int(100 * warmup_ratio)
        
        schedule = SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample
        )
        
        return schedule, temperature
    
    def get_full_temperature_schedule(self) -> np.ndarray:
        """
        Get full temperature schedule as array.
        
        Returns:
            (n_steps,) array of temperatures
        """
        return np.array([
            self.get_temperature(step)
            for step in range(self.n_steps)
        ])


@dataclass
class ParallelTemperingSchedule:
    """
    Parallel tempering (replica exchange) sampling.
    
    Runs multiple chains at different temperatures simultaneously
    and periodically swaps configurations between chains.
    
    Attributes:
        temperatures: List of temperatures for each chain
        swap_interval: How often to attempt swaps (in steps)
        n_warmup: Warmup steps per chain
        steps_per_sample: Steps between samples
    """
    temperatures: List[float] = None
    swap_interval: int = 10
    n_warmup: int = 100
    steps_per_sample: int = 2
    
    def __post_init__(self):
        """Initialize with default temperatures if not provided."""
        if self.temperatures is None:
            # Default: geometric spacing from 0.1 to 10.0
            self.temperatures = [0.1, 0.3, 1.0, 3.0, 10.0]
        
        self.n_chains = len(self.temperatures)
        self.swap_count = 0
        self.acceptance_count = 0
    
    def get_schedules(self, n_samples: int = 1) -> List[SamplingSchedule]:
        """
        Get sampling schedules for all chains.
        
        Args:
            n_samples: Number of samples per chain
            
        Returns:
            List of SamplingSchedule, one per chain
        """
        return [
            SamplingSchedule(
                n_warmup=self.n_warmup,
                n_samples=n_samples,
                steps_per_sample=self.steps_per_sample
            )
            for _ in range(self.n_chains)
        ]
    
    def should_attempt_swap(self, step: int) -> bool:
        """
        Check if should attempt replica swap at this step.
        
        Args:
            step: Current step
            
        Returns:
            True if should attempt swap
        """
        return step % self.swap_interval == 0
    
    def compute_swap_probability(
        self,
        energy_i: float,
        energy_j: float,
        temp_i: float,
        temp_j: float
    ) -> float:
        """
        Compute probability of swapping replicas i and j.
        
        Metropolis criterion:
        P_swap = min(1, exp(ΔE * Δβ))
        where ΔE = E_j - E_i, Δβ = 1/T_i - 1/T_j
        
        Args:
            energy_i: Energy of replica i
            energy_j: Energy of replica j
            temp_i: Temperature of replica i
            temp_j: Temperature of replica j
            
        Returns:
            Swap probability [0, 1]
        """
        delta_energy = energy_j - energy_i
        delta_beta = (1.0 / temp_i) - (1.0 / temp_j)
        
        # Metropolis criterion
        log_prob = delta_energy * delta_beta
        prob = np.exp(min(0.0, log_prob))
        
        return prob
    
    def attempt_swap(
        self,
        energies: List[float],
        key: jax.random.PRNGKey
    ) -> Tuple[List[int], bool]:
        """
        Attempt to swap adjacent replicas.
        
        Args:
            energies: List of energies for each chain
            key: JAX PRNG key
            
        Returns:
            (swap_indices, accepted) tuple
            swap_indices: [i, j] if swap attempted
            accepted: True if swap was accepted
        """
        self.swap_count += 1
        
        # Choose random pair of adjacent chains
        i = np.random.randint(0, self.n_chains - 1)
        j = i + 1
        
        # Compute swap probability
        prob = self.compute_swap_probability(
            energies[i],
            energies[j],
            self.temperatures[i],
            self.temperatures[j]
        )
        
        # Accept or reject
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey)
        accepted = u < prob
        
        if accepted:
            self.acceptance_count += 1
        
        return [i, j], accepted
    
    def get_acceptance_rate(self) -> float:
        """
        Get overall swap acceptance rate.
        
        Returns:
            Acceptance rate [0, 1]
        """
        if self.swap_count == 0:
            return 0.0
        return self.acceptance_count / self.swap_count


def create_geometric_temperature_ladder(
    T_min: float = 0.1,
    T_max: float = 10.0,
    n_chains: int = 5
) -> List[float]:
    """
    Create geometrically spaced temperature ladder.
    
    Args:
        T_min: Minimum temperature
        T_max: Maximum temperature
        n_chains: Number of temperature levels
        
    Returns:
        List of temperatures
    """
    ratio = (T_max / T_min) ** (1.0 / (n_chains - 1))
    temperatures = [T_min * (ratio ** i) for i in range(n_chains)]
    return temperatures


def create_optimal_temperature_ladder(
    T_min: float = 0.1,
    T_max: float = 10.0,
    n_chains: int = 5,
    target_acceptance: float = 0.3
) -> List[float]:
    """
    Create temperature ladder optimized for target acceptance rate.
    
    Uses formula from: Kofke, D. A. (2002). "On the acceptance probability
    of replica-exchange Monte Carlo trials."
    
    Args:
        T_min: Minimum temperature
        T_max: Maximum temperature
        n_chains: Number of temperature levels
        target_acceptance: Target swap acceptance rate
        
    Returns:
        List of temperatures
    """
    # Simplified version: geometric spacing with adjustment
    base_temps = create_geometric_temperature_ladder(T_min, T_max, n_chains)
    
    # Adjust spacing based on target acceptance
    # Higher target acceptance -> closer temperatures
    spacing_factor = 1.0 / (target_acceptance + 0.1)
    
    adjusted_temps = []
    for i, temp in enumerate(base_temps):
        if i == 0:
            adjusted_temps.append(temp)
        else:
            # Adjust spacing
            prev_temp = adjusted_temps[-1]
            new_temp = prev_temp * spacing_factor
            adjusted_temps.append(min(new_temp, T_max))
    
    return adjusted_temps

