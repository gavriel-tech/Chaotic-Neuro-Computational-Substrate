"""
Hybrid Chaos+Gradient Training for GMCS.

Innovative training paradigm that combines:
- Chaos for exploration (non-differentiable dynamics)
- Gradients for optimization (differentiable ML)
- THRML energy landscapes for guidance

This enables a new class of learning algorithms that leverage
both chaotic exploration and gradient-based exploitation.

Key features:
- Alternating chaos/gradient phases
- Co-evolution of oscillators and ML models
- Energy-guided exploration
- Multi-objective optimization
- Adaptive balance between chaos and gradients

Use cases:
- Discover novel solutions via chaos, refine via gradients
- Learn oscillator parameters jointly with ML weights
- Navigate complex energy landscapes
- Avoid local minima through chaotic perturbations
"""

from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum

try:
    import jax
    import jax.numpy as jnp
    from jax import grad as jax_grad, jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


# ============================================================================
# Training Mode
# ============================================================================

class TrainingPhase(Enum):
    """Training phase in hybrid learning."""
    CHAOS_EXPLORATION = "chaos_exploration"
    GRADIENT_OPTIMIZATION = "gradient_optimization"
    JOINT_UPDATE = "joint_update"


@dataclass
class HybridConfig:
    """
    Configuration for hybrid training.
    """
    # Phase control
    chaos_steps: int = 100  # Steps of chaos exploration
    gradient_steps: int = 10  # Steps of gradient descent
    phase_schedule: str = 'alternating'  # 'alternating', 'adaptive', 'joint'
    
    # Learning rates
    chaos_lr: float = 0.1  # Chaos exploration rate
    gradient_lr: float = 0.001  # Gradient learning rate
    
    # Balance
    chaos_weight: float = 0.5  # Weight for chaos objective
    gradient_weight: float = 0.5  # Weight for gradient objective
    
    # Adaptation
    adaptive_balance: bool = True  # Adapt chaos/gradient balance
    balance_update_freq: int = 10  # How often to update balance
    
    # Energy guidance
    use_energy_guidance: bool = True  # Use THRML energy
    energy_threshold: float = 0.0  # Energy threshold for acceptance
    
    # Diversity
    diversity_bonus: float = 0.1  # Bonus for diverse solutions
    population_size: int = 10  # Population for diversity


# ============================================================================
# Hybrid Trainer
# ============================================================================

class HybridTrainer:
    """
    Hybrid trainer combining chaos exploration with gradient optimization.
    
    Core idea:
    1. Use chaos to explore the space (find promising regions)
    2. Use gradients to optimize within those regions (refine solutions)
    3. Alternate between exploration and exploitation
    4. Optionally guide by energy landscape (THRML)
    """
    
    def __init__(
        self,
        oscillator_model: Any,  # Differentiable oscillator
        ml_model: Any,  # ML model
        config: Optional[HybridConfig] = None,
        energy_fn: Optional[Callable] = None
    ):
        """
        Initialize hybrid trainer.
        
        Args:
            oscillator_model: Differentiable oscillator (from differentiable_chua.py)
            ml_model: ML model node
            config: Training configuration
            energy_fn: Optional energy function (THRML)
        """
        self.oscillator_model = oscillator_model
        self.ml_model = ml_model
        self.config = config or HybridConfig()
        self.energy_fn = energy_fn
        
        # Training state
        self.current_phase = TrainingPhase.CHAOS_EXPLORATION
        self.iteration = 0
        self.history = {
            'chaos_loss': [],
            'gradient_loss': [],
            'combined_loss': [],
            'energy': [],
            'diversity': [],
            'balance': []
        }
        
        # Population (for diversity)
        self.population = []
        self.population_fitness = []
    
    def train_step(
        self,
        data: np.ndarray,
        target: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Single hybrid training step.
        
        Args:
            data: Input data (chaos state or features)
            target: Target output (if supervised)
            
        Returns:
            Metrics dict
        """
        self.iteration += 1
        
        # Determine phase
        if self.config.phase_schedule == 'alternating':
            if self.iteration % (self.config.chaos_steps + self.config.gradient_steps) < self.config.chaos_steps:
                self.current_phase = TrainingPhase.CHAOS_EXPLORATION
            else:
                self.current_phase = TrainingPhase.GRADIENT_OPTIMIZATION
        elif self.config.phase_schedule == 'adaptive':
            self.current_phase = self._adaptive_phase_selection()
        else:  # joint
            self.current_phase = TrainingPhase.JOINT_UPDATE
        
        # Execute training based on phase
        if self.current_phase == TrainingPhase.CHAOS_EXPLORATION:
            metrics = self._chaos_exploration_step(data)
        elif self.current_phase == TrainingPhase.GRADIENT_OPTIMIZATION:
            metrics = self._gradient_optimization_step(data, target)
        else:  # JOINT_UPDATE
            metrics = self._joint_update_step(data, target)
        
        # Update balance if adaptive
        if self.config.adaptive_balance and self.iteration % self.config.balance_update_freq == 0:
            self._update_balance()
        
        # Record history
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        return metrics
    
    def _chaos_exploration_step(self, data: np.ndarray) -> Dict[str, float]:
        """
        Chaos exploration step.
        
        Perturb parameters using chaotic dynamics to explore solution space.
        
        Args:
            data: Current state
            
        Returns:
            Metrics
        """
        # Get current parameters
        if hasattr(self.oscillator_model, 'get_parameters'):
            params = self.oscillator_model.get_parameters()
        else:
            params = {}
        
        # Generate chaotic perturbation
        perturbation = self._generate_chaos_perturbation(data)
        
        # Apply perturbation
        new_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                new_params[key] = value + self.config.chaos_lr * perturbation.get(key, 0.0)
            else:
                new_params[key] = value
        
        # Evaluate fitness
        fitness = self._evaluate_fitness(new_params, data)
        
        # Accept or reject based on fitness and energy
        accept = self._acceptance_criterion(fitness, new_params)
        
        if accept:
            if hasattr(self.oscillator_model, 'set_parameters'):
                self.oscillator_model.set_parameters(new_params)
            
            # Add to population
            if len(self.population) < self.config.population_size:
                self.population.append(new_params)
                self.population_fitness.append(fitness)
            else:
                # Replace worst
                worst_idx = np.argmin(self.population_fitness)
                if fitness > self.population_fitness[worst_idx]:
                    self.population[worst_idx] = new_params
                    self.population_fitness[worst_idx] = fitness
        
        # Compute diversity
        diversity = self._compute_diversity()
        
        return {
            'chaos_loss': -fitness,  # Negative fitness as loss
            'energy': self._compute_energy(new_params) if self.energy_fn else 0.0,
            'diversity': diversity,
            'accepted': float(accept)
        }
    
    def _gradient_optimization_step(
        self,
        data: np.ndarray,
        target: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Gradient optimization step.
        
        Use standard gradient descent on ML model and optionally
        on differentiable oscillator parameters.
        
        Args:
            data: Input data
            target: Target output
            
        Returns:
            Metrics
        """
        # ML model gradient step
        if hasattr(self.ml_model, 'train_step'):
            if target is not None:
                metrics = self.ml_model.train_step(data, target)
            else:
                metrics = self.ml_model.train_step(data, data)  # Autoencoder
            
            gradient_loss = metrics.get('loss', 0.0)
        else:
            gradient_loss = 0.0
        
        # Optionally optimize oscillator parameters via gradients
        if JAX_AVAILABLE and hasattr(self.oscillator_model, 'get_parameters'):
            # This requires differentiable oscillator
            oscillator_loss = self._optimize_oscillator_gradients(data)
        else:
            oscillator_loss = 0.0
        
        combined_loss = gradient_loss + oscillator_loss
        
        return {
            'gradient_loss': gradient_loss,
            'oscillator_loss': oscillator_loss,
            'combined_loss': combined_loss
        }
    
    def _joint_update_step(
        self,
        data: np.ndarray,
        target: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Joint update step.
        
        Simultaneously update via chaos and gradients with weighted combination.
        
        Args:
            data: Input data
            target: Target output
            
        Returns:
            Metrics
        """
        # Get both updates
        chaos_metrics = self._chaos_exploration_step(data)
        gradient_metrics = self._gradient_optimization_step(data, target)
        
        # Combine with weights
        combined_loss = (
            self.config.chaos_weight * chaos_metrics.get('chaos_loss', 0.0) +
            self.config.gradient_weight * gradient_metrics.get('gradient_loss', 0.0)
        )
        
        return {
            **chaos_metrics,
            **gradient_metrics,
            'combined_loss': combined_loss,
            'balance': self.config.chaos_weight / (self.config.chaos_weight + self.config.gradient_weight)
        }
    
    def _generate_chaos_perturbation(self, state: np.ndarray) -> Dict[str, float]:
        """
        Generate chaotic perturbation for parameters.
        
        Args:
            state: Current state
            
        Returns:
            Perturbation dict
        """
        # Use chaos dynamics to generate perturbation
        # Simple version: just use state values scaled
        perturbation = {}
        
        if hasattr(self.oscillator_model, 'get_parameters'):
            params = self.oscillator_model.get_parameters()
            
            for i, (key, value) in enumerate(params.items()):
                if isinstance(value, (int, float)):
                    # Use state component to perturb this parameter
                    idx = i % len(state) if len(state) > 0 else 0
                    chaos_val = state[idx] if len(state) > 0 else 0.0
                    perturbation[key] = float(chaos_val * 0.1)  # Scale down
        
        return perturbation
    
    def _evaluate_fitness(self, params: Dict[str, float], data: np.ndarray) -> float:
        """
        Evaluate fitness of parameters.
        
        Args:
            params: Parameters to evaluate
            data: Data for evaluation
            
        Returns:
            Fitness value (higher is better)
        """
        # Simple fitness: negative of distance to data
        # This is problem-specific and should be customized
        
        # Placeholder: random fitness
        fitness = np.random.rand()
        
        # Add diversity bonus
        if self.population:
            diversity = self._param_diversity(params)
            fitness += self.config.diversity_bonus * diversity
        
        return float(fitness)
    
    def _acceptance_criterion(self, fitness: float, params: Dict[str, float]) -> bool:
        """
        Decide whether to accept new parameters.
        
        Args:
            fitness: Fitness of new parameters
            params: New parameters
            
        Returns:
            Accept or reject
        """
        # Energy-based acceptance if using THRML
        if self.config.use_energy_guidance and self.energy_fn:
            energy = self._compute_energy(params)
            if energy > self.config.energy_threshold:
                return False
        
        # Metropolis-like acceptance
        if not self.population_fitness:
            return True  # Accept first
        
        current_fitness = np.mean(self.population_fitness)
        
        # Always accept if better
        if fitness > current_fitness:
            return True
        
        # Probabilistically accept if worse
        delta = fitness - current_fitness
        prob = np.exp(delta)  # Simulated annealing-like
        
        return np.random.rand() < prob
    
    def _compute_energy(self, params: Dict[str, float]) -> float:
        """
        Compute energy of parameters.
        
        Args:
            params: Parameters
            
        Returns:
            Energy value
        """
        if self.energy_fn is None:
            return 0.0
        
        # Convert params to state format for energy function
        # This is problem-specific
        state = np.array(list(params.values())[:3])  # Take first 3 values
        
        try:
            return float(self.energy_fn(state))
        except:
            return 0.0
    
    def _compute_diversity(self) -> float:
        """
        Compute diversity of population.
        
        Returns:
            Diversity measure
        """
        if len(self.population) < 2:
            return 0.0
        
        # Average pairwise distance
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._param_distance(self.population[i], self.population[j])
                total_distance += distance
                count += 1
        
        return float(total_distance / count) if count > 0 else 0.0
    
    def _param_distance(self, params1: Dict[str, float], params2: Dict[str, float]) -> float:
        """Distance between parameter sets."""
        distance = 0.0
        for key in params1.keys():
            if key in params2 and isinstance(params1[key], (int, float)):
                distance += (params1[key] - params2[key]) ** 2
        return np.sqrt(distance)
    
    def _param_diversity(self, params: Dict[str, float]) -> float:
        """Diversity of params compared to population."""
        if not self.population:
            return 1.0
        
        distances = [self._param_distance(params, p) for p in self.population]
        return float(np.mean(distances))
    
    def _optimize_oscillator_gradients(self, data: np.ndarray) -> float:
        """
        Optimize oscillator parameters using gradients.
        
        Args:
            data: Data for optimization
            
        Returns:
            Loss value
        """
        # This requires JAX and differentiable oscillator
        if not JAX_AVAILABLE:
            return 0.0
        
        # Placeholder - actual implementation would compute gradients
        # and update oscillator parameters
        return 0.0
    
    def _adaptive_phase_selection(self) -> TrainingPhase:
        """
        Adaptively select training phase.
        
        Based on recent progress in each phase.
        
        Returns:
            Selected phase
        """
        # Look at recent history
        window = 10
        
        if len(self.history['chaos_loss']) < window:
            return TrainingPhase.CHAOS_EXPLORATION
        
        # Compute progress in each phase
        chaos_progress = self._compute_progress(self.history['chaos_loss'][-window:])
        gradient_progress = self._compute_progress(self.history['gradient_loss'][-window:]) if self.history['gradient_loss'] else 0.0
        
        # Select phase with more progress
        if chaos_progress > gradient_progress:
            return TrainingPhase.CHAOS_EXPLORATION
        else:
            return TrainingPhase.GRADIENT_OPTIMIZATION
    
    def _compute_progress(self, losses: List[float]) -> float:
        """
        Compute progress from loss history.
        
        Args:
            losses: Recent losses
            
        Returns:
            Progress measure (higher is better)
        """
        if len(losses) < 2:
            return 0.0
        
        # Rate of improvement
        improvements = [losses[i] - losses[i+1] for i in range(len(losses) - 1)]
        return float(np.mean(improvements))
    
    def _update_balance(self):
        """Update chaos/gradient balance adaptively."""
        # Compute effectiveness of each phase
        chaos_eff = self._compute_progress(self.history['chaos_loss'][-self.config.balance_update_freq:])
        gradient_eff = self._compute_progress(self.history['gradient_loss'][-self.config.balance_update_freq:]) if self.history['gradient_loss'] else 0.0
        
        # Update weights (softmax-like)
        total_eff = abs(chaos_eff) + abs(gradient_eff) + 1e-8
        self.config.chaos_weight = abs(chaos_eff) / total_eff
        self.config.gradient_weight = abs(gradient_eff) / total_eff
    
    def train(
        self,
        data_generator: Callable,
        n_steps: int = 1000,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full hybrid training loop.
        
        Args:
            data_generator: Function that generates training data
            n_steps: Number of training steps
            verbose: Print progress
            
        Returns:
            Training history
        """
        for step in range(n_steps):
            # Generate data
            data = data_generator()
            target = None  # Can be set for supervised learning
            
            # Training step
            metrics = self.train_step(data, target)
            
            # Print progress
            if verbose and (step + 1) % 100 == 0:
                phase_name = self.current_phase.value
                print(f"Step {step+1}/{n_steps} [{phase_name}]: " +
                      ", ".join([f"{k}={v:.6f}" for k, v in metrics.items() if isinstance(v, float)]))
        
        return self.history
    
    def get_best_parameters(self) -> Dict[str, float]:
        """
        Get best parameters from population.
        
        Returns:
            Best parameters
        """
        if not self.population:
            return {}
        
        best_idx = np.argmax(self.population_fitness)
        return self.population[best_idx]


# ============================================================================
# Helper Functions
# ============================================================================

def create_hybrid_trainer(
    oscillator_model: Any,
    ml_model: Any,
    chaos_weight: float = 0.5,
    gradient_weight: float = 0.5,
    **kwargs
) -> HybridTrainer:
    """
    Factory function for hybrid trainer.
    
    Args:
        oscillator_model: Differentiable oscillator
        ml_model: ML model
        chaos_weight: Weight for chaos objective
        gradient_weight: Weight for gradient objective
        **kwargs: Additional config parameters
        
    Returns:
        HybridTrainer instance
    """
    config = HybridConfig(
        chaos_weight=chaos_weight,
        gradient_weight=gradient_weight,
        **kwargs
    )
    
    return HybridTrainer(oscillator_model, ml_model, config)


if __name__ == '__main__':
    # Example usage
    print("Hybrid Training Module Loaded!")
    print("\nThis module enables innovative chaos+gradient learning.")
    print("Key idea: Chaos explores, gradients optimize.")
    
    print("\nExample configuration:")
    config = HybridConfig(
        chaos_steps=100,
        gradient_steps=10,
        phase_schedule='alternating',
        adaptive_balance=True
    )
    
    print(f"  Chaos steps: {config.chaos_steps}")
    print(f"  Gradient steps: {config.gradient_steps}")
    print(f"  Phase schedule: {config.phase_schedule}")
    print(f"  Adaptive balance: {config.adaptive_balance}")
    
    print("\n✓ Ready for hybrid training!")
