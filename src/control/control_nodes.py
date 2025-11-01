"""
Control Nodes for GMCS Node Graph.

Provides real-time control and optimization capabilities:
- Parameter Optimizer: Gradient-based parameter tuning
- Chaos Controller: Stabilize/control chaotic dynamics
- PID Controller: Classic feedback control

These nodes enable active control of oscillators, THRML,
and other system components for achieving desired behaviors.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Parameter Optimizer
# ============================================================================

class OptimizerType(Enum):
    """Supported optimizer types."""
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"


@dataclass
class ParameterOptimizerConfig:
    """Configuration for Parameter Optimizer node."""
    learning_rate: float = 0.01
    target_metric: str = 'energy'
    optimizer: str = 'adam'
    momentum: float = 0.9
    beta1: float = 0.9  # Adam parameter
    beta2: float = 0.999  # Adam parameter
    epsilon: float = 1e-8


class ParameterOptimizer:
    """
    Gradient-based parameter optimizer using various optimization algorithms.
    
    Adjusts parameters to minimize error between current and target values.
    Useful for THRML energy minimization, oscillator stabilization, etc.
    """
    
    def __init__(self, config: ParameterOptimizerConfig):
        self.config = config
        
        # Optimizer state
        self.velocity = 0.0  # For momentum-based optimizers
        self.m = 0.0  # First moment (Adam)
        self.v = 0.0  # Second moment (Adam)
        self.t = 0  # Time step
        
        # History
        self.error_history = []
    
    def process(self, current_value: float, target_value: float) -> Dict[str, float]:
        """
        Compute control signal to drive current toward target.
        
        Args:
            current_value: Current metric value
            target_value: Desired metric value
            
        Returns:
            Dictionary with control_signal and error
        """
        # Compute error
        error = current_value - target_value
        self.error_history.append(error)
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        
        # Compute gradient (simplified - just use error)
        gradient = error
        
        # Apply optimizer
        if self.config.optimizer == 'sgd':
            control = self._sgd_step(gradient)
        elif self.config.optimizer == 'adam':
            control = self._adam_step(gradient)
        elif self.config.optimizer == 'rmsprop':
            control = self._rmsprop_step(gradient)
        else:
            control = -self.config.learning_rate * gradient
        
        return {
            'control_signal': float(control),
            'error': float(error)
        }
    
    def _sgd_step(self, gradient: float) -> float:
        """SGD with momentum."""
        self.velocity = self.config.momentum * self.velocity - self.config.learning_rate * gradient
        return self.velocity
    
    def _adam_step(self, gradient: float) -> float:
        """Adam optimizer."""
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.config.beta1 * self.m + (1 - self.config.beta1) * gradient
        
        # Update biased second moment estimate
        self.v = self.config.beta2 * self.v + (1 - self.config.beta2) * (gradient ** 2)
        
        # Compute bias-corrected moments
        m_hat = self.m / (1 - self.config.beta1 ** self.t)
        v_hat = self.v / (1 - self.config.beta2 ** self.t)
        
        # Compute update
        control = -self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.config.epsilon)
        
        return control
    
    def _rmsprop_step(self, gradient: float) -> float:
        """RMSprop optimizer."""
        # Update moving average of squared gradients
        self.v = self.config.beta2 * self.v + (1 - self.config.beta2) * (gradient ** 2)
        
        # Compute update
        control = -self.config.learning_rate * gradient / (np.sqrt(self.v) + self.config.epsilon)
        
        return control


# ============================================================================
# Chaos Controller
# ============================================================================

class ControlMethod(Enum):
    """Chaos control methods."""
    PYRAGAS = "pyragas"  # Time-delayed feedback
    OGY = "ogy"  # Ott-Grebogi-Yorke
    PINNING = "pinning"  # Pinning control


@dataclass
class ChaosControllerConfig:
    """Configuration for Chaos Controller node."""
    method: str = 'pyragas'
    strength: float = 0.1
    delay: int = 100  # For Pyragas method


class ChaosController:
    """
    Control and stabilize chaotic dynamics using advanced chaos control methods.
    
    Can stabilize unstable periodic orbits embedded in chaotic attractors
    using small perturbations (Pyragas, OGY methods).
    """
    
    def __init__(self, config: ChaosControllerConfig):
        self.config = config
        
        # Delay buffer for Pyragas method
        self.delay_buffer = []
        
        # OGY state
        self.previous_state = None
        self.target_orbit = None
    
    def process(self, state: np.ndarray, target: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute control signal to stabilize or direct chaotic dynamics.
        
        Args:
            state: Current system state (N-dimensional)
            target: Optional target state/orbit
            
        Returns:
            Dictionary with control signal array
        """
        if self.config.method == 'pyragas':
            control = self._pyragas_control(state)
        elif self.config.method == 'ogy':
            control = self._ogy_control(state, target)
        elif self.config.method == 'pinning':
            control = self._pinning_control(state, target)
        else:
            control = np.zeros_like(state)
        
        return {'control': control}
    
    def _pyragas_control(self, state: np.ndarray) -> np.ndarray:
        """
        Pyragas time-delayed feedback control.
        
        Stabilizes unstable periodic orbits by feeding back the difference
        between current and delayed states.
        """
        # Add current state to delay buffer
        self.delay_buffer.append(state.copy())
        
        # Maintain delay buffer size
        if len(self.delay_buffer) > self.config.delay:
            delayed_state = self.delay_buffer.pop(0)
        else:
            # Not enough history yet
            return np.zeros_like(state)
        
        # Compute control: K * (x(t - τ) - x(t))
        control = self.config.strength * (delayed_state - state)
        
        return control
    
    def _ogy_control(self, state: np.ndarray, target: Optional[np.ndarray]) -> np.ndarray:
        """
        OGY (Ott-Grebogi-Yorke) chaos control.
        
        Applies small perturbations when trajectory passes near
        desired unstable periodic orbit.
        """
        if target is None:
            # If no target specified, use zero control
            return np.zeros_like(state)
        
        # Check if near target orbit (Poincaré section crossing)
        if self.previous_state is not None:
            # Simplified - check if crossing a threshold
            distance = np.linalg.norm(state - target)
            
            if distance < 0.5:  # Near target
                # Apply control to push onto stable manifold
                control = -self.config.strength * (state - target)
            else:
                control = np.zeros_like(state)
        else:
            control = np.zeros_like(state)
        
        self.previous_state = state.copy()
        return control
    
    def _pinning_control(self, state: np.ndarray, target: Optional[np.ndarray]) -> np.ndarray:
        """
        Pinning control for network synchronization.
        
        Applies control to a subset of nodes to drive system to target.
        """
        if target is None:
            return np.zeros_like(state)
        
        # Simple proportional control
        control = -self.config.strength * (state - target)
        
        # Apply only to first half of nodes (pinning subset)
        control[len(control)//2:] = 0
        
        return control


# ============================================================================
# PID Controller
# ============================================================================

@dataclass
class PIDControllerConfig:
    """Configuration for PID Controller node."""
    Kp: float = 1.0  # Proportional gain
    Ki: float = 0.1  # Integral gain  
    Kd: float = 0.01  # Derivative gain
    setpoint: float = 0.0  # Target value
    output_limit: Optional[Tuple[float, float]] = None  # (min, max)


class PIDController:
    """
    Classic PID (Proportional-Integral-Derivative) feedback controller.
    
    Maintains a desired setpoint through three control terms:
    - P: Responds to current error
    - I: Eliminates steady-state error
    - D: Reduces overshoot
    """
    
    def __init__(self, config: PIDControllerConfig):
        self.config = config
        
        # PID state
        self.integral = 0.0
        self.previous_error = 0.0
        self.error_history = []
    
    def process(self, measurement: float, dt: float = 0.01) -> Dict[str, float]:
        """
        Compute PID control signal.
        
        Args:
            measurement: Current process variable
            dt: Time step (seconds)
            
        Returns:
            Dictionary with control signal and error
        """
        # Compute error
        error = self.config.setpoint - measurement
        self.error_history.append(error)
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        
        # Proportional term
        p_term = self.config.Kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        # Simple anti-windup: clamp integral
        max_integral = 10.0 / (self.config.Ki + 1e-10)
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        i_term = self.config.Ki * self.integral
        
        # Derivative term (with filtering)
        derivative = (error - self.previous_error) / (dt + 1e-10)
        d_term = self.config.Kd * derivative
        
        # Compute total control
        control = p_term + i_term + d_term
        
        # Apply output limits if specified
        if self.config.output_limit is not None:
            control = np.clip(control, self.config.output_limit[0], self.config.output_limit[1])
        
        # Update state
        self.previous_error = error
        
        return {
            'control': float(control),
            'error': float(error)
        }
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.previous_error = 0.0
        self.error_history = []


# ============================================================================
# Node Wrappers for Test Compatibility
# ============================================================================

class OptimizerControlNode:
    """Wrapper for ParameterOptimizer to match test calling convention."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        config = ParameterOptimizerConfig(**kwargs)
        self._impl = ParameterOptimizer(config)
    
    def process(self, **inputs):
        return self._impl.process(**inputs)


class ChaosControlNode:
    """Wrapper for ChaosController to match test calling convention."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        # Map 'control_strength' to 'strength' for config
        if 'control_strength' in kwargs and 'strength' not in kwargs:
            kwargs['strength'] = kwargs.pop('control_strength')
        config = ChaosControllerConfig(**kwargs)
        self._impl = ChaosController(config)
    
    def process(self, **inputs):
        return self._impl.process(**inputs)


class PIDControlNode:
    """Wrapper for PIDController to match test calling convention."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        # Map lowercase parameters to config format
        if 'kp' in kwargs:
            kwargs['Kp'] = kwargs.pop('kp')
        if 'ki' in kwargs:
            kwargs['Ki'] = kwargs.pop('ki')
        if 'kd' in kwargs:
            kwargs['Kd'] = kwargs.pop('kd')
        config = PIDControllerConfig(**kwargs)
        self._impl = PIDController(config)
    
    def process(self, **inputs):
        return self._impl.process(**inputs)


# ============================================================================
# Node Factory
# ============================================================================

def create_control_node(node_type: str, config: dict):
    """
    Factory function to create control nodes.
    
    Args:
        node_type: Type of control node
        config: Configuration dictionary
        
    Returns:
        Control node instance
    """
    if node_type == 'Parameter Optimizer':
        return ParameterOptimizer(ParameterOptimizerConfig(**config))
    elif node_type == 'Chaos Controller':
        return ChaosController(ChaosControllerConfig(**config))
    elif node_type == 'PID Controller':
        return PIDController(PIDControllerConfig(**config))
    else:
        raise ValueError(f"Unknown control node type: {node_type}")

