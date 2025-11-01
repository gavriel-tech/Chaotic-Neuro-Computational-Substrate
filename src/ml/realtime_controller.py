"""
Real-Time ML Controller for GMCS Simulation Loop.

Provides ML models that run IN the simulation loop for active control,
prediction, and online learning.
"""

from collections import deque
from typing import Optional, Any, Dict, Callable
import jax.numpy as jnp
import numpy as np


class RealtimeMLController:
    """
    ML controller that executes in the simulation loop for real-time control.
    
    This controller can:
    - Predict future states
    - Stabilize chaotic dynamics
    - Optimize system parameters
    - Learn online from experience
    
    Control modes:
    - 'prediction': Predict next state and compute control
    - 'stabilization': Drive oscillators toward target state
    - 'optimization': Optimize a cost function
    - 'none': Passive observation only
    
    Example:
        >>> from src.ml.supervised import create_mlp
        >>> model = create_mlp(input_dim=192, hidden_dims=[128, 64], output_dim=64)
        >>> controller = RealtimeMLController(model, control_mode='prediction')
        >>>
        >>> # In simulation loop:
        >>> for step in range(1000):
        >>>     state_dict = {'oscillator_state': state.oscillator_state, 'field_p': state.field_p}
        >>>     control = controller.step(state_dict)
        >>>     state = apply_control(state, control)
        >>>     
        >>>     if step % 10 == 0:
        >>>         controller.train_online()
    """
    
    def __init__(
        self,
        model: Any,
        control_mode: str = 'prediction',
        history_size: int = 1000,
        target_state: Optional[np.ndarray] = None
    ):
        """
        Initialize real-time ML controller.
        
        Args:
            model: ML model with forward() or predict() method
            control_mode: One of ['prediction', 'stabilization', 'optimization', 'none']
            history_size: Number of recent states to store
            target_state: Target state for stabilization mode (default: origin)
        """
        self.model = model
        self.control_mode = control_mode
        self.history = deque(maxlen=history_size)
        self.step_count = 0
        
        # Target state for stabilization
        if target_state is not None:
            self.target_state = jnp.array(target_state)
        else:
            self.target_state = jnp.array([1.0, 0.0, 0.0])  # Default: x=1 attractor
        
        # Control gains
        self.Kp = 0.1  # Proportional gain for stabilization
        self.Ki = 0.01  # Integral gain
        self.Kd = 0.05  # Derivative gain
        
        # PID state
        self.integral_error = None
        self.previous_error = None
        
        # Performance tracking
        self.control_history = []
        self.loss_history = []
        
    def step(self, state_dict: Dict[str, Any]) -> jnp.ndarray:
        """
        Process one simulation step and return control signal.
        
        Args:
            state_dict: Dictionary with simulation state, should contain:
                - 'oscillator_state': (N, 3) array of oscillator states
                - 'field_p': (GRID_W, GRID_H) wave field (optional)
                - 'time': scalar time value (optional)
                
        Returns:
            (N,) control signal to apply to oscillators
        """
        # Extract state
        oscillators = state_dict.get('oscillator_state', jnp.zeros((64, 3)))
        wave_field = state_dict.get('field_p', None)
        time = state_dict.get('time', self.step_count)
        
        n_oscillators = len(oscillators)
        
        # Compute control based on mode
        if self.control_mode == 'prediction':
            control = self._predict_control(oscillators, wave_field)
        elif self.control_mode == 'stabilization':
            control = self._stabilize(oscillators)
        elif self.control_mode == 'optimization':
            control = self._optimize(oscillators, wave_field)
        else:  # 'none' or passive
            control = jnp.zeros(n_oscillators)
        
        # Store for learning
        self.history.append({
            'state': np.array(oscillators),
            'control': np.array(control),
            'step': self.step_count,
            'time': time
        })
        
        self.control_history.append(float(jnp.mean(jnp.abs(control))))
        self.step_count += 1
        
        return control
    
    def _predict_control(
        self,
        osc: jnp.ndarray,
        field: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Predict optimal control from ML model.
        
        Uses model to predict next state or directly predict control.
        """
        try:
            # Build feature vector
            features = osc.flatten()  # (N*3,)
            
            # Optionally include wave field statistics
            if field is not None:
                field_stats = jnp.array([
                    jnp.mean(field),
                    jnp.std(field),
                    jnp.max(field),
                    jnp.min(field)
                ])
                features = jnp.concatenate([features, field_stats])
            
            # Model inference
            if hasattr(self.model, 'forward'):
                prediction = self.model.forward(features)
            elif hasattr(self.model, 'predict'):
                prediction = self.model.predict(features)
            elif hasattr(self.model, '__call__'):
                prediction = self.model(features)
            else:
                # No valid inference method
                return jnp.zeros(len(osc))
            
            # Extract control from prediction
            # Assume prediction is either:
            # - Next state (N*3) → compute control as difference
            # - Control signal (N) → use directly
            
            if len(prediction) == len(osc) * 3:
                # Predicted next state
                predicted_state = prediction.reshape(osc.shape)
                control = (predicted_state - osc)[:, 0]  # Control on x component
            elif len(prediction) == len(osc):
                # Direct control prediction
                control = prediction
            else:
                # Unexpected shape, zero control
                control = jnp.zeros(len(osc))
            
            return control
            
        except Exception as e:
            print(f"[ML Controller] Prediction failed: {e}")
            return jnp.zeros(len(osc))
    
    def _stabilize(self, osc: jnp.ndarray) -> jnp.ndarray:
        """
        PID control to stabilize oscillators toward target state.
        
        Uses proportional-integral-derivative control to drive
        oscillators toward self.target_state.
        """
        n_osc = len(osc)
        
        # Expand target to match oscillator count
        if len(self.target_state) == 3:
            # Single target for all oscillators
            target = jnp.tile(self.target_state, (n_osc, 1))
        else:
            target = self.target_state.reshape(n_osc, 3)
        
        # Compute error (on x, y, z components)
        error = target - osc
        
        # Initialize integral and derivative terms
        if self.integral_error is None:
            self.integral_error = jnp.zeros_like(error)
        if self.previous_error is None:
            self.previous_error = error
        
        # Update integral
        self.integral_error = self.integral_error + error
        
        # Compute derivative
        derivative_error = error - self.previous_error
        
        # PID control
        control = (
            self.Kp * error[:, 0] +  # Proportional (x only)
            self.Ki * self.integral_error[:, 0] +  # Integral
            self.Kd * derivative_error[:, 0]  # Derivative
        )
        
        # Update for next step
        self.previous_error = error
        
        # Clip control to reasonable range
        control = jnp.clip(control, -1.0, 1.0)
        
        return control
    
    def _optimize(
        self,
        osc: jnp.ndarray,
        field: Optional[jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Optimize a cost function using gradient-based control.
        
        Minimizes a cost function that can include:
        - State deviation from target
        - Energy dissipation
        - Wave field alignment
        """
        # Simple gradient descent on energy
        # Cost: J = 0.5 * ||x - target||^2
        
        target = self.target_state
        if len(target) == 3:
            target = jnp.tile(target, (len(osc), 1))
        else:
            target = target.reshape(len(osc), 3)
        
        # Gradient of cost w.r.t. x
        grad = osc - target
        
        # Control is negative gradient (move toward minimum)
        control = -self.Kp * grad[:, 0]
        
        return control
    
    def train_online(self, batch_size: int = 32) -> Optional[float]:
        """
        Train model from recent experience using online learning.
        
        Samples a batch from history and updates model weights.
        
        Args:
            batch_size: Number of samples for training batch
            
        Returns:
            Training loss (float) or None if insufficient data
        """
        if len(self.history) < batch_size:
            return None
        
        try:
            # Sample batch from history
            indices = np.random.choice(len(self.history), batch_size, replace=False)
            batch = [self.history[i] for i in indices]
            
            # Prepare training data
            states = np.array([item['state'] for item in batch])  # (batch_size, N, 3)
            controls = np.array([item['control'] for item in batch])  # (batch_size, N)
            
            # Next states (shifted by 1)
            indices_next = np.clip(indices + 1, 0, len(self.history) - 1)
            next_states = np.array([self.history[i]['state'] for i in indices_next])
            
            # Train model (if it has train_step method)
            if hasattr(self.model, 'train_step'):
                loss = self.model.train_step({
                    'states': states,
                    'controls': controls,
                    'next_states': next_states
                })
                self.loss_history.append(float(loss))
                return float(loss)
            else:
                return None
                
        except Exception as e:
            print(f"[ML Controller] Online training failed: {e}")
            return None
    
    def reset(self):
        """Reset controller state (PID terms, history)."""
        self.integral_error = None
        self.previous_error = None
        self.step_count = 0
        self.history.clear()
        self.control_history.clear()
        self.loss_history.clear()
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get controller diagnostics for monitoring."""
        recent_control = self.control_history[-100:] if self.control_history else [0.0]
        recent_loss = self.loss_history[-100:] if self.loss_history else [0.0]
        
        return {
            'step_count': self.step_count,
            'control_mode': self.control_mode,
            'history_size': len(self.history),
            'mean_control_magnitude': float(np.mean(recent_control)),
            'std_control_magnitude': float(np.std(recent_control)),
            'mean_loss': float(np.mean(recent_loss)) if recent_loss else None,
            'model_type': type(self.model).__name__
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_simple_controller(
    n_oscillators: int = 64,
    control_mode: str = 'stabilization'
) -> RealtimeMLController:
    """
    Create a simple controller with no ML model (pure control).
    
    Useful for testing and baseline comparisons.
    
    Args:
        n_oscillators: Number of oscillators to control
        control_mode: Control mode ('stabilization', 'prediction', etc.)
        
    Returns:
        RealtimeMLController with no ML model
    """
    # Dummy model (just returns zeros)
    class DummyModel:
        def forward(self, x):
            return jnp.zeros(n_oscillators)
    
    controller = RealtimeMLController(
        model=DummyModel(),
        control_mode=control_mode
    )
    
    return controller


def create_mlp_controller(
    n_oscillators: int = 64,
    hidden_dims: list = [128, 64],
    control_mode: str = 'prediction'
) -> RealtimeMLController:
    """
    Create controller with MLP model for prediction/control.
    
    Args:
        n_oscillators: Number of oscillators
        hidden_dims: Hidden layer dimensions
        control_mode: Control mode
        
    Returns:
        RealtimeMLController with MLP model
    """
    from .supervised import MLPNode
    
    # Input: oscillator states (N*3) + wave field stats (4)
    input_dim = n_oscillators * 3 + 4
    output_dim = n_oscillators  # Control for each oscillator
    
    model = MLPNode({
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'output_dim': output_dim,
        'activation': 'tanh'
    })
    
    controller = RealtimeMLController(
        model=model,
        control_mode=control_mode
    )
    
    return controller

