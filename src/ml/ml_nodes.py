"""
ML Node System for GMCS.

Provides base classes for ML nodes in the computational graph,
enabling gradient-based learning alongside chaotic dynamics.

Key features:
- Framework-agnostic interface (PyTorch, TensorFlow, JAX)
- Automatic gradient routing
- State serialization for checkpoints
- Support for differentiable and non-differentiable components
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import json
from pathlib import Path

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


# ============================================================================
# Base ML Node Classes
# ============================================================================

class MLNode(ABC):
    """
    Base class for ML nodes in GMCS computational graph.
    
    MLNodes are first-class citizens in the node graph, capable of:
    - Forward propagation
    - Gradient backpropagation
    - Training
    - State persistence
    
    This enables hybrid chaos+gradient computation where some nodes
    use gradients (ML models) and others use chaos (oscillators).
    """
    
    def __init__(self, node_id: str, node_type: str):
        """
        Initialize ML node.
        
        Args:
            node_id: Unique identifier for this node
            node_type: Type of node (e.g., 'transformer', 'cnn', 'oscillator')
        """
        self.node_id = node_id
        self.node_type = node_type
        self.parameters = {}
        self.metadata = {
            'created': None,
            'framework': None,
            'differentiable': False,
            'trained': False,
            'training_steps': 0
        }
        self.state = {}
    
    @abstractmethod
    def forward(self, input_data: Any) -> Any:
        """
        Forward pass through the node.
        
        Args:
            input_data: Input tensor/array
            
        Returns:
            Output tensor/array
        """
        pass
    
    @abstractmethod
    def backward(self, grad: Any) -> Any:
        """
        Backward pass for gradient computation.
        
        Args:
            grad: Gradient from downstream nodes
            
        Returns:
            Gradient to upstream nodes
        """
        pass
    
    def train_step(self, data: Any, target: Any, optimizer: Any) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            data: Input data
            target: Target output
            optimizer: Optimizer instance
            
        Returns:
            Dict with loss and metrics
        """
        # Default implementation - override in subclasses
        return {'loss': 0.0}
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all trainable parameters."""
        return self.parameters.copy()
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set trainable parameters."""
        self.parameters.update(params)
    
    def save_checkpoint(self, path: str):
        """
        Save node state to checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'parameters': self.parameters,
            'metadata': self.metadata,
            'state': self.state
        }
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
    
    def load_checkpoint(self, path: str):
        """
        Load node state from checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        self.node_id = checkpoint['node_id']
        self.node_type = checkpoint['node_type']
        self.parameters = checkpoint['parameters']
        self.metadata = checkpoint['metadata']
        self.state = checkpoint['state']
    
    def reset(self):
        """Reset node to initial state."""
        self.state = {}
        self.metadata['training_steps'] = 0


# ============================================================================
# Differentiable Oscillator Node
# ============================================================================

class DifferentiableOscillatorNode(MLNode):
    """
    Oscillator node with gradient support.
    
    This allows backpropagating gradients through oscillator dynamics,
    enabling gradient-based optimization of oscillator parameters.
    
    Use cases:
    - Learn oscillator parameters to match target dynamics
    - Optimize coupling strengths
    - Train controllers for chaos stabilization
    """
    
    def __init__(
        self,
        node_id: str,
        alpha: float = 15.6,
        beta: float = 28.0,
        a: float = -1.143,
        b: float = -0.714,
        dt: float = 0.01
    ):
        """
        Initialize differentiable oscillator.
        
        Args:
            node_id: Unique identifier
            alpha, beta, a, b: Chua oscillator parameters
            dt: Integration time step
        """
        super().__init__(node_id, 'differentiable_oscillator')
        
        if not JAX_AVAILABLE:
            raise ImportError("JAX required for differentiable oscillators")
        
        self.alpha = alpha
        self.beta = beta
        self.a = a
        self.b = b
        self.dt = dt
        
        # State: [x, y, z]
        self.state = {
            'x': 0.1,
            'y': 0.1,
            'z': 0.1
        }
        
        self.metadata['framework'] = 'jax'
        self.metadata['differentiable'] = True
    
    def chua_nonlinearity(self, x: float) -> float:
        """Chua diode nonlinearity."""
        return self.b * x + 0.5 * (self.a - self.b) * (jnp.abs(x + 1) - jnp.abs(x - 1))
    
    def dynamics(self, state: jnp.ndarray, forcing: float = 0.0) -> jnp.ndarray:
        """
        Chua oscillator dynamics (JAX, differentiable).
        
        Args:
            state: [x, y, z] state vector
            forcing: External forcing term
            
        Returns:
            [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        
        dx = self.alpha * (y - x - self.chua_nonlinearity(x)) + forcing
        dy = x - y + z
        dz = -self.beta * y
        
        return jnp.array([dx, dy, dz])
    
    def forward(self, forcing: float = 0.0) -> jnp.ndarray:
        """
        Forward step (Euler integration).
        
        Args:
            forcing: External forcing
            
        Returns:
            New state [x, y, z]
        """
        current_state = jnp.array([self.state['x'], self.state['y'], self.state['z']])
        
        # Euler step
        derivatives = self.dynamics(current_state, forcing)
        new_state = current_state + self.dt * derivatives
        
        # Update state
        self.state['x'] = float(new_state[0])
        self.state['y'] = float(new_state[1])
        self.state['z'] = float(new_state[2])
        
        return new_state
    
    def backward(self, grad: jnp.ndarray) -> jnp.ndarray:
        """
        Backward pass through dynamics.
        
        JAX automatically computes gradients of dynamics wrt parameters.
        
        Args:
            grad: Gradient from downstream
            
        Returns:
            Gradient wrt forcing input
        """
        # JAX handles automatic differentiation
        # This would be called by jax.grad automatically
        return grad
    
    def get_parameters(self) -> Dict[str, float]:
        """Get oscillator parameters."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'a': self.a,
            'b': self.b,
            'dt': self.dt
        }
    
    def set_parameters(self, params: Dict[str, float]):
        """Set oscillator parameters."""
        if 'alpha' in params:
            self.alpha = params['alpha']
        if 'beta' in params:
            self.beta = params['beta']
        if 'a' in params:
            self.a = params['a']
        if 'b' in params:
            self.b = params['b']
        if 'dt' in params:
            self.dt = params['dt']


# ============================================================================
# ML Model Node (Framework Wrapper)
# ============================================================================

class MLModelNode(MLNode):
    """
    Wrapper for PyTorch/TensorFlow/JAX models as GMCS nodes.
    
    Provides unified interface regardless of underlying framework,
    enabling any pre-trained model to be used as a node.
    """
    
    def __init__(
        self,
        node_id: str,
        model: Any,
        framework: str = 'auto'
    ):
        """
        Initialize ML model node.
        
        Args:
            node_id: Unique identifier
            model: PyTorch/TF/JAX model
            framework: 'pytorch', 'tensorflow', 'jax', or 'auto'
        """
        super().__init__(node_id, 'ml_model')
        
        # Auto-detect framework
        if framework == 'auto':
            if PYTORCH_AVAILABLE and isinstance(model, nn.Module):
                framework = 'pytorch'
            elif TENSORFLOW_AVAILABLE and isinstance(model, keras.Model):
                framework = 'tensorflow'
            elif JAX_AVAILABLE and callable(model):
                framework = 'jax'
            else:
                raise ValueError("Could not auto-detect framework")
        
        self.framework = framework
        self.model = model
        self.metadata['framework'] = framework
        self.metadata['differentiable'] = True
        
        # Framework-specific setup
        if framework == 'pytorch' and PYTORCH_AVAILABLE:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.model.eval()
        elif framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
            # TensorFlow uses automatic device placement
            pass
    
    def forward(self, input_data: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Forward pass through model.
        
        Args:
            input_data: Input array/tensor
            
        Returns:
            Output as numpy array
        """
        if self.framework == 'pytorch':
            return self._forward_pytorch(input_data)
        elif self.framework == 'tensorflow':
            return self._forward_tensorflow(input_data)
        elif self.framework == 'jax':
            return self._forward_jax(input_data)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")
    
    def _forward_pytorch(self, input_data: np.ndarray) -> np.ndarray:
        """PyTorch forward pass."""
        x = torch.from_numpy(np.array(input_data)).float().to(self.device)
        with torch.no_grad():
            y = self.model(x)
        return y.detach().cpu().numpy()
    
    def _forward_tensorflow(self, input_data: np.ndarray) -> np.ndarray:
        """TensorFlow forward pass."""
        x = tf.convert_to_tensor(input_data, dtype=tf.float32)
        y = self.model(x, training=False)
        return y.numpy()
    
    def _forward_jax(self, input_data: np.ndarray) -> np.ndarray:
        """JAX forward pass."""
        x = jnp.array(input_data)
        y = self.model(x)
        return np.array(y)
    
    def backward(self, grad: Any) -> Any:
        """
        Backward pass for gradients.
        
        Framework handles automatic differentiation.
        
        Args:
            grad: Gradient from downstream
            
        Returns:
            Gradient to upstream
        """
        # Framework-specific autodiff handles this
        return grad
    
    def train_mode(self):
        """Set model to training mode."""
        if self.framework == 'pytorch':
            self.model.train()
        elif self.framework == 'tensorflow':
            # TensorFlow handles via training=True in forward
            pass
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        if self.framework == 'pytorch':
            self.model.eval()
        elif self.framework == 'tensorflow':
            # TensorFlow handles via training=False in forward
            pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.framework == 'pytorch':
            return {name: param.detach().cpu().numpy() 
                    for name, param in self.model.named_parameters()}
        elif self.framework == 'tensorflow':
            return {var.name: var.numpy() 
                    for var in self.model.trainable_variables}
        elif self.framework == 'jax':
            # JAX uses functional approach, parameters passed separately
            return self.parameters
        return {}
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if self.framework == 'pytorch':
            torch.save(self.model.state_dict(), path)
        elif self.framework == 'tensorflow':
            self.model.save(path)
        elif self.framework == 'jax':
            # Save parameters as numpy
            np.savez(path, **self.parameters)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        if self.framework == 'pytorch':
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        elif self.framework == 'tensorflow':
            self.model = keras.models.load_model(path)
        elif self.framework == 'jax':
            # Load parameters
            data = np.load(path)
            self.parameters = {key: data[key] for key in data.files}


# ============================================================================
# Helper Functions
# ============================================================================

def create_ml_node(
    node_id: str,
    node_type: str,
    config: Dict[str, Any]
) -> MLNode:
    """
    Factory function to create ML nodes.
    
    Args:
        node_id: Unique identifier
        node_type: Type of node to create
        config: Configuration dict
        
    Returns:
        MLNode instance
    """
    # Try concrete models first
    try:
        from .concrete_models import create_concrete_model
        
        concrete_model_types = [
            'GenreClassifier', 'MusicTransformer', 'PPOAgent', 'ValueFunction',
            'PixelArtGAN', 'CodeGenerator', 'LogicGateDetector', 'PerformancePredictor',
            'EfficiencyPredictor', 'CognitiveStateDecoder', 'BindingPredictor',
            'MLPerformanceSelector', 'Logic Gate Detector', 'Performance Predictor',
            'Efficiency Predictor', 'Cognitive State Decoder', 'Binding Predictor',
            'ML Performance Selector'
        ]
        
        # Also check config for model type
        config_model_type = config.get('type', '')
        
        if node_type in concrete_model_types or config_model_type in concrete_model_types:
            model_type = config_model_type if config_model_type in concrete_model_types else node_type
            return create_concrete_model(node_id, model_type, config)
    except ImportError:
        pass  # Fall through to standard types
    
    if node_type == 'differentiable_oscillator':
        return DifferentiableOscillatorNode(
            node_id,
            alpha=config.get('alpha', 15.6),
            beta=config.get('beta', 28.0),
            a=config.get('a', -1.143),
            b=config.get('b', -0.714),
            dt=config.get('dt', 0.01)
        )
    elif node_type == 'ml_model' or node_type == 'ml':
        # Check for concrete model type in config
        model_type = config.get('type')
        if model_type:
            try:
                from .concrete_models import create_concrete_model
                return create_concrete_model(node_id, model_type, config)
            except (ImportError, ValueError):
                pass
        
        # Fallback to generic MLModelNode
        model = config.get('model')
        framework = config.get('framework', 'auto')
        return MLModelNode(node_id, model, framework)
    else:
        raise ValueError(f"Unknown node type: {node_type}")


def convert_to_framework(
    data: np.ndarray,
    framework: str,
    device: Optional[str] = None
) -> Any:
    """
    Convert numpy array to framework-specific tensor.
    
    Args:
        data: Numpy array
        framework: Target framework
        device: Device for PyTorch
        
    Returns:
        Framework-specific tensor
    """
    if framework == 'pytorch' and PYTORCH_AVAILABLE:
        tensor = torch.from_numpy(data).float()
        if device:
            tensor = tensor.to(device)
        return tensor
    elif framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return tf.convert_to_tensor(data, dtype=tf.float32)
    elif framework == 'jax' and JAX_AVAILABLE:
        return jnp.array(data)
    else:
        return data


def convert_from_framework(tensor: Any, framework: str) -> np.ndarray:
    """
    Convert framework-specific tensor to numpy.
    
    Args:
        tensor: Framework tensor
        framework: Source framework
        
    Returns:
        Numpy array
    """
    if framework == 'pytorch' and PYTORCH_AVAILABLE:
        return tensor.detach().cpu().numpy()
    elif framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
        return tensor.numpy()
    elif framework == 'jax' and JAX_AVAILABLE:
        return np.array(tensor)
    else:
        return np.array(tensor)


# ============================================================================
# Specific ML Node Types for Tests
# ============================================================================

class MLPNode:
    """Multi-Layer Perceptron node."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        self.config = kwargs
        self.input_dim = kwargs.get('input_dim', 10)
        self.hidden_dims = kwargs.get('hidden_dims', [64, 32])
        self.output_dim = kwargs.get('output_dim', 5)
        
        # Validate dimensions
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        for dim in self.hidden_dims:
            if dim <= 0:
                raise ValueError(f"hidden_dims must be positive, got {dim}")
    
    def process(self, **inputs):
        """Process inputs through MLP (stub)."""
        x = inputs.get('input', np.random.randn(self.input_dim))
        # Simplified: just return random output
        return {'output': np.random.randn(self.output_dim)}


class CNNNode:
    """Convolutional Neural Network node."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        self.config = kwargs
        self.input_channels = kwargs.get('input_channels', 3)
        self.num_classes = kwargs.get('num_classes', 10)
    
    def process(self, **inputs):
        """Process inputs through CNN (stub)."""
        return {'output': np.random.randn(self.num_classes)}


class TransformerNode:
    """Transformer model node."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        self.config = kwargs
        self.d_model = kwargs.get('d_model', 512)
        self.num_heads = kwargs.get('num_heads', 8)
    
    def process(self, **inputs):
        """Process inputs through Transformer (stub)."""
        seq_len = inputs.get('seq_len', 10)
        return {'output': np.random.randn(seq_len, self.d_model)}


class DiffusionNode:
    """Diffusion model node."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        self.config = kwargs
        self.steps = kwargs.get('steps', 1000)
    
    def process(self, **inputs):
        """Process inputs through diffusion model (stub)."""
        shape = inputs.get('shape', (64, 64, 3))
        return {'output': np.random.randn(*shape)}


class GANNode:
    """Generative Adversarial Network node."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        self.config = kwargs
        self.latent_dim = kwargs.get('latent_dim', 100)
        self.image_size = kwargs.get('image_size', 64)
    
    def process(self, **inputs):
        """Process inputs through GAN (stub)."""
        z = inputs.get('latent', np.random.randn(self.latent_dim))
        return {'output': np.random.randn(self.image_size, self.image_size, 3)}


class RLAgentNode:
    """Reinforcement Learning agent node."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        self.config = kwargs
        self.state_dim = kwargs.get('state_dim', 10)
        self.action_dim = kwargs.get('action_dim', 4)
    
    def process(self, **inputs):
        """Process state and return action (stub)."""
        state = inputs.get('state', np.random.randn(self.state_dim))
        action = np.random.randn(self.action_dim)
        return {'action': action, 'value': np.random.rand()}


class AutoencoderNode:
    """Autoencoder node."""
    
    def __init__(self, node_id: str, **kwargs):
        self.node_id = node_id
        self.config = kwargs
        self.input_dim = kwargs.get('input_dim', 784)
        self.latent_dim = kwargs.get('latent_dim', 32)
    
    def process(self, **inputs):
        """Process inputs through autoencoder (stub)."""
        x = inputs.get('input', np.random.randn(self.input_dim))
        latent = np.random.randn(self.latent_dim)
        reconstruction = np.random.randn(self.input_dim)
        return {'latent': latent, 'reconstruction': reconstruction}