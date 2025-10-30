"""
TensorFlow/Keras Integration for GMCS.

Enables loading TensorFlow models and integrating them with the GMCS simulation.
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    print("TensorFlow not available. Install with: pip install tensorflow")

import jax.numpy as jnp


# Stub classes when TensorFlow is not available
if not TENSORFLOW_AVAILABLE:
    class TensorFlowModelWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow is required for this module. Install with: pip install tensorflow")
    
    class GMCSTensorFlowTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow is required for this module. Install with: pip install tensorflow")
else:
    # Real implementations when TensorFlow is available
    class TensorFlowModelWrapper:
        """
        Wrapper for TensorFlow/Keras models to integrate with GMCS.
        """
        
        def __init__(self, model: Optional[keras.Model] = None):
            """
            Initialize TensorFlow model wrapper.
            
            Args:
                model: Keras model
            """
            self.model = model
        
        def jax_to_tf(self, x: jnp.ndarray) -> tf.Tensor:
            """Convert JAX array to TensorFlow tensor."""
            return tf.convert_to_tensor(np.array(x))
        
        def tf_to_jax(self, x: tf.Tensor) -> jnp.ndarray:
            """Convert TensorFlow tensor to JAX array."""
            return jnp.array(x.numpy())
        
        def forward(self, x: jnp.ndarray) -> jnp.ndarray:
            """Run forward pass through model."""
            if self.model is None:
                raise ValueError("No model loaded")
            
            x_tf = self.jax_to_tf(x)
            y_tf = self.model(x_tf, training=False)
            return self.tf_to_jax(y_tf)
        
        def load_model(self, path: str):
            """Load model from file."""
            self.model = keras.models.load_model(path)
        
        def save_model(self, path: str):
            """Save model to file."""
            if self.model is None:
                raise ValueError("No model to save")
            self.model.save(path)


    class GMCSTensorFlowTrainer:
        """
        Trainer for TensorFlow models using GMCS data.
        """
        
        def __init__(
            self,
            model: keras.Model,
            optimizer: Optional[keras.optimizers.Optimizer] = None,
            loss_fn: Optional[keras.losses.Loss] = None
        ):
            """
            Initialize trainer.
            
            Args:
                model: Keras model to train
                optimizer: Keras optimizer (default: Adam)
                loss_fn: Loss function (default: MSE)
            """
            self.model = model
            self.optimizer = optimizer or keras.optimizers.Adam(learning_rate=0.001)
            self.loss_fn = loss_fn or keras.losses.MeanSquaredError()
            
            self.train_losses = []
            self.val_losses = []
        
        def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
            """Single training step."""
            x_tf = tf.convert_to_tensor(x)
            y_tf = tf.convert_to_tensor(y)
            
            with tf.GradientTape() as tape:
                y_pred = self.model(x_tf, training=True)
                loss = self.loss_fn(y_tf, y_pred)
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            return float(loss.numpy())
        
        def validate(self, x: np.ndarray, y: np.ndarray) -> float:
            """Validation step."""
            x_tf = tf.convert_to_tensor(x)
            y_tf = tf.convert_to_tensor(y)
            
            y_pred = self.model(x_tf, training=False)
            loss = self.loss_fn(y_tf, y_pred)
            
            return float(loss.numpy())
        
        def fit(
            self,
            train_data: List[Tuple[np.ndarray, np.ndarray]],
            val_data: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
            epochs: int = 100,
            verbose: bool = True
        ) -> Dict[str, List[float]]:
            """
            Train model for multiple epochs.
            
            Args:
                train_data: List of (input, target) tuples
                val_data: Optional validation data
                epochs: Number of epochs to train
                verbose: Print progress
            
            Returns:
                Dictionary with 'train_losses' and 'val_losses'
            """
            for epoch in range(epochs):
                epoch_loss = 0.0
                for x, y in train_data:
                    loss = self.train_step(x, y)
                    epoch_loss += loss
                train_loss = epoch_loss / len(train_data)
                self.train_losses.append(train_loss)
                
                if val_data is not None:
                    val_loss = sum(self.validate(x, y) for x, y in val_data) / len(val_data)
                    self.val_losses.append(val_loss)
                    
                    if verbose and epoch % 10 == 0:
                        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                else:
                    if verbose and epoch % 10 == 0:
                        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
