"""
PyTorch Integration for GMCS.

Enables loading PyTorch models and integrating them with the GMCS simulation
for bidirectional data flow and training.
"""

from typing import Optional, Dict, Any, Tuple, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    print("PyTorch not available. Install with: pip install torch")

import jax.numpy as jnp


# Stub classes when PyTorch is not available
if not PYTORCH_AVAILABLE:
    class PyTorchModelWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for this module. Install with: pip install torch")
    
    class GMCSPyTorchTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for this module. Install with: pip install torch")
else:
    # Real implementations when PyTorch is available
    class PyTorchModelWrapper:
        """
        Wrapper for PyTorch models to integrate with GMCS.
        
        Handles conversion between JAX and PyTorch tensors, model inference,
        and gradient flow.
        """
        
        def __init__(self, model: Optional[nn.Module] = None, device: str = "cuda"):
            """
            Initialize PyTorch model wrapper.
            
            Args:
                model: PyTorch model (nn.Module)
                device: Device to run model on ('cuda' or 'cpu')
            """
            self.model = model
            self.device = device if torch.cuda.is_available() else "cpu"
            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()
        
        def jax_to_torch(self, x: jnp.ndarray) -> torch.Tensor:
            """Convert JAX array to PyTorch tensor."""
            return torch.from_numpy(np.array(x)).to(self.device)
        
        def torch_to_jax(self, x: torch.Tensor) -> jnp.ndarray:
            """Convert PyTorch tensor to JAX array."""
            return jnp.array(x.detach().cpu().numpy())
        
        def forward(self, x: jnp.ndarray) -> jnp.ndarray:
            """Run forward pass through model."""
            if self.model is None:
                raise ValueError("No model loaded")
            
            x_torch = self.jax_to_torch(x)
            with torch.no_grad():
                y_torch = self.model(x_torch)
            return self.torch_to_jax(y_torch)
        
        def load_model(self, path: str):
            """Load model from checkpoint."""
            self.model = torch.load(path, map_location=self.device)
            self.model.eval()
        
        def save_model(self, path: str):
            """Save model to checkpoint."""
            if self.model is None:
                raise ValueError("No model to save")
            torch.save(self.model, path)


    class GMCSPyTorchTrainer:
        """
        Trainer for PyTorch models using GMCS data.
        """
        
        def __init__(
            self,
            model: nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            loss_fn: Optional[callable] = None,
            device: str = "cuda"
        ):
            """
            Initialize trainer.
            
            Args:
                model: PyTorch model to train
                optimizer: PyTorch optimizer (default: Adam)
                loss_fn: Loss function (default: MSE)
                device: Device to train on
            """
            self.model = model
            self.device = device if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
            self.loss_fn = loss_fn or nn.MSELoss()
            
            self.train_losses = []
            self.val_losses = []
        
        def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
            """Single training step."""
            self.model.train()
            
            x_torch = torch.from_numpy(x).to(self.device)
            y_torch = torch.from_numpy(y).to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(x_torch)
            loss = self.loss_fn(y_pred, y_torch)
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        
        def validate(self, x: np.ndarray, y: np.ndarray) -> float:
            """Validation step."""
            self.model.eval()
            
            x_torch = torch.from_numpy(x).to(self.device)
            y_torch = torch.from_numpy(y).to(self.device)
            
            with torch.no_grad():
                y_pred = self.model(x_torch)
                loss = self.loss_fn(y_pred, y_torch)
            
            return loss.item()
        
        def train_epoch(self, train_data: List[Tuple[np.ndarray, np.ndarray]]) -> float:
            """Train for one epoch."""
            epoch_loss = 0.0
            for x, y in train_data:
                loss = self.train_step(x, y)
                epoch_loss += loss
            return epoch_loss / len(train_data)
        
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
                train_loss = self.train_epoch(train_data)
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
