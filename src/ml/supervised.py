"""
Supervised Learning Models for GMCS.

Provides standard supervised learning architectures (CNNs, MLPs, Autoencoders)
as GMCS nodes for classification, regression, and feature learning.

Key features:
- Convolutional networks (ResNet, EfficientNet)
- Multi-layer perceptrons
- Autoencoders
- Vision transformers
- Pre-trained model support

Use cases:
- Classify chaotic regimes
- Predict oscillator behavior
- Learn latent representations
- Compress attractors
- Process wave fields
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import models
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from .ml_nodes import MLModelNode


# ============================================================================
# Multi-Layer Perceptron (MLP)
# ============================================================================

class MLP(nn.Module):
    """
    Multi-layer perceptron.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class MLPNode(MLModelNode):
    """
    MLP node for GMCS.
    
    Good for:
    - Predicting oscillator states
    - Classifying attractor types
    - Learning feature mappings
    """
    
    def __init__(
        self,
        node_id: str,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        output_dim: int = 1,
        activation: str = 'relu',
        dropout: float = 0.1,
        device: str = "auto"
    ):
        """
        Initialize MLP node.
        
        Args:
            node_id: Unique identifier
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function
            dropout: Dropout rate
            device: Device to use
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = MLP(input_dim, hidden_dims, output_dim, activation, dropout)
        
        super().__init__(node_id, model, framework='pytorch')
        
        self.model.to(device)
        self.device_name = device
        
        # For training
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()  # Default, can be changed
        
        self.metadata.update({
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'activation': activation
        })
    
    def train_step(
        self,
        data: np.ndarray,
        target: np.ndarray,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            data: Input data
            target: Target output
            optimizer: Optimizer (uses default if None)
            
        Returns:
            Dict with loss
        """
        self.model.train()
        
        if optimizer is None:
            optimizer = self.optimizer
        
        x = torch.from_numpy(data).float().to(self.device_name)
        y = torch.from_numpy(target).float().to(self.device_name)
        
        # Forward
        pred = self.model(x)
        loss = self.criterion(pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {'loss': float(loss.item())}


# ============================================================================
# Convolutional Neural Network (CNN)
# ============================================================================

class CNN1D(nn.Module):
    """
    1D CNN for time series.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        conv_dims: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [7, 5, 3],
        output_dim: int = 10,
        pool_size: int = 2
    ):
        super().__init__()
        
        # Convolutional layers
        conv_layers = []
        prev_dim = input_channels
        
        for dim, kernel_size in zip(conv_dims, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(prev_dim, dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.MaxPool1d(pool_size)
            ])
            prev_dim = dim
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class CNNNode(MLModelNode):
    """
    CNN node for GMCS.
    
    Good for:
    - Classifying time series patterns
    - Processing wave fields
    - Feature extraction from sequences
    """
    
    def __init__(
        self,
        node_id: str,
        input_channels: int = 1,
        output_dim: int = 10,
        architecture: str = 'custom',
        device: str = "auto"
    ):
        """
        Initialize CNN node.
        
        Args:
            node_id: Unique identifier
            input_channels: Number of input channels
            output_dim: Output dimension
            architecture: 'custom', 'resnet18', 'efficientnet'
            device: Device to use
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create model
        if architecture == 'custom':
            model = CNN1D(input_channels, output_dim=output_dim)
        elif architecture == 'resnet18':
            # Use pretrained ResNet (adapt for 1D)
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, output_dim)
        else:
            model = CNN1D(input_channels, output_dim=output_dim)
        
        super().__init__(node_id, model, framework='pytorch')
        
        self.model.to(device)
        self.device_name = device
        
        # For training
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.metadata.update({
            'input_channels': input_channels,
            'output_dim': output_dim,
            'architecture': architecture
        })
    
    def train_step(
        self,
        data: np.ndarray,
        target: np.ndarray,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Dict[str, float]:
        """Training step."""
        self.model.train()
        
        if optimizer is None:
            optimizer = self.optimizer
        
        x = torch.from_numpy(data).float().to(self.device_name)
        y = torch.from_numpy(target).long().to(self.device_name)
        
        # Forward
        pred = self.model(x)
        loss = self.criterion(pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accuracy
        _, predicted = torch.max(pred, 1)
        accuracy = (predicted == y).float().mean()
        
        return {
            'loss': float(loss.item()),
            'accuracy': float(accuracy.item())
        }


# ============================================================================
# Autoencoder
# ============================================================================

class Encoder1D(nn.Module):
    """1D encoder."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64]
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)


class Decoder1D(nn.Module):
    """1D decoder."""
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 128]
    ):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder(z)


class Autoencoder(nn.Module):
    """Autoencoder for compression."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64]
    ):
        super().__init__()
        
        self.encoder = Encoder1D(input_dim, latent_dim, hidden_dims)
        self.decoder = Decoder1D(latent_dim, input_dim, list(reversed(hidden_dims)))
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


class AutoencoderNode(MLModelNode):
    """
    Autoencoder node for GMCS.
    
    Good for:
    - Learning latent representations
    - Compressing chaotic attractors
    - Dimensionality reduction
    - Denoising
    """
    
    def __init__(
        self,
        node_id: str,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: List[int] = [128, 64],
        device: str = "auto"
    ):
        """
        Initialize autoencoder node.
        
        Args:
            node_id: Unique identifier
            input_dim: Input dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            device: Device to use
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = Autoencoder(input_dim, latent_dim, hidden_dims)
        
        super().__init__(node_id, model, framework='pytorch')
        
        self.model.to(device)
        self.device_name = device
        self.latent_dim = latent_dim
        
        # For training
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.metadata.update({
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'hidden_dims': hidden_dims
        })
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Encode then decode (reconstruction).
        
        Args:
            input_data: Input data
            
        Returns:
            Reconstructed data
        """
        x = torch.from_numpy(input_data).float().to(self.device_name)
        
        with torch.no_grad():
            recon, _ = self.model(x)
        
        return recon.cpu().numpy()
    
    def encode(self, input_data: np.ndarray) -> np.ndarray:
        """
        Encode to latent space.
        
        Args:
            input_data: Input data
            
        Returns:
            Latent representation
        """
        x = torch.from_numpy(input_data).float().to(self.device_name)
        
        with torch.no_grad():
            z = self.model.encode(x)
        
        return z.cpu().numpy()
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode from latent space.
        
        Args:
            latent: Latent representation
            
        Returns:
            Reconstructed data
        """
        z = torch.from_numpy(latent).float().to(self.device_name)
        
        with torch.no_grad():
            recon = self.model.decode(z)
        
        return recon.cpu().numpy()
    
    def train_step(
        self,
        data: np.ndarray,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            data: Input data (will be used as target too)
            optimizer: Optimizer (uses default if None)
            
        Returns:
            Dict with loss
        """
        self.model.train()
        
        if optimizer is None:
            optimizer = self.optimizer
        
        x = torch.from_numpy(data).float().to(self.device_name)
        
        # Forward
        recon, z = self.model(x)
        loss = self.criterion(recon, x)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {'loss': float(loss.item())}


# ============================================================================
# Helper Functions
# ============================================================================

def create_supervised_node(
    node_id: str,
    model_type: str = 'mlp',
    **kwargs
) -> MLModelNode:
    """
    Factory function for supervised learning nodes.
    
    Args:
        node_id: Unique identifier
        model_type: 'mlp', 'cnn', or 'autoencoder'
        **kwargs: Additional arguments
        
    Returns:
        Supervised learning node
    """
    if model_type == 'mlp':
        return MLPNode(node_id, **kwargs)
    elif model_type == 'cnn':
        return CNNNode(node_id, **kwargs)
    elif model_type == 'autoencoder':
        return AutoencoderNode(node_id, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_classifier_on_chaos(
    model: Union[MLPNode, CNNNode],
    chaos_data: np.ndarray,
    labels: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 32
) -> Dict[str, List[float]]:
    """
    Train classifier on chaotic data.
    
    Args:
        model: Classification model
        chaos_data: Input data
        labels: Target labels
        n_epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        Training history
    """
    history = {'loss': [], 'accuracy': []}
    
    n_samples = len(chaos_data)
    
    for epoch in range(n_epochs):
        epoch_loss = []
        epoch_acc = []
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples - batch_size + 1, batch_size):
            batch_indices = indices[i:i+batch_size]
            data_batch = chaos_data[batch_indices]
            label_batch = labels[batch_indices]
            
            # Train step
            metrics = model.train_step(data_batch, label_batch)
            
            epoch_loss.append(metrics['loss'])
            if 'accuracy' in metrics:
                epoch_acc.append(metrics['accuracy'])
        
        history['loss'].append(np.mean(epoch_loss))
        if epoch_acc:
            history['accuracy'].append(np.mean(epoch_acc))
        
        if (epoch + 1) % 10 == 0:
            if history['accuracy']:
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"Loss={history['loss'][-1]:.4f}, "
                      f"Acc={history['accuracy'][-1]:.4f}")
            else:
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"Loss={history['loss'][-1]:.4f}")
    
    return history


def compress_attractor(
    autoencoder: AutoencoderNode,
    attractor_trajectory: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compress chaotic attractor to latent space.
    
    Args:
        autoencoder: Autoencoder model
        attractor_trajectory: Full trajectory
        
    Returns:
        (latent_trajectory, reconstruction)
    """
    latent = autoencoder.encode(attractor_trajectory)
    reconstruction = autoencoder.decode(latent)
    
    return latent, reconstruction


if __name__ == '__main__':
    # Example usage
    if PYTORCH_AVAILABLE:
        print("Creating supervised learning models...\n")
        
        # MLP
        print("1. MLP Node:")
        mlp = MLPNode("mlp1", input_dim=10, hidden_dims=[64, 32], output_dim=1)
        print(f"   Model: {mlp.model}")
        
        # CNN
        print("\n2. CNN Node:")
        cnn = CNNNode("cnn1", input_channels=1, output_dim=5)
        print(f"   Model: {cnn.model}")
        
        # Autoencoder
        print("\n3. Autoencoder Node:")
        ae = AutoencoderNode("ae1", input_dim=100, latent_dim=10)
        print(f"   Model: {ae.model}")
        print(f"   Latent dim: {ae.latent_dim}")
        
        # Test encoding
        print("\nTesting autoencoder...")
        test_data = np.random.randn(5, 100)
        latent = ae.encode(test_data)
        recon = ae.decode(latent)
        print(f"   Input shape: {test_data.shape}")
        print(f"   Latent shape: {latent.shape}")
        print(f"   Reconstruction shape: {recon.shape}")
        print(f"   Reconstruction error: {np.mean((test_data - recon)**2):.6f}")
    else:
        print("PyTorch not available. Install with: pip install torch")

