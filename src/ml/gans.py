"""
Generative Adversarial Networks (GANs) for GMCS.

Provides GAN architectures for adversarial generation, using chaos
as a noise source for enhanced diversity.

Key features:
- DCGAN architecture
- Chaos-driven noise
- Style transfer
- Conditional generation
- Wasserstein GAN variant

Use cases:
- Generate patterns from chaos
- Style transfer with oscillators
- Adversarial training
- Novel pattern discovery
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from .ml_nodes import MLModelNode


# ============================================================================
# Generator Networks
# ============================================================================

class Generator1D(nn.Module):
    """
    1D Generator for time series generation.
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        output_length: int = 256,
        hidden_dims: List[int] = [256, 512, 256, 128]
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_length = output_length
        
        # Initial projection
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * (output_length // 16))
        
        # Upsampling blocks
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
                nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(inplace=True)
            ])
        
        # Final layer
        layers.extend([
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.Conv1d(hidden_dims[-1], 1, kernel_size=5, padding=2),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim) noise
            
        Returns:
            (batch, 1, length) generated sequence
        """
        x = self.fc(z)
        x = x.view(x.size(0), -1, self.output_length // 16)
        x = self.model(x)
        return x


class Discriminator1D(nn.Module):
    """
    1D Discriminator for time series.
    """
    
    def __init__(
        self,
        input_length: int = 256,
        hidden_dims: List[int] = [64, 128, 256, 512]
    ):
        super().__init__()
        
        # Downsampling blocks
        layers = []
        in_channels = 1
        
        for dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_channels, dim, kernel_size=5, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            in_channels = dim
        
        self.features = nn.Sequential(*layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 1, length) input sequence
            
        Returns:
            (batch, 1) real/fake probability
        """
        features = self.features(x)
        return self.classifier(features)


# ============================================================================
# GAN Node
# ============================================================================

class GANNode(MLModelNode):
    """
    GAN node for GMCS.
    
    Implements adversarial training between generator and discriminator.
    Can use chaos as noise source for enhanced diversity.
    """
    
    def __init__(
        self,
        node_id: str,
        latent_dim: int = 100,
        output_length: int = 256,
        generator: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        device: str = "auto",
        loss_type: str = 'bce'  # 'bce' or 'wasserstein'
    ):
        """
        Initialize GAN node.
        
        Args:
            node_id: Unique identifier
            latent_dim: Dimension of latent noise
            output_length: Length of generated sequence
            generator: Generator network (created if None)
            discriminator: Discriminator network (created if None)
            device: Device to use
            loss_type: 'bce' (standard) or 'wasserstein'
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for GANs")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create networks if not provided
        if generator is None:
            generator = Generator1D(latent_dim, output_length)
        
        if discriminator is None:
            discriminator = Discriminator1D(output_length)
        
        # Initialize with generator (for forward pass)
        super().__init__(node_id, generator, framework='pytorch')
        
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.device_name = device
        self.loss_type = loss_type
        
        self.generator.to(device)
        self.discriminator.to(device)
        
        # Create optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.metadata.update({
            'latent_dim': latent_dim,
            'output_length': output_length,
            'loss_type': loss_type
        })
    
    def forward(self, noise: Optional[np.ndarray] = None, num_samples: int = 1) -> np.ndarray:
        """
        Generate samples.
        
        Args:
            noise: Optional noise (generated if None)
            num_samples: Number of samples if noise is None
            
        Returns:
            Generated samples
        """
        self.generator.eval()
        
        if noise is None:
            noise = np.random.randn(num_samples, self.latent_dim)
        
        z = torch.from_numpy(noise).float().to(self.device_name)
        
        with torch.no_grad():
            fake = self.generator(z)
        
        return fake.cpu().numpy()
    
    def generate_from_chaos(self, chaos_sequence: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """
        Generate using chaos as noise source.
        
        Args:
            chaos_sequence: Chaotic time series
            num_samples: Number of samples
            
        Returns:
            Generated samples
        """
        # Convert chaos to latent noise
        # Simple approach: reshape and normalize
        chaos_flat = chaos_sequence.flatten()
        
        # Resample to latent_dim
        indices = np.linspace(0, len(chaos_flat) - 1, self.latent_dim * num_samples, dtype=int)
        noise = chaos_flat[indices].reshape(num_samples, self.latent_dim)
        
        # Normalize
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)
        
        return self.forward(noise)
    
    def train_step(
        self,
        real_data: np.ndarray,
        chaos_noise: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Single GAN training step.
        
        Args:
            real_data: Real training data
            chaos_noise: Optional chaos for noise (random if None)
            
        Returns:
            Dict with losses
        """
        self.generator.train()
        self.discriminator.train()
        
        real = torch.from_numpy(real_data).float().to(self.device_name)
        batch_size = real.shape[0]
        
        # =================
        # Train Discriminator
        # =================
        
        # Real samples
        d_real = self.discriminator(real)
        
        # Fake samples
        if chaos_noise is not None:
            z = torch.from_numpy(chaos_noise).float().to(self.device_name)
        else:
            z = torch.randn(batch_size, self.latent_dim, device=self.device_name)
        
        fake = self.generator(z).detach()  # Detach to not train generator
        d_fake = self.discriminator(fake)
        
        # Discriminator loss
        if self.loss_type == 'bce':
            real_labels = torch.ones_like(d_real)
            fake_labels = torch.zeros_like(d_fake)
            d_loss_real = F.binary_cross_entropy(d_real, real_labels)
            d_loss_fake = F.binary_cross_entropy(d_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake
        else:  # Wasserstein
            d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        
        # Update discriminator
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()
        
        # Clip weights for Wasserstein
        if self.loss_type == 'wasserstein':
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
        
        # =================
        # Train Generator
        # =================
        
        # Generate new fakes
        if chaos_noise is not None:
            z = torch.from_numpy(chaos_noise).float().to(self.device_name)
        else:
            z = torch.randn(batch_size, self.latent_dim, device=self.device_name)
        
        fake = self.generator(z)
        d_fake = self.discriminator(fake)
        
        # Generator loss (fool discriminator)
        if self.loss_type == 'bce':
            real_labels = torch.ones_like(d_fake)
            g_loss = F.binary_cross_entropy(d_fake, real_labels)
        else:  # Wasserstein
            g_loss = -torch.mean(d_fake)
        
        # Update generator
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': float(d_loss.item()),
            'g_loss': float(g_loss.item()),
            'd_real': float(d_real.mean().item()),
            'd_fake': float(d_fake.mean().item())
        }
    
    def save_checkpoint(self, path: str):
        """Save GAN checkpoint."""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'metadata': self.metadata
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load GAN checkpoint."""
        checkpoint = torch.load(path, map_location=self.device_name)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.metadata = checkpoint['metadata']


# ============================================================================
# Conditional GAN
# ============================================================================

class ConditionalGenerator1D(nn.Module):
    """
    Conditional generator (can condition on labels or chaos).
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        condition_dim: int = 10,
        output_length: int = 256,
        hidden_dims: List[int] = [256, 512, 256, 128]
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_length = output_length
        
        # Combined input
        input_dim = latent_dim + condition_dim
        
        # Initial projection
        self.fc = nn.Linear(input_dim, hidden_dims[0] * (output_length // 16))
        
        # Upsampling blocks
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
                nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(inplace=True)
            ])
        
        # Final layer
        layers.extend([
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.Conv1d(hidden_dims[-1], 1, kernel_size=5, padding=2),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z, condition):
        """
        Args:
            z: (batch, latent_dim) noise
            condition: (batch, condition_dim) condition
            
        Returns:
            (batch, 1, length) generated sequence
        """
        x = torch.cat([z, condition], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, self.output_length // 16)
        x = self.model(x)
        return x


class ConditionalGANNode(GANNode):
    """
    Conditional GAN that can generate based on chaos state.
    """
    
    def __init__(
        self,
        node_id: str,
        latent_dim: int = 100,
        condition_dim: int = 10,
        output_length: int = 256,
        device: str = "auto"
    ):
        """
        Initialize conditional GAN.
        
        Args:
            node_id: Unique identifier
            latent_dim: Noise dimension
            condition_dim: Condition dimension
            output_length: Output length
            device: Device to use
        """
        generator = ConditionalGenerator1D(latent_dim, condition_dim, output_length)
        discriminator = Discriminator1D(output_length)
        
        super().__init__(node_id, latent_dim, output_length, generator, discriminator, device)
        
        self.condition_dim = condition_dim
    
    def forward(
        self,
        noise: Optional[np.ndarray] = None,
        condition: Optional[np.ndarray] = None,
        num_samples: int = 1
    ) -> np.ndarray:
        """
        Generate conditioned on input.
        
        Args:
            noise: Optional noise
            condition: Conditioning vector
            num_samples: Number of samples
            
        Returns:
            Generated samples
        """
        self.generator.eval()
        
        if noise is None:
            noise = np.random.randn(num_samples, self.latent_dim)
        
        if condition is None:
            condition = np.random.randn(num_samples, self.condition_dim)
        
        z = torch.from_numpy(noise).float().to(self.device_name)
        c = torch.from_numpy(condition).float().to(self.device_name)
        
        with torch.no_grad():
            fake = self.generator(z, c)
        
        return fake.cpu().numpy()


# ============================================================================
# Helper Functions
# ============================================================================

def create_gan_node(
    node_id: str,
    gan_type: str = 'standard',
    **kwargs
) -> GANNode:
    """
    Factory function for GAN nodes.
    
    Args:
        node_id: Unique identifier
        gan_type: 'standard' or 'conditional'
        **kwargs: Additional arguments
        
    Returns:
        GANNode instance
    """
    if gan_type == 'standard':
        return GANNode(node_id, **kwargs)
    elif gan_type == 'conditional':
        return ConditionalGANNode(node_id, **kwargs)
    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")


def train_gan_on_chaos(
    gan: GANNode,
    chaos_data: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 32,
    use_chaos_noise: bool = True
) -> Dict[str, List[float]]:
    """
    Train GAN on chaotic data.
    
    Args:
        gan: GAN node
        chaos_data: Chaotic time series data
        n_epochs: Number of epochs
        batch_size: Batch size
        use_chaos_noise: Use chaos as GAN noise
        
    Returns:
        Training history
    """
    history = {'d_loss': [], 'g_loss': [], 'd_real': [], 'd_fake': []}
    
    n_samples = len(chaos_data)
    
    for epoch in range(n_epochs):
        epoch_d_loss = []
        epoch_g_loss = []
        epoch_d_real = []
        epoch_d_fake = []
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples - batch_size + 1, batch_size):
            batch_indices = indices[i:i+batch_size]
            real_batch = chaos_data[batch_indices]
            
            # Optionally use chaos as noise
            if use_chaos_noise:
                # Use different part of chaos as noise
                noise_indices = np.random.randint(0, n_samples, batch_size)
                chaos_noise_batch = chaos_data[noise_indices]
                # Reshape to latent dim
                chaos_noise_batch = chaos_noise_batch[:, :gan.latent_dim]
            else:
                chaos_noise_batch = None
            
            # Train step
            losses = gan.train_step(real_batch, chaos_noise_batch)
            
            epoch_d_loss.append(losses['d_loss'])
            epoch_g_loss.append(losses['g_loss'])
            epoch_d_real.append(losses['d_real'])
            epoch_d_fake.append(losses['d_fake'])
        
        # Record epoch metrics
        history['d_loss'].append(np.mean(epoch_d_loss))
        history['g_loss'].append(np.mean(epoch_g_loss))
        history['d_real'].append(np.mean(epoch_d_real))
        history['d_fake'].append(np.mean(epoch_d_fake))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"D_loss={history['d_loss'][-1]:.4f}, "
                  f"G_loss={history['g_loss'][-1]:.4f}")
    
    return history


if __name__ == '__main__':
    # Example usage
    if PYTORCH_AVAILABLE:
        print("Creating GAN...")
        
        gan = GANNode(
            "gan1",
            latent_dim=100,
            output_length=256
        )
        
        print(f"Generator: {gan.generator}")
        print(f"Discriminator: {gan.discriminator}")
        
        # Generate samples
        print("\nGenerating samples...")
        samples = gan.forward(num_samples=4)
        print(f"Generated samples shape: {samples.shape}")
        
        # Create chaos noise
        chaos_noise = np.random.randn(4, 100)
        chaos_samples = gan.generate_from_chaos(chaos_noise)
        print(f"Chaos-guided samples shape: {chaos_samples.shape}")
    else:
        print("PyTorch not available. Install with: pip install torch")

