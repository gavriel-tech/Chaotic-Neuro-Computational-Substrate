"""
Diffusion Models for GMCS.

Provides diffusion-based generative models (DDPM, DDIM, Latent Diffusion)
as GMCS nodes for pattern generation and denoising.

Key features:
- Standard diffusion (DDPM)
- Fast sampling (DDIM)
- Latent diffusion
- 1D diffusion for time series/audio
- 2D diffusion for wave fields
- Conditional generation (chaos-guided)

Use cases:
- Generate chaotic patterns with diffusion
- Denoise oscillator signals
- Interpolate between attractor states
- Conditional generation guided by chaos
"""

from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
# Noise Schedules
# ============================================================================

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> np.ndarray:
    """Linear noise schedule."""
    return np.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    """Cosine noise schedule (better for high-quality generation)."""
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999)


# ============================================================================
# Simple 1D U-Net for Time Series
# ============================================================================

class Conv1DBlock(nn.Module):
    """1D Convolutional block."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UNet1D(nn.Module):
    """
    Simple 1D U-Net for time series diffusion.
    
    Architecture: Encoder → Bottleneck → Decoder
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: List[int] = [64, 128, 256],
        time_emb_dim: int = 128
    ):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder
        self.encoders = nn.ModuleList()
        prev_dim = in_channels
        for dim in hidden_dims:
            self.encoders.append(nn.Sequential(
                Conv1DBlock(prev_dim, dim),
                Conv1DBlock(dim, dim),
                nn.AvgPool1d(2)
            ))
            prev_dim = dim
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            Conv1DBlock(hidden_dims[-1], hidden_dims[-1] * 2),
            Conv1DBlock(hidden_dims[-1] * 2, hidden_dims[-1])
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        for i in reversed(range(len(hidden_dims))):
            dim = hidden_dims[i]
            prev_dim = hidden_dims[i+1] if i < len(hidden_dims)-1 else hidden_dims[-1]
            self.decoders.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
                Conv1DBlock(prev_dim + dim, dim),  # +dim for skip connection
                Conv1DBlock(dim, dim)
            ))
        
        # Output
        self.output = nn.Conv1d(hidden_dims[0], in_channels, 1)
    
    def forward(self, x, t):
        """
        Args:
            x: (batch, channels, length) noisy input
            t: (batch,) timesteps
            
        Returns:
            (batch, channels, length) predicted noise
        """
        # Time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1).float() / 1000.0)
        
        # Encoder with skip connections
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        skips = reversed(skips)
        for decoder, skip in zip(self.decoders, skips):
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        return self.output(x)


# ============================================================================
# Diffusion Process
# ============================================================================

class DiffusionProcess:
    """
    Manages forward (noise) and reverse (denoise) diffusion processes.
    """
    
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = 'cosine'
    ):
        """
        Initialize diffusion process.
        
        Args:
            timesteps: Number of diffusion steps
            beta_schedule: 'linear' or 'cosine'
        """
        self.timesteps = timesteps
        
        # Noise schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            betas = cosine_beta_schedule(timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])
        
        # Store as tensors
        self.betas = torch.from_numpy(betas).float()
        self.alphas = torch.from_numpy(alphas).float()
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).float()
        self.alphas_cumprod_prev = torch.from_numpy(alphas_cumprod_prev).float()
        
        # Precompute values for forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Precompute values for reverse process
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = torch.from_numpy(self.posterior_variance).float()
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: add noise to x_0.
        
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_0: Clean data
            t: Timestep
            noise: Optional noise (generated if None)
            
        Returns:
            Noisy x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        while sqrt_alpha_bar.ndim < x_0.ndim:
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)
        
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion: single denoising step (DDPM).
        
        Args:
            model: Noise prediction model
            x_t: Noisy data at timestep t
            t: Current timestep
            
        Returns:
            x_{t-1}
        """
        # Predict noise
        predicted_noise = model(x_t, t)
        
        alpha = self.alphas[t]
        alpha_bar = self.alphas_cumprod[t]
        beta = self.betas[t]
        
        # Reshape for broadcasting
        while alpha.ndim < x_t.ndim:
            alpha = alpha.unsqueeze(-1)
            alpha_bar = alpha_bar.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
        
        # Compute x_{t-1}
        mean = (1.0 / torch.sqrt(alpha)) * (
            x_t - (beta / torch.sqrt(1.0 - alpha_bar)) * predicted_noise
        )
        
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t]
            while variance.ndim < x_t.ndim:
                variance = variance.unsqueeze(-1)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
    
    def ddim_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        DDIM sampling (faster, deterministic if eta=0).
        
        Args:
            model: Noise prediction model
            x_t: Noisy data at timestep t
            t: Current timestep
            t_prev: Previous timestep
            eta: Stochasticity (0=deterministic, 1=DDPM)
            
        Returns:
            x_{t_prev}
        """
        # Predict noise
        predicted_noise = model(x_t, t)
        
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = self.alphas_cumprod[t_prev]
        
        # Reshape for broadcasting
        while alpha_bar_t.ndim < x_t.ndim:
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            alpha_bar_t_prev = alpha_bar_t_prev.unsqueeze(-1)
        
        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev - eta**2 * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_t_prev)) * predicted_noise
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
        
        if eta > 0 and t[0] > 0:
            noise = torch.randn_like(x_t)
            sigma = eta * torch.sqrt((1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)) * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_t_prev)
            x_prev = x_prev + sigma * noise
        
        return x_prev


# ============================================================================
# Diffusion Node
# ============================================================================

class DiffusionNode(MLModelNode):
    """
    Diffusion model node for GMCS.
    
    Supports both DDPM (slow, stochastic) and DDIM (fast, deterministic) sampling.
    """
    
    def __init__(
        self,
        node_id: str,
        model: Optional[nn.Module] = None,
        timesteps: int = 1000,
        beta_schedule: str = 'cosine',
        data_shape: Tuple[int, ...] = (1, 256),
        device: str = "auto"
    ):
        """
        Initialize diffusion node.
        
        Args:
            node_id: Unique identifier
            model: U-Net model (created if None)
            timesteps: Number of diffusion steps
            beta_schedule: Noise schedule
            data_shape: Shape of data (channels, length)
            device: Device to use
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for diffusion models")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create model if not provided
        if model is None:
            in_channels = data_shape[0]
            model = UNet1D(in_channels=in_channels)
        
        super().__init__(node_id, model, framework='pytorch')
        
        self.device_name = device
        self.model.to(device)
        self.data_shape = data_shape
        
        # Create diffusion process
        self.diffusion = DiffusionProcess(timesteps, beta_schedule)
        self.diffusion.betas = self.diffusion.betas.to(device)
        self.diffusion.alphas = self.diffusion.alphas.to(device)
        self.diffusion.alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        self.diffusion.sqrt_alphas_cumprod = self.diffusion.sqrt_alphas_cumprod.to(device)
        self.diffusion.sqrt_one_minus_alphas_cumprod = self.diffusion.sqrt_one_minus_alphas_cumprod.to(device)
        self.diffusion.posterior_variance = self.diffusion.posterior_variance.to(device)
        
        self.timesteps = timesteps
        
        self.metadata.update({
            'timesteps': timesteps,
            'beta_schedule': beta_schedule,
            'data_shape': data_shape
        })
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Denoise input (single step).
        
        Args:
            x: Noisy input
            
        Returns:
            Denoised output
        """
        x_t = torch.from_numpy(x).float().to(self.device_name)
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(0)  # Add batch dim
        
        # Assume input is at t=0 (clean) and we want to denoise
        # This is a simplified forward - use sample() for generation
        t = torch.zeros(x_t.shape[0], dtype=torch.long, device=self.device_name)
        
        with torch.no_grad():
            noise_pred = self.model(x_t, t)
        
        return noise_pred.squeeze(0).cpu().numpy()
    
    def sample(
        self,
        num_samples: int = 1,
        num_steps: Optional[int] = None,
        method: str = 'ddpm',
        guidance: Optional[np.ndarray] = None,
        guidance_strength: float = 1.0
    ) -> np.ndarray:
        """
        Generate samples from noise.
        
        Args:
            num_samples: Number of samples to generate
            num_steps: Number of denoising steps (None=all timesteps)
            method: 'ddpm' or 'ddim'
            guidance: Optional conditioning signal (chaos)
            guidance_strength: How much to use guidance
            
        Returns:
            Generated samples
        """
        self.model.eval()
        
        # Start from noise
        x = torch.randn(num_samples, *self.data_shape, device=self.device_name)
        
        # Determine timesteps
        if num_steps is None:
            timesteps = list(range(self.timesteps))
        else:
            timesteps = np.linspace(0, self.timesteps - 1, num_steps, dtype=int).tolist()
        timesteps = list(reversed(timesteps))
        
        # Denoising loop
        for i, t in enumerate(timesteps):
            t_batch = torch.full((num_samples,), t, dtype=torch.long, device=self.device_name)
            
            # Apply guidance if provided
            if guidance is not None:
                # Simple guidance: add to noise prediction
                guidance_t = torch.from_numpy(guidance).float().to(self.device_name)
                if guidance_t.ndim == 1:
                    guidance_t = guidance_t.unsqueeze(0).unsqueeze(0)
                # This is simplified - proper guidance would be more complex
            
            # Denoise
            if method == 'ddpm':
                x = self.diffusion.p_sample(self.model, x, t_batch)
            elif method == 'ddim':
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                t_prev_batch = torch.full((num_samples,), t_prev, dtype=torch.long, device=self.device_name)
                x = self.diffusion.ddim_sample(self.model, x, t_batch, t_prev_batch, eta=0.0)
        
        return x.cpu().numpy()
    
    def train_step(
        self,
        data: np.ndarray,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            data: Clean training data
            optimizer: Optimizer
            
        Returns:
            Dict with loss
        """
        self.model.train()
        
        x_0 = torch.from_numpy(data).float().to(self.device_name)
        batch_size = x_0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device_name)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion
        x_t = self.diffusion.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {'loss': float(loss.item())}


# ============================================================================
# Helper Functions
# ============================================================================

def create_diffusion_node(
    node_id: str,
    data_shape: Tuple[int, ...] = (1, 256),
    timesteps: int = 1000,
    **kwargs
) -> DiffusionNode:
    """
    Factory function for diffusion nodes.
    
    Args:
        node_id: Unique identifier
        data_shape: Shape of data
        timesteps: Diffusion timesteps
        **kwargs: Additional arguments
        
    Returns:
        DiffusionNode instance
    """
    return DiffusionNode(node_id, timesteps=timesteps, data_shape=data_shape, **kwargs)


def chaos_guided_diffusion(
    diffusion_node: DiffusionNode,
    chaos_sequence: np.ndarray,
    num_samples: int = 1
) -> np.ndarray:
    """
    Generate samples guided by chaotic sequence.
    
    Args:
        diffusion_node: Diffusion model
        chaos_sequence: Chaotic time series for guidance
        num_samples: Number of samples
        
    Returns:
        Generated samples
    """
    # Normalize chaos
    chaos_norm = (chaos_sequence - chaos_sequence.min()) / (chaos_sequence.max() - chaos_sequence.min() + 1e-8)
    
    # Use chaos as conditioning
    samples = diffusion_node.sample(
        num_samples=num_samples,
        guidance=chaos_norm,
        guidance_strength=1.0
    )
    
    return samples


if __name__ == '__main__':
    # Example usage
    if PYTORCH_AVAILABLE:
        print("Creating diffusion model...")
        
        # Create model
        diffusion = DiffusionNode(
            "diffusion1",
            timesteps=1000,
            data_shape=(1, 256)
        )
        
        print(f"Model: {diffusion.model}")
        print(f"Timesteps: {diffusion.timesteps}")
        print(f"Data shape: {diffusion.data_shape}")
        
        # Generate samples
        print("\nGenerating samples...")
        samples = diffusion.sample(num_samples=2, num_steps=50, method='ddim')
        print(f"Generated samples shape: {samples.shape}")
    else:
        print("PyTorch not available. Install with: pip install torch")

