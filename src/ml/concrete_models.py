"""
Concrete ML Model Implementations for GMCS.

This module provides domain-specific ML models used across various presets:
- Music Analysis: GenreClassifier, MusicTransformer
- RL: PPOAgent, ValueFunction
- Generative: PixelArtGAN, CodeGenerator
- Scientific: LogicGateDetector, PerformancePredictor, EfficiencyPredictor
- Neuroscience: CognitiveStateDecoder
- Chemistry: BindingPredictor
- General: MLPerformanceSelector

All models support:
- PyTorch backend (primary)
- Training from scratch with random initialization
- State serialization for checkpointing
- Integration with GMCS node graph
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod

# Framework imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam, RMSprop
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("[ConcreteModels] PyTorch not available - ML models will be disabled")

try:
    from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# Base Classes
# ============================================================================

class BaseMLModel(ABC):
    """Base class for all concrete ML models."""
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        self.node_id = node_id
        self.config = config
        self.model = None
        self.optimizer = None
        self.training_step = 0
        
    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward pass."""
        pass
    
    @abstractmethod
    def get_loss(self, predictions: Any, targets: Any) -> float:
        """Compute loss for training."""
        pass
    
    def train_step(self, x: Any, y: Any) -> float:
        """Perform one training step."""
        if not PYTORCH_AVAILABLE or self.model is None:
            return 0.0
        
        self.model.train()
        self.optimizer.zero_grad()
        
        predictions = self.forward(x)
        loss = self.get_loss(predictions, y)
        
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        return float(loss.item())
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        if not PYTORCH_AVAILABLE or self.model is None:
            return
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'training_step': self.training_step,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        if not PYTORCH_AVAILABLE or self.model is None:
            return
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)


# ============================================================================
# Music Analysis Models
# ============================================================================

class GenreClassifier(BaseMLModel):
    """
    2D CNN for music genre classification from spectrograms.
    
    Input: Mel-spectrogram (1, 128, 128)
    Output: 10 genre probabilities
    """
    
    DEFAULT_GENRES = ['rock', 'pop', 'classical', 'jazz', 'electronic', 
                      'hip_hop', 'country', 'blues', 'reggae', 'metal']
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not PYTORCH_AVAILABLE:
            return
        
        self.genres = config.get('class_names', self.DEFAULT_GENRES)
        self.num_classes = len(self.genres)
        
        # Build 2D CNN
        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            
            # Flatten and classify
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1)
        )
        
        self.optimizer = Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-3))
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """Forward pass."""
        if not PYTORCH_AVAILABLE:
            return np.zeros(self.num_classes)
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Ensure shape is (batch, 1, 128, 128)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        
        return output
    
    def get_loss(self, predictions: 'torch.Tensor', targets: 'torch.Tensor') -> 'torch.Tensor':
        """Compute cross-entropy loss."""
        return self.criterion(predictions, targets)
    
    def process(self, spectrogram: np.ndarray) -> Dict[str, Any]:
        """Process spectrogram and return genre predictions."""
        probs = self.forward(spectrogram)
        
        if PYTORCH_AVAILABLE:
            probs_np = probs.cpu().numpy().flatten()
        else:
            probs_np = probs
        
        predicted_idx = int(np.argmax(probs_np))
        
        return {
            'genre': self.genres[predicted_idx],
            'probabilities': probs_np.tolist(),
            'confidence': float(probs_np[predicted_idx]),
            'top_3': [(self.genres[i], float(probs_np[i])) 
                      for i in np.argsort(probs_np)[-3:][::-1]]
        }


class MusicTransformer(BaseMLModel):
    """
    Transformer encoder for music feature analysis.
    
    Input: Sequence of audio features (seq_len, 512)
    Output: Embeddings for downstream tasks
    """
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not PYTORCH_AVAILABLE:
            return
        
        self.d_model = config.get('d_model', 512)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dim_feedforward = config.get('dim_feedforward', 2048)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        
        self.model = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.optimizer = Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-4))
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """Forward pass."""
        if not PYTORCH_AVAILABLE:
            return np.zeros((x.shape[0], self.d_model)) if isinstance(x, np.ndarray) else x
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Ensure shape is (batch, seq_len, d_model)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        
        return output
    
    def get_loss(self, predictions: 'torch.Tensor', targets: 'torch.Tensor') -> 'torch.Tensor':
        """MSE loss for reconstruction."""
        return F.mse_loss(predictions, targets)
    
    def process(self, features: np.ndarray) -> Dict[str, Any]:
        """Process audio features and return embeddings."""
        embeddings = self.forward(features)
        
        if PYTORCH_AVAILABLE:
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings
        
        # Global pooling for sequence-level representation
        global_embedding = np.mean(embeddings_np, axis=1) if embeddings_np.ndim == 3 else embeddings_np
        
        return {
            'embeddings': embeddings_np,
            'global_embedding': global_embedding,
            'sequence_length': embeddings_np.shape[1] if embeddings_np.ndim == 3 else 1
        }


# ============================================================================
# Reinforcement Learning Models
# ============================================================================

class PPOAgent(BaseMLModel):
    """
    Proximal Policy Optimization (PPO) agent with actor-critic architecture.
    
    Input: State vector (variable dimension)
    Output: Action probabilities (actor) and state value (critic)
    """
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not PYTORCH_AVAILABLE:
            return
        
        self.state_dim = config.get('state_dim', 64)
        self.action_dim = config.get('action_dim', 4)
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        self.model = nn.ModuleDict({'actor': self.actor, 'critic': self.critic})
        
        self.optimizer = Adam(self.model.parameters(), lr=config.get('learning_rate', 3e-4))
        
        # PPO hyperparameters
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
    
    def forward(self, state: Union[np.ndarray, 'torch.Tensor']) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Forward pass returns (action_probs, state_value)."""
        if not PYTORCH_AVAILABLE:
            return (np.ones(self.action_dim) / self.action_dim, 0.0)
        
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            action_probs = self.actor(state)
            state_value = self.critic(state)
        
        return action_probs, state_value
    
    def get_loss(self, predictions: Dict[str, 'torch.Tensor'], 
                 data: Dict[str, 'torch.Tensor']) -> 'torch.Tensor':
        """Compute PPO loss."""
        if not PYTORCH_AVAILABLE:
            return torch.tensor(0.0)
        
        # Extract data
        states = data['states']
        actions = data['actions']
        old_log_probs = data['old_log_probs']
        advantages = data['advantages']
        returns = data['returns']
        
        # Forward pass
        action_probs = self.actor(states)
        state_values = self.critic(states).squeeze()
        
        # Compute log probabilities
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # PPO clipped surrogate objective
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(state_values, returns)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        return loss
    
    def process(self, state: np.ndarray) -> Dict[str, Any]:
        """Process state and return action distribution."""
        action_probs, state_value = self.forward(state)
        
        if PYTORCH_AVAILABLE:
            action_probs_np = action_probs.cpu().numpy().flatten()
            state_value_np = float(state_value.cpu().numpy())
        else:
            action_probs_np = action_probs
            state_value_np = state_value
        
        # Sample action
        action = np.random.choice(self.action_dim, p=action_probs_np)
        
        return {
            'action': int(action),
            'action_probs': action_probs_np.tolist(),
            'state_value': state_value_np,
            'entropy': float(-np.sum(action_probs_np * np.log(action_probs_np + 1e-10)))
        }


class ValueFunction(BaseMLModel):
    """
    MLP-based value function for state value estimation.
    
    Input: State features
    Output: State value estimate
    """
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not PYTORCH_AVAILABLE:
            return
        
        self.input_dim = config.get('input_dim', 64)
        self.hidden_dims = config.get('hidden_dims', [256, 256, 128])
        
        # Build MLP
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        self.optimizer = Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-3))
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """Forward pass."""
        if not PYTORCH_AVAILABLE:
            return np.array([0.0])
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        
        return output
    
    def get_loss(self, predictions: 'torch.Tensor', targets: 'torch.Tensor') -> 'torch.Tensor':
        """MSE loss."""
        return F.mse_loss(predictions.squeeze(), targets)
    
    def process(self, state: np.ndarray) -> Dict[str, Any]:
        """Estimate value of state."""
        value = self.forward(state)
        
        if PYTORCH_AVAILABLE:
            value_np = float(value.cpu().numpy())
        else:
            value_np = float(value)
        
        return {
            'value': value_np,
            'state_dim': self.input_dim
        }


# ============================================================================
# Generative Models
# ============================================================================

class PixelArtGAN(BaseMLModel):
    """
    GAN generator for 32x32 pixel art generation.
    
    Input: Latent vector (128D)
    Output: 32x32 RGB image
    """
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not PYTORCH_AVAILABLE:
            return
        
        self.latent_dim = config.get('latent_dim', 128)
        self.img_size = config.get('img_size', 32)
        self.channels = config.get('channels', 3)
        
        # Generator network
        self.generator = nn.Sequential(
            # Project and reshape
            nn.Linear(self.latent_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            
            # Upsample to 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Upsample to 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Upsample to 32x32
            nn.ConvTranspose2d(64, self.channels, 4, 2, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Discriminator (for training)
        self.discriminator = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )
        
        self.model = nn.ModuleDict({
            'generator': self.generator,
            'discriminator': self.discriminator
        })
        
        self.optimizer_g = Adam(self.generator.parameters(), lr=config.get('lr_g', 2e-4), betas=(0.5, 0.999))
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=config.get('lr_d', 2e-4), betas=(0.5, 0.999))
    
    def forward(self, z: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """Generate image from latent vector."""
        if not PYTORCH_AVAILABLE:
            return np.zeros((self.channels, self.img_size, self.img_size))
        
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        
        if z.ndim == 1:
            z = z.unsqueeze(0)
        
        self.generator.eval()
        with torch.no_grad():
            image = self.generator(z)
        
        return image
    
    def get_loss(self, real_images: 'torch.Tensor', fake_images: 'torch.Tensor') -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Compute GAN losses."""
        if not PYTORCH_AVAILABLE:
            return torch.tensor(0.0), torch.tensor(0.0)
        
        # Discriminator loss
        real_preds = self.discriminator(real_images)
        fake_preds = self.discriminator(fake_images.detach())
        
        d_loss = -torch.mean(torch.log(real_preds + 1e-8) + torch.log(1 - fake_preds + 1e-8))
        
        # Generator loss
        fake_preds_g = self.discriminator(fake_images)
        g_loss = -torch.mean(torch.log(fake_preds_g + 1e-8))
        
        return g_loss, d_loss
    
    def process(self, latent: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate pixel art image."""
        if latent is None:
            latent = np.random.randn(self.latent_dim)
        
        image = self.forward(latent)
        
        if PYTORCH_AVAILABLE:
            image_np = image.cpu().numpy().squeeze()
        else:
            image_np = image
        
        # Convert from [-1, 1] to [0, 255]
        image_uint8 = ((image_np + 1) * 127.5).astype(np.uint8)
        
        return {
            'image': image_uint8,
            'latent': latent.tolist() if isinstance(latent, np.ndarray) else latent,
            'size': (self.img_size, self.img_size)
        }


class CodeGenerator(BaseMLModel):
    """
    Transformer-based code generator using pretrained GPT-2.
    
    Input: Problem specification (text)
    Output: Generated code
    """
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        self.max_length = config.get('max_length', 512)
        self.temperature = config.get('temperature', 0.8)
        
        if not TRANSFORMERS_AVAILABLE or not PYTORCH_AVAILABLE:
            print("[CodeGenerator] Transformers library not available, using stub")
            self.model = None
            self.tokenizer = None
            return
        
        # Use pretrained GPT-2 (small)
        model_name = config.get('model_name', 'gpt2')
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"[CodeGenerator] Failed to load pretrained model: {e}")
            self.model = None
            self.tokenizer = None
    
    def forward(self, prompt: str) -> str:
        """Generate code from prompt."""
        if not TRANSFORMERS_AVAILABLE or not PYTORCH_AVAILABLE or self.model is None:
            return f"# Generated code stub\ndef solution():\n    pass  # {prompt}"
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors='pt', padding=True)
            
            # Generate
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            print(f"[CodeGenerator] Generation failed: {e}")
            return f"# Error: {e}"
    
    def get_loss(self, predictions: 'torch.Tensor', targets: 'torch.Tensor') -> 'torch.Tensor':
        """Language modeling loss."""
        if not PYTORCH_AVAILABLE:
            return torch.tensor(0.0)
        return F.cross_entropy(predictions.view(-1, predictions.size(-1)), targets.view(-1))
    
    def process(self, specification: str) -> Dict[str, Any]:
        """Generate code from specification."""
        # Format prompt for code generation
        prompt = f"# Task: {specification}\n# Solution:\n"
        
        generated_code = self.forward(prompt)
        
        return {
            'code': generated_code,
            'specification': specification,
            'model': 'gpt2' if self.model else 'stub'
        }


# ============================================================================
# Scientific/NAS Models
# ============================================================================

class LogicGateDetector(BaseMLModel):
    """
    1D CNN for detecting logic gate behavior in oscillator time series.
    
    Input: Time series (200 timesteps, 3 features)
    Output: 4 classes (AND, OR, XOR, NAND)
    """
    
    GATE_CLASSES = ['AND', 'OR', 'XOR', 'NAND']
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not PYTORCH_AVAILABLE:
            return
        
        self.seq_length = config.get('seq_length', 200)
        self.input_features = config.get('input_features', 3)
        self.num_classes = len(self.GATE_CLASSES)
        
        # 1D CNN architecture
        self.model = nn.Sequential(
            # Conv block 1
            nn.Conv1d(self.input_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Conv block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Conv block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            
            # Classifier
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_classes),
            nn.Softmax(dim=1)
        )
        
        self.optimizer = Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-3))
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """Forward pass."""
        if not PYTORCH_AVAILABLE:
            return np.ones(self.num_classes) / self.num_classes
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Ensure shape is (batch, features, seq_length)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.shape[1] != self.input_features:
            x = x.transpose(1, 2)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        
        return output
    
    def get_loss(self, predictions: 'torch.Tensor', targets: 'torch.Tensor') -> 'torch.Tensor':
        """Cross-entropy loss."""
        return self.criterion(predictions, targets)
    
    def process(self, time_series: np.ndarray) -> Dict[str, Any]:
        """Detect logic gate from time series."""
        probs = self.forward(time_series)
        
        if PYTORCH_AVAILABLE:
            probs_np = probs.cpu().numpy().flatten()
        else:
            probs_np = probs
        
        predicted_idx = int(np.argmax(probs_np))
        
        return {
            'gate_type': self.GATE_CLASSES[predicted_idx],
            'class_probs': {gate: float(probs_np[i]) for i, gate in enumerate(self.GATE_CLASSES)},
            'confidence': float(probs_np[predicted_idx])
        }


class PerformancePredictor(BaseMLModel):
    """
    MLP for predicting neural architecture performance.
    
    Input: Architecture encoding (100D)
    Output: [accuracy, latency, params]
    """
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not PYTORCH_AVAILABLE:
            return
        
        self.input_dim = config.get('input_dim', 100)
        self.output_dim = 3  # accuracy, latency, params
        
        # Deep MLP with residual connections
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, self.output_dim)
        )
        
        self.optimizer = Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-3))
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """Forward pass."""
        if not PYTORCH_AVAILABLE:
            return np.array([0.9, 0.5, 1e6])  # Default predictions
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        
        return output
    
    def get_loss(self, predictions: 'torch.Tensor', targets: 'torch.Tensor') -> 'torch.Tensor':
        """MSE loss."""
        return F.mse_loss(predictions, targets)
    
    def process(self, architecture_encoding: np.ndarray) -> Dict[str, Any]:
        """Predict architecture performance."""
        predictions = self.forward(architecture_encoding)
        
        if PYTORCH_AVAILABLE:
            preds_np = predictions.cpu().numpy().flatten()
        else:
            preds_np = predictions
        
        return {
            'accuracy': float(preds_np[0]),
            'latency': float(preds_np[1]),
            'params': float(preds_np[2]),
            'score': float(preds_np[0] / (preds_np[1] + 0.1))  # Accuracy/latency trade-off
        }


class EfficiencyPredictor(BaseMLModel):
    """
    Physics-informed neural network for solar cell efficiency prediction.
    
    Input: Material parameters (bandgap, thickness, doping, etc.)
    Output: Efficiency percentage
    """
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not PYTORCH_AVAILABLE:
            return
        
        self.input_dim = config.get('input_dim', 10)  # Material parameters
        
        # Physics-informed architecture
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.Tanh(),  # Smooth activation for physical constraints
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Efficiency in [0, 1]
        )
        
        self.optimizer = Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-3))
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """Forward pass."""
        if not PYTORCH_AVAILABLE:
            return np.array([0.15])  # Default 15% efficiency
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        
        return output
    
    def get_loss(self, predictions: 'torch.Tensor', targets: 'torch.Tensor') -> 'torch.Tensor':
        """MSE loss with physics constraints."""
        mse_loss = F.mse_loss(predictions, targets)
        
        # Physics constraint: efficiency should be bounded
        constraint_loss = torch.mean(torch.relu(predictions - 0.4))  # Penalize >40% (unrealistic)
        
        return mse_loss + 0.1 * constraint_loss
    
    def process(self, material_params: np.ndarray) -> Dict[str, Any]:
        """Predict solar cell efficiency."""
        efficiency = self.forward(material_params)
        
        if PYTORCH_AVAILABLE:
            eff_value = float(efficiency.cpu().numpy())
        else:
            eff_value = float(efficiency)
        
        return {
            'efficiency': eff_value * 100,  # Convert to percentage
            'efficiency_normalized': eff_value,
            'quality_score': 'excellent' if eff_value > 0.2 else 'good' if eff_value > 0.15 else 'moderate'
        }


class CognitiveStateDecoder(BaseMLModel):
    """
    2D CNN for decoding cognitive states from spatial EEG maps.
    
    Input: EEG topographic map (64x64)
    Output: 5 cognitive states
    """
    
    COGNITIVE_STATES = ['resting', 'focused', 'drowsy', 'active_thinking', 'meditation']
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not PYTORCH_AVAILABLE:
            return
        
        self.num_classes = len(self.COGNITIVE_STATES)
        
        # 2D CNN for spatial EEG patterns
        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Global pooling and classification
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, self.num_classes),
            nn.Softmax(dim=1)
        )
        
        self.optimizer = Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-3))
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """Forward pass."""
        if not PYTORCH_AVAILABLE:
            return np.ones(self.num_classes) / self.num_classes
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Ensure shape is (batch, 1, 64, 64)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        
        return output
    
    def get_loss(self, predictions: 'torch.Tensor', targets: 'torch.Tensor') -> 'torch.Tensor':
        """Cross-entropy loss."""
        return self.criterion(predictions, targets)
    
    def process(self, eeg_map: np.ndarray) -> Dict[str, Any]:
        """Decode cognitive state from EEG map."""
        probs = self.forward(eeg_map)
        
        if PYTORCH_AVAILABLE:
            probs_np = probs.cpu().numpy().flatten()
        else:
            probs_np = probs
        
        predicted_idx = int(np.argmax(probs_np))
        
        return {
            'state': self.COGNITIVE_STATES[predicted_idx],
            'probabilities': {state: float(probs_np[i]) for i, state in enumerate(self.COGNITIVE_STATES)},
            'confidence': float(probs_np[predicted_idx])
        }


class BindingPredictor(BaseMLModel):
    """
    Deep MLP for predicting molecular binding affinity.
    
    Input: Molecular fingerprints (200D)
    Output: Binding affinity score
    """
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not PYTORCH_AVAILABLE:
            return
        
        self.input_dim = config.get('input_dim', 200)
        
        # Deep MLP with batch normalization
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 1)  # Binding affinity score
        )
        
        self.optimizer = Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-3))
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """Forward pass."""
        if not PYTORCH_AVAILABLE:
            return np.array([5.0])  # Default binding affinity
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
        
        return output
    
    def get_loss(self, predictions: 'torch.Tensor', targets: 'torch.Tensor') -> 'torch.Tensor':
        """MSE loss."""
        return F.mse_loss(predictions.squeeze(), targets)
    
    def process(self, fingerprint: np.ndarray) -> Dict[str, Any]:
        """Predict binding affinity from molecular fingerprint."""
        affinity = self.forward(fingerprint)
        
        if PYTORCH_AVAILABLE:
            affinity_value = float(affinity.cpu().numpy())
        else:
            affinity_value = float(affinity)
        
        # Convert to binding energy (kcal/mol)
        binding_energy = -1.36 * affinity_value  # RT * ln(K_d)
        
        return {
            'affinity': affinity_value,
            'binding_energy': binding_energy,
            'strength': 'strong' if affinity_value > 7 else 'moderate' if affinity_value > 5 else 'weak'
        }


class MLPerformanceSelector(BaseMLModel):
    """
    Multi-output MLP for predicting algorithm performance metrics.
    
    Input: Algorithm features (50D)
    Output: [speed, memory, quality] scores
    """
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        if not PYTORCH_AVAILABLE:
            return
        
        self.input_dim = config.get('input_dim', 50)
        self.output_dim = 3  # speed, memory, quality
        
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.speed_head = nn.Linear(128, 1)
        self.memory_head = nn.Linear(128, 1)
        self.quality_head = nn.Linear(128, 1)
        
        self.model = nn.ModuleDict({
            'shared': self.shared,
            'speed': self.speed_head,
            'memory': self.memory_head,
            'quality': self.quality_head
        })
        
        self.optimizer = Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-3))
    
    def forward(self, x: Union[np.ndarray, 'torch.Tensor']) -> 'torch.Tensor':
        """Forward pass."""
        if not PYTORCH_AVAILABLE:
            return np.array([0.8, 0.6, 0.9])  # Default predictions
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            shared_repr = self.shared(x)
            speed = self.speed_head(shared_repr)
            memory = self.memory_head(shared_repr)
            quality = self.quality_head(shared_repr)
            
            output = torch.cat([speed, memory, quality], dim=1)
        
        return output
    
    def get_loss(self, predictions: 'torch.Tensor', targets: 'torch.Tensor') -> 'torch.Tensor':
        """MSE loss for multi-output regression."""
        return F.mse_loss(predictions, targets)
    
    def process(self, algorithm_features: np.ndarray) -> Dict[str, Any]:
        """Predict performance metrics."""
        predictions = self.forward(algorithm_features)
        
        if PYTORCH_AVAILABLE:
            preds_np = predictions.cpu().numpy().flatten()
        else:
            preds_np = predictions
        
        return {
            'speed': float(preds_np[0]),
            'memory': float(preds_np[1]),
            'quality': float(preds_np[2]),
            'overall_score': float(np.mean(preds_np))
        }


# ============================================================================
# Factory Function
# ============================================================================

def create_concrete_model(node_id: str, model_type: str, config: Dict[str, Any]) -> BaseMLModel:
    """
    Factory function to create concrete ML models.
    
    Args:
        node_id: Unique node identifier
        model_type: Type of model to create
        config: Model configuration
        
    Returns:
        Instance of BaseMLModel
    """
    
    models = {
        # Music
        'GenreClassifier': GenreClassifier,
        'MusicTransformer': MusicTransformer,
        
        # RL
        'PPOAgent': PPOAgent,
        'ValueFunction': ValueFunction,
        
        # Generative
        'PixelArtGAN': PixelArtGAN,
        'CodeGenerator': CodeGenerator,
        
        # Scientific/NAS
        'LogicGateDetector': LogicGateDetector,
        'PerformancePredictor': PerformancePredictor,
        'EfficiencyPredictor': EfficiencyPredictor,
        'CognitiveStateDecoder': CognitiveStateDecoder,
        'BindingPredictor': BindingPredictor,
        'MLPerformanceSelector': MLPerformanceSelector,
        
        # Aliases
        'Logic Gate Detector': LogicGateDetector,
        'Performance Predictor': PerformancePredictor,
        'Efficiency Predictor': EfficiencyPredictor,
        'Cognitive State Decoder': CognitiveStateDecoder,
        'Binding Predictor': BindingPredictor,
        'ML Performance Selector': MLPerformanceSelector,
    }
    
    model_class = models.get(model_type)
    if model_class is None:
        raise ValueError(f"Unknown ML model type: {model_type}")
    
    return model_class(node_id, config)

