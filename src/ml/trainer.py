"""
Unified Trainer for GMCS ML Models.

Provides training loops and utilities for all ML model types,
with support for chaos data, callbacks, logging, and checkpointing.

Key features:
- Unified interface for all model types
- Chaos data loaders
- Training callbacks
- Automatic checkpointing
- TensorBoard logging
- Early stopping
- Learning rate scheduling

Use cases:
- Train any ML model on chaotic data
- Standardized training workflow
- Experiment tracking
- Production training pipelines
"""

from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from pathlib import Path
from datetime import datetime
import json
import time
import numpy as np

try:
    import torch
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from .ml_nodes import MLNode
from .losses import get_loss_fn, CombinedLoss


# ============================================================================
# Chaos Data Loaders
# ============================================================================

class ChaosDataset:
    """
    Dataset for chaotic time series.
    
    Generates sequences from chaotic dynamics for training.
    """
    
    def __init__(
        self,
        oscillator_fn: Callable,
        n_samples: int = 1000,
        sequence_length: int = 100,
        initial_state_fn: Optional[Callable] = None
    ):
        """
        Initialize chaos dataset.
        
        Args:
            oscillator_fn: Function that generates chaotic trajectory
            n_samples: Number of samples to generate
            sequence_length: Length of each sequence
            initial_state_fn: Function to generate initial states
        """
        self.oscillator_fn = oscillator_fn
        self.n_samples = n_samples
        self.sequence_length = sequence_length
        self.initial_state_fn = initial_state_fn or (lambda: np.random.randn(3) * 0.1)
        
        # Pre-generate data
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[np.ndarray]:
        """Generate dataset."""
        data = []
        for _ in range(self.n_samples):
            initial = self.initial_state_fn()
            trajectory = self.oscillator_fn(initial, self.sequence_length)
            data.append(trajectory)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[idx]


class ChaosDataLoader:
    """
    Data loader for chaos data.
    
    Provides batching, shuffling, and iteration over chaos datasets.
    """
    
    def __init__(
        self,
        dataset: Union[ChaosDataset, List, np.ndarray],
        batch_size: int = 32,
        shuffle: bool = True
    ):
        """
        Initialize data loader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        if isinstance(dataset, ChaosDataset):
            self.data = dataset.data
        elif isinstance(dataset, list):
            self.data = dataset
        else:
            # Assume numpy array
            self.data = [dataset[i] for i in range(len(dataset))]
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(self.data)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        
        self._reset_indices()
    
    def _reset_indices(self):
        """Reset iteration indices."""
        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_batch = 0
    
    def __iter__(self):
        """Reset and return self."""
        self._reset_indices()
        return self
    
    def __next__(self) -> np.ndarray:
        """Get next batch."""
        if self.current_batch >= self.n_batches:
            raise StopIteration
        
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        batch_indices = self.indices[start_idx:end_idx]
        batch = np.array([self.data[i] for i in batch_indices])
        
        self.current_batch += 1
        
        return batch
    
    def __len__(self) -> int:
        return self.n_batches


# ============================================================================
# Training Callbacks
# ============================================================================

class Callback:
    """Base class for training callbacks."""
    
    def on_train_begin(self, trainer: 'Trainer'):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: 'Trainer'):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, trainer: 'Trainer'):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer'):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, trainer: 'Trainer'):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, loss: float, trainer: 'Trainer'):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback.
    
    Stops training if validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer'):
        """Check if should stop."""
        val_loss = metrics.get('val_loss', metrics.get('loss', float('inf')))
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                print(f"\nEarly stopping at epoch {epoch}")


class ModelCheckpoint(Callback):
    """
    Model checkpointing callback.
    
    Saves model when validation loss improves.
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            save_best_only: Only save when metric improves
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = float('inf')
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer'):
        """Save checkpoint if improved."""
        current_value = metrics.get(self.monitor, float('inf'))
        
        if not self.save_best_only or current_value < self.best_value:
            self.best_value = current_value
            
            # Create checkpoint path
            path = self.filepath.format(epoch=epoch, **metrics)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            trainer.model.save_checkpoint(path)
            print(f"\nSaved checkpoint to {path}")


class LearningRateScheduler(Callback):
    """
    Learning rate scheduler callback.
    """
    
    def __init__(
        self,
        schedule_fn: Callable[[int], float],
        optimizer: Any
    ):
        """
        Initialize LR scheduler.
        
        Args:
            schedule_fn: Function that takes epoch and returns LR
            optimizer: Optimizer to update
        """
        self.schedule_fn = schedule_fn
        self.optimizer = optimizer
    
    def on_epoch_begin(self, epoch: int, trainer: 'Trainer'):
        """Update learning rate."""
        new_lr = self.schedule_fn(epoch)
        
        if PYTORCH_AVAILABLE and isinstance(self.optimizer, optim.Optimizer):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        print(f"Learning rate: {new_lr:.6f}")


class ProgressLogger(Callback):
    """
    Progress logging callback.
    """
    
    def __init__(self, print_every: int = 1):
        """
        Initialize logger.
        
        Args:
            print_every: Print frequency (epochs)
        """
        self.print_every = print_every
        self.epoch_start_time = None
    
    def on_epoch_begin(self, epoch: int, trainer: 'Trainer'):
        """Record start time."""
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: 'Trainer'):
        """Print progress."""
        if (epoch + 1) % self.print_every == 0:
            elapsed = time.time() - self.epoch_start_time
            
            metrics_str = ", ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
            print(f"Epoch {epoch+1}/{trainer.epochs}: {metrics_str} ({elapsed:.2f}s)")


# ============================================================================
# Unified Trainer
# ============================================================================

class Trainer:
    """
    Unified trainer for all ML models.
    
    Provides standardized training loop with callbacks, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: MLNode,
        loss_fn: Union[str, Callable, CombinedLoss],
        optimizer: Optional[Any] = None,
        device: str = "auto"
    ):
        """
        Initialize trainer.
        
        Args:
            model: ML model node
            loss_fn: Loss function (name, callable, or CombinedLoss)
            optimizer: Optimizer (created if None)
            device: Device to use
        """
        self.model = model
        
        # Get loss function
        if isinstance(loss_fn, str):
            self.loss_fn = get_loss_fn(loss_fn)
        else:
            self.loss_fn = loss_fn
        
        # Create optimizer if not provided
        if optimizer is None and hasattr(model, 'model'):
            if PYTORCH_AVAILABLE and hasattr(model.model, 'parameters'):
                self.optimizer = optim.Adam(model.model.parameters(), lr=0.001)
            else:
                self.optimizer = None
        else:
            self.optimizer = optimizer
        
        self.device = device
        
        # Training state
        self.history = {
            'loss': [],
            'val_loss': [],
            'metrics': []
        }
        self.epochs = 0
        self.stop_training = False
        
        # Callbacks
        self.callbacks = []
    
    def add_callback(self, callback: Callback):
        """Add training callback."""
        self.callbacks.append(callback)
    
    def train(
        self,
        train_data: Union[ChaosDataLoader, Any],
        epochs: int,
        val_data: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train model.
        
        Args:
            train_data: Training data loader
            epochs: Number of epochs
            val_data: Validation data (optional)
            callbacks: Training callbacks
            verbose: Print progress
            
        Returns:
            Training history
        """
        self.epochs = epochs
        self.stop_training = False
        
        # Add callbacks
        if callbacks:
            for cb in callbacks:
                self.add_callback(cb)
        
        # Add default progress logger if verbose
        if verbose and not any(isinstance(cb, ProgressLogger) for cb in self.callbacks):
            self.add_callback(ProgressLogger())
        
        # Training begin callbacks
        for cb in self.callbacks:
            cb.on_train_begin(self)
        
        # Training loop
        for epoch in range(epochs):
            if self.stop_training:
                break
            
            # Epoch begin callbacks
            for cb in self.callbacks:
                cb.on_epoch_begin(epoch, self)
            
            # Train epoch
            epoch_losses = []
            
            for batch_idx, batch_data in enumerate(train_data):
                # Batch begin callbacks
                for cb in self.callbacks:
                    cb.on_batch_begin(batch_idx, self)
                
                # Training step
                if hasattr(self.model, 'train_step'):
                    # Use model's train_step if available
                    if isinstance(batch_data, tuple):
                        metrics = self.model.train_step(batch_data[0], batch_data[1], self.optimizer)
                    else:
                        # Autoencoder case: data is both input and target
                        metrics = self.model.train_step(batch_data, self.optimizer)
                    
                    loss = metrics.get('loss', 0.0)
                else:
                    # Generic training step
                    loss = self._generic_train_step(batch_data)
                
                epoch_losses.append(loss)
                
                # Batch end callbacks
                for cb in self.callbacks:
                    cb.on_batch_end(batch_idx, loss, self)
            
            # Compute epoch metrics
            metrics = {'loss': np.mean(epoch_losses)}
            
            # Validation
            if val_data is not None:
                val_loss = self.evaluate(val_data)
                metrics['val_loss'] = val_loss
            
            # Record history
            self.history['loss'].append(metrics['loss'])
            if 'val_loss' in metrics:
                self.history['val_loss'].append(metrics['val_loss'])
            
            # Epoch end callbacks
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, metrics, self)
        
        # Training end callbacks
        for cb in self.callbacks:
            cb.on_train_end(self)
        
        return self.history
    
    def _generic_train_step(self, batch_data: np.ndarray) -> float:
        """
        Generic training step for models without train_step method.
        
        Args:
            batch_data: Batch of data
            
        Returns:
            Loss value
        """
        # This is a simplified version - override for specific needs
        if self.optimizer is None:
            return 0.0
        
        # Assume batch_data is (input, target) tuple
        if isinstance(batch_data, tuple):
            inputs, targets = batch_data
        else:
            # Autoregressive or autoencoder
            inputs = targets = batch_data
        
        # Forward pass
        predictions = self.model.forward(inputs)
        
        # Compute loss
        loss = self.loss_fn(predictions, targets)
        
        return loss
    
    def evaluate(self, data: Union[ChaosDataLoader, Any]) -> float:
        """
        Evaluate model on data.
        
        Args:
            data: Evaluation data
            
        Returns:
            Average loss
        """
        losses = []
        
        for batch_data in data:
            if isinstance(batch_data, tuple):
                inputs, targets = batch_data
            else:
                inputs = targets = batch_data
            
            # Forward pass
            predictions = self.model.forward(inputs)
            
            # Compute loss
            loss = self.loss_fn(predictions, targets)
            losses.append(loss)
        
        return float(np.mean(losses))
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data: Input data
            
        Returns:
            Predictions
        """
        return self.model.forward(data)
    
    def save(self, path: str):
        """
        Save trainer state.
        
        Args:
            path: Path to save
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'history': self.history,
            'epochs': self.epochs
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save model
        model_path = str(Path(path).with_suffix('.model'))
        self.model.save_checkpoint(model_path)
    
    def load(self, path: str):
        """
        Load trainer state.
        
        Args:
            path: Path to load from
        """
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.history = state['history']
        self.epochs = state['epochs']
        
        # Load model
        model_path = str(Path(path).with_suffix('.model'))
        self.model.load_checkpoint(model_path)


# ============================================================================
# Training Utilities
# ============================================================================

def train_on_chaos(
    model: MLNode,
    chaos_generator: Callable,
    n_samples: int = 1000,
    epochs: int = 50,
    batch_size: int = 32,
    val_split: float = 0.2,
    callbacks: Optional[List[Callback]] = None
) -> Dict[str, List[float]]:
    """
    Train model on chaotic data.
    
    Args:
        model: ML model to train
        chaos_generator: Function that generates chaos data
        n_samples: Number of samples
        epochs: Number of epochs
        batch_size: Batch size
        val_split: Validation split ratio
        callbacks: Training callbacks
        
    Returns:
        Training history
    """
    # Generate data
    print("Generating chaotic data...")
    data = [chaos_generator() for _ in range(n_samples)]
    
    # Split into train/val
    n_val = int(n_samples * val_split)
    train_data = data[:-n_val] if n_val > 0 else data
    val_data = data[-n_val:] if n_val > 0 else None
    
    # Create data loaders
    train_loader = ChaosDataLoader(train_data, batch_size=batch_size)
    val_loader = ChaosDataLoader(val_data, batch_size=batch_size) if val_data else None
    
    # Create trainer
    trainer = Trainer(model, loss_fn='mse')
    
    # Train
    print(f"Training for {epochs} epochs...")
    history = trainer.train(
        train_loader,
        epochs=epochs,
        val_data=val_loader,
        callbacks=callbacks
    )
    
    return history


if __name__ == '__main__':
    # Example usage
    print("Trainer module loaded successfully!")
    print("\nExample: Train MLP on chaos data")
    
    if PYTORCH_AVAILABLE:
        from .supervised import MLPNode
        
        # Create model
        model = MLPNode("test_mlp", input_dim=3, hidden_dims=[32, 16], output_dim=3)
        
        # Create synthetic chaos data
        def generate_chaos():
            return np.random.randn(3)
        
        # Create trainer
        trainer = Trainer(model, loss_fn='mse')
        
        # Create data loader
        data = [generate_chaos() for _ in range(100)]
        data_loader = ChaosDataLoader(data, batch_size=16)
        
        # Train
        print("\nTraining...")
        history = trainer.train(data_loader, epochs=5, verbose=True)
        
        print(f"\nFinal loss: {history['loss'][-1]:.6f}")
        print("✓ Training complete!")
    else:
        print("PyTorch not available - skipping example")

