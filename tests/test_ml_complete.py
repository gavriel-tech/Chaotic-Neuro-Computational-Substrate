"""
Comprehensive Test Suite for ML + Gradient Integration.

Tests all major ML components, training infrastructure, and hybrid training.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Test ML Nodes
# ============================================================================

def test_ml_nodes():
    """Test ML node base classes."""
    from src.ml.ml_nodes import MLNode, MLModelNode
    
    # Base class should be abstract
    assert MLNode is not None
    print("[OK] ML node base classes imported")


def test_differentiable_oscillator_node():
    """Test differentiable oscillator node."""
    try:
        from src.ml.ml_nodes import DifferentiableOscillatorNode
        import jax.numpy as jnp
        
        node = DifferentiableOscillatorNode("test_osc", alpha=15.6, beta=28.0)
        assert node.node_id == "test_osc"
        assert node.metadata['differentiable'] == True
        
        # Test forward pass
        new_state = node.forward(forcing=0.1)
        assert new_state.shape == (3,)
        
        print("[OK] Differentiable oscillator node works")
    except ImportError as e:
        pytest.skip(f"JAX not available: {e}")


# ============================================================================
# Test Differentiable Chaos
# ============================================================================

def test_differentiable_chua():
    """Test differentiable Chua oscillator."""
    try:
        from src.core.differentiable_chua import chua_trajectory, default_params
        import jax.numpy as jnp
        
        params = default_params()
        initial = jnp.array([0.1, 0.1, 0.1])
        forcing = jnp.zeros(10)
        
        trajectory = chua_trajectory(initial, params, forcing, 0.01, 10)
        
        assert trajectory.shape == (11, 3)
        print("[OK] Differentiable Chua trajectory generation works")
    except ImportError as e:
        pytest.skip(f"JAX not available: {e}")


def test_parameter_optimization():
    """Test gradient-based parameter optimization."""
    try:
        from src.core.differentiable_chua import DifferentiableChuaOptimizer, default_params
        import jax.numpy as jnp
        
        params = default_params()
        optimizer = DifferentiableChuaOptimizer(params, learning_rate=0.1)
        
        assert optimizer.params == params
        print("[OK] Parameter optimizer initialized")
    except ImportError as e:
        pytest.skip(f"JAX not available: {e}")


# ============================================================================
# Test ML Models
# ============================================================================

def test_supervised_models():
    """Test supervised learning models."""
    try:
        from src.ml.supervised import MLPNode, CNNNode, AutoencoderNode
        
        # MLP
        mlp = MLPNode("mlp1", input_dim=10, hidden_dims=[32, 16], output_dim=1)
        assert mlp.node_id == "mlp1"
        
        # CNN
        cnn = CNNNode("cnn1", input_channels=1, output_dim=5)
        assert cnn.node_id == "cnn1"
        
        # Autoencoder
        ae = AutoencoderNode("ae1", input_dim=100, latent_dim=10)
        assert ae.latent_dim == 10
        
        print("[OK] Supervised models initialized")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")


def test_transformers():
    """Test transformer models."""
    try:
        from src.ml.transformers import TransformerNode
        
        # Just test import and class definition
        assert TransformerNode is not None
        print("[OK] Transformer models available")
    except ImportError as e:
        pytest.skip(f"Transformers not available: {e}")


def test_diffusion():
    """Test diffusion models."""
    try:
        from src.ml.diffusion import DiffusionNode
        
        diffusion = DiffusionNode(
            "diff1",
            timesteps=100,  # Small for testing
            data_shape=(1, 64)
        )
        
        assert diffusion.timesteps == 100
        print("[OK] Diffusion model initialized")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")


def test_gans():
    """Test GAN models."""
    try:
        from src.ml.gans import GANNode
        
        gan = GANNode("gan1", latent_dim=50, output_length=64)
        
        assert gan.latent_dim == 50
        print("[OK] GAN model initialized")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")


def test_reinforcement():
    """Test RL controllers."""
    try:
        from src.ml.reinforcement import RLControllerNode
        
        controller = RLControllerNode(
            "rl1",
            state_dim=3,
            action_dim=1,
            algorithm='ppo'
        )
        
        assert controller.state_dim == 3
        print("[OK] RL controller initialized")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")


# ============================================================================
# Test Training Infrastructure
# ============================================================================

def test_loss_functions():
    """Test loss functions."""
    from src.ml.losses import trajectory_mse_loss, LOSS_REGISTRY
    
    # Test MSE
    pred = np.random.randn(10, 3)
    target = np.random.randn(10, 3)
    loss = trajectory_mse_loss(pred, target)
    
    assert isinstance(loss, float)
    assert loss >= 0
    
    # Test registry
    assert 'trajectory_mse' in LOSS_REGISTRY
    assert 'lyapunov' in LOSS_REGISTRY
    
    print("[OK] Loss functions work")


def test_trainer():
    """Test unified trainer."""
    try:
        from src.ml.trainer import Trainer
        from src.ml.supervised import MLPNode
        
        model = MLPNode("test_mlp", input_dim=5, hidden_dims=[16], output_dim=1)
        trainer = Trainer(model, loss_fn='mse')
        
        assert trainer.model == model
        print("[OK] Trainer initialized")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")


def test_callbacks():
    """Test training callbacks."""
    from src.ml.trainer import EarlyStopping, ModelCheckpoint, LearningRateScheduler
    
    # Early stopping
    early_stop = EarlyStopping(patience=10)
    assert early_stop.patience == 10
    
    # Model checkpoint
    checkpoint = ModelCheckpoint('test_checkpoint.pt')
    assert checkpoint.filepath == 'test_checkpoint.pt'
    
    print("[OK] Callbacks initialized")


def test_hybrid_training():
    """Test hybrid chaos+gradient training."""
    from src.ml.hybrid_training import HybridConfig, HybridTrainer
    
    config = HybridConfig(
        chaos_steps=10,
        gradient_steps=5,
        phase_schedule='alternating'
    )
    
    assert config.chaos_steps == 10
    assert config.gradient_steps == 5
    
    print("[OK] Hybrid training config works")


# ============================================================================
# Test Model Registry
# ============================================================================

def test_model_registry():
    """Test model registry."""
    from src.ml.model_registry import get_registry
    
    registry = get_registry()
    
    # List models
    models = registry.list_models()
    assert len(models) > 0
    
    # Search
    bert_models = registry.search_models('bert')
    assert len(bert_models) > 0
    
    # Statistics
    stats = registry.get_statistics()
    assert 'total_models' in stats
    
    print(f"✓ Model registry works ({stats['total_models']} models)")


def test_model_conversion():
    """Test framework conversion."""
    from src.ml.model_conversion import torch_to_numpy, numpy_to_torch
    
    # Test numpy conversion
    arr = np.random.randn(5, 10)
    
    try:
        tensor = numpy_to_torch(arr)
        back = torch_to_numpy(tensor)
        
        assert np.allclose(arr, back)
        print("[OK] Framework conversion works")
    except ImportError:
        pytest.skip("PyTorch not available")


# ============================================================================
# Test Gradient Flow
# ============================================================================

def test_gradient_flow_graph():
    """Test gradient flow graph."""
    from src.ml.gradient_manager import GradientFlowGraph
    
    graph = GradientFlowGraph()
    
    # Add nodes
    graph.add_node("input", differentiable=False)
    graph.add_node("ml", differentiable=True)
    graph.add_node("output", differentiable=True)
    
    # Add edges
    graph.add_edge("input", "ml")
    graph.add_edge("ml", "output")
    
    # Test topology
    order = graph._compute_topological_order()
    assert len(order) == 3
    assert order[0] == "input"
    
    print("[OK] Gradient flow graph works")


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_chaos_ml():
    """Test end-to-end chaos → ML pipeline."""
    try:
        from src.core.differentiable_chua import chua_trajectory, default_params
        from src.ml.supervised import MLPNode
        import jax.numpy as jnp
        
        # Generate chaos
        params = default_params()
        initial = jnp.array([0.1, 0.1, 0.1])
        trajectory = chua_trajectory(initial, params, jnp.zeros(20), 0.01, 20)
        
        # Create ML model
        mlp = MLPNode("predictor", input_dim=3, hidden_dims=[16], output_dim=3)
        
        # Test forward pass
        data = np.array(trajectory[:10])
        output = mlp.forward(data)
        
        assert output.shape[0] == 10
        print("[OK] End-to-end chaos → ML works")
    except ImportError as e:
        pytest.skip(f"Dependencies not available: {e}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("GMCS ML + Gradient Integration - Comprehensive Test Suite")
    print("="*70 + "\n")
    
    tests = [
        ("ML Nodes", test_ml_nodes),
        ("Differentiable Oscillator", test_differentiable_oscillator_node),
        ("Differentiable Chua", test_differentiable_chua),
        ("Parameter Optimization", test_parameter_optimization),
        ("Supervised Models", test_supervised_models),
        ("Transformers", test_transformers),
        ("Diffusion Models", test_diffusion),
        ("GANs", test_gans),
        ("Reinforcement Learning", test_reinforcement),
        ("Loss Functions", test_loss_functions),
        ("Trainer", test_trainer),
        ("Callbacks", test_callbacks),
        ("Hybrid Training", test_hybrid_training),
        ("Model Registry", test_model_registry),
        ("Model Conversion", test_model_conversion),
        ("Gradient Flow Graph", test_gradient_flow_graph),
        ("End-to-End", test_end_to_end_chaos_ml)
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        try:
            print(f"\nRunning: {name}")
            test_func()
            passed += 1
        except pytest.skip.Exception as e:
            print(f"[SKIP] {e}")
            skipped += 1
        except Exception as e:
            print(f"[FAIL] {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*70 + "\n")
    
    if failed > 0:
        sys.exit(1)

