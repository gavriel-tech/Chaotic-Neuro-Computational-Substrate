"""
HuggingFace Transformers Integration for GMCS.

Enables loading HuggingFace models and integrating them with the GMCS simulation.
"""

from typing import Optional, Dict, Any, List
import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer, PreTrainedModel
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    AutoModel = None
    AutoTokenizer = None
    PreTrainedModel = None
    print("HuggingFace Transformers not available. Install with: pip install transformers")

import jax.numpy as jnp


# Stub classes when HuggingFace is not available
if not HUGGINGFACE_AVAILABLE:
    class HuggingFaceModelWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("HuggingFace Transformers is required for this module. Install with: pip install transformers")
else:
    # Real implementations when HuggingFace is available
    class HuggingFaceModelWrapper:
        """
        Wrapper for HuggingFace models to integrate with GMCS.
        """
        
        def __init__(self, model_name: Optional[str] = None, model: Optional[PreTrainedModel] = None):
            """
            Initialize HuggingFace model wrapper.
            
            Args:
                model_name: Name of pretrained model (e.g., 'bert-base-uncased')
                model: Pretrained model instance
            """
            if model is not None:
                self.model = model
            elif model_name is not None:
                self.model = AutoModel.from_pretrained(model_name)
            else:
                self.model = None
            
            self.tokenizer = None
            if model_name is not None:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                except:
                    pass
        
        def forward(self, x: jnp.ndarray) -> jnp.ndarray:
            """Run forward pass through model."""
            if self.model is None:
                raise ValueError("No model loaded")
            
            # Convert JAX array to numpy
            x_np = np.array(x)
            
            # Run model
            import torch
            x_torch = torch.from_numpy(x_np)
            outputs = self.model(x_torch)
            
            # Convert back to JAX
            if hasattr(outputs, 'last_hidden_state'):
                result = outputs.last_hidden_state.detach().numpy()
            elif hasattr(outputs, 'logits'):
                result = outputs.logits.detach().numpy()
            else:
                result = outputs[0].detach().numpy()
            
            return jnp.array(result)
        
        def encode_text(self, text: str) -> jnp.ndarray:
            """Encode text to embeddings."""
            if self.tokenizer is None:
                raise ValueError("No tokenizer available")
            
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
            elif hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output.detach().numpy()
            else:
                embeddings = outputs[0][:, 0, :].detach().numpy()
            
            return jnp.array(embeddings)
        
        def load_model(self, model_name: str):
            """Load model from HuggingFace Hub."""
            self.model = AutoModel.from_pretrained(model_name)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except:
                self.tokenizer = None
        
        def save_model(self, path: str):
            """Save model to directory."""
            if self.model is None:
                raise ValueError("No model to save")
            self.model.save_pretrained(path)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(path)
