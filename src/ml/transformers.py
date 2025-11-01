"""
Transformer Models for GMCS.

Provides transformer-based models (BERT, GPT, T5, etc.) as GMCS nodes
for sequence modeling, embeddings, and attention-based control.

Key features:
- Pre-trained model loading from HuggingFace
- Standalone attention mechanisms
- Sequence-to-sequence models
- Integration with chaotic time series

Use cases:
- Learn patterns in chaotic time series
- Generate sequences guided by chaos
- Attention-based control signals
- Feature extraction from oscillator states
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        GPT2Model,
        GPT2LMHeadModel,
        BertModel,
        BertTokenizer,
        T5Model,
        T5Tokenizer,
        pipeline
    )
    import torch
    import torch.nn as nn
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from .ml_nodes import MLModelNode


# ============================================================================
# Transformer Node Base Class
# ============================================================================

class TransformerNode(MLModelNode):
    """
    Base class for transformer models in GMCS.
    
    Wraps HuggingFace transformers for use in node graph.
    """
    
    def __init__(
        self,
        node_id: str,
        model_name: str = "bert-base-uncased",
        device: str = "auto"
    ):
        """
        Initialize transformer node.
        
        Args:
            node_id: Unique identifier
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or 'auto'
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers torch")
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_name = model_name
        self.device_name = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Initialize base class
        super().__init__(node_id, self.model, framework='pytorch')
        
        self.model.to(device)
        self.model.eval()
        
        # Get model config
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        
        self.metadata.update({
            'model_name': model_name,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        })
    
    def encode_sequence(
        self,
        sequence: Union[str, List[float], np.ndarray]
    ) -> np.ndarray:
        """
        Encode sequence to embeddings.
        
        Args:
            sequence: Text string or numerical sequence
            
        Returns:
            (seq_len, hidden_size) embeddings
        """
        if isinstance(sequence, str):
            # Text input
            inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device_name) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get last hidden state
            embeddings = outputs.last_hidden_state.squeeze(0)
            return embeddings.cpu().numpy()
        
        else:
            # Numerical sequence - convert to tokens
            # Map values to vocabulary (simple binning)
            sequence = np.array(sequence)
            normalized = (sequence - sequence.min()) / (sequence.max() - sequence.min() + 1e-8)
            token_ids = (normalized * (self.tokenizer.vocab_size - 1)).astype(int)
            
            inputs = torch.tensor(token_ids).unsqueeze(0).to(self.device_name)
            
            with torch.no_grad():
                outputs = self.model(inputs)
            
            embeddings = outputs.last_hidden_state.squeeze(0)
            return embeddings.cpu().numpy()
    
    def forward(self, input_data: Union[str, np.ndarray]) -> np.ndarray:
        """
        Forward pass through transformer.
        
        Args:
            input_data: Input sequence (text or numerical)
            
        Returns:
            Output embeddings
        """
        return self.encode_sequence(input_data)


# ============================================================================
# BERT Node
# ============================================================================

class BERTNode(TransformerNode):
    """
    BERT model for embeddings and sequence understanding.
    
    BERT is bidirectional and excellent for:
    - Feature extraction from sequences
    - Understanding context
    - Sentence/sequence embeddings
    """
    
    def __init__(
        self,
        node_id: str,
        model_name: str = "bert-base-uncased",
        device: str = "auto",
        pooling: str = "mean"
    ):
        """
        Initialize BERT node.
        
        Args:
            node_id: Unique identifier
            model_name: BERT variant
            device: Device to use
            pooling: How to pool embeddings ('mean', 'max', 'cls')
        """
        super().__init__(node_id, model_name, device)
        self.pooling = pooling
    
    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """
        Get sentence-level embedding.
        
        Args:
            text: Input text
            
        Returns:
            (hidden_size,) sentence embedding
        """
        token_embeddings = self.encode_sequence(text)
        
        if self.pooling == "mean":
            return np.mean(token_embeddings, axis=0)
        elif self.pooling == "max":
            return np.max(token_embeddings, axis=0)
        elif self.pooling == "cls":
            return token_embeddings[0]  # [CLS] token
        else:
            return np.mean(token_embeddings, axis=0)


# ============================================================================
# GPT Node
# ============================================================================

class GPTNode(TransformerNode):
    """
    GPT model for autoregressive generation.
    
    GPT is causal/unidirectional and excellent for:
    - Sequence generation
    - Next token prediction
    - Continuation of patterns
    """
    
    def __init__(
        self,
        node_id: str,
        model_name: str = "gpt2",
        device: str = "auto",
        max_length: int = 50
    ):
        """
        Initialize GPT node.
        
        Args:
            node_id: Unique identifier
            model_name: GPT variant
            device: Device to use
            max_length: Maximum generation length
        """
        # Override with causal LM
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_name = model_name
        self.device_name = device
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Initialize MLModelNode (skip TransformerNode init)
        MLModelNode.__init__(self, node_id, self.model, framework='pytorch')
        
        self.model.to(device)
        self.model.eval()
        
        self.hidden_size = self.model.config.n_embd
        self.num_layers = self.model.config.n_layer
    
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> str:
        """
        Generate text continuation.
        
        Args:
            prompt: Starting prompt
            max_length: Maximum length
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated text
        """
        max_length = max_length or self.max_length
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_from_chaos(
        self,
        chaos_sequence: np.ndarray,
        seed_text: str = "The pattern is",
        **kwargs
    ) -> str:
        """
        Generate text guided by chaotic sequence.
        
        Uses chaos to modulate generation temperature.
        
        Args:
            chaos_sequence: Chaotic time series
            seed_text: Starting text
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        # Normalize chaos to temperature range [0.5, 1.5]
        chaos_norm = (chaos_sequence - chaos_sequence.min()) / (chaos_sequence.max() - chaos_sequence.min() + 1e-8)
        temperature = 0.5 + chaos_norm.mean()
        
        return self.generate(seed_text, temperature=temperature, **kwargs)


# ============================================================================
# Attention Mechanism Node
# ============================================================================

class AttentionNode(MLModelNode):
    """
    Standalone multi-head attention mechanism.
    
    Can be used independently or as part of larger models.
    """
    
    def __init__(
        self,
        node_id: str,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: str = "auto"
    ):
        """
        Initialize attention node.
        
        Args:
            node_id: Unique identifier
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            device: Device to use
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers/torch required")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create attention module
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        super().__init__(node_id, self.attention, framework='pytorch')
        
        self.attention.to(device)
        self.device_name = device
        self.hidden_size = hidden_size
        self.num_heads = num_heads
    
    def forward(
        self,
        query: np.ndarray,
        key: Optional[np.ndarray] = None,
        value: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute attention.
        
        Args:
            query: Query tensor (batch, seq_len, hidden_size)
            key: Key tensor (optional, defaults to query)
            value: Value tensor (optional, defaults to query)
            
        Returns:
            Attended output
        """
        query_t = torch.from_numpy(query).float().to(self.device_name)
        
        if key is None:
            key_t = query_t
        else:
            key_t = torch.from_numpy(key).float().to(self.device_name)
        
        if value is None:
            value_t = query_t
        else:
            value_t = torch.from_numpy(value).float().to(self.device_name)
        
        # Ensure 3D: (batch, seq, hidden)
        if query_t.ndim == 2:
            query_t = query_t.unsqueeze(0)
            key_t = key_t.unsqueeze(0)
            value_t = value_t.unsqueeze(0)
        
        with torch.no_grad():
            output, _ = self.attention(query_t, key_t, value_t)
        
        output_np = output.squeeze(0).cpu().numpy()
        return output_np


# ============================================================================
# Sequence-to-Sequence Node (T5)
# ============================================================================

class Seq2SeqNode(TransformerNode):
    """
    Sequence-to-sequence model (T5).
    
    Useful for:
    - Translation between representations
    - Conditional generation
    - Pattern transformation
    """
    
    def __init__(
        self,
        node_id: str,
        model_name: str = "t5-small",
        device: str = "auto"
    ):
        """
        Initialize seq2seq node.
        
        Args:
            node_id: Unique identifier
            model_name: T5 variant
            device: Device to use
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required")
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_name = model_name
        self.device_name = device
        
        # Load T5
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5Model.from_pretrained(model_name)
        
        MLModelNode.__init__(self, node_id, self.model, framework='pytorch')
        
        self.model.to(device)
        self.model.eval()
    
    def encode_decode(
        self,
        input_text: str,
        max_length: int = 50
    ) -> str:
        """
        Encode-decode transformation.
        
        Args:
            input_text: Input sequence
            max_length: Maximum output length
            
        Returns:
            Transformed sequence
        """
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================================
# Helper Functions
# ============================================================================

def create_transformer_node(
    node_id: str,
    model_type: str = "bert",
    **kwargs
) -> TransformerNode:
    """
    Factory function for transformer nodes.
    
    Args:
        node_id: Unique identifier
        model_type: 'bert', 'gpt', 't5', or 'attention'
        **kwargs: Additional arguments
        
    Returns:
        TransformerNode instance
    """
    if model_type == "bert":
        return BERTNode(node_id, **kwargs)
    elif model_type == "gpt":
        return GPTNode(node_id, **kwargs)
    elif model_type == "t5":
        return Seq2SeqNode(node_id, **kwargs)
    elif model_type == "attention":
        return AttentionNode(node_id, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def chaos_to_embeddings(
    chaos_sequence: np.ndarray,
    transformer_node: TransformerNode,
    window_size: int = 10
) -> np.ndarray:
    """
    Convert chaotic time series to transformer embeddings.
    
    Args:
        chaos_sequence: (seq_len,) or (seq_len, dim) chaos trajectory
        transformer_node: Transformer node for encoding
        window_size: Window size for chunking
        
    Returns:
        (n_windows, hidden_size) embeddings
    """
    if chaos_sequence.ndim == 1:
        chaos_sequence = chaos_sequence.reshape(-1, 1)
    
    n_steps = len(chaos_sequence)
    embeddings = []
    
    for i in range(0, n_steps - window_size + 1, window_size):
        window = chaos_sequence[i:i+window_size]
        embedding = transformer_node.encode_sequence(window.flatten())
        
        # Pool to single vector
        pooled = np.mean(embedding, axis=0)
        embeddings.append(pooled)
    
    return np.array(embeddings)


if __name__ == '__main__':
    # Example usage
    if TRANSFORMERS_AVAILABLE:
        print("Creating BERT node...")
        bert = BERTNode("bert1")
        
        text = "Chaotic dynamics are fascinating."
        embedding = bert.get_sentence_embedding(text)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.2f}")
    else:
        print("Transformers library not available. Install with: pip install transformers torch")

