"""
HuggingFace Integration for THRML

Enables using pre-trained transformers and embeddings with THRML:
- Extract embeddings from text/images using HuggingFace models
- Use embeddings as THRML node biases or initialization
- Pattern recognition and classification with THRML
- Transfer learning from large language models to THRML

Key features:
- Text embeddings (BERT, RoBERTa, GPT-2)
- Image embeddings (CLIP, Vision Transformers)
- Sentence embeddings (Sentence-BERT)
- Zero-shot classification
- Fine-tuning support
"""

import numpy as np
import jax.numpy as jnp
from typing import List, Optional, Dict, Any, Union, Tuple
import logging

logger = logging.getLogger(__name__)

# Try importing transformers
try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoConfig,
        pipeline,
        CLIPProcessor,
        CLIPModel
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    HUGGINGFACE_AVAILABLE = False
    logger.warning("transformers not available, HuggingFace integration disabled")


# ============================================================================
# Text Embedding Extractors
# ============================================================================

if TRANSFORMERS_AVAILABLE:
    
    class TextEmbeddingExtractor:
        """
        Extract embeddings from text using HuggingFace transformers.
        
        Supports:
        - BERT, RoBERTa, DistilBERT (sentence encoders)
        - GPT-2, GPT-Neo (autoregressive)
        - Sentence-BERT (optimized for similarity)
        """
        
        def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            device: str = "auto"
        ):
            """
            Initialize text embedding extractor.
            
            Args:
                model_name: HuggingFace model name
                device: Device ('cuda', 'cpu', 'auto')
            """
            self.model_name = model_name
            
            # Determine device
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            # Load model and tokenizer
            logger.info(f"[HF-Text] Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            
            logger.info(f"[HF-Text] Loaded {model_name} on {self.device}")
            logger.info(f"  Embedding dim: {self.embedding_dim}")
        
        def extract(
            self,
            texts: Union[str, List[str]],
            pooling: str = "mean"
        ) -> np.ndarray:
            """
            Extract embeddings from text(s).
            
            Args:
                texts: Single text or list of texts
                pooling: Pooling strategy ('mean', 'max', 'cls')
                
            Returns:
                (n_texts, embedding_dim) array
            """
            if isinstance(texts, str):
                texts = [texts]
            
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Pool
            if pooling == "mean":
                # Mean pooling over tokens
                embeddings = outputs.last_hidden_state.mean(dim=1)
            elif pooling == "max":
                # Max pooling
                embeddings = outputs.last_hidden_state.max(dim=1)[0]
            elif pooling == "cls":
                # Use CLS token
                embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            
            return embeddings.cpu().numpy()
        
        def similarity(
            self,
            text1: Union[str, List[str]],
            text2: Union[str, List[str]]
        ) -> np.ndarray:
            """
            Compute cosine similarity between texts.
            
            Args:
                text1: First text(s)
                text2: Second text(s)
                
            Returns:
                Similarity scores
            """
            emb1 = self.extract(text1)
            emb2 = self.extract(text2)
            
            # Normalize
            emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
            emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
            
            # Cosine similarity
            similarity = np.sum(emb1 * emb2, axis=1)
            
            return similarity
    
    
    class ZeroShotClassifier:
        """
        Zero-shot classification using HuggingFace models.
        
        Classify text into arbitrary categories without training.
        """
        
        def __init__(
            self,
            model_name: str = "facebook/bart-large-mnli"
        ):
            """
            Initialize zero-shot classifier.
            
            Args:
                model_name: Model for zero-shot classification
            """
            self.model_name = model_name
            
            logger.info(f"[HF-ZeroShot] Loading {model_name}...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name
            )
            
            logger.info(f"[HF-ZeroShot] Ready")
        
        def classify(
            self,
            texts: Union[str, List[str]],
            candidate_labels: List[str],
            multi_label: bool = False
        ) -> List[Dict[str, Any]]:
            """
            Classify texts into candidate labels.
            
            Args:
                texts: Text(s) to classify
                candidate_labels: Possible labels
                multi_label: Whether multiple labels can apply
                
            Returns:
                List of classification results
            """
            if isinstance(texts, str):
                texts = [texts]
            
            results = []
            for text in texts:
                result = self.classifier(
                    text,
                    candidate_labels=candidate_labels,
                    multi_label=multi_label
                )
                results.append(result)
            
            return results
    
    
    class CLIPEmbeddingExtractor:
        """
        Extract embeddings from text and images using CLIP.
        
        CLIP provides joint text-image embeddings, enabling:
        - Image classification with text labels
        - Text-to-image retrieval
        - Cross-modal similarity
        """
        
        def __init__(
            self,
            model_name: str = "openai/clip-vit-base-patch32"
        ):
            """
            Initialize CLIP extractor.
            
            Args:
                model_name: CLIP model name
            """
            self.model_name = model_name
            
            logger.info(f"[HF-CLIP] Loading {model_name}...")
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            
            self.embedding_dim = self.model.config.projection_dim
            
            logger.info(f"[HF-CLIP] Loaded on {self.device}")
            logger.info(f"  Embedding dim: {self.embedding_dim}")
        
        def extract_text(
            self,
            texts: Union[str, List[str]]
        ) -> np.ndarray:
            """Extract text embeddings."""
            if isinstance(texts, str):
                texts = [texts]
            
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy()
        
        def extract_image(
            self,
            images: List[Any]  # PIL Images or arrays
        ) -> np.ndarray:
            """Extract image embeddings."""
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()


# ============================================================================
# THRML Integration (outside if block to work with stubs)
# ============================================================================

if TRANSFORMERS_AVAILABLE:
    class THRMLHuggingFaceAdapter:
        """
        Adapter for using HuggingFace embeddings with THRML.
        
        Enables:
        - Using embeddings as THRML node biases
        - Initializing THRML weights from embedding similarity
        - Pattern recognition with THRML
        """
        
        def __init__(
            self,
            thrml_wrapper,
            embedding_extractor: Optional[Union[TextEmbeddingExtractor, CLIPEmbeddingExtractor]] = None
        ):
            """
            Initialize adapter.
            
            Args:
                thrml_wrapper: THRMLWrapper instance
                embedding_extractor: HuggingFace embedding extractor
            """
            self.thrml_wrapper = thrml_wrapper
        self.embedding_extractor = embedding_extractor
        
        if embedding_extractor is not None:
            self.embedding_dim = embedding_extractor.embedding_dim
        else:
            self.embedding_dim = None
        
        logger.info(f"[THRML-HF] Adapter initialized")
        logger.info(f"  THRML nodes: {thrml_wrapper.n_nodes}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
    
    def set_biases_from_text(
        self,
        texts: List[str],
        scaling: float = 1.0,
        reduction: str = "pca"
    ):
        """
        Set THRML biases from text embeddings.
        
        Args:
            texts: List of texts (one per node or one for all)
            scaling: Scale factor for biases
            reduction: Dimensionality reduction ('pca', 'mean', 'sum')
        """
        if self.embedding_extractor is None:
            raise ValueError("No embedding extractor provided")
        
        # Extract embeddings
        embeddings = self.embedding_extractor.extract(texts)
        
        # Reduce to n_nodes dimension
        if embeddings.shape[0] == 1:
            # Single embedding, expand to all nodes
            biases = np.tile(embeddings[0][:self.thrml_wrapper.n_nodes], (1, 1))[0]
        elif embeddings.shape[0] == self.thrml_wrapper.n_nodes:
            # One embedding per node, reduce embedding dim
            if reduction == "mean":
                biases = np.mean(embeddings, axis=1)
            elif reduction == "sum":
                biases = np.sum(embeddings, axis=1)
            elif reduction == "pca":
                # Simple PCA: take first principal component
                centered = embeddings - np.mean(embeddings, axis=0)
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                pc1 = eigenvectors[:, -1]  # Largest eigenvalue
                biases = embeddings @ pc1
            else:
                raise ValueError(f"Unknown reduction: {reduction}")
        else:
            raise ValueError(f"Expected {self.thrml_wrapper.n_nodes} texts, got {embeddings.shape[0]}")
        
        # Scale and update
        biases = biases * scaling
        biases = np.nan_to_num(biases, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.thrml_wrapper.update_biases(biases)
        
        logger.info(f"[THRML-HF] Updated biases from {len(texts)} texts")
    
    def initialize_weights_from_similarity(
        self,
        texts: List[str],
        scaling: float = 0.1
    ):
        """
        Initialize THRML weights based on text similarity.
        
        Nodes with similar text get positive coupling.
        
        Args:
            texts: One text per node
            scaling: Scale factor for weights
        """
        if self.embedding_extractor is None:
            raise ValueError("No embedding extractor provided")
        
        if len(texts) != self.thrml_wrapper.n_nodes:
            raise ValueError(f"Expected {self.thrml_wrapper.n_nodes} texts, got {len(texts)}")
        
        # Extract embeddings
        embeddings = self.embedding_extractor.extract(texts)
        
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity = embeddings @ embeddings.T
        
        # Scale and symmetrize
        weights = similarity * scaling
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        
        # Update THRML weights
        # This requires rebuilding the wrapper with new weights
        # For now, just log
        logger.info(f"[THRML-HF] Computed similarity-based weights")
        logger.info(f"  Mean similarity: {np.mean(similarity):.3f}")
        logger.info(f"  Min/Max: {np.min(similarity):.3f} / {np.max(similarity):.3f}")
        
        return weights
    
    def classify_with_thrml(
        self,
        texts: List[str],
        class_labels: List[str],
        n_gibbs_steps: int = 100,
        temperature: float = 1.0,
        key: Any = None
    ) -> List[str]:
        """
        Classify texts using THRML + embeddings.
        
        1. Extract embeddings for texts and class labels
        2. Use similarity as energy bias
        3. Sample from THRML
        4. Assign to class with highest spin
        
        Args:
            texts: Texts to classify
            class_labels: Possible classes
            n_gibbs_steps: THRML sampling steps
            temperature: Sampling temperature
            key: JAX random key
            
        Returns:
            Predicted class labels
        """
        import jax.random as random
        
        if key is None:
            key = random.PRNGKey(0)
        
        if self.embedding_extractor is None:
            raise ValueError("No embedding extractor provided")
        
        # Extract embeddings
        text_embeddings = self.embedding_extractor.extract(texts)
        label_embeddings = self.embedding_extractor.extract(class_labels)
        
        # Normalize
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        label_embeddings = label_embeddings / np.linalg.norm(label_embeddings, axis=1, keepdims=True)
        
        predictions = []
        
        for text_emb in text_embeddings:
            # Compute similarity to each class
            similarities = label_embeddings @ text_emb
            
            # Use as biases (one node per class)
            class_biases = np.zeros(self.thrml_wrapper.n_nodes)
            n_classes = len(class_labels)
            class_biases[:n_classes] = similarities
            
            # Update THRML biases
            self.thrml_wrapper.update_biases(class_biases)
            
            # Sample
            key, subkey = random.split(key)
            samples = self.thrml_wrapper.sample_gibbs(
                n_steps=n_gibbs_steps,
                temperature=temperature,
                key=subkey
            )
            
            # Predict class with highest spin
            class_spins = samples[:n_classes]
            predicted_idx = np.argmax(class_spins)
            predicted_label = class_labels[predicted_idx]
            
            predictions.append(predicted_label)
        
        return predictions


# ============================================================================
# Convenience Functions
# ============================================================================

def create_text_embedding_thrml(
    thrml_wrapper,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> "THRMLHuggingFaceAdapter":
    """
    Create THRML-HuggingFace adapter with text embeddings.
    
    Args:
        thrml_wrapper: THRMLWrapper instance
        model_name: HuggingFace model name
        
    Returns:
        THRMLHuggingFaceAdapter
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers not available")
    
    extractor = TextEmbeddingExtractor(model_name=model_name)
    adapter = THRMLHuggingFaceAdapter(
        thrml_wrapper=thrml_wrapper,
        embedding_extractor=extractor
    )
    
    return adapter


def create_clip_thrml(
    thrml_wrapper,
    model_name: str = "openai/clip-vit-base-patch32"
) -> "THRMLHuggingFaceAdapter":
    """
    Create THRML-HuggingFace adapter with CLIP embeddings.
    
    Args:
        thrml_wrapper: THRMLWrapper instance
        model_name: CLIP model name
        
    Returns:
        THRMLHuggingFaceAdapter
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers not available")
    
    extractor = CLIPEmbeddingExtractor(model_name=model_name)
    adapter = THRMLHuggingFaceAdapter(
        thrml_wrapper=thrml_wrapper,
        embedding_extractor=extractor
    )
    
    return adapter


# ============================================================================
# Example Usage
# ============================================================================

def example_text_classification():
    """Example: Text classification with THRML + HuggingFace."""
    if not TRANSFORMERS_AVAILABLE:
        print("transformers not available, skipping example")
        return
    
    print("Text Classification Example\n")
    
    # Create THRML model
    from src.core.thrml_integration import create_thrml_model
    
    n_nodes = 16
    weights = np.random.randn(n_nodes, n_nodes) * 0.01
    weights = (weights + weights.T) / 2
    np.fill_diagonal(weights, 0)
    
    thrml_wrapper = create_thrml_model(
        n_nodes=n_nodes,
        weights=weights,
        biases=np.zeros(n_nodes),
        beta=1.0
    )
    
    # Create adapter
    adapter = create_text_embedding_thrml(thrml_wrapper)
    
    # Classify texts
    texts = [
        "The movie was excellent and entertaining",
        "This film was terrible and boring"
    ]
    class_labels = ["positive", "negative"]
    
    print(f"Classifying {len(texts)} texts into {class_labels}...\n")
    
    predictions = adapter.classify_with_thrml(
        texts=texts,
        class_labels=class_labels,
        n_gibbs_steps=50,
        temperature=1.0
    )
    
    for text, pred in zip(texts, predictions):
        print(f"Text: '{text}'")
        print(f"Prediction: {pred}\n")
    
    print("Example complete!")


if not TRANSFORMERS_AVAILABLE:
    # Stub classes when transformers not available
    class TextEmbeddingExtractor:
        pass
    
    class ZeroShotClassifier:
        pass
    
    class CLIPEmbeddingExtractor:
        pass
    
    class THRMLHuggingFaceAdapter:
        pass
    
    class HuggingFaceModelWrapper:
        pass
    
    class GMCSTextInterface:
        pass
    
    class GMCSPatternRecognizer:
        pass


if __name__ == '__main__':
    if TRANSFORMERS_AVAILABLE:
        example_text_classification()
    else:
        print("Transformers not available, skipping example")
