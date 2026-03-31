"""
Embedding service abstraction.
Supports OpenAI and can be extended to other providers.
"""

import numpy as np
from typing import List, Optional
import os


class EmbeddingService:
    """
    Abstract interface for embedding generation.
    Vector-first design requires consistent embeddings.
    """
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        raise NotImplementedError
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError


class OpenAIEmbeddingService(EmbeddingService):
    """
    OpenAI embedding service using text-embedding-ada-002.
    Dimension: 1536
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
        """
        Initialize OpenAI embedding service.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Embedding model to use
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]


class MockEmbeddingService(EmbeddingService):
    """
    Mock embedding service for testing without API calls.
    Generates random but consistent embeddings based on text hash.
    """
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
    
    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic random embedding from text hash."""
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.dimension).astype(np.float32)
        # Normalize
        return embedding / np.linalg.norm(embedding)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]
