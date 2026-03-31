"""
Local LLM integration using Ollama.
Supports both chat completion and embeddings with local models.
"""

import requests
import numpy as np
from typing import List, Optional
import json


class OllamaLLM:
    """
    Ollama LLM service for local model inference.
    Works with models running in WSL or local Ollama instance.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b-instruct-q4_K_M"):
        """
        Initialize Ollama LLM client.
        
        Args:
            base_url: Ollama API base URL (default: localhost:11434)
                     For WSL: http://localhost:11434
            model: Model name in Ollama (e.g., llama3.1:8b-instruct-q4_K_M)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        
        # Test connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if self.model not in model_names:
                print(f"⚠️  Model '{self.model}' not found in Ollama")
                print(f"Available models: {', '.join(model_names)}")
                print(f"\nTo pull the model, run in WSL:")
                print(f"  ollama pull {self.model}")
            else:
                print(f"✓ Connected to Ollama - Model: {self.model}")
        
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Cannot connect to Ollama at {self.base_url}")
            print(f"Error: {e}")
            print("\nMake sure Ollama is running:")
            print("  In WSL: ollama serve")
            print("  Or: systemctl start ollama")
    
    def generate(self, prompt: str, temperature: float = 0.7, stream: bool = False, timeout: int = 60) -> str:
        """
        Generate text completion using local model.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            stream: Whether to stream response
            timeout: Request timeout in seconds (default 60)
        
        Returns:
            Generated text
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": stream
                },
                timeout=timeout
            )
            response.raise_for_status()
            
            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'response' in data:
                            full_response += data['response']
                return full_response
            else:
                return response.json()['response']
        
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    def chat(self, messages: List[dict], temperature: float = 0.7) -> str:
        """
        Chat completion using local model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
        
        Returns:
            Assistant's response
        """
        # Format messages into a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt_parts.append(f"System: {content}\n")
            elif role == 'user':
                prompt_parts.append(f"User: {content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        
        return self.generate(prompt, temperature=temperature)


class OllamaEmbedding:
    """
    Ollama embedding service using local models.
    Uses Ollama's embedding endpoint for consistent vector representations.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1:8b-instruct-q4_K_M"):
        """
        Initialize Ollama embedding client.
        
        Args:
            base_url: Ollama API base URL
            model: Model name for embeddings
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.dimension = None  # Will be determined on first embed
        
        print(f"✓ Ollama Embeddings initialized - Model: {self.model}")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Numpy array of embedding vector
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            
            embedding = np.array(response.json()['embedding'], dtype=np.float32)
            
            # Cache dimension on first call
            if self.dimension is None:
                self.dimension = len(embedding)
                print(f"✓ Embedding dimension: {self.dimension}")
            
            return embedding
        
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            # Fallback: return zero vector with expected dimension
            if self.dimension:
                return np.zeros(self.dimension, dtype=np.float32)
            else:
                # Default to 4096 (common for Llama models)
                return np.zeros(4096, dtype=np.float32)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension (lazy loaded).
        
        Returns:
            Embedding dimension
        """
        if self.dimension is None:
            # Test with a short text to determine dimension
            test_embedding = self.embed("test")
            self.dimension = len(test_embedding)
        return self.dimension


def create_local_system(model: str = "llama3.1:8b-instruct-q4_K_M", base_url: str = "http://localhost:11434"):
    """
    Factory function to create a memory system with local Ollama models.
    
    Args:
        model: Ollama model name
        base_url: Ollama API URL (use http://localhost:11434 for WSL)
    
    Returns:
        Tuple of (llm_callable, embedding_service, dimension)
    """
    from .embeddings import EmbeddingService
    
    # Initialize Ollama services
    llm = OllamaLLM(base_url=base_url, model=model)
    embedding_service = OllamaEmbedding(base_url=base_url, model=model)
    
    # Get dimension (will test connection)
    dimension = embedding_service.get_dimension()
    
    # Create LLM callable for memory system with low temperature to reduce hallucination
    def llm_callable(prompt: str) -> str:
        return llm.generate(prompt, temperature=0.4)
    
    return llm_callable, embedding_service, dimension
