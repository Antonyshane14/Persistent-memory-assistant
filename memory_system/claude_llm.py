"""
Claude API integration via OpenRouter for better JSON parsing and memory extraction.
OpenRouter provides unified access to Claude and other LLMs.
"""

import os
from typing import List, Optional
from openai import OpenAI


class ClaudeLLM:
    """
    Claude API service via OpenRouter.
    Better at structured output (JSON) compared to local models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-3.5-sonnet"
    ):
        """
        Initialize Claude LLM client via OpenRouter.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model to use (default: anthropic/claude-3.5-sonnet)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided and OPENROUTER_API_KEY not set")

        # Initialize OpenAI client with OpenRouter endpoint
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model = model
        print(f"✓ Connected to OpenRouter - Model: {self.model}")

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """
        Generate text completion using Claude via OpenRouter.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract text from response
            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ OpenRouter API call failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now."

    def chat(self, messages: List[dict], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """
        Chat completion using Claude via OpenRouter.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Assistant's response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ OpenRouter chat failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now."


def create_claude_system(model: str = None):
    """
    Factory function to create a memory system with Claude via OpenRouter.

    Args:
        model: Model name (OpenRouter format: provider/model).
              If None, uses OPENROUTER_MODEL from .env or defaults to anthropic/claude-3.5-sonnet

    Returns:
        Tuple of (llm_callable, embedding_service, dimension)
    """
    from .embeddings import MockEmbeddingService

    # Get model from env if not provided
    if model is None:
        model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

    # Initialize Claude LLM via OpenRouter
    claude = ClaudeLLM(model=model)

    # Use mock embeddings (4096 dimension to match Ollama)
    # In production, you'd use a real embedding service
    embedding_service = MockEmbeddingService(dimension=4096)
    dimension = 4096

    print(f"✓ Using OpenRouter API for LLM + Mock embeddings (dimension: {dimension})")

    # Create LLM callable with low temperature for factual responses
    def llm_callable(prompt: str) -> str:
        return claude.generate(prompt, temperature=0.3, max_tokens=4096)

    return llm_callable, embedding_service, dimension
