"""
Dual-Agent Graph-First Memory System
Provides long-term memory for LLMs using graph-based semantic retrieval.
"""

__version__ = "2.0.0"

from .memory_system import MemorySystem
from .models import Memory, MemoryCategory, ConversationTurn
from .agent_a import AgentA
from .agent_b import AgentB
from .graph_store import GraphStore
from .embeddings import EmbeddingService, OpenAIEmbeddingService, MockEmbeddingService

__all__ = [
    "MemorySystem",
    "Memory",
    "MemoryCategory",
    "ConversationTurn",
    "AgentA",
    "AgentB",
    "GraphStore",
    "EmbeddingService",
    "OpenAIEmbeddingService",
    "MockEmbeddingService",
]
