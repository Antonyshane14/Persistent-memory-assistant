"""
Main orchestrator - coordinates Agent A and Agent B.
This is the entry point for the memory system.
"""

from typing import Optional, Callable
import os

from .models import ConversationTurn
from .graph_store import GraphStore
from .embeddings import EmbeddingService, OpenAIEmbeddingService, MockEmbeddingService
from .agent_a import AgentA
from .agent_b import AgentB


class MemorySystem:
    """
    Dual-agent memory system orchestrator.

    Architecture:
    - Agent A: Handles user interaction + retrieval (read-only)
    - Agent B: Processes conversations + writes memories (write-only)
    - Graph DB: Primary memory store (Kuzu - unified memories + facts + relationships)
    """

    def __init__(
        self,
        llm_callable: Callable[[str], str],
        embedding_service: Optional[EmbeddingService] = None,
        storage_path: str = "./data",
        use_mock_embeddings: bool = False
    ):
        """
        Initialize the memory system.

        Args:
            llm_callable: Function that takes a prompt and returns LLM response
            embedding_service: Optional custom embedding service
            storage_path: Where to store graph DB
            use_mock_embeddings: Use mock embeddings for testing (no API calls)
        """
        # Initialize embedding service
        if embedding_service:
            self.embedding_service = embedding_service
        elif use_mock_embeddings:
            print("Using mock embeddings (no API calls)")
            self.embedding_service = MockEmbeddingService(dimension=1536)
        else:
            self.embedding_service = OpenAIEmbeddingService()

        # Determine embedding dimension (for local models)
        if hasattr(self.embedding_service, 'get_dimension'):
            dimension = self.embedding_service.get_dimension()
        elif hasattr(self.embedding_service, 'dimension'):
            dimension = self.embedding_service.dimension
        else:
            dimension = 1536  # Default for OpenAI

        # Initialize graph store (replaces both VectorStore and FactsDatabase)
        self.graph_store = GraphStore(
            dimension=dimension,
            storage_path=storage_path
        )

        # Aliases for backwards compatibility with agents
        self.vector_store = self.graph_store  # GraphStore implements VectorStore interface
        self.facts_db = self.graph_store       # GraphStore implements FactsDatabase interface

        # Initialize agents (they use vector_store and facts_db interfaces)
        self.agent_a = AgentA(self.graph_store, self.embedding_service, llm_callable, self.graph_store)
        self.agent_b = AgentB(self.graph_store, self.embedding_service, llm_callable, self.graph_store)

        # Storage path for conversation logs
        self.storage_path = os.path.abspath(storage_path)
        self.conversations_dir = os.path.join(self.storage_path, "conversations")
        os.makedirs(self.conversations_dir, exist_ok=True)

        print(f"✓ Memory system initialized (Graph DB: Kuzu)")
        print(f"  - Storage: {storage_path}")
        print(f"  - Memories loaded: {len(self.graph_store.memories)}")
        print(f"  - Facts loaded: {len(self.graph_store.facts)}")
    
    def chat(self, user_message: str, llm_callable: Callable[[str], str]) -> str:
        """
        Process a user message and return response.
        
        This is the main entry point for conversation.
        Agent A handles retrieval and response generation.
        
        Args:
            user_message: User's input
            llm_callable: Function to generate LLM response
        
        Returns:
            Assistant's response
        """
        return self.agent_a.process_user_message(user_message, llm_callable)
    
    def save_conversation_to_file(self) -> str:
        """
        Save current conversation to a TXT file.
        
        Returns:
            Path to saved file
        """
        from datetime import datetime
        
        session_history = self.agent_a.get_session_history()
        if not session_history:
            return None
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.txt"
        filepath = os.path.join(self.conversations_dir, filename)
        
        # Write conversation to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== Conversation Log ===\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total turns: {len(session_history)}\n")
            f.write(f"\n{'='*50}\n\n")
            
            for i, turn in enumerate(session_history, 1):
                f.write(f"Turn {i}:\n")
                f.write(f"You: {turn.user_message}\n")
                f.write(f"Assistant: {turn.assistant_message}\n")
                f.write(f"\n{'-'*50}\n\n")
        
        return filepath
    
    def process_memories(self, chunk_size: int = 5, save_to_file: bool = True) -> int:
        """
        Process conversation history and extract memories.
        
        This should be called periodically (e.g., after N turns or end of session).
        Agent B handles memory extraction and writing.
        
        Args:
            chunk_size: Number of conversation turns per processing chunk
            save_to_file: Whether to save conversation to TXT file first
        
        Returns:
            Number of new memories written
        """
        session_history = self.agent_a.get_session_history()
        
        if not session_history:
            print("No conversation history to process")
            return 0
        
        # Save conversation to file
        if save_to_file:
            filepath = self.save_conversation_to_file()
            if filepath:
                print(f"💾 Conversation saved: {os.path.basename(filepath)}")
        
        print(f"🧠 Processing {len(session_history)} conversation turns...")
        memories_written = self.agent_b.process_session(session_history, chunk_size)
        print(f"✓ Extracted {memories_written} new memories")
        
        # Create conversation-level summary (captures overall theme)
        print("📊 Creating conversation summary...")
        if self.agent_b.create_conversation_summary(session_history):
            print("✓ Conversation summary created\n")
        else:
            print("✓ Conversation too short for summary\n")
        
        # Save facts database if updated
        if self.facts_db and len(self.facts_db.facts) > 0:
            self.facts_db.save()
        
        return memories_written
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "vector_store": self.vector_store.get_stats(),
            "current_session_turns": len(self.agent_a.session_history)
        }
    
    def clear_session(self):
        """Clear current session (start fresh conversation)."""
        self.agent_a.clear_session()
        print("✓ Session cleared")
    
    def list_memories(self, limit: int = 10):
        """
        List recent memories for debugging.
        
        Args:
            limit: Maximum number of memories to show
        """
        memories = self.vector_store.memories[-limit:]
        
        if not memories:
            print("No memories stored yet")
            return
        
        print(f"\n📚 Recent memories (showing {len(memories)}):")
        for i, mem in enumerate(reversed(memories), 1):
            # Use event_date if available (when event happened), otherwise created_at (when stored)
            display_date = mem.event_date if mem.event_date else mem.created_at
            date_str = display_date.strftime('%b %d, %Y %I:%M %p')
            date_label = "Event" if mem.event_date else "Stored"
            
            print(f"\n{i}. [{mem.category.value}] confidence: {mem.confidence:.2f}")
            print(f"   {mem.summary}")
            print(f"   📅 {date_label}: {date_str}")
