"""
Core data models for the dual-agent memory system.
Vector-first design: most data lives in the vector DB.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class MemoryCategory(Enum):
    """Memory categories for semantic organization."""
    IDENTITY = "identity"
    PREFERENCE = "preference"
    TECH_STACK = "tech_stack"
    DECISION = "decision"
    EPISODIC = "episodic"
    CONSTRAINT = "constraint"


@dataclass
class Memory:
    """
    A single memory unit - the atomic element of long-term memory.
    Stored in vector DB with its embedding.
    
    Design principle: Summaries must be LLM-readable and concise.
    """
    memory_id: str
    summary: str  # The actual memory content (gets embedded)
    category: MemoryCategory
    confidence: float  # 0.0 - 1.0
    source: str
    created_at: datetime
    last_reinforced: datetime
    event_date: Optional[datetime] = None  # When the event actually happened (vs when memory was stored)
    
    def __post_init__(self):
        """Validate memory constraints."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if len(self.summary) < 10:
            raise ValueError("Summary too short - must be semantic unit")
    
    @staticmethod
    def create_new(summary: str, category: MemoryCategory, confidence: float = 0.6, event_date: Optional[datetime] = None) -> 'Memory':
        """Factory method for creating new memories with defaults."""
        now = datetime.now()
        return Memory(
            memory_id=str(uuid.uuid4()),
            summary=summary,
            category=category,
            confidence=confidence,
            source="conversation",
            created_at=now,
            last_reinforced=now,
            event_date=event_date  # When event actually happened (None = unknown/general knowledge)
        )
    
    def reinforce(self, boost: float = 0.05):
        """Reinforce existing memory - increases confidence."""
        self.confidence = min(1.0, self.confidence + boost)
        self.last_reinforced = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = {
            "memory_id": self.memory_id,
            "summary": self.summary,
            "category": self.category.value,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "last_reinforced": self.last_reinforced.isoformat()
        }
        # Add event_date only if it exists
        if self.event_date:
            data["event_date"] = self.event_date.isoformat()
        return data
    
    @staticmethod
    def from_dict(data: dict) -> 'Memory':
        """Reconstruct from dictionary."""
        # Parse event_date if it exists (backward compatible)
        event_date = None
        if "event_date" in data and data["event_date"]:
            event_date = datetime.fromisoformat(data["event_date"])
        
        return Memory(
            memory_id=data["memory_id"],
            summary=data["summary"],
            category=MemoryCategory(data["category"]),
            confidence=data["confidence"],
            source=data["source"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_reinforced=datetime.fromisoformat(data["last_reinforced"]),
            event_date=event_date
        )


@dataclass
class RetrievedMemory:
    """
    Memory + retrieval metadata for ranking and context building.
    Used by Agent A during retrieval.
    """
    memory: Memory
    similarity: float
    final_score: float
    
    def format_for_prompt(self) -> str:
        """Format memory for inclusion in LLM prompt."""
        return f"- {self.memory.summary} (category: {self.memory.category.value}, confidence: {self.memory.confidence:.2f})"


@dataclass
class ConversationTurn:
    """Single conversation exchange for Agent B processing."""
    user_message: str
    assistant_message: str
    timestamp: datetime
    
    @staticmethod
    def create(user_msg: str, assistant_msg: str) -> 'ConversationTurn':
        return ConversationTurn(user_msg, assistant_msg, datetime.now())
