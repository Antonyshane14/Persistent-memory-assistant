"""
Vector store abstraction using Qdrant.
This is the PRIMARY memory store - source of truth.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from .models import Memory, RetrievedMemory


class VectorStore:
    """
    Qdrant-based vector store for semantic memory retrieval.

    Design:
    - Uses Qdrant in local mode (file-based persistence)
    - Unified storage: vectors + metadata in single database
    - Native support for in-place updates and filtering
    - Automatic persistence (no manual save/load needed)
    """

    def __init__(self, dimension: int = 1536, storage_path: str = "./data"):
        """
        Initialize vector store.

        Args:
            dimension: Embedding dimension (OpenAI ada-002 = 1536, Ollama llama3.1 = 4096)
            storage_path: Where to persist Qdrant database
        """
        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Initialize Qdrant client in local mode (embedded)
        qdrant_path = self.storage_path / "qdrant_db"
        self.client = QdrantClient(path=str(qdrant_path))
        self.collection_name = "memories"

        # Metadata store - local cache for fast access
        self.memories: List[Memory] = []

        # Initialize collection and load existing data
        self._initialize_collection()
        self._load_memories()

    def _initialize_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE  # Direct cosine similarity
                )
            )
            print(f"✓ Created Qdrant collection '{self.collection_name}' with dimension {self.dimension}")
        else:
            print(f"✓ Using existing Qdrant collection '{self.collection_name}'")

    def add_memory(self, memory: Memory, embedding: np.ndarray):
        """
        Add a new memory with its embedding.

        Args:
            memory: Memory object
            embedding: Vector embedding of memory.summary
        """
        if embedding.shape[0] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.shape[0]}")

        # Normalize for cosine similarity (recommended for consistency)
        embedding = self._normalize(embedding)

        # Generate unique point ID from memory_id
        point_id = abs(hash(memory.memory_id)) % (10 ** 10)

        # Create point with vector + payload (metadata)
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=memory.to_dict()
        )

        # Upsert to Qdrant (automatically persists)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        # Update local cache
        self.memories.append(memory)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 8,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[Memory, float]]:
        """
        Search for similar memories.

        Args:
            query_embedding: Query vector
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score (cosine)

        Returns:
            List of (Memory, similarity_score) tuples
        """
        if len(self.memories) == 0:
            return []

        # Normalize query
        query_embedding = self._normalize(query_embedding)

        # Search Qdrant with native score threshold
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=top_k,
            score_threshold=similarity_threshold
        )

        # Convert to Memory objects with similarity scores
        results = []
        for hit in search_results.points:
            memory = Memory.from_dict(hit.payload)
            similarity = hit.score
            results.append((memory, float(similarity)))

        return results

    def find_duplicates(self, embedding: np.ndarray, threshold: float = 0.9) -> Optional[Tuple[Memory, float]]:
        """
        Check if a memory already exists (for deduplication).

        Args:
            embedding: Embedding of potential new memory
            threshold: Similarity threshold for duplicate detection

        Returns:
            (Memory, similarity) if duplicate found, else None
        """
        results = self.search(embedding, top_k=1, similarity_threshold=threshold)
        return results[0] if results else None

    def update_memory(self, memory_id: str, updated_memory: Memory):
        """
        Update an existing memory (e.g., after reinforcement).
        Qdrant supports native in-place updates of metadata.
        """
        # Find point by memory_id using scroll with filter
        search_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="memory_id",
                        match=MatchValue(value=memory_id)
                    )
                ]
            ),
            limit=1
        )

        if not search_result[0]:
            raise ValueError(f"Memory {memory_id} not found")

        point_id = search_result[0][0].id

        # Update payload (vector remains unchanged)
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=updated_memory.to_dict(),
            points=[point_id]
        )

        # Update local cache
        for idx, mem in enumerate(self.memories):
            if mem.memory_id == memory_id:
                self.memories[idx] = updated_memory
                break

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _load_memories(self):
        """Load memories from Qdrant to populate local cache."""
        try:
            # Check if collection has any points
            collection_info = self.client.get_collection(self.collection_name)
            if collection_info.points_count == 0:
                print("✓ No existing memories found (new database)")
                return

            # Scroll through all points to load into cache
            offset = None
            loaded_count = 0

            while True:
                records, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset
                )

                for record in records:
                    memory = Memory.from_dict(record.payload)
                    self.memories.append(memory)
                    loaded_count += 1

                if offset is None:
                    break

            print(f"✓ Loaded {loaded_count} memories from Qdrant")
        except Exception as e:
            print(f"⚠ Error loading memories: {e}")

    def get_stats(self) -> dict:
        """Get statistics about the memory store."""
        return {
            "total_memories": len(self.memories),
            "dimension": self.dimension,
            "categories": {cat.value: sum(1 for m in self.memories if m.category == cat)
                          for cat in set(m.category for m in self.memories)}
        }
