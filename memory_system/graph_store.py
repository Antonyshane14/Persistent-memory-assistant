"""
Graph-based memory store using Kuzu (embedded graph database).
Replaces both Qdrant (VectorStore) and SQLite (FactsDatabase).

Graph schema
------------
Nodes:
  Memory   — summary text + embedding vector
  Entity   — people, places, organisations Antony mentions
  DateNode — calendar date nodes (YYYY-MM-DD)
  FactNode — structured (subject, predicate, value) triples

Relationships:
  Memory  -[MENTIONS]->   Entity    (confidence DOUBLE)
  Memory  -[OCCURRED_ON]-> DateNode
  Entity  -[KNOWS]->      Entity    (relationship_type STRING, confidence DOUBLE)

Drop-in compatible with:
  VectorStore  — add_memory, search, find_duplicates, update_memory, get_stats, .memories
  FactsDatabase — add_fact, search_facts, query_by_date, query_relationships, save, .facts
"""

import json
import uuid
import numpy as np
import kuzu
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .models import Memory, MemoryCategory, RetrievedMemory


# ──────────────────────────────────────────────────────────────────────────────
# Fact dataclass (mirrors facts_db_sql.Fact for agent_b.py compatibility)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Fact:
    """
    Structured fact stored as a FactNode in the graph.
    Drop-in replacement for facts_db_sql.Fact.
    """
    fact_id: str
    fact_type: str        # person | relationship | organization | event | achievement
    subject: str          # "Antony"
    predicate: str        # "has girlfriend" | "knows person" | "event on date"
    value: Any            # name, date string, number, etc.
    date: Optional[str]   # ISO date when fact occurred (YYYY-MM-DD or None)
    confidence: float
    created_at: str       # ISO datetime string

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "fact_type": self.fact_type,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": str(self.value),
            "date": self.date,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }


# ──────────────────────────────────────────────────────────────────────────────
# GraphStore
# ──────────────────────────────────────────────────────────────────────────────

class GraphStore:
    """
    Kuzu-backed graph store — single source of truth for memories and facts.

    Embedding vectors are serialised as JSON strings inside Memory nodes.
    Cosine similarity search is done in NumPy over the local cache (fine for
    personal-assistant scale; no external server required).
    """

    def __init__(self, dimension: int = 1536, storage_path: str = "./data"):
        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        db_path = str(self.storage_path / "graph_db")
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)

        # In-memory caches — mirrors VectorStore.memories / FactsDatabase.facts
        self.memories: List[Memory] = []
        self.facts: List[Fact] = []

        # Embedding cache for fast similarity search (memory_id → normalised vector)
        self._embedding_cache: dict = {}

        self._initialize_schema()
        self._load_memories()
        self._load_facts()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _initialize_schema(self):
        """Create node and relationship tables (idempotent)."""
        node_tables = [
            """CREATE NODE TABLE IF NOT EXISTS Memory(
                memory_id   STRING,
                summary     STRING,
                category    STRING,
                confidence  DOUBLE,
                source      STRING,
                created_at  STRING,
                last_reinforced STRING,
                event_date  STRING,
                embedding   STRING,
                PRIMARY KEY(memory_id)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS Entity(
                name            STRING,
                entity_type     STRING,
                first_mentioned STRING,
                mention_count   INT64,
                PRIMARY KEY(name)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS DateNode(
                date_str STRING,
                PRIMARY KEY(date_str)
            )""",
            """CREATE NODE TABLE IF NOT EXISTS FactNode(
                fact_id   STRING,
                fact_type STRING,
                subject   STRING,
                predicate STRING,
                value     STRING,
                date      STRING,
                confidence DOUBLE,
                created_at STRING,
                PRIMARY KEY(fact_id)
            )""",
        ]

        rel_tables = [
            "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Memory TO Entity, confidence DOUBLE)",
            "CREATE REL TABLE IF NOT EXISTS OCCURRED_ON(FROM Memory TO DateNode)",
            "CREATE REL TABLE IF NOT EXISTS KNOWS(FROM Entity TO Entity, relationship_type STRING, confidence DOUBLE)",
        ]

        for stmt in node_tables + rel_tables:
            try:
                self.conn.execute(stmt)
            except Exception:
                pass  # Table already exists

        print("✓ Graph schema ready")

    # ── VectorStore interface ─────────────────────────────────────────────────

    def add_memory(self, memory: Memory, embedding: np.ndarray):
        """Insert a Memory node with its embedding vector."""
        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.shape[0]}"
            )

        vec = self._normalize(embedding)
        embedding_json = json.dumps(vec.tolist())
        event_date_str = memory.event_date.isoformat() if memory.event_date else ""

        self.conn.execute(
            """CREATE (:Memory {
                memory_id: $id, summary: $summary, category: $category,
                confidence: $conf, source: $source,
                created_at: $created_at, last_reinforced: $lr,
                event_date: $event_date, embedding: $embedding
            })""",
            {
                "id": memory.memory_id,
                "summary": memory.summary,
                "category": memory.category.value,
                "conf": memory.confidence,
                "source": memory.source,
                "created_at": memory.created_at.isoformat(),
                "lr": memory.last_reinforced.isoformat(),
                "event_date": event_date_str,
                "embedding": embedding_json,
            },
        )

        # Link to DateNode for temporal graph queries
        if memory.event_date:
            date_str = memory.event_date.strftime("%Y-%m-%d")
            self._ensure_date_node(date_str)
            try:
                self.conn.execute(
                    """MATCH (m:Memory {memory_id: $mid}), (d:DateNode {date_str: $ds})
                       CREATE (m)-[:OCCURRED_ON]->(d)""",
                    {"mid": memory.memory_id, "ds": date_str},
                )
            except Exception:
                pass

        self.memories.append(memory)
        self._embedding_cache[memory.memory_id] = vec

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 8,
        similarity_threshold: float = 0.7,
    ) -> List[Tuple[Memory, float]]:
        """
        Semantic search using cosine similarity.
        Uses in-memory embedding cache for speed; falls back to graph DB if cache is cold.
        """
        if not self.memories:
            return []

        query_vec = self._normalize(query_embedding)

        # Warm cache if empty (e.g., after restart)
        if not self._embedding_cache:
            self._warm_embedding_cache()

        mem_map = {m.memory_id: m for m in self.memories}
        candidates = []

        for mem_id, vec in self._embedding_cache.items():
            sim = float(np.dot(query_vec, vec))
            if sim >= similarity_threshold:
                candidates.append((mem_id, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)

        return [
            (mem_map[mid], sim)
            for mid, sim in candidates[:top_k]
            if mid in mem_map
        ]

    def find_duplicates(
        self, embedding: np.ndarray, threshold: float = 0.9
    ) -> Optional[Tuple[Memory, float]]:
        """Return the most similar memory if cosine similarity ≥ threshold."""
        results = self.search(embedding, top_k=1, similarity_threshold=threshold)
        return results[0] if results else None

    def update_memory(self, memory_id: str, updated_memory: Memory):
        """Update confidence and last_reinforced for an existing Memory node."""
        self.conn.execute(
            """MATCH (m:Memory {memory_id: $id})
               SET m.confidence = $conf, m.last_reinforced = $lr""",
            {
                "id": memory_id,
                "conf": updated_memory.confidence,
                "lr": updated_memory.last_reinforced.isoformat(),
            },
        )
        for idx, mem in enumerate(self.memories):
            if mem.memory_id == memory_id:
                self.memories[idx] = updated_memory
                break

    def get_stats(self) -> dict:
        """Return memory/entity/fact counts and category breakdown."""
        r = self.conn.execute("MATCH (e:Entity) RETURN count(e)")
        entity_count = r.get_next()[0] if r.has_next() else 0

        r2 = self.conn.execute("MATCH (f:FactNode) RETURN count(f)")
        fact_count = r2.get_next()[0] if r2.has_next() else 0

        cats = {}
        for m in self.memories:
            cats[m.category.value] = cats.get(m.category.value, 0) + 1

        return {
            "total_memories": len(self.memories),
            "dimension": self.dimension,
            "entity_count": entity_count,
            "fact_count": fact_count,
            "categories": cats,
        }

    # ── FactsDatabase interface ───────────────────────────────────────────────

    def add_fact(self, fact: Fact) -> bool:
        """
        Add a structured fact.
        Skips duplicates (same subject + predicate + value).
        Creates Entity nodes and KNOWS edges for persons and relationships.
        """
        # Deduplication
        dup = self.conn.execute(
            """MATCH (f:FactNode)
               WHERE f.subject = $s AND f.predicate = $p AND f.value = $v
               RETURN f.fact_id LIMIT 1""",
            {"s": fact.subject, "p": fact.predicate, "v": str(fact.value)},
        )
        if dup.has_next():
            return False

        try:
            self.conn.execute(
                """CREATE (:FactNode {
                    fact_id: $fid, fact_type: $ftype, subject: $subj,
                    predicate: $pred, value: $val, date: $date,
                    confidence: $conf, created_at: $cat
                })""",
                {
                    "fid": fact.fact_id,
                    "ftype": fact.fact_type,
                    "subj": fact.subject,
                    "pred": fact.predicate,
                    "val": str(fact.value),
                    "date": fact.date or "",
                    "conf": fact.confidence,
                    "cat": fact.created_at,
                },
            )
        except Exception:
            return False  # Duplicate fact_id

        # Build graph edges for persons and relationships
        if fact.fact_type in ("person", "relationship"):
            self._ensure_entity(str(fact.value), "person")
            from_name = fact.subject if fact.subject.lower() not in ("antony", "antony shane") else "Antony Shane"
            self._ensure_entity(from_name, "person")
            try:
                self.conn.execute(
                    """MATCH (a:Entity {name: $fn}), (b:Entity {name: $tn})
                       CREATE (a)-[:KNOWS {relationship_type: $rt, confidence: $conf}]->(b)""",
                    {
                        "fn": from_name,
                        "tn": str(fact.value),
                        "rt": fact.predicate,
                        "conf": fact.confidence,
                    },
                )
            except Exception:
                pass

        elif fact.fact_type == "organization":
            self._ensure_entity(str(fact.value), "organization")

        self.facts.append(fact)
        return True

    def search_facts(self, query: str) -> List[Fact]:
        """Full-text keyword search across subject, predicate, and value."""
        result = self.conn.execute(
            """MATCH (f:FactNode)
               WHERE lower(f.subject) CONTAINS $q
                  OR lower(f.predicate) CONTAINS $q
                  OR lower(f.value) CONTAINS $q
               RETURN f.fact_id, f.fact_type, f.subject, f.predicate,
                      f.value, f.date, f.confidence, f.created_at
               LIMIT 20""",
            {"q": query.lower()},
        )
        return self._rows_to_facts(result)

    def query_by_date(self, date: str) -> List[Fact]:
        """Return all facts whose date field contains the given date string."""
        result = self.conn.execute(
            """MATCH (f:FactNode)
               WHERE f.date CONTAINS $date
               RETURN f.fact_id, f.fact_type, f.subject, f.predicate,
                      f.value, f.date, f.confidence, f.created_at
               LIMIT 20""",
            {"date": date},
        )
        return self._rows_to_facts(result)

    def query_relationships(self, person_name: str = None) -> List[Fact]:
        """
        Graph traversal: find all KNOWS edges involving a person.
        Returns Fact objects for drop-in compatibility with FactsDatabase.
        """
        if person_name:
            result = self.conn.execute(
                """MATCH (a:Entity {name: $name})-[r:KNOWS]-(b:Entity)
                   RETURN a.name, r.relationship_type, b.name, r.confidence""",
                {"name": person_name},
            )
        else:
            result = self.conn.execute(
                """MATCH (a:Entity)-[r:KNOWS]-(b:Entity)
                   RETURN a.name, r.relationship_type, b.name, r.confidence"""
            )

        facts = []
        while result.has_next():
            row = result.get_next()
            facts.append(
                Fact(
                    fact_id=str(uuid.uuid4()),
                    fact_type="relationship",
                    subject=row[0],
                    predicate=row[1] or "knows",
                    value=row[2],
                    date=None,
                    confidence=row[3] or 0.7,
                    created_at=datetime.now().isoformat(),
                )
            )
        return facts

    def save(self):
        """No-op: Kuzu auto-persists every write. Kept for FactsDatabase API compatibility."""
        pass

    # ── Graph-native extras (no SQLite/Qdrant equivalent) ────────────────────

    def get_entity_graph(self, entity_name: str) -> dict:
        """
        Return all first-degree connections and their relationship types.
        Unique to the graph model — enables richer context for the LLM.
        """
        result = self.conn.execute(
            """MATCH (a:Entity {name: $name})-[r:KNOWS]-(b:Entity)
               RETURN a.name, r.relationship_type, b.name""",
            {"name": entity_name},
        )
        nodes = {entity_name}
        edges = []
        while result.has_next():
            row = result.get_next()
            nodes.add(row[2])
            edges.append({"from": row[0], "rel": row[1] or "knows", "to": row[2]})
        return {"nodes": list(nodes), "edges": edges}

    def get_memories_by_entity(self, entity_name: str) -> List[Memory]:
        """Return all Memory nodes that MENTION a specific entity."""
        result = self.conn.execute(
            """MATCH (m:Memory)-[:MENTIONS]->(e:Entity {name: $name})
               RETURN m.memory_id""",
            {"name": entity_name},
        )
        mem_map = {m.memory_id: m for m in self.memories}
        memories = []
        while result.has_next():
            mid = result.get_next()[0]
            if mid in mem_map:
                memories.append(mem_map[mid])
        return memories

    def link_memory_to_entity(
        self, memory_id: str, entity_name: str, confidence: float = 0.8
    ):
        """Create a MENTIONS edge between a Memory node and an Entity node."""
        self._ensure_entity(entity_name, "person")
        try:
            self.conn.execute(
                """MATCH (m:Memory {memory_id: $mid}), (e:Entity {name: $en})
                   CREATE (m)-[:MENTIONS {confidence: $conf}]->(e)""",
                {"mid": memory_id, "en": entity_name, "conf": confidence},
            )
        except Exception:
            pass

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_entity(self, name: str, entity_type: str = "person"):
        """Upsert: create Entity node if absent, else increment mention_count."""
        try:
            self.conn.execute(
                """CREATE (:Entity {
                    name: $name, entity_type: $et,
                    first_mentioned: $now, mention_count: 1
                })""",
                {"name": name, "et": entity_type, "now": datetime.now().isoformat()},
            )
        except Exception:
            try:
                self.conn.execute(
                    "MATCH (e:Entity {name: $name}) SET e.mention_count = e.mention_count + 1",
                    {"name": name},
                )
            except Exception:
                pass

    def _ensure_date_node(self, date_str: str):
        """Create a DateNode for a given YYYY-MM-DD string if it doesn't exist."""
        try:
            self.conn.execute(
                "CREATE (:DateNode {date_str: $ds})", {"ds": date_str}
            )
        except Exception:
            pass  # Already exists

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def _warm_embedding_cache(self):
        """Load all embeddings from graph DB into in-memory cache."""
        try:
            result = self.conn.execute(
                "MATCH (m:Memory) RETURN m.memory_id, m.embedding"
            )
            while result.has_next():
                row = result.get_next()
                mem_id, emb_json = row[0], row[1]
                if emb_json:
                    vec = np.array(json.loads(emb_json), dtype=np.float32)
                    self._embedding_cache[mem_id] = vec
        except Exception as e:
            print(f"⚠ Embedding cache warm-up failed: {e}")

    def _rows_to_facts(self, result) -> List[Fact]:
        """Convert a Kuzu query result (8 columns) into a list of Fact objects."""
        facts = []
        while result.has_next():
            row = result.get_next()
            facts.append(
                Fact(
                    fact_id=row[0],
                    fact_type=row[1],
                    subject=row[2],
                    predicate=row[3],
                    value=row[4],
                    date=row[5] if row[5] else None,
                    confidence=row[6],
                    created_at=row[7],
                )
            )
        return facts

    def _load_memories(self):
        """Populate local memory cache and embedding cache from graph DB on startup."""
        try:
            result = self.conn.execute(
                """MATCH (m:Memory)
                   RETURN m.memory_id, m.summary, m.category, m.confidence,
                          m.source, m.created_at, m.last_reinforced, m.event_date,
                          m.embedding"""
            )
            count = 0
            while result.has_next():
                row = result.get_next()
                memory = self._row_to_memory(row[:8])
                self.memories.append(memory)
                # Populate embedding cache
                emb_json = row[8]
                if emb_json:
                    self._embedding_cache[memory.memory_id] = np.array(
                        json.loads(emb_json), dtype=np.float32
                    )
                count += 1
            print(f"✓ Loaded {count} memories from graph DB")
        except Exception as e:
            print(f"⚠ Graph memory load: {e}")

    def _load_facts(self):
        """Populate local facts cache from graph DB on startup."""
        try:
            result = self.conn.execute(
                """MATCH (f:FactNode)
                   RETURN f.fact_id, f.fact_type, f.subject, f.predicate,
                          f.value, f.date, f.confidence, f.created_at"""
            )
            count = 0
            while result.has_next():
                row = result.get_next()
                self.facts.append(
                    Fact(
                        fact_id=row[0],
                        fact_type=row[1],
                        subject=row[2],
                        predicate=row[3],
                        value=row[4],
                        date=row[5] if row[5] else None,
                        confidence=row[6],
                        created_at=row[7],
                    )
                )
                count += 1
            print(f"✓ Loaded {count} facts from graph DB")
        except Exception as e:
            print(f"⚠ Graph facts load: {e}")

    def _row_to_memory(self, row) -> Memory:
        """Convert an 8-column Kuzu row into a Memory dataclass."""
        memory_id, summary, cat_str, confidence, source, created_str, lr_str, event_str = row
        return Memory(
            memory_id=memory_id,
            summary=summary,
            category=MemoryCategory(cat_str) if cat_str else MemoryCategory.EPISODIC,
            confidence=float(confidence),
            source=source or "conversation",
            created_at=datetime.fromisoformat(created_str) if created_str else datetime.now(),
            last_reinforced=datetime.fromisoformat(lr_str) if lr_str else datetime.now(),
            event_date=datetime.fromisoformat(event_str) if event_str else None,
        )
