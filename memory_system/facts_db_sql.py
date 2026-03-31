"""
SQLite-based Structured Facts Database - For exact dates, names, numbers, and temporal info.
Replaces JSON-based facts storage with a proper relational database.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Any
from dataclasses import dataclass


@dataclass
class Fact:
    """
    A structured fact with exact information.
    Unlike memories (fuzzy/semantic), facts are precise and queryable.
    """
    fact_id: str
    fact_type: str  # date, person, number, location, event
    subject: str  # who/what this fact is about
    predicate: str  # relationship/property
    value: Any  # the exact value
    date: Optional[str]  # when this fact is from (ISO format)
    confidence: float
    created_at: str
    
    def to_dict(self) -> dict:
        return {
            'fact_id': self.fact_id,
            'fact_type': self.fact_type,
            'subject': self.subject,
            'predicate': self.predicate,
            'value': str(self.value),
            'date': self.date,
            'confidence': self.confidence,
            'created_at': self.created_at
        }


class FactsDatabase:
    """
    SQLite-based structured database for exact information.
    
    Tables:
    - facts: Main facts table with indexes on subject, date, type
    - entities: Named entities (people, places, organizations)
    - events: Timestamped events
    
    Examples:
    - "Antony" "met" "Joshna" on "2022-08-07"
    - "Antony" "rank" "jumped from 40 to 4"
    - "Joshna" "wore" "pink dress" on first meeting
    """
    
    def __init__(self, storage_path: str = "./assistant_memory"):
        self.storage_path = storage_path
        self.db_file = os.path.join(storage_path, "facts.db")
        os.makedirs(storage_path, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_file)
        self.conn.row_factory = sqlite3.Row  # Return rows as dicts
        self._create_tables()
        self.facts = []  # Cached facts for compatibility
        self._load_facts()
    
    def _create_tables(self):
        """Create database schema with indexes."""
        cursor = self.conn.cursor()
        
        # Main facts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                fact_type TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value TEXT NOT NULL,
                date TEXT,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # Indexes for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subject ON facts(subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON facts(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON facts(fact_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_subject_pred ON facts(subject, predicate)")
        
        # Entities table for named entities (people, places)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                entity_type TEXT NOT NULL,
                first_mentioned TEXT,
                mention_count INTEGER DEFAULT 1
            )
        """)
        
        # Events table for timestamped events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date TEXT NOT NULL,
                description TEXT NOT NULL,
                participants TEXT,
                location TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_date ON events(event_date)")
        
        self.conn.commit()
    
    def _load_facts(self):
        """Load all facts into memory for compatibility."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM facts")
        rows = cursor.fetchall()
        
        self.facts = []
        for row in rows:
            fact = Fact(
                fact_id=row['fact_id'],
                fact_type=row['fact_type'],
                subject=row['subject'],
                predicate=row['predicate'],
                value=row['value'],
                date=row['date'],
                confidence=row['confidence'],
                created_at=row['created_at']
            )
            self.facts.append(fact)
    
    def save(self):
        """Commit any pending changes (auto-commits in SQLite)."""
        self.conn.commit()
        self._load_facts()  # Refresh cache
    
    def add_fact(self, fact: Fact) -> bool:
        """Add a new fact to the database."""
        cursor = self.conn.cursor()
        
        # Check for duplicates
        cursor.execute("""
            SELECT fact_id FROM facts 
            WHERE subject = ? AND predicate = ? AND value = ?
        """, (fact.subject, fact.predicate, str(fact.value)))
        
        if cursor.fetchone():
            return False  # Duplicate
        
        # Insert fact
        cursor.execute("""
            INSERT INTO facts (fact_id, fact_type, subject, predicate, value, date, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fact.fact_id,
            fact.fact_type,
            fact.subject,
            fact.predicate,
            str(fact.value),
            fact.date,
            fact.confidence,
            fact.created_at
        ))
        
        # Track entity if it's a person
        if fact.fact_type == 'person' or 'met' in fact.predicate.lower() or 'know' in fact.predicate.lower():
            self._track_entity(fact.value, 'person')
        
        self.conn.commit()
        self.facts.append(fact)
        return True
    
    def _track_entity(self, name: str, entity_type: str = 'person'):
        """Track a named entity (person, place, etc.)."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO entities (name, entity_type, first_mentioned)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET mention_count = mention_count + 1
        """, (name, entity_type, datetime.now().isoformat()))
        
        self.conn.commit()
    
    def query_by_subject(self, subject: str) -> List[Fact]:
        """Get all facts about a subject."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM facts 
            WHERE subject LIKE ? 
            ORDER BY date DESC, created_at DESC
        """, (f'%{subject}%',))
        
        rows = cursor.fetchall()
        return [Fact(**dict(row)) for row in rows]
    
    def query_by_date(self, date: str) -> List[Fact]:
        """Get all facts from a specific date."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM facts WHERE date = ? ORDER BY created_at DESC", (date,))
        rows = cursor.fetchall()
        return [Fact(**dict(row)) for row in rows]
    
    def query_by_type(self, fact_type: str) -> List[Fact]:
        """Get all facts of a specific type."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM facts WHERE fact_type = ? ORDER BY created_at DESC", (fact_type,))
        rows = cursor.fetchall()
        return [Fact(**dict(row)) for row in rows]
    
    def query_date_range(self, subject: str, start_date: str = None, end_date: str = None) -> List[Fact]:
        """Get all dated facts about a subject in a date range."""
        cursor = self.conn.cursor()
        
        if start_date and end_date:
            cursor.execute("""
                SELECT * FROM facts 
                WHERE subject LIKE ? AND date BETWEEN ? AND ?
                ORDER BY date ASC
            """, (f'%{subject}%', start_date, end_date))
        elif start_date:
            cursor.execute("""
                SELECT * FROM facts 
                WHERE subject LIKE ? AND date >= ?
                ORDER BY date ASC
            """, (f'%{subject}%', start_date))
        else:
            cursor.execute("""
                SELECT * FROM facts 
                WHERE subject LIKE ? AND date IS NOT NULL
                ORDER BY date ASC
            """, (f'%{subject}%',))
        
        rows = cursor.fetchall()
        return [Fact(**dict(row)) for row in rows]
    
    def get_fact(self, subject: str, predicate: str) -> Optional[Fact]:
        """Get a specific fact."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM facts 
            WHERE subject LIKE ? AND predicate LIKE ?
            ORDER BY created_at DESC LIMIT 1
        """, (f'%{subject}%', f'%{predicate}%'))
        
        row = cursor.fetchone()
        return Fact(**dict(row)) if row else None
    
    def query_relationships(self, person_name: str = None) -> List[Fact]:
        """
        Get relationship facts (girlfriend, friend, colleague, etc.).
        If person_name is provided, get facts about that specific person.
        """
        cursor = self.conn.cursor()
        
        if person_name:
            # Get specific person's relationship
            cursor.execute("""
                SELECT * FROM facts 
                WHERE fact_type = 'relationship' 
                AND value LIKE ?
                ORDER BY confidence DESC, created_at DESC
            """, (f'%{person_name}%',))
        else:
            # Get all relationships
            cursor.execute("""
                SELECT * FROM facts 
                WHERE fact_type = 'relationship'
                ORDER BY confidence DESC, created_at DESC
            """)
        
        rows = cursor.fetchall()
        return [Fact(**dict(row)) for row in rows]
    
    def query_by_predicate(self, predicate_pattern: str) -> List[Fact]:
        """Get all facts matching a predicate pattern."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM facts 
            WHERE predicate LIKE ?
            ORDER BY confidence DESC, created_at DESC
        """, (f'%{predicate_pattern}%',))
        
        rows = cursor.fetchall()
        return [Fact(**dict(row)) for row in rows]
    
    def search_facts(self, query: str) -> List[Fact]:
        """Full-text search across all facts."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM facts 
            WHERE subject LIKE ? OR predicate LIKE ? OR value LIKE ?
            ORDER BY confidence DESC, created_at DESC
            LIMIT 20
        """, (f'%{query}%', f'%{query}%', f'%{query}%'))
        
        rows = cursor.fetchall()
        return [Fact(**dict(row)) for row in rows]
    
    def get_entities(self, entity_type: str = None) -> List[dict]:
        """Get all tracked entities."""
        cursor = self.conn.cursor()
        
        if entity_type:
            cursor.execute("""
                SELECT * FROM entities 
                WHERE entity_type = ?
                ORDER BY mention_count DESC
            """, (entity_type,))
        else:
            cursor.execute("SELECT * FROM entities ORDER BY mention_count DESC")
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_timeline(self, subject: str = None) -> List[Fact]:
        """Get chronological timeline of dated facts."""
        cursor = self.conn.cursor()
        
        if subject:
            cursor.execute("""
                SELECT * FROM facts 
                WHERE date IS NOT NULL AND subject LIKE ?
                ORDER BY date ASC
            """, (f'%{subject}%',))
        else:
            cursor.execute("""
                SELECT * FROM facts 
                WHERE date IS NOT NULL
                ORDER BY date ASC
            """)
        
        rows = cursor.fetchall()
        return [Fact(**dict(row)) for row in rows]
    
    def format_for_prompt(self, facts: List[Fact]) -> str:
        """Format facts for inclusion in LLM prompt."""
        if not facts:
            return ""
        
        lines = ["[Structured Facts - Exact Information]:"]
        for fact in facts:
            date_str = f" (on {fact.date})" if fact.date else ""
            lines.append(f"- {fact.subject} {fact.predicate}: {fact.value}{date_str}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as count FROM facts")
        total_facts = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM facts WHERE date IS NOT NULL")
        dated_facts = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM entities")
        total_entities = cursor.fetchone()['count']
        
        return {
            'total_facts': total_facts,
            'dated_facts': dated_facts,
            'total_entities': total_entities
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
