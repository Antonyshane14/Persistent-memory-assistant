# Persistent Memory System

A dual-agent memory system for LLMs that provides long-term, persistent memory using an embedded graph database (Kuzu).

## Features

- **Graph-based memory storage** - Kuzu embedded graph DB for memories, entities, and relationships
- **Dual-agent architecture** - Agent A (read-only retrieval) + Agent B (write-only memory extraction)
- **Semantic search** - Vector embeddings for similarity-based memory retrieval
- **Temporal awareness** - Resolves "yesterday", "last week" to actual dates
- **Entity relationships** - Tracks connections between people, places, and events
- **Multiple LLM backends** - Supports Ollama (local), Claude (cloud), or hybrid

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment template
cp .env.example .env
# Edit .env with your API keys

# 3. Run the assistant
python assistant.py          # Local mode (Ollama)
python assistant_claude.py   # Cloud mode (Claude via OpenRouter)
```

## Architecture

### Dual-Agent Design

**Agent A (Conversational - read-only)**
- Expands user queries into context-aware keywords
- Searches graph DB for relevant memories
- Re-ranks results by similarity, confidence, and recency
- Injects memories into LLM prompt for context-aware responses

**Agent B (Memory Curator - write-only)**
- Processes conversation chunks asynchronously
- Extracts structured memories with categories and confidence scores
- Deduplicates against existing memories
- Creates entity relationships in the graph

### Graph Schema

```
Nodes:
  - Memory (id, summary, category, confidence, embedding, timestamps)
  - Entity (name, type: person/place/company)
  - DateNode (date)
  - FactNode (subject, predicate, object)

Relationships:
  - Memory -[MENTIONS]-> Entity
  - Memory -[OCCURRED_ON]-> DateNode
  - Entity -[KNOWS]-> Entity
```

## Memory Categories

| Category | Description |
|----------|-------------|
| `identity` | Personal info, background, who they are |
| `preference` | Likes, dislikes, opinions |
| `tech_stack` | Technical skills, tools, frameworks |
| `decision` | Choices made, reasoning |
| `episodic` | Events, experiences, stories |
| `constraint` | Limitations, boundaries, requirements |

## Requirements

- Python 3.12+ (Kuzu doesn't have wheels for 3.14 yet)
- Kuzu (embedded graph database)
- OpenRouter API key (for Claude) or Ollama (for local LLMs)

## Project Structure

```
persistent-memory/
├── assistant.py           # Main entry point (local mode)
├── assistant_claude.py    # Cloud mode entry point
├── requirements.txt       # Dependencies
├── .env.example           # Environment template
└── memory_system/
    ├── __init__.py
    ├── memory_system.py   # Main orchestrator
    ├── agent_a.py         # Retrieval agent
    ├── agent_b.py         # Memory extraction agent
    ├── graph_store.py     # Kuzu graph database
    ├── vector_store.py    # Qdrant vector store (legacy)
    ├── embeddings.py      # Embedding services
    ├── models.py          # Data models
    ├── claude_llm.py      # Claude LLM wrapper
    └── local_llm.py       # Ollama LLM wrapper
```

## License

MIT
