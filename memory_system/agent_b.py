"""
Agent B - Memory Curator
Responsibilities: Summarization + Memory Writing (WRITE ONLY)
NEVER interacts with user directly.
"""

from typing import List, Optional, Dict, Union, TYPE_CHECKING
import json
from datetime import datetime, timedelta

from .models import Memory, MemoryCategory, ConversationTurn
from .embeddings import EmbeddingService

if TYPE_CHECKING:
    from .graph_store import GraphStore
    from .vector_store import VectorStore


def normalize_category(category_str: str) -> str:
    """
    Normalize LLM output to valid MemoryCategory values.
    Handles common variations and typos.
    """
    category_map = {
        # Direct mappings
        "identity": "identity",
        "preference": "preference",
        "tech_stack": "tech_stack",
        "decision": "decision",
        "episodic": "episodic",
        "constraint": "constraint",
        # Common LLM variations
        "episode": "episodic",
        "event": "episodic",
        "fact": "episodic",
        "tech": "tech_stack",
        "technology": "tech_stack",
        "choice": "decision",
        "like": "preference",
        "dislike": "preference",
        "limitation": "constraint",
        "personal": "identity",
    }
    
    normalized = category_str.lower().strip()
    return category_map.get(normalized, "episodic")  # Default to episodic


class AgentB:
    """
    The memory curator that processes conversations and writes to memory.
    
    Design principles:
    - Runs in background (synchronous in v1)
    - Processes conversation in chunks
    - Extracts durable memories
    - Deduplicates against existing memories
    - NEVER reads memory for retrieval (that's Agent A's job)
    """
    
    # Memory extraction prompt template
    EXTRACTION_PROMPT = """You are a memory extraction specialist for Antony Shane's personal AI assistant.
Your job is to analyze conversations and extract memories about Antony's life, preferences, and experiences.

CURRENT DATE: {current_date}
CURRENT TIME: {current_time}

🔴 CRITICAL - WHO IS WHO:
- ANTONY SHANE (the user) = "User:", "You:" in the conversation
- When ANTONY says "I", "my", "me", "mine" → that refers to ANTONY SHANE, the user
- "Assistant:" messages are from the AI, NOT from Antony
- Extract memories about ANTONY's experiences, NOT the AI's
- Example: If user says "I met Joshna" → Memory should be "Antony met Joshna", NOT "User told me about meeting Joshna"

PERSPECTIVE RULES:
- Write memories from 3rd person about ANTONY
- "Antony did X", "Antony's friend is Y", "Antony likes Z"
- NOT "I learned that", "User told me", "We discussed"
- The memories are FACTS ABOUT ANTONY'S LIFE, not facts about conversations with AI

CONTEXT: You're extracting memories about ANTONY SHANE (the user). When he says "I", "my", "me" - that refers to Antony.

TEMPORAL AWARENESS:
- When Antony says "today" → this means the event happened on {current_date}
- When he says "yesterday" → calculate the date
- When he says "last week", "last month" → include approximate timeframe
- ALWAYS include temporal context in summaries (e.g., "Antony got scolded by sir today")

BE LENIENT - Extract ANY meaningful information that helps remember Antony better:

ACCEPT (extract these):
- Facts about Antony's life, relationships, experiences
- People he mentions (friends, family, colleagues, girlfriend, etc.) - ALWAYS capture their names
- Companies, workplaces, organizations - ALWAYS capture company names and what they do
- Places, locations, addresses
- Events, meetings, dates (especially first meetings, important dates)
- His preferences, opinions, feelings about people/things
- His decisions, plans, goals
- His work, studies, projects - with specific names and details
- Technology he uses or mentions
- Stories, anecdotes, memories he shares
- Conversations about other people (e.g., Joshna, friends)
- Details provided when asked clarifying questions (WHO, WHAT, WHERE, WHEN, WHY)
- Temporal events ("today I...", "yesterday I...", "last week...")

SPECIAL - TEMPORAL EVENTS:
- If he says "today I got scolded" → "Antony got scolded by teacher on {current_date}"
- If he says "I called her yesterday" → include date reference
- For recent events, ALWAYS include the timeframe in the memory

SPECIAL: When assistant asks clarifying questions and user provides details, extract ALL context:
- If asked "Who is that?" → extract person's name, relationship, description
- If asked "What company?" → extract company name, industry, role
- If asked "What happened?" → extract event details, date, location, people involved
- If asked "Tell me more" → extract all additional context provided

REJECT (skip these):
- Pure questions with no information
- Hypotheticals without context
- Just greetings/pleasantries

INSTRUCTIONS:
1. Read the conversation chunk
2. Extract 1-5 memories (be generous - more is better than missing important context)
3. For each memory:
   - Write a clear summary from 3rd person perspective about ANTONY (use "Antony" or subject like "Antony's girlfriend")
   - If mentioning people: include their FULL names and relationships (e.g., "talked about Joshna, his girlfriend")
   - If companies: include company NAME and what they do
   - If dates/times mentioned: include them in exact format
   - If temporal words (today, yesterday): include the actual date context
   - If locations: include specific place names
   - Assign category: identity, preference, tech_stack, decision, episodic, constraint
   - Rate confidence (0.5-0.8 depending on clarity)
   - Remember: memories are about ANTONY's life, not about the AI assistant

Return JSON only:
{
  "memories": [
    {
      "summary": "Antony got scolded by professor",
      "category": "episodic",
      "confidence": 0.7,
      "event_date": "2026-02-08"  // REQUIRED: When event happened (YYYY-MM-DD format)
    }
  ]
}

IMPORTANT - event_date field:
- If user says "today" → use {current_date} 
- If user says "yesterday" → calculate yesterday's date
- If user says "last week" → estimate the date
- If no temporal reference → use {current_date} (assume today)
- Format: YYYY-MM-DD

If nothing meaningful, return: {"memories": []}

CONVERSATION CHUNK:
"""

    def __init__(
        self,
        vector_store: "Union[GraphStore, VectorStore]",
        embedding_service: EmbeddingService,
        llm_callable,
        facts_db=None
    ):
        """
        Initialize Agent B.

        Args:
            vector_store: Graph/Vector DB for memory storage
            embedding_service: Service for embedding generation
            llm_callable: Function that takes prompt and returns LLM response
            facts_db: Optional facts database for structured data (or GraphStore)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_callable = llm_callable
        self.facts_db = facts_db
        self.user_name = "Antony Shane"  # User identity context
    
    def process_conversation_chunk(self, turns: List[ConversationTurn]) -> int:
        """
        Process a chunk of conversation and extract memories.
        
        Args:
            turns: List of conversation turns to process
        
        Returns:
            Number of new memories written
        """
        if not turns:
            return 0
        
        # Step 1: Format conversation for LLM
        conversation_text = self._format_conversation(turns)
        
        # Step 2: Extract memories using LLM
        extracted_memories = self._extract_memories(conversation_text)
        
        if not extracted_memories:
            return 0
        
        # Step 3: Process each extracted memory
        memories_written = 0
        for mem_data in extracted_memories:
            if self._write_memory(mem_data):
                memories_written += 1
        
        return memories_written
    
    def create_conversation_summary(self, turns: List[ConversationTurn]) -> bool:
        """
        Create a high-level summary memory of the entire conversation.
        Called at END of chat session to capture overall themes/topics.
        
        Example: If talked about dogs for an hour → "User had extended discussion about dogs, 
        covering breeds, training, and personal experiences on Feb 8, 2026"
        
        Args:
            turns: All conversation turns from the session
        
        Returns:
            True if summary memory was created, False otherwise
        """
        if len(turns) < 3:  # Need at least 3 turns for meaningful summary
            return False
        
        now = datetime.now()
        current_date = now.strftime("%A, %B %d, %Y")
        
        # Build conversation text
        conversation_text = self._format_conversation(turns)
        
        # Prompt LLM to create conversation-level summary
        prompt = f"""Analyze this entire conversation and create ONE high-level summary that captures:
1. Main topics/themes discussed (what was the conversation mostly about?)
2. Key subjects (people, places, things mentioned repeatedly)
3. Overall context (casual chat, deep discussion, problem-solving, etc.)

CONVERSATION DATE: {current_date}
TOTAL TURNS: {len(turns)}

CONVERSATION:
{conversation_text}

Create a summary that would help remember this conversation later. Focus on WHAT was discussed, not individual details.

Examples:
- "User had extended discussion about dogs, covering training methods and breed preferences"
- "User talked about relationship with girlfriend Joshna, discussed feelings and future plans"
- "User discussed work projects and technical challenges with NIT placement process"
- "Casual conversation about weekend plans and social activities"

Return JSON:
{{
  "summary": "<one-sentence summary of what this conversation was about>",
  "confidence": 0.8
}}

If conversation was just greetings/small talk with no real substance, return: {{"summary": null}}
"""
        
        try:
            response = self.llm_callable(prompt).strip()
            original_response = response  # Keep for debugging
            
            # Parse JSON - handle markdown blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            # Try to find JSON object if there's extra text
            if not response.startswith("{"):
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1:
                    response = response[start:end+1]
            
            data = json.loads(response)
            
            if not data.get("summary"):
                return False
            
            # Create conversation summary memory
            summary_text = f"Conversation on {current_date}: {data['summary']}"
            confidence = float(data.get("confidence", 0.7))
            
            # Create memory with event_date as conversation date
            embedding = self.embedding_service.embed(summary_text)
            memory = Memory.create_new(
                summary=summary_text,
                category=MemoryCategory.EPISODIC,
                confidence=confidence,
                event_date=now
            )
            
            self.vector_store.add_memory(memory, embedding)
            print(f"📝 Conversation summary: {summary_text[:80]}...")
            return True
            
        except json.JSONDecodeError as e:
            print(f"⚠ Failed to create conversation summary - Invalid JSON")
            print(f"[DEBUG] Response (first 300 chars): {original_response[:300]}")
            return False
            
        except Exception as e:
            print(f"⚠ Failed to create conversation summary: {e}")
            return False
    
    def _format_conversation(self, turns: List[ConversationTurn]) -> str:
        """Format conversation turns into readable text."""
        lines = []
        for turn in turns:
            lines.append(f"User: {turn.user_message}")
            lines.append(f"Assistant: {turn.assistant_message}")
            lines.append("")
        return "\n".join(lines)
    
    def _extract_memories(self, conversation_text: str) -> List[Dict]:
        """
        Use LLM to extract memories from conversation.
        
        Returns:
            List of memory dictionaries with summary, category, confidence
        """
        from datetime import datetime
        
        # Get current date/time for temporal context
        now = datetime.now()
        current_date = now.strftime("%A, %B %d, %Y")  # e.g., "Friday, February 08, 2026"
        current_time = now.strftime("%I:%M %p")  # e.g., "02:30 PM"
        
        # Inject current date/time into prompt
        prompt = self.EXTRACTION_PROMPT.format(
            current_date=current_date,
            current_time=current_time
        ) + conversation_text
        
        try:
            response = self.llm_callable(prompt)
            
            # Parse JSON response
            # Handle common LLM issues: markdown code blocks, extra text
            response = response.strip()
            original_response = response  # Keep for debugging
            
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            # Try to find JSON object if there's extra text
            if not response.startswith("{"):
                # Look for first { and last }
                start = response.find("{")
                end = response.rfind("}")
                if start != -1 and end != -1:
                    response = response[start:end+1]
            
            data = json.loads(response)
            return data.get("memories", [])
        
        except json.JSONDecodeError as e:
            print(f"⚠ Memory extraction failed - Invalid JSON")
            print(f"[DEBUG] LLM Response (first 500 chars):\n{original_response[:500]}")
            print(f"[DEBUG] JSON Error: {e}")
            return []
        
        except Exception as e:
            print(f"⚠ Memory extraction failed: {e}")
            print(f"[DEBUG] Response: {response[:200] if 'response' in locals() else 'No response'}")
            return []
    
    def _write_memory(self, mem_data: Dict) -> bool:
        """
        Write a single memory to vector store with deduplication.
        
        Args:
            mem_data: Dictionary with summary, category, confidence, event_date
        
        Returns:
            True if memory was written, False if duplicate/invalid
        """
        try:
            # Validate required fields
            if not all(k in mem_data for k in ["summary", "category", "confidence"]):
                print(f"⚠ Invalid memory data: {mem_data}")
                return False
            
            summary = mem_data["summary"]
            # Normalize category to handle LLM variations (e.g., "episode" -> "episodic")
            normalized_cat = normalize_category(mem_data["category"])
            category = MemoryCategory(normalized_cat)
            confidence = float(mem_data["confidence"])
            
            # Parse event_date if provided
            event_date = None
            if "event_date" in mem_data and mem_data["event_date"]:
                try:
                    event_date = datetime.strptime(mem_data["event_date"], "%Y-%m-%d")
                except:
                    print(f"⚠ Invalid event_date format: {mem_data['event_date']}, using None")
            
            # Generate embedding
            embedding = self.embedding_service.embed(summary)
            
            # Step 5: Deduplication - check for existing similar memory
            duplicate = self.vector_store.find_duplicates(embedding, threshold=0.9)
            
            if duplicate:
                # Reinforce existing memory instead of creating new one
                existing_memory, similarity = duplicate
                print(f"✓ Reinforcing existing memory (similarity: {similarity:.2f})")
                existing_memory.reinforce(boost=0.05)
                self.vector_store.update_memory(existing_memory.memory_id, existing_memory)
                return False  # Didn't write new memory
            
            # Create new memory with event_date
            memory = Memory.create_new(summary, category, confidence, event_date=event_date)
            
            # Write to vector store
            self.vector_store.add_memory(memory, embedding)
            event_info = f" [event: {event_date.strftime('%b %d, %Y')}]" if event_date else ""
            print(f"✓ New memory: {summary[:60]}... [{category.value}]{event_info}")
            
            # Extract structured facts if facts_db is available
            if self.facts_db:
                self._extract_facts_from_summary(summary)
            
            return True
        
        except Exception as e:
            print(f"⚠ Failed to write memory: {e}")
            return False
    
    def _extract_facts_from_summary(self, summary: str):
        """
        Extract structured facts from memory summary.
        Looks for dates, names, numbers, and stores them in facts DB.
        NOW with temporal awareness: "today", "yesterday", etc.
        """
        import re
        import uuid
        from datetime import datetime as dt, timedelta
        from .facts_db_sql import Fact
        
        now = dt.now()
        
        # Detect temporal references and extract dates
        temporal_patterns = [
            (r'\btoday\b', 0),  # today = 0 days ago
            (r'\byesterday\b', 1),  # yesterday = 1 day ago
            (r'\bthis morning\b', 0),  # this morning = today
            (r'\bthis afternoon\b', 0),
            (r'\bthis evening\b', 0),
            (r'\btonigh?t\b', 0),
            (r'\blast night\b', 1),  # last night = yesterday
        ]
        
        # Check for temporal references and create dated fact
        for pattern, days_ago in temporal_patterns:
            if re.search(pattern, summary, re.IGNORECASE):
                event_date = now - timedelta(days=days_ago)
                date_str = event_date.strftime('%Y-%m-%d')
                
                # Create temporal event fact
                fact = Fact(
                    fact_id=str(uuid.uuid4()),
                    fact_type='event',
                    subject='Antony',
                    predicate='event on date',
                    value=summary[:150],  # Store full context
                    date=date_str,
                    confidence=0.85,
                    created_at=now.isoformat()
                )
                self.facts_db.add_fact(fact)
                print(f"  📅 Temporal fact: {date_str}")
                break  # Only create one temporal fact per summary
        
        # Extract explicit dates (various formats)
        date_patterns = [
            (r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', '%Y-%m-%d'),  # 2022-08-07
            (r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4})', '%B %d, %Y'),  # August 7, 2022
            (r'\(([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\)', '%B %d, %Y'),  # (Feb 8, 2026)
        ]
        
        for pattern, date_format in date_patterns:
            matches = re.findall(pattern, summary, re.IGNORECASE)
            for match in matches:
                try:
                    # Parse and normalize date
                    parsed_date = dt.strptime(match, date_format)
                    date_str = parsed_date.strftime('%Y-%m-%d')
                    
                    # Create fact about event with this date
                    fact = Fact(
                        fact_id=str(uuid.uuid4()),
                        fact_type='event',
                        subject='Antony',
                        predicate='event on date',
                        value=summary[:150],
                        date=date_str,
                        confidence=0.8,
                        created_at=now.isoformat()
                    )
                    self.facts_db.add_fact(fact)
                except:
                    pass
        
        # Extract person names and relationships
        # First pass: Look for explicit relationships
        relationship_patterns = [
            (r'(?:girlfriend|dating|with|love)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', 'girlfriend'),
            (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s+(?:his|her|user\'?s)?\s*girlfriend\b', 'girlfriend'),
            (r'(?:best friend|close friend)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', 'best friend'),
            (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s+(?:his|her|user\'?s)?\s*(?:best friend|close friend)\b', 'best friend'),
            (r'(?:colleague|coworker)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', 'colleague'),
            (r'(?:brother|sister|sibling)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', 'sibling'),
            (r'\bfriend\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', 'friend'),
        ]
        
        extracted_names = set()  # Track which names we've already added
        
        for pattern, relationship_type in relationship_patterns:
            matches = re.findall(pattern, summary, re.IGNORECASE)
            for name in matches:
                name = name.strip()
                # Skip common false positives
                if name.lower() in ['user', 'antony', 'shane', 'antony shane'] or len(name) < 2:
                    continue
                
                extracted_names.add(name)
                
                # Store relationship fact
                fact = Fact(
                    fact_id=str(uuid.uuid4()),
                    fact_type='relationship',
                    subject='Antony',
                    predicate=f'has {relationship_type}',
                    value=name,
                    date=None,
                    confidence=0.85,
                    created_at=dt.now().isoformat()
                )
                self.facts_db.add_fact(fact)
                
                # Also store generic "knows person" fact for discovery
                fact2 = Fact(
                    fact_id=str(uuid.uuid4()),
                    fact_type='person',
                    subject='Antony',
                    predicate='knows person',
                    value=name,
                    date=None,
                    confidence=0.7,
                    created_at=dt.now().isoformat()
                )
                self.facts_db.add_fact(fact2)
        
        # Second pass: Generic name extraction for people not captured above
        name_patterns = [
            r'(?:met|know|talked to|with|about)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:said|told|asked)\b',
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, summary)
            for name in matches:
                name = name.strip()
                # Skip common false positives, pronouns, and already extracted names
                skip_words = ['user', 'antony', 'shane', 'antony shane', 'him', 'her', 
                             'his', 'their', 'them', 'who', 'what', 'when', 'where', 
                             'which', 'are', 'and', 'the', 'that', 'this', 'to', 'misses']
                if name.lower() in skip_words or name in extracted_names or len(name) < 2:
                    continue
                
                # Skip if name ends with common words
                if any(name.lower().endswith(' ' + word) for word in ['and', 'to', 'who', 'are']):
                    continue
                
                extracted_names.add(name)
                
                fact = Fact(
                    fact_id=str(uuid.uuid4()),
                    fact_type='person',
                    subject='Antony',
                    predicate='knows person',
                    value=name,
                    date=None,
                    confidence=0.7,
                    created_at=dt.now().isoformat()
                )
                self.facts_db.add_fact(fact)
        
        # Extract numbers (rankings, scores, etc.)
        number_patterns = [
            (r'rank(?:ed)?\s+(?:from\s+)?(\d+)(?:\s+to\s+(\d+))?', 'rank'),
            (r'score(?:d)?\s+(\d+)', 'score'),
        ]
        
        for pattern, fact_type in number_patterns:
            matches = re.findall(pattern, summary, re.IGNORECASE)
            for match in matches:
                value = match if isinstance(match, str) else ' → '.join(m for m in match if m)
                fact = Fact(
                    fact_id=str(uuid.uuid4()),
                    fact_type='achievement',
                    subject='Antony',
                    predicate=fact_type,
                    value=value,
                    date=None,
                    confidence=0.75,
                    created_at=dt.now().isoformat()
                )
                self.facts_db.add_fact(fact)
        
        # Extract companies/organizations
        # Look for: "works at X", "company X", "joined X", etc.
        company_patterns = [
            r'(?:works at|working at|joined|company|employer|firm)\s+([A-Z][A-Za-z0-9\s&]+(?:Inc|LLC|Ltd|Corp|Company|Technologies|Tech|Systems)?)',
            r'at\s+([A-Z][A-Za-z0-9\s&]+(?:Inc|LLC|Ltd|Corp|Company|Technologies|Tech|Systems))',
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, summary)
            for company_name in matches:
                company_name = company_name.strip()
                # Skip common false positives
                if len(company_name) < 3 or company_name.lower() in ['user', 'antony', 'the']:
                    continue
                
                fact = Fact(
                    fact_id=str(uuid.uuid4()),
                    fact_type='organization',
                    subject='Antony',
                    predicate='associated with company',
                    value=company_name,
                    date=None,
                    confidence=0.7,
                    created_at=dt.now().isoformat()
                )
                self.facts_db.add_fact(fact)
    
    def process_session(self, turns: List[ConversationTurn], chunk_size: int = 5) -> int:
        """
        Process entire session in chunks.
        
        Args:
            turns: All conversation turns from session
            chunk_size: Number of turns per chunk
        
        Returns:
            Total memories written
        """
        total_written = 0
        
        # Process in chunks
        for i in range(0, len(turns), chunk_size):
            chunk = turns[i:i+chunk_size]
            
            # Filter out small talk (very short exchanges)
            meaningful_chunk = [
                turn for turn in chunk 
                if len(turn.user_message) > 10  # Simple heuristic
            ]
            
            if meaningful_chunk:
                written = self.process_conversation_chunk(meaningful_chunk)
                total_written += written
        
        return total_written
