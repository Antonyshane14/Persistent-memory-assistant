"""
Agent A - Conversational Agent
Responsibilities: Retrieval + Interaction (READ ONLY)
NEVER writes to memory.
"""

from typing import List, Optional, Union, TYPE_CHECKING
from datetime import datetime
import numpy as np

from .models import Memory, RetrievedMemory, ConversationTurn
from .embeddings import EmbeddingService

if TYPE_CHECKING:
    from .graph_store import GraphStore
    from .vector_store import VectorStore


class AgentA:
    """
    The conversational agent that interacts with users.

    Design principles:
    - Retrieves relevant memories from graph/vector DB
    - Builds context-aware prompts
    - NEVER writes to memory (that's Agent B's job)
    - Maintains session conversation history
    """

    def __init__(self, vector_store: "Union[GraphStore, VectorStore]", embedding_service: EmbeddingService, llm_callable=None, facts_db=None):
        """
        Initialize Agent A.
        
        Args:
            vector_store: Vector DB for memory retrieval
            embedding_service: Service for query embedding
            llm_callable: Optional LLM for query expansion
            facts_db: Optional facts database for structured data
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_callable = llm_callable
        self.facts_db = facts_db
        self.session_history: List[ConversationTurn] = []
        self.max_history_turns = 50  # Keep last 50 turns for better context continuity
    
    def process_user_message(self, user_message: str, llm_callable) -> str:
        """
        Main entry point: user sends message, get response with memory context.
        
        Args:
            user_message: User's input
            llm_callable: Function that takes prompt and returns LLM response
        
        Returns:
            Assistant's response
        """
        from datetime import datetime, timedelta
        
        # Check if this is a meta-question about memories
        is_meta_question = self._is_meta_question(user_message)
        
        # Step 0.5: Expand query first to get context-aware keywords
        # NOTE: This uses the FULL conversation history (last 5 turns), not just the current message
        num_context_turns = min(5, len(self.session_history))
        expanded_keywords = self._expand_query(user_message)
        print(f"[DEBUG] Generated {len(expanded_keywords)} keywords using {num_context_turns} conversation turns + temporal awareness:")
        print(f"[DEBUG] Keywords: {expanded_keywords}")
        
        # Step 1: Retrieve relevant memories from vector DB using expanded keywords
        if is_meta_question and len(self.vector_store.memories) > 0:
            # For meta-questions, provide a summary of all memories instead of filtering
            retrieved_memories = self._get_memory_summary()
            print(f"\n[DEBUG] Meta-question detected - providing summary of {len(self.vector_store.memories)} memories")
        else:
            # Use expanded keywords for better search
            retrieved_memories = self._retrieve_memories_with_keywords(expanded_keywords)
            
            # DEBUG: Show what was retrieved
            if retrieved_memories:
                print(f"\n[DEBUG] Found {len(retrieved_memories)} relevant memories:")
                highly_relevant = [m for m in retrieved_memories if m.final_score > 0.2]
                for mem in retrieved_memories:
                    prefix = "✓" if mem.final_score > 0.2 else "✗"
                    # Show FULL memory text with EVENT date (when it happened, not when stored)
                    display_date = mem.memory.event_date if mem.memory.event_date else mem.memory.created_at
                    date_str = display_date.strftime("%b %d, %Y")
                    print(f"  {prefix} [{date_str}] {mem.memory.summary} (score: {mem.final_score:.2f})")
                print(f"[DEBUG] {len(highly_relevant)} memories above 0.2 threshold will be shown to LLM")
            else:
                print(f"\n[DEBUG] No relevant memories found (db has {len(self.vector_store.memories)} total)")
        
        # Step 1.5: Query structured facts database using expanded keywords
        relevant_facts = []
        if self.facts_db and len(self.facts_db.facts) > 0:
            # Use expanded keywords to search facts database
            now = datetime.now()
            
            # Extract dates from keywords for temporal search
            temporal_dates = []
            for keyword in expanded_keywords:
                # Check if keyword looks like a date
                if any(char.isdigit() for char in keyword):
                    # Try to parse as date
                    for fmt in ["%B %d, %Y", "%Y-%m-%d", "%B %Y"]:
                        try:
                            parsed = datetime.strptime(keyword, fmt)
                            temporal_dates.append(parsed.strftime("%Y-%m-%d"))
                            break
                        except:
                            pass
            
            # Search facts using all expanded keywords
            for keyword in expanded_keywords:
                # Search for keyword in facts database
                keyword_facts = self.facts_db.search_facts(keyword)
                relevant_facts.extend(keyword_facts[:3])
            
            # If temporal dates found, get facts from those dates
            if temporal_dates:
                print(f"[DEBUG] Searching facts for dates: {temporal_dates}")
                for date_str in temporal_dates[:3]:  # Limit to 3 dates
                    date_facts = self.facts_db.query_by_date(date_str)
                    relevant_facts.extend(date_facts[:5])
            
            # Remove duplicates while preserving order
            seen_ids = set()
            unique_facts = []
            for fact in relevant_facts:
                if fact.fact_id not in seen_ids:
                    seen_ids.add(fact.fact_id)
                    unique_facts.append(fact)
            relevant_facts = unique_facts[:12]  # Limit to 12 facts
            
            if relevant_facts:
                print(f"[DEBUG] Found {len(relevant_facts)} structured facts using keywords")
                for fact in relevant_facts[:8]:  # Show first 8
                    date_part = f" on {fact.date}" if fact.date else ""
                    print(f"  • {fact.subject} {fact.predicate}: {fact.value}{date_part}")
        
        # Step 2: Build prompt with memory context and facts
        prompt = self._build_prompt(user_message, retrieved_memories, relevant_facts)
        
        # DEBUG: Show token estimate
        total_tokens = len(prompt) // 4
        print(f"[DEBUG] Estimated prompt tokens: ~{total_tokens} (limit: 8000)")
        
        # DEBUG: Show prompt structure with MORE context (first 2000 chars to see full conversation)
        if len(self.session_history) > 0:
            preview_length = 2000  # Show much more to see full conversation history
            preview = prompt[:preview_length] + "..." if len(prompt) > preview_length else prompt
            print(f"[DEBUG] Prompt preview:\n{preview}\n")
        
        # Step 3: Get LLM response
        assistant_response = llm_callable(prompt)
        
        # Step 4: Log interaction to session history with sliding window
        turn = ConversationTurn.create(user_message, assistant_response)
        self.session_history.append(turn)
        
        # Sliding window: keep only last N turns to prevent memory bloat
        if len(self.session_history) > self.max_history_turns:
            self.session_history = self.session_history[-self.max_history_turns:]
        
        return assistant_response
    
    def _is_meta_question(self, message: str) -> bool:
        """
        Detect if user is asking about what we know about them (meta-question).
        """
        message_lower = message.lower()
        meta_patterns = [
            "what do you know",
            "what you know",
            "tell me what you know",
            "do you know me",
            "know about me",
            "remember about me",
            "what do you remember",
            "tell me about me"
        ]
        return any(pattern in message_lower for pattern in meta_patterns)
    
    def _get_memory_summary(self) -> List[RetrievedMemory]:
        """
        Get a summary sample of memories for meta-questions.
        Returns recent/important memories without filtering by query relevance.
        """
        now = datetime.now()
        summary_memories = []
        
        # Sort memories by recency and confidence
        sorted_memories = sorted(
            self.vector_store.memories,
            key=lambda m: (m.last_reinforced, m.confidence),
            reverse=True
        )
        
        # Take top 5-6 for summary
        for memory in sorted_memories[:6]:
            days_old = (now - memory.last_reinforced).days
            recency = max(0.0, 1.0 - (days_old / 30.0))
            
            # Create pseudo-score for display
            final_score = 0.5 * memory.confidence + 0.5 * recency
            
            summary_memories.append(RetrievedMemory(
                memory=memory,
                similarity=0.8,  # Placeholder since we're not using semantic search
                final_score=final_score
            ))
        
        return summary_memories
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand user query into relevant keywords and phrases using LLM.
        NOW CONTEXT-AWARE: Uses full conversation history for better understanding.
        
        Example:
        - "what do I study?" → ["study", "college", "education", "university", "NIT Rourkela", "degree"]
        - "where do I work?" → ["work", "job", "office", "company", "workplace", "home"]
        - "yesterday" (after talking about Joshna) → ["yesterday", "Feb 7 2026", "joshna", "relationship"]
        """
        from datetime import datetime, timedelta
        
        # Get current date for temporal awareness
        now = datetime.now()
        today_str = now.strftime("%B %d, %Y")
        
        # Extract person names from query first
        person_names = []
        words = query.split()
        for word in words:
            clean_word = word.strip('.,!?;:()"\'')
            if clean_word and len(clean_word) > 2 and clean_word[0].isupper():
                if clean_word.lower() not in ['the', 'and', 'but', 'for', 'with', 'i']:
                    person_names.append(clean_word.lower())
        
        # Detect temporal references and convert to dates
        temporal_keywords = []
        query_lower = query.lower()
        
        if 'yesterday' in query_lower:
            yesterday = now - timedelta(days=1)
            temporal_keywords.append(yesterday.strftime("%B %d, %Y"))  # "February 07, 2026"
            temporal_keywords.append(yesterday.strftime("%Y-%m-%d"))   # "2026-02-07"
            temporal_keywords.append("yesterday")
        
        if 'today' in query_lower:
            temporal_keywords.append(today_str)
            temporal_keywords.append(now.strftime("%Y-%m-%d"))
            temporal_keywords.append("today")
        
        if 'last week' in query_lower or 'past week' in query_lower:
            last_week_start = now - timedelta(days=7)
            temporal_keywords.append(last_week_start.strftime("%B %Y"))
            temporal_keywords.append("last week")
            temporal_keywords.append("past week")
        
        if 'last month' in query_lower:
            last_month = now - timedelta(days=30)
            temporal_keywords.append(last_month.strftime("%B %Y"))
            temporal_keywords.append("last month")
        
        if not self.llm_callable:
            # Fallback: include detected names + temporal keywords + original query
            return [query] + person_names + temporal_keywords
        
        # Build context-aware prompt with conversation summary
        conversation_context = ""
        if len(self.session_history) > 0:
            # Get last 5 turns for context
            recent_turns = self.session_history[-5:]
            context_lines = []
            for turn in recent_turns:
                context_lines.append(f"User: {turn.user_message}")
                context_lines.append(f"Assistant: {turn.assistant_message}")
            conversation_context = "\n".join(context_lines)
            print(f"[DEBUG] Using {len(recent_turns)} conversation turns for context-aware keyword generation")
        
        # Use LLM to generate relevant search terms - focus on semantic concepts
        prompt = f"""You are a keyword generator. Your ONLY job is to return comma-separated search terms based on the conversation context.

CURRENT DATE: {today_str}

=== CONVERSATION CONTEXT ===
{conversation_context if conversation_context else "(No previous conversation)"}

=== CURRENT MESSAGE ===
"{query}"

Generate 8-10 search keywords based on the FULL conversation above (not just current message).
Include: people/entities mentioned, topics discussed, relationships, emotions, temporal references.

Examples:
- Conversation about Joshna + "yesterday" → joshna, girlfriend, relationship, february 7, yesterday, feelings, dating, love
- "what do I study" → study, college, education, degree, university, engineering, academics
- "tell me about vinay" → vinay, friend, best friend, friendship, relationship

RETURN ONLY COMMA-SEPARATED KEYWORDS, NOTHING ELSE:
"""
        
        try:
            response = self.llm_callable(prompt).strip()
            # Parse comma-separated keywords - filter out garbage
            keywords = []
            for kw in response.split(','):
                kw = kw.strip().lower()
                # Filter out: empty, too long (>50 chars), contains newlines, or template text
                if kw and len(kw) <= 50 and '\n' not in kw and 'based on' not in kw:
                    keywords.append(kw)
            
            # If we got bad output, log it and fall back
            if len(keywords) == 0:
                print(f"[DEBUG] LLM returned bad keywords, using fallback. Response: {response[:100]}")
                keywords = []
            
            # Always include original query and detected temporal keywords
            all_queries = [query] + keywords + person_names + temporal_keywords
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in all_queries:
                q_lower = q.lower()
                if q_lower not in seen:
                    seen.add(q_lower)
                    unique_queries.append(q)
            
            return unique_queries[:15]  # Return up to 15 keywords for comprehensive search
        except Exception as e:
            print(f"[DEBUG] Keyword generation failed: {e}")
            return [query] + person_names + temporal_keywords
    
    def _retrieve_memories_with_keywords(self, keywords: List[str]) -> List[RetrievedMemory]:
        """
        Retrieve memories using pre-expanded keywords from context-aware query expansion.
        
        Args:
            keywords: List of context-aware keywords from _expand_query()
        
        Returns:
            List of RetrievedMemory objects with memories above threshold
        """
        now = datetime.now()
        
        # Detect if this is about current positive feelings (filter out old drama)
        # Use session_history which stores ConversationTurn objects
        recent_turns = self.session_history[-3:] if len(self.session_history) >= 3 else self.session_history
        recent_text = " ".join([turn.user_message + " " + turn.assistant_message for turn in recent_turns]).lower()
        positive_keywords = ['love', 'miss', 'hug', 'kiss', 'feel', 'want', 'like', 'happy', 'excited']
        is_positive_context = any(word in recent_text for word in positive_keywords)
        
        # Detect if user is explicitly referencing old events
        temporal_indicators = ['old', 'years ago', 'back then', 'past', 'used to', 'before', 'previous']
        references_old_explicitly = any(indicator in recent_text for indicator in temporal_indicators)
        
        # Search with each expanded keyword
        all_results = {}
        for keyword in keywords:
            results = self._retrieve_memories(keyword, top_k=20, similarity_threshold=0.05)
            for result in results:
                mem_id = result.memory.memory_id
                mem_date = result.memory.event_date if result.memory.event_date else result.memory.created_at
                days_old = (now - mem_date).days
                
                # Heavily penalize very old memories (>180 days) unless explicitly asked about
                if not references_old_explicitly and days_old > 180:
                    # Reduce score by 70% for memories older than 6 months
                    result = RetrievedMemory(
                        memory=result.memory,
                        similarity=result.similarity * 0.3,
                        final_score=result.final_score * 0.3
                    )
                
                # Filter out negative/dramatic memories if talking about positive current feelings
                if is_positive_context:
                    negative_indicators = ['situationship', 'back off', 'decided to move on', 
                                          'insult', 'humiliation', 'cringe', 'ex ', ' sn ', 'sn,']
                    summary_lower = result.memory.summary.lower()
                    if any(indicator in summary_lower for indicator in negative_indicators):
                        # Heavily penalize negative memories in positive contexts
                        result = RetrievedMemory(
                            memory=result.memory,
                            similarity=result.similarity * 0.3,  # Reduce score by 70%
                            final_score=result.final_score * 0.3
                        )
                
                # Keep the best score for each memory
                if mem_id not in all_results or result.final_score > all_results[mem_id].final_score:
                    all_results[mem_id] = result
        
        # Sort by score and return top results
        combined_results = sorted(all_results.values(), key=lambda x: x.final_score, reverse=True)
        return combined_results[:12]
    
    def _retrieve_memories(self, query: str, top_k: int = 20, similarity_threshold: float = 0.05) -> List[RetrievedMemory]:
        """
        Retrieve and rerank memories for the query.
        SMART FILTERING: Only returns truly relevant memories.
        
        Retrieval strategy:
        1. Semantic search (top_k = 20, threshold 0.05 for Ollama embeddings)
        2. Rerank using: 0.5*similarity + 0.3*confidence + 0.2*recency
        3. Filter by relevance threshold (0.2)
        4. Return top 12 only if highly relevant
        """
        # Step 1: Embed query
        query_embedding = self.embedding_service.embed(query)
        
        # Step 2: Semantic search
        search_results = self.vector_store.search(
            query_embedding, 
            top_k=top_k, 
            similarity_threshold=similarity_threshold
        )
        
        if not search_results:
            return []
        
        # Step 3: Rerank with multi-factor scoring
        retrieved = []
        now = datetime.now()
        
        for memory, similarity in search_results:
            # Calculate recency score (0-1, decays over 30 days)
            days_old = (now - memory.last_reinforced).days
            recency = max(0.0, 1.0 - (days_old / 30.0))
            
            # Combined score
            final_score = (
                0.5 * similarity +
                0.3 * memory.confidence +
                0.2 * recency
            )
            
            retrieved.append(RetrievedMemory(
                memory=memory,
                similarity=similarity,
                final_score=final_score
            ))
        
        # Sort by final score
        retrieved.sort(key=lambda x: x.final_score, reverse=True)
        
        # SMART FILTERING: Only keep relevant memories (adjusted for Ollama embeddings)
        # Threshold: final_score > 0.2 means it's useful (Ollama has lower scores)
        relevant_memories = [m for m in retrieved if m.final_score > 0.2]
        
        # Return top 12 max, only if they're truly relevant
        return relevant_memories[:12] if relevant_memories else []
    
    def _build_prompt(self, user_message: str, retrieved_memories: List[RetrievedMemory], relevant_facts: List = None) -> str:
        """
        Construct the final prompt with memory context and structured facts.
        SMART: Only includes memories if they're actually relevant to current message.
        Keeps total prompt under 8k tokens.
        """
        MAX_TOKENS = 8000  # Leave room for response
        
        if relevant_facts is None:
            relevant_facts = []
        
        prompt_parts = []
        
        # Current date/time context for temporal awareness
        now = datetime.now()
        current_date = now.strftime("%A, %B %d, %Y")  # e.g., "Friday, February 08, 2026"
        current_time = now.strftime("%I:%M %p")  # e.g., "02:30 PM"
        
        # System role - prioritize conversation flow, let LLM decide memory relevance
        prompt_parts.append(
            f"You are Antony Shane's close friend and digital twin - talk like a real Gen Z friend, not a formal assistant.\n"
            f"Current date: {current_date}\n"
            f"Current time: {current_time}\n"
            "\n⏰ CRITICAL - TEMPORAL AWARENESS:\n"
            f"- TODAY is {current_date} at {current_time}\n"
            "- Memories below show WHEN they happened (today, yesterday, weeks ago, or specific dates like 'Aug 2022')\n"
            "- If a memory says 'Aug 2022' or '2023' → that was YEARS AGO, don't treat it as recent\n"
            "- If a memory says 'Feb 2026' and today is Feb 2026 → that's this month\n"
            "- DO NOT bring up events from months/years ago unless he explicitly asks about them\n"
            "- DO NOT confuse old events with current conversation\n"
            "\n👤 PERSPECTIVE - YOU ARE AN AI, NOT A PARTICIPANT:\n"
            "- The memories below are things ANTONY TOLD YOU about HIS life\n"
            "- YOU were NOT there when these events happened to him\n"
            "- DON'T say 'we went through', 'I remember when we', 'our conversation'\n"
            "- DO say 'you told me about', 'you mentioned', 'from what you said'\n"
            "- You're learning about his life, NOT living it with him\n"
            "- Example: 'yo you mentioned that SN thing from way back' NOT 'remember when we talked about SN?'\n"
            "\n🎯 RESPONSE RULES:\n"
            "1. ONLY respond to what he's saying RIGHT NOW in the current conversation\n"
            "2. Talk naturally - use Gen Z slang (fr, ngl, yk, lowkey, bruh, deadass, etc.)\n"
            "3. Keep it SHORT (1-2 sentences max, like texting)\n"
            "4. Match his energy - hyped if he's excited, supportive if he's down\n"
            "5. OLD memories (months/years ago) are BACKGROUND CONTEXT ONLY\n"
            "   → Don't mention them unless he specifically asks about that time period\n"
            "6. If he says something is 'old' or 'years ago' → BELIEVE HIM, don't argue\n"
            "\n⚠️ NEVER DO THIS:\n"
            "- Don't bring up drama from 2022-2023 unless he asks\n"
            "- Don't assume events from old memories happened recently\n"
            "- Don't reference people from old memories as if they're current\n"
            "- Don't ask about things that happened years ago as if they're new\n"
            "- Don't act like YOU were there when events happened to HIM\n"
            "\n✅ DO THIS:\n"
            "- Focus on the CURRENT conversation (the text right below)\n"
            "- If he mentions something new → respond to THAT\n"
            "- Be his homie, keep it real and chill\n"
            "- If confused about timing → ask him directly\n"
            "- Position yourself as someone he's TELLING his story to, not someone who lived it\n"
        )
        
        # Start conversation section - put this BEFORE memories to prioritize context
        prompt_parts.append("\n---\nRECENT CONVERSATION (this is what matters most):")
        
        # Include previous conversation turns for context (with token limit)
        if len(self.session_history) > 0:
            # Start with last 15 turns for better continuity
            context_window = min(15, len(self.session_history))
            recent_turns = self.session_history[-context_window:]
            
            # Approximate token count (4 chars ≈ 1 token)
            estimated_tokens = len("\n".join(prompt_parts)) // 4
            
            # Add conversation turns, but prioritize recent ones
            conversation_parts = []
            for turn in recent_turns:
                turn_text = f"User: {turn.user_message}\nAssistant: {turn.assistant_message}"
                turn_tokens = len(turn_text) // 4
                
                # More generous token budget - prioritize conversation context over memories
                if estimated_tokens + turn_tokens > MAX_TOKENS - 1500:  # Reserve only 1500 for memories and response
                    break
                    
                conversation_parts.append(f"User: {turn.user_message}")
                conversation_parts.append(f"Assistant: {turn.assistant_message}")
                estimated_tokens += turn_tokens
            
            # Add all conversation parts
            prompt_parts.extend(conversation_parts)
            
            # DEBUG: Show conversation context stats
            turns_included = len(conversation_parts) // 2  # Each turn = 2 parts (User + Assistant)
            print(f"[DEBUG] Including {turns_included}/{len(self.session_history)} conversation turns for context")
        else:
            prompt_parts.append("(This is the start of your conversation)")
        
        # Include memories above threshold (0.2) - let LLM decide relevance (respect token limit)
        if retrieved_memories:
            relevant = [m for m in retrieved_memories if m.final_score > 0.2]
            if relevant:
                # Show top 12 memories from semantic search, respecting token limit
                memories_to_add = []
                current_tokens = len("\n".join(prompt_parts)) // 4
                
                for mem in relevant[:12]:
                    # Use event_date if available (when event happened), otherwise created_at
                    mem_date = mem.memory.event_date if mem.memory.event_date else mem.memory.created_at
                    days_ago = (now - mem_date).days
                    
                    # Format timestamp based on recency
                    if days_ago == 0:
                        time_context = "today"
                    elif days_ago == 1:
                        time_context = "yesterday"
                    elif days_ago < 7:
                        time_context = f"{days_ago} days ago"
                    elif days_ago < 30:
                        weeks = days_ago // 7
                        time_context = f"{weeks} week{'s' if weeks > 1 else ''} ago"
                    elif days_ago < 365:
                        # Show month and year for older memories
                        time_context = mem_date.strftime("%b %Y")  # e.g., "Aug 2022"
                    else:
                        # Show full date for very old memories
                        time_context = mem_date.strftime("%b %d, %Y")  # e.g., "Aug 7, 2022"
                    
                    mem_text = f"[{time_context}] {mem.memory.summary}"
                    mem_tokens = len(mem_text) // 4
                    if current_tokens + mem_tokens > MAX_TOKENS - 800:  # Reserve 800 for user message and response
                        break
                    memories_to_add.append((mem, time_context))
                    current_tokens += mem_tokens
                
                if memories_to_add:
                    prompt_parts.append("\n[OLD BACKGROUND MEMORIES - DO NOT bring these up unless he asks about that specific time:]:")
                    for mem, time_context in memories_to_add:
                        prompt_parts.append(f"- [{time_context}] {mem.memory.summary}")
                    prompt_parts.append("\n^ These are OLD memories for context. Focus on current conversation above, NOT these old events.")
        
        # Add structured facts if available (exact information)
        if relevant_facts:
            current_tokens = len("\n".join(prompt_parts)) // 4
            facts_to_add = []
            
            for fact in relevant_facts[:8]:  # Limit to 8 facts
                fact_text = f"{fact.subject} {fact.predicate}: {fact.value}"
                if fact.date:
                    fact_text += f" (on {fact.date})"
                fact_tokens = len(fact_text) // 4
                
                if current_tokens + fact_tokens > MAX_TOKENS - 600:
                    break
                facts_to_add.append(fact_text)
                current_tokens += fact_tokens
            
            if facts_to_add:
                prompt_parts.append("\n[Quick Facts - Names, dates, relationships (use current date for time relativity)]:")
                for fact_text in facts_to_add:
                    prompt_parts.append(f"- {fact_text}")
                prompt_parts.append("")
        
        # Current message - make it prominent
        prompt_parts.append("\n---\n[CURRENT MESSAGE - Respond to THIS:]")
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def get_session_history(self) -> List[ConversationTurn]:
        """Return current session history for Agent B processing."""
        return self.session_history.copy()
    
    def clear_session(self):
        """Clear session history (e.g., start new conversation)."""
        self.session_history = []
    
    def get_memory_stats(self) -> dict:
        """Get statistics about available memories."""
        return self.vector_store.get_stats()
