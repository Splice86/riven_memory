"""
Context management for the Memory API.

Single Context class that handles:
- add(): adds context messages, auto-summarizes if needed
- get(): returns summary + unsummarized turns for LLM
"""

import os
from datetime import datetime, timezone
from typing import Optional

# Shared layered config (config.yaml < secrets < env vars)
import riven_memory_config as cfg

# Try to import tiktoken for token counting
try:
    import tiktoken
    tiktoken_available = True
except ImportError:
    tiktoken_available = False

# Try to import OpenAI for LLM calls
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False


# ============================================================================
# Config (derived from shared memory_config)
# ============================================================================

LLM_URL = cfg.get('llm.url', 'http://127.0.0.1:8000/v1/')
LLM_API_KEY = cfg.get('llm.api_key', 'sk-dummy')
LLM_MODEL = cfg.get('llm.model', 'nvidia/MiniMax-M2.5-NVFP4')
MAX_TOKENS_DEFAULT = cfg.get('context.max_tokens', 32000)
MIN_CLUSTER_SIZE = cfg.get('context.min_cluster_size', 3)


# ============================================================================
# Token Counting
# ============================================================================

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken, fallback to rough estimate."""
    if not text:
        return 0
    
    if tiktoken_available:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    
    return len(text) // 4


def count_message_tokens(role: str, content: str) -> int:
    """Count tokens for a message including overhead."""
    return count_tokens(content) + 4


# ============================================================================
# LLM Client
# ============================================================================

class SummarizerLLM:
    """LLM client for generating summaries."""
    
    def __init__(
        self,
        llm_url: str = LLM_URL,
        llm_api_key: str = LLM_API_KEY,
        model: str = LLM_MODEL
    ):
        self.llm_url = llm_url
        self.llm_api_key = llm_api_key
        self.model = model
        
        if openai_available:
            self.client = OpenAI(base_url=f"{self.llm_url}/v1", api_key=self.llm_api_key)
        else:
            self.client = None
    
    def summarize(self, text: str) -> str:
        """Summarize text using the LLM."""
        if not self.client:
            return f"[Summary of {len(text)} chars]"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes text concisely."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize the following in 1-2 paragraphs:\n\n{text}"
                    }
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Summary failed: {e}]"
    
    def health_check(self) -> bool:
        """Check if LLM is available."""
        if not self.client:
            return False
        try:
            self.summarize("test")
            return True
        except Exception:
            return False


# ============================================================================
# Context
# ============================================================================

class Context:
    """
    Handles adding and retrieving context for the LLM.
    
    - add(): adds a context message, auto-summarizes if needed
    - get(): returns summary + unsummarized turns for LLM context
    """
    
    VALID_ROLES = {"user", "assistant", "system", "tool"}
    
    def __init__(self, db, max_tokens: int = MAX_TOKENS_DEFAULT, min_cluster_size: int = MIN_CLUSTER_SIZE):
        self.db = db
        self.max_tokens = max_tokens
        self.min_cluster_size = min_cluster_size
    
    def add(self, role: str, text: str, created_at: str = None, session: str = None) -> dict:
        """
        Add a context message.
        
        Automatically checks and runs summarization if needed.
        
        Args:
            role: Message role (user, assistant, system, tool)
            text: Message content
            created_at: Optional timestamp (ISO format)
            session: Optional session ID to group memories (stored as property)
            
        Returns:
            Dict with id, role, token_count, created_at, and summarization result
        """
        if role not in self.VALID_ROLES:
            raise ValueError(f"Invalid role. Must be one of: {self.VALID_ROLES}")
        
        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat()
        
        token_count = count_message_tokens(role, text)
        
        memory_id = self.db.add_memory(
            content=text,
            keywords=["context", role],
            properties={
                "role": role,
                "node_type": "context",
                "token_count": str(token_count)
            },
            created_at=created_at,
            session=session
        )
        
        # Check if summarization is needed
        summary_result = self._maybe_summarize(session)
        
        return {
            "id": memory_id,
            "role": role,
            "token_count": token_count,
            "created_at": created_at,
            "summarized": summary_result.get("summarized", False),
            "summary_id": summary_result.get("summary_id"),
            "memories_summarized": summary_result.get("memories_summarized", 0)
        }
    
    def get(self, limit: int = 100, session: str = None) -> list[dict]:
        """
        Get context for LLM: summary first, then unsummarized turns.
        
        Args:
            limit: Maximum number of unsummarized turns to return
            session: Optional session ID to filter by
            
        Returns:
            List of memory dicts with id, role, content, created_at
        """
        # Get last summary (filtered by session if provided)
        summary = self._get_last_summary(session)
        
        # Get unsummarized (filtered by session if provided)
        unsummarized = self._get_unsummarized(limit, session)
        
        # Build context: summary first, then unsummarized
        context = []
        
        if summary:
            context.append({
                "id": summary["id"],
                "role": "summary",
                "content": summary["content"],
                "created_at": summary["created_at"]
            })
        
        for mem in unsummarized:
            context.append({
                "id": mem["id"],
                "role": mem["role"],
                "content": mem["content"],
                "created_at": mem["created_at"]
            })
        
        return context
    
    def get_token_count(self, session: str = None) -> int:
        """Get total tokens in unsummarized context."""
        # Build query with session filter if provided (property filter)
        query_parts = ["k:context"]
        if session:
            query_parts.append(f"p:session={session}")
        query = " AND ".join(query_parts)
        results = self.db.search(query, limit=10000)
        
        total = 0
        for mem in results:
            props = mem.get("properties", {})
            # Skip if already summarized (any level)
            if props.get("was_summarized") == "true":
                continue
            if props.get("summary_level"):
                continue
            
            token_count = props.get("token_count", "0")
            try:
                total += int(token_count)
            except ValueError:
                total += count_tokens(mem.get("content", ""))
        
        return total
    
    def _maybe_summarize(self, session: str = None) -> dict:
        """Check token count and summarize if needed."""
        # Build query with session filter if provided (property filter)
        query_parts = ["k:context"]
        if session:
            query_parts.append(f"p:session={session}")
        query = " AND ".join(query_parts)
        results = self.db.search(query, limit=10000)
        
        unsummarized = []
        for mem in results:
            props = mem.get("properties", {})
            if props.get("was_summarized") == "true":
                continue
            
            token_count = props.get("token_count", "0")
            try:
                token_count = int(token_count)
            except ValueError:
                token_count = count_tokens(mem.get("content", ""))
            
            unsummarized.append({
                "id": mem["id"],
                "content": mem["content"],
                "created_at": mem["created_at"],
                "token_count": token_count
            })
        
        if len(unsummarized) < self.min_cluster_size:
            return {"summarized": False}
        
        total_tokens = sum(m["token_count"] for m in unsummarized)
        
        if total_tokens <= self.max_tokens:
            return {"summarized": False}
        
        return self._summarize(unsummarized, session)
    
    def _summarize(self, memories: list[dict], session: str = None, level: int = None) -> dict:
        """Summarize the given memories at the specified level."""
        if not memories:
            return {"summarized": False}
        
        # Determine level: use provided level or calculate from input memories
        if level is None:
            existing_level = memories[0].get("properties", {}).get("summary_level")
            level = (int(existing_level) + 1) if existing_level else 1
        else:
            level = int(level)
        
        combined = "\n\n".join(m["content"] for m in memories)
        llm = SummarizerLLM()
        summary_text = llm.summarize(combined)
        
        total_tokens = 0
        for m in memories:
            try:
                total_tokens += int(m.get("properties", {}).get("token_count", "0"))
            except (ValueError, TypeError):
                pass
        
        # Use the most recent timestamp from the memories being summarized
        created_at = max(m["created_at"] for m in memories)
        
        # Get session from first memory's properties if not provided
        if not session:
            session = memories[0].get("properties", {}).get("session")
        
        # Estimate token count for the summary
        summary_token_count = count_message_tokens("summary", summary_text)
        
        summary_id = self.db.add_memory(
            content=summary_text,
            keywords=["context", "summary"],
            properties={
                "role": "summary",
                "summary_level": str(level),
                "token_count": str(summary_token_count),
                "summarized_count": str(len(memories)),
                "summarized_tokens": str(total_tokens)
            },
            created_at=created_at,
            session=session
        )
        
        for memory in memories:
            self.db.update_memory(
                memory["id"],
                properties={"was_summarized": "true"}
            )
            self.db.add_link(summary_id, memory["id"], "summary_of")
        
        return {
            "summarized": True,
            "summary_id": summary_id,
            "memories_summarized": len(memories)
        }
    
    def _get_last_summary(self, session: str = None) -> Optional[dict]:
        """Get the most recent summary memory."""
        # Build query with session filter if provided (property filter)
        query_parts = ["k:summary"]
        if session:
            query_parts.append(f"p:session={session}")
        query = " AND ".join(query_parts)
        results = self.db.search(query, limit=10)
        
        if not results:
            return None
        
        # Sort by created_at desc to get most recent
        results.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return results[0]
    
    def _get_unsummarized(self, limit: int, session: str = None) -> list[dict]:
        """Get unsummarized context memories."""
        # Build query with session filter if provided (property filter)
        query_parts = ["k:context"]
        if session:
            query_parts.append(f"p:session={session}")
        query = " AND ".join(query_parts)
        results = self.db.search(query, limit=10000)
        
        unsummarized = []
        for mem in results:
            props = mem.get("properties", {})
            # Skip if already summarized (original messages marked after summarization)
            if props.get("was_summarized") == "true":
                continue
            # For level 1, only get original messages (not summaries)
            # Summaries have role="summary" and summary_level set
            if props.get("role") == "summary":
                continue
            
            unsummarized.append({
                "id": mem["id"],
                "role": props.get("role", "unknown"),
                "content": mem["content"],
                "created_at": mem["created_at"],
                "properties": {"token_count": props.get("token_count", "0")}
            })
        
        unsummarized.sort(key=lambda m: m["created_at"])
        return unsummarized[-limit:]
    
    def _get_by_level(self, level: int, session: str = None) -> list[dict]:
        """Get summaries at a specific level for hierarchical clustering.
        
        Args:
            level: Summary level to fetch (e.g., 1 for first-level summaries)
            session: Optional session filter
            
        Returns:
            List of summary memories at the specified level
        """
        # Build query for summaries at this level
        query_parts = [f"k:summary", f"summary_level:{level}"]
        if session:
            query_parts.append(f"p:session={session}")
        query = " AND ".join(query_parts)
        results = self.db.search(query, limit=10000)
        
        summaries = []
        for mem in results:
            props = mem.get("properties", {})
            # Skip if already summarized (prevents re-clustering)
            if props.get("was_summarized") == "true":
                continue
            
            summaries.append({
                "id": mem["id"],
                "role": props.get("role", "summary"),
                "content": mem["content"],
                "created_at": mem["created_at"],
                "properties": {
                    "token_count": props.get("token_count", "0"),
                    "summary_level": props.get("summary_level"),
                    "was_summarized": props.get("was_summarized", ""),
                    "session": props.get("session", "")
                }
            })
        
        return summaries
    
    # Hierarchical max_gap: time window for each summary level
    # Level 1: ~1 minute (cluster messages within a conversation)
    # Level 2: ~30 minutes (cluster summaries of conversations)
    # Level 3: ~3 hours (cluster summary-of-summaries)
    # Level 4+: ~1 day, ~1 week, etc.
    MAX_GAP_BY_LEVEL = {
        1: 60,        # 1 minute
        2: 1800,      # 30 minutes
        3: 10800,     # 3 hours
        4: 86400,     # 1 day
        5: 604800,    # 1 week
    }
    
    def _get_max_gap_for_level(self, level: int) -> int:
        """Get the appropriate max_gap for a given summary level."""
        return self.MAX_GAP_BY_LEVEL.get(level, 86400)  # default to 1 day
    
    def _get_lower_level_in_time_window(self, primary_memories: list[dict], max_gap: int) -> list[dict]:
        """Get unsummarized memories from lower levels that are OLDER than the oldest primary memory.
        
        This pulls in older unsummarized memories that are within the time window
        of the oldest primary memory, ensuring old memories get clustered first.
        
        Args:
            primary_memories: The primary memories being clustered (from level-1)
            max_gap: Time window in seconds
            
        Returns:
            List of older unsummarized memories from lower levels in time window
        """
        from datetime import datetime, timedelta
        
        if not primary_memories:
            return []
        
        # Get the oldest primary memory's timestamp
        oldest_primary = min(primary_memories, key=lambda m: m.get("created_at", ""))
        oldest_ts_str = oldest_primary.get("created_at", "")
        
        if not oldest_ts_str:
            return []
        
        try:
            if "+" in oldest_ts_str or oldest_ts_str.endswith("Z"):
                oldest_ts = datetime.fromisoformat(oldest_ts_str.replace("Z", "+00:00"))
            else:
                oldest_ts = datetime.fromisoformat(oldest_ts_str)
        except (ValueError, TypeError):
            return []
        
        # Get all unsummarized context memories
        session = primary_memories[0].get("properties", {}).get("session")
        query_parts = ["k:context"]
        if session:
            query_parts.append(f"p:session={session}")
        query = " AND ".join(query_parts)
        results = self.db.search(query, limit=10000)
        
        memories = []
        
        for mem in results:
            props = mem.get("properties", {})
            
            # Skip if already summarized
            if props.get("was_summarized") == "true":
                continue
            
            # Skip if this IS a summary
            if props.get("role") == "summary":
                continue
            
            created_at_str = mem.get("created_at", "")
            if not created_at_str:
                continue
            
            try:
                if "+" in created_at_str or created_at_str.endswith("Z"):
                    created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                else:
                    created_at = datetime.fromisoformat(created_at_str)
                
                # Only pull in OLDER memories (before the oldest primary memory)
                # and within the time window
                time_diff = (oldest_ts - created_at).total_seconds()
                if 0 < time_diff <= max_gap:
                    memories.append({
                        "id": mem["id"],
                        "role": props.get("role", "unknown"),
                        "content": mem["content"],
                        "created_at": mem["created_at"],
                        "properties": {"token_count": props.get("token_count", "0")}
                    })
            except (ValueError, TypeError):
                continue
        
        return memories
    
    def _group_by_time(self, memories: list[dict], max_gap: int = 30) -> list[list[dict]]:
        """Group memories by temporal proximity.
        
        Args:
            memories: List of memories sorted by created_at
            max_gap: Max seconds between messages to be in same group
            
        Returns:
            List of groups (each group is a list of memories)
        """
        if not memories:
            return []
        
        from datetime import datetime
        
        groups = []
        current_group = [memories[0]]
        
        for i in range(1, len(memories)):
            prev_time = datetime.fromisoformat(memories[i-1]["created_at"].replace("Z", "+00:00"))
            curr_time = datetime.fromisoformat(memories[i]["created_at"].replace("Z", "+00:00"))
            
            # Calculate gap in seconds
            gap = (curr_time - prev_time).total_seconds()
            
            if abs(gap) <= max_gap:
                # Within gap - add to current group
                current_group.append(memories[i])
            else:
                # Gap too large - start new group
                groups.append(current_group)
                current_group = [memories[i]]
        
        # Don't forget the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def force_cluster(self, target_tokens: int = 5000, min_live_tokens: int = 1000, max_gap: int = None, level: int = 1, session: str = None) -> dict:
        """Force temporal clustering to reduce context to target token count.
        
        Groups messages by temporal proximity and summarizes oldest groups first,
        while keeping at least min_live_tokens in unsummarized form.
        
        Uses hierarchical time windows based on level:
        - Level 1: ~1 minute (messages in a conversation)
        - Level 2: ~30 minutes (summaries of conversations)
        - Level 3: ~3 hours (summaries of summaries)
        - etc.
        
        Args:
            target_tokens: Target token count for summarized context (default 5000)
            min_live_tokens: Minimum tokens to keep unsummarized (default 1000)
            max_gap: Max seconds between messages to cluster (auto-calculated from level if None)
            level: Summary level to create (1 = summary, 2 = summary of summaries, etc.)
            session: Optional session to cluster
            
        Returns:
            Dict with iterations, memories_summarized, final_token_count
        """
        if min_live_tokens >= target_tokens:
            return {"error": "min_live_tokens must be less than target_tokens"}
        
        # Use hierarchical max_gap based on level if not provided
        if max_gap is None:
            max_gap = self._get_max_gap_for_level(level)
        
        iterations = 0
        total_summarized = 0
        
        while True:
            # Get memories to summarize based on level
            if level == 1:
                # Get unsummarized original messages
                memories = self._get_unsummarized(limit=10000, session=session)
            else:
                # Get summaries from previous level for higher levels
                primary_memories = self._get_by_level(level - 1, session)
                
                # Also pull in older unsummarized memories from lower levels that are in time window
                lower_level_memories = self._get_lower_level_in_time_window(
                    primary_memories, max_gap
                )
                
                # Combine primary + lower level memories
                memories = primary_memories + lower_level_memories
            
            if not memories:
                break
            
            # Sort oldest first
            memories.sort(key=lambda m: m.get("created_at", ""))
            
            # Calculate current token count
            current_tokens = 0
            for m in memories:
                try:
                    current_tokens += int(m.get("properties", {}).get("token_count", "0"))
                except (ValueError, TypeError):
                    pass
            
            # Check if we're below threshold
            if current_tokens <= target_tokens:
                break
            
            # Group by temporal proximity
            groups = self._group_by_time(memories, max_gap)
            
            if not groups:
                break
            
            # Take the oldest group to summarize
            to_summarize = groups[0]
            
            if len(to_summarize) < 2:
                break
            
            # Check remaining tokens after this summarization
            group_tokens = sum(int(m.get("properties", {}).get("token_count", "0")) for m in to_summarize)
            remaining_tokens = current_tokens - group_tokens
            
            # Only enforce min_live if there will be more unsummarized memories after this
            # If we're at the last group and above target, allow summarization anyway
            remaining_count = len(memories) - len(to_summarize)
            if remaining_count > 0 and remaining_tokens < min_live_tokens:
                break
            
            # Summarize these at the specified level
            result = self._summarize(to_summarize, session, level)
            
            if not result.get("summarized"):
                break
            
            iterations += 1
            total_summarized += result.get("memories_summarized", 0)
            
            # Safety limit
            if iterations > 20:
                break
        
        # Get final token count (remaining after clustering)
        if level == 1:
            final_memories = self._get_unsummarized(limit=10000, session=session)
        else:
            final_memories = self._get_by_level(level - 1, session)
        
        final_tokens = 0
        for m in final_memories:
            try:
                final_tokens += int(m.get("properties", {}).get("token_count", "0"))
            except (ValueError, TypeError):
                pass
        
        return {
            "iterations": iterations,
            "memories_summarized": total_summarized,
            "final_token_count": final_tokens
        }


# Example usage:
# from context import Context
#
# ctx = Context(db)
# ctx.add("user", "Hello!")
# ctx.add("assistant", "Hi there!")
#
# # Get context for LLM (summary + unsummarized turns)
# context = ctx.get()