"""Memory API server - FastAPI endpoints for memory storage and search."""

import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone


from database import MemoryDB, init_db
from context import Context
import numpy as np

# Config (layered: config.yaml < secrets_template.yaml < env vars)
import config as cfg

# Database settings
DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    cfg.get('database.db_dir', 'database')
)
os.makedirs(DB_DIR, exist_ok=True)

# Try to import tiktoken for token counting
try:
    import tiktoken
    tiktoken_available = True
except ImportError:
    tiktoken_available = False

app = FastAPI(title="Riven Memory API")

# Default DB name - read from config like the rest of the app
DEFAULT_DB = cfg.get('database.default_db', 'riven.db')

def get_db_path() -> str:
    """Get full path for the DB file."""
    db_name = DEFAULT_DB
    # Only append .db if not already present
    if not db_name.endswith(".db"):
        db_name = f"{db_name}.db"
    return os.path.join(DB_DIR, db_name)


class AddMemoryRequest(BaseModel):
    """Request to add a memory with tags/properties."""
    content: str
    keywords: list[str] | None = None
    properties: dict[str, str] | None = None
    created_at: str | None = None  # Optional timestamp (ISO format)


class AddSummaryRequest(BaseModel):
    """Request to add a summary memory with links to target memories."""
    content: str
    keywords: list[str] | None = None
    properties: dict[str, str] | None = None
    created_at: str  # Required timestamp (ISO format) - set by agent
    target_ids: list[int]  # List of memory IDs to link to
    link_type: str = "summary_of"


class AddLinkRequest(BaseModel):
    """Request to add a link between two memories."""
    source_id: int
    target_id: int
    link_type: str = "related_to"


class SearchRequest(BaseModel):
    """Request to search memories."""
    query: str
    limit: int = 50



class EmbedRequest(BaseModel):
    """Request to get embedding for text."""
    text: str


class AddContextRequest(BaseModel):
    """Request to add a context message."""
    content: str
    role: str  # "user", "assistant", "system", "tool"
    created_at: str | None = None  # Optional timestamp
    session: str | None = None  # Optional session ID (stored as property)
    tool_call_id: str | None = None  # Optional tool call ID for tracing
    function: str | None = None  # Optional function name for tool results
    max_live_messages: int | None = None  # Hard cap on unsummarized messages (force summary if exceeded)


class GetContextRequest(BaseModel):
    """Request to get context."""
    limit: int = 100
    max_summaries: int = 3  # How many top-level summaries to include
    session: str | None = None  # Optional session ID to filter by


# Database singleton - no more db_name selection
_db_instance: MemoryDB | None = None


def get_db() -> MemoryDB:
    """Get or create the single database instance."""
    global _db_instance
    if _db_instance is None:
        db_path = get_db_path()
        init_db(db_path)
        _db_instance = MemoryDB(db_path=db_path)
    return _db_instance


@app.get("/db/info")
async def get_db_info() -> dict:
    """Get current database info."""
    db_path = get_db_path()
    return {"db_name": DEFAULT_DB, "db_path": db_path, "exists": os.path.exists(db_path)}


class ExecuteSQLRequest(BaseModel):
    """Request to execute raw SQL."""
    sql: str
    params: list | None = None


@app.post("/db/execute")
async def execute_sql(request: ExecuteSQLRequest) -> dict:
    """Execute raw SQL against the database.
    
    WARNING: This is a powerful and potentially dangerous operation.
    Use only for debugging or direct database inspection.
    
    Args:
        request: SQL statement and optional parameters
        
    Returns:
        Query results or row count
    """
    import sqlite3
    
    db_path = get_db_path()
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if request.params:
                cursor.execute(request.sql, request.params)
            else:
                cursor.execute(request.sql)
            
            # Check if it's a SELECT query
            if request.sql.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    results.append(dict(row))
                return {"type": "select", "rows": results, "count": len(results)}
            else:
                conn.commit()
                return {"type": "execute", "rows_affected": cursor.rowcount}
                
    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/memories")
async def add_memory(request: AddMemoryRequest) -> dict:
    """Add a new memory with optional keywords and properties.
    
    Args:
        request: Memory content, optional keywords, properties, and created_at timestamp
        
    Returns:
        The ID of the created memory
    """
    db = get_db()
    memory_id = db.add_memory(
        content=request.content,
        keywords=request.keywords,
        properties=request.properties,
        created_at=request.created_at
    )
    
    return {"id": memory_id, "content": request.content[:100]}


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken, fallback to rough estimate."""
    if not text:
        return 0
    
    if tiktoken_available:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    
    # Fallback: ~4 chars per token
    return len(text) // 4



def _count_message_tokens(role: str, content: str) -> int:
    """Count tokens for a message including overhead."""
    return _count_tokens(content) + 4  # ~4 tokens for role/formatting


@app.post("/context")
async def add_context(
    request: AddContextRequest,
    min_cluster_size: int | None = Query(None, description="Min messages before summarization"),
) -> dict:
    """Add a context message and auto-summarize if needed.
    
    Uses the Context class which handles:
    - Adding the message with context keyword and role
    - Token counting
    - Auto-summarization when threshold is exceeded
    - Force summarization when max_live_messages is exceeded
    
    Args:
        request: Context content and role
        min_cluster_size: Override min messages before summarization (default 3)
        
    Returns:
        Memory ID, role, token count, and summarization status
    """
    db = get_db()
    ctx = Context(db, min_cluster_size=min_cluster_size or 3)
    result = ctx.add(
        request.role, 
        request.content, 
        request.created_at, 
        request.session, 
        request.tool_call_id, 
        request.function,
        max_live_messages=request.max_live_messages
    )
    
    return result


@app.get("/context")
async def get_context(
    max_summaries: int = Query(3, description="Max top-level summaries to include"),
    session: str | None = Query(None, description="Session ID to filter context"),
) -> dict:
    """Get context for LLM: top summaries first, then all unsummarized turns.
    
    Summaries are fetched as the "top level" of the summary tree - summaries
    that have no parent summary pointing to them. Only the most recent
    max_summaries are included, oldest first.
    
    All unsummarized messages are returned.
    
    Args:
        max_summaries: Maximum number of top-level summaries to include
        session: Session ID to filter context (optional)
        
    Returns:
        List of context messages (summaries + unsummarized) with timestamps
    """
    db = get_db()
    ctx = Context(db)
    context = ctx.get(max_summaries=max_summaries, session=session)
    
    # Get timestamps for the context metadata
    now = datetime.now(timezone.utc)
    earliest = min((item["created_at"] for item in context), default=None)
    latest = max((item["created_at"] for item in context), default=None)
    
    # Count summaries vs messages for metadata
    summary_count = sum(1 for item in context if item.get("role") == "summary")
    
    return {
        "context": context,
        "count": len(context),
        "summary_count": summary_count,
        "message_count": len(context) - summary_count,
        "retrieved_at": now.isoformat(),
        "earliest_created_at": earliest,
        "latest_created_at": latest,
    }





@app.post("/context/cluster")
async def cluster_context(
    target_tokens: int = Query(5000, description="Target token count for summarized context"),
    min_live_tokens: int = Query(1000, description="Minimum tokens to keep unsummarized"),
    max_gap: int | None = Query(None, description="Max seconds between messages (auto-calculated from level if None)"),
    level: int = Query(1, description="Summary level (1 = summary, 2 = summary of summaries)"),
    session: str | None = Query(None, description="Session ID to cluster (optional)"),
) -> dict:
    """Force temporal clustering to reduce context to target token count.
    
    Uses hierarchical time windows based on level if max_gap not provided.
    oldest groups first, while keeping at least min_live_tokens in unsummarized form.
    
    Args:
        target_tokens: Target token count (default 5000)
        min_live_tokens: Minimum tokens to keep live (default 1000)
        max_gap: Max seconds between messages to cluster (default 30)
        level: Summary level (1 = summary, 2 = summary of summaries, etc.)
        session: Optional session to cluster
        
    Returns:
        Iterations, memories summarized, final token count
    """
    from context import Context
    db = get_db()
    ctx = Context(db)
    return ctx.force_cluster(target_tokens, min_live_tokens, max_gap, level, session)


@app.post("/memories/summary")
async def add_summary(request: AddSummaryRequest) -> dict:
    """Add a summary memory and link it to target memories.
    
    The created_at timestamp is required and should be set by the agent
    making the API call to time-bound the summary.
    
    Args:
        request: Summary content, keywords, properties, created_at, target_ids, link_type
        
    Returns:
        The ID of the created summary memory
    """
    db = get_db()
    
    # Add the summary memory
    summary_id = db.add_memory(
        content=request.content,
        keywords=request.keywords,
        properties=request.properties,
        created_at=request.created_at
    )
    
    # Link to each target memory
    for target_id in request.target_ids:
        db.add_link(
            source_id=summary_id,
            target_id=target_id,
            link_type=request.link_type
        )
    
    return {"id": summary_id, "content": request.content[:100], "linked_to": request.target_ids}


@app.post("/memories/link")
async def add_link(request: AddLinkRequest) -> dict:
    """Add a link between two memories.
    
    Args:
        request: source_id, target_id, link_type
        
    Returns:
        Success message with link details
    """
    db = get_db()
    
    db.add_link(
        source_id=request.source_id,
        target_id=request.target_id,
        link_type=request.link_type
    )
    
    return {
        "source_id": request.source_id,
        "target_id": request.target_id,
        "link_type": request.link_type,
        "message": "Link created successfully"
    }


@app.post("/memories/search")
async def search_memories(request: SearchRequest) -> dict:
    """Search memories using the query DSL.
    
    Args:
        request: Query string and limit
        
    Returns:
        List of matching memories
    """
    db = get_db()
    
    results = db.search(request.query, limit=request.limit)
    
    return {"memories": results, "count": len(results)}


@app.get("/memories")
async def list_memories(limit: int = 50, offset: int = 0) -> dict:
    """List all memories with pagination.
    
    Args:
        limit: Maximum number of memories to return
        offset: Number of memories to skip
        
    Returns:
        List of memories
    """
    db = get_db()
    
    results = db.search("", limit=limit + offset)
    return {
        "memories": results[offset:offset + limit],
        "count": len(results),
        "limit": limit,
        "offset": offset
    }


@app.get("/memories/{memory_id}")
async def get_memory(memory_id: int) -> dict:
    """Get a memory by ID.
    
    Args:
        memory_id: The ID of the memory
        
    Returns:
        The memory data
    """
    db = get_db()
    
    memory = db.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return memory


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int) -> dict:
    """Delete a memory by ID.
    
    Args:
        memory_id: The ID of the memory to delete
        
    Returns:
        Success message
    """
    db = get_db()
    
    deleted = db.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {"deleted": memory_id, "message": "Memory deleted successfully"}


class UpdateMemoryRequest(BaseModel):
    """Request to update a memory."""
    properties: dict[str, str] | None = None
    keywords: list[str] | None = None


@app.put("/memories/{memory_id}")
async def update_memory(memory_id: int, request: UpdateMemoryRequest) -> dict:
    """Update a memory's properties and/or keywords.
    
    Args:
        memory_id: The ID of the memory to update
        request: Update request with properties and/or keywords
        
    Returns:
        Updated memory data
    """
    db = get_db()
    
    updated = db.update_memory(memory_id, request.properties, request.keywords)
    if not updated:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return updated


@app.post("/embed")
async def get_embedding(request: EmbedRequest) -> dict:
    """Get embedding vector for text.
    
    Args:
        request: Text to embed
        
    Returns:
        Embedding vector as list of floats
    """
    db = get_db()
    
    embedding = db.embedding.get(request.text)
    
    return {
        "text": request.text,
        "embedding": embedding.tolist(),
        "dimension": len(embedding)
    }


@app.get("/stats")
async def get_stats() -> dict:
    """Get memory statistics.
    
    Returns:
        Count of memories
    """
    db = get_db()
    
    results = db.search("", limit=10000)
    
    return {"count": len(results)}


@app.get("/embed/model")
async def get_embedding_model_info() -> dict:
    """Get information about the embedding model.
    
    Returns:
        Model name, dimension, and other info
    """
    from embedding import MODELS, DEFAULT_MODEL_SIZE
    
    # Get info from the embedding model if loaded
    try:
        from embedding import _default_model
        if _default_model:
            return {
                "model_name": _default_model.model_name,
                "model_size": _default_model.model_size,
                "dimension": _default_model.dimension,
                "device": _default_model.device,
                "cache_db": _default_model.cache_db,
            }
    except Exception:
        pass
    
    # Return defaults
    return {
        "model_name": MODELS[DEFAULT_MODEL_SIZE]["name"],
        "model_size": DEFAULT_MODEL_SIZE,
        "dimension": MODELS[DEFAULT_MODEL_SIZE]["dimension"],
    }


@app.get("/embed/cache")
async def get_embedding_cache_info() -> dict:
    """Get embedding cache statistics.
    
    Returns:
        Cache count and info
    """
    try:
        from embedding import get_embedding_model
        model = get_embedding_model()
        return model.get_cache_stats()
    except Exception as e:
        return {"error": str(e)}



@app.delete("/embed/cache")
async def clear_embedding_cache() -> dict:
    """Clear the embedding cache.
    
    Returns:
        Number of entries deleted
    """
    try:
        from embedding import get_embedding_model
        model = get_embedding_model()
        count = model.clear_cache()
        return {"deleted": count}
    except Exception as e:
        return {"error": str(e)}


@app.get("/docs/search-syntax")
async def get_search_syntax() -> dict:
    """Get documentation for the search query syntax.
    
    Returns:
        Search syntax documentation
    """
    return {
        "title": "Memory Search Query Syntax",
        "version": "1.0",
        "operators": {
            "AND": "Both conditions must match. Default between terms.",
            "OR": "Either condition can match.",
            "NOT": "Negate a condition."
        },
        "filters": {
            "keyword": {
                "syntax": "k:<keyword> or keyword:<keyword>",
                "example": "k:python or python",
                "description": "Search by keyword tag"
            },
            "property": {
                "syntax": "p:<key>=<value> or p:<key><op><value>",
                "example": "p:status=active, p:rating>=4, p:opinion<0, p:status=active*, p:name=*test*",
                "description": "Filter by property. Supports string equality (with wildcards: * = any chars, ? = single char), and numeric comparisons: <, >, <=, >=, !="
            },
            "date": {
                "syntax": "d:last <n> days or d:<date>",
                "example": "d:last 7 days, d:2025-01-01",
                "description": "Filter by creation date. 'd:last N days' finds memories created in the last N days."
            },
            "similarity": {
                "syntax": "q:<query> or q:<query>@<threshold> or s:<keyword>@<threshold>",
                "example": "q:async programming, q:async@0.7, s:python@0.5",
                "description": "Semantic similarity search. Lower threshold = more permissive. Requires vector embedding."
            },
            "link": {
                "syntax": "l:<link_type> or l:direction:<link_type> or l:<link_type>:(filter)",
                "example": "l:related_to, l:summary_of, l:source:related_to, l:target:related_to, l:summary_of:(k:python)",
                "description": "Find memories by link relationships. Direction: source=links TO others, target=IS LINKED TO."
            },
            "id": {
                "syntax": "id:<memory_id>",
                "example": "id:123",
                "description": "Find a specific memory by ID"
            }
        },
        "conditionals": {
            "syntax": "IF <condition> THEN <query> ELSE <query>",
            "example": "IF k:python THEN k:asyncio ELSE k:docker",
            "description": "Conditional queries based on whether the first condition returns results"
        },
        "grouping": {
            "syntax": "(<query>) AND/OR (<query>)",
            "example": "(k:python OR k:javascript) AND d:last 7 days",
            "description": "Use parentheses to group conditions and control precedence"
        },
        "examples": [
            "k:python AND k:asyncio - memories with both python and asyncio keywords",
            "p:status=active AND k:python - active memories about python",
            "p:opinion<0 - memories with negative opinion (numeric comparison)",
            "p:rating>=4 AND k:positive - highly rated positive memories",
            "d:last 7 days AND (k:python OR k:javascript) - recent Python or JS",
            "l:summary_of:(k:python) - summaries linked to python memories",
            "l:source:related_to - memories that link TO other memories",
            "l:target:related_to - memories that ARE LINKED TO by others",
            "IF k:python THEN k:asyncio ELSE k:docker - conditional based on keyword",
            "(k:python OR k:javascript) AND (d:last 7 days OR p:status=active) - complex nested"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)
