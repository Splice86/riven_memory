"""Memory API server - FastAPI endpoints for memory storage and search."""

import os
from fastapi import FastAPI, HTTPException, Query, Depends
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

def get_db_path(db_name: str) -> str:
    """Get full path for a DB file."""
    if not db_name:
        db_name = DEFAULT_DB
    # Only append .db if not already present
    if not db_name.endswith(".db"):
        db_name = f"{db_name}.db"
    return os.path.join(DB_DIR, db_name)


def get_or_create_db(db_name: str) -> MemoryDB:
    """Get or create a database instance."""
    db_path = get_db_path(db_name)
    init_db(db_path)
    return MemoryDB(db_path=db_path)


# Cache for DB instances (per process)
_db_cache: dict[str, MemoryDB] = {}


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



class GetContextRequest(BaseModel):
    """Request to get context."""
    limit: int = 100
    session: str | None = None  # Optional session ID to filter by


# Database dependency - gets DB from query param
def get_db(db_name: str = Query(DEFAULT_DB, description="Database name")) -> MemoryDB:
    """Get or create a database instance for the requested DB name.
    
    Auto-creates the database if it doesn't exist.
    """
    if db_name not in _db_cache:
        _db_cache[db_name] = get_or_create_db(db_name)
    return _db_cache[db_name]


@app.post("/db/create")
async def create_database(name: str = Query(..., description="Database name to create")) -> dict:
    """Create a new database.
    
    Args:
        name: Name for the new database (without .db extension)
        
    Returns:
        Success message with DB path
    """
    db_path = get_db_path(name)
    
    if os.path.exists(db_path):
        return {"message": "Database already exists", "name": name, "path": db_path}
    
    init_db(db_path)
    # Initialize the DB instance
    _db_cache[name] = MemoryDB(db_path=db_path)
    
    return {"message": "Database created", "name": name, "path": db_path}


@app.get("/db/list")
async def list_databases() -> dict:
    """List all existing databases."""
    db_files = [f[:-3] for f in os.listdir(DB_DIR) if f.endswith(".db")]
    return {"databases": db_files}


@app.get("/db/exists/{name}")
async def check_database_exists(name: str) -> dict:
    """Check if a database exists."""
    db_path = get_db_path(name)
    return {"exists": os.path.exists(db_path), "name": name}


class ExecuteSQLRequest(BaseModel):
    """Request to execute raw SQL."""
    sql: str
    params: list | None = None


@app.post("/db/execute")
async def execute_sql(request: ExecuteSQLRequest, db_name: str = Query("default", description="Database name")) -> dict:
    """Execute raw SQL against the database.
    
    WARNING: This is a powerful and potentially dangerous operation.
    Use only for debugging or direct database inspection.
    
    Args:
        request: SQL statement and optional parameters
        db_name: Database name (query param)
        
    Returns:
        Query results or row count
    """
    import sqlite3
    
    db_path = get_db_path(db_name)
    
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail=f"Database {db_name} not found")
    
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
async def add_memory(request: AddMemoryRequest, db: MemoryDB = Depends(get_db)) -> dict:
    """Add a new memory with optional keywords and properties.
    
    Args:
        request: Memory content, optional keywords, properties, and created_at timestamp
        db: Database name (query param)
        
    Returns:
        The ID of the created memory
    """
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
    max_tokens: int | None = Query(None, description="Max tokens before summarization"),
    min_cluster_size: int | None = Query(None, description="Min messages before summarization"),
    db: MemoryDB = Depends(get_db)
) -> dict:
    """Add a context message and auto-summarize if needed.
    
    Uses the Context class which handles:
    - Adding the message with context keyword and role
    - Token counting
    - Auto-summarization when threshold is exceeded
    
    Args:
        request: Context content and role
        max_tokens: Override max tokens before summarization (default 32000)
        min_cluster_size: Override min messages before summarization (default 3)
        db: Database name (query param)
        
    Returns:
        Memory ID, role, token count, and summarization status
    """
    ctx = Context(db, max_tokens=max_tokens or 32000, min_cluster_size=min_cluster_size or 3)
    result = ctx.add(request.role, request.content, request.created_at, request.session)
    
    return result


@app.get("/context")
async def get_context(
    limit: int = Query(100, description="Max messages to return"),
    max_tokens: int | None = Query(None, description="Max tokens threshold (for get)"),
    session: str | None = Query(None, description="Session ID to filter context"),
    db: MemoryDB = Depends(get_db)
) -> dict:
    """Get context for LLM: summary first, then unsummarized turns.
    
    Args:
        limit: Maximum number of unsummarized turns to return
        max_tokens: Override max tokens (optional, for metadata)
        session: Session ID to filter context (optional)
        db: Database name (query param)
        
    Returns:
        List of context messages (summary + unsummarized)
    """
    ctx = Context(db)
    context = ctx.get(limit=limit, session=session)
    
    return {"context": context, "count": len(context)}





@app.post("/context/cluster")
async def cluster_context(
    target_tokens: int = Query(5000, description="Target token count for summarized context"),
    min_live_tokens: int = Query(1000, description="Minimum tokens to keep unsummarized"),
    max_gap: int | None = Query(None, description="Max seconds between messages (auto-calculated from level if None)"),
    level: int = Query(1, description="Summary level (1 = summary, 2 = summary of summaries)"),
    session: str | None = Query(None, description="Session ID to cluster (optional)"),
    db: MemoryDB = Depends(get_db)
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
    ctx = Context(db)
    return ctx.force_cluster(target_tokens, min_live_tokens, max_gap, level, session)


@app.post("/memories/summary")
async def add_summary(request: AddSummaryRequest, db: MemoryDB = Depends(get_db)) -> dict:
    """Add a summary memory and link it to target memories.
    
    The created_at timestamp is required and should be set by the agent
    making the API call to time-bound the summary.
    
    Args:
        request: Summary content, keywords, properties, created_at, target_ids, link_type
        db: Database name (query param)
        
    Returns:
        The ID of the created summary memory
    """
    
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
async def add_link(request: AddLinkRequest, db: MemoryDB = Depends(get_db)) -> dict:
    """Add a link between two memories.
    
    Args:
        request: source_id, target_id, link_type
        db: Database name (query param)
        
    Returns:
        Success message with link details
    """
    
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
async def search_memories(request: SearchRequest, db: MemoryDB = Depends(get_db)) -> dict:
    """Search memories using the query DSL.
    
    Args:
        request: Query string and limit
        db: Database name (query param)
        
    Returns:
        List of matching memories
    """
    
    results = db.search(request.query, limit=request.limit)
    
    return {"memories": results, "count": len(results)}


@app.get("/memories")
async def list_memories(limit: int = 50, offset: int = 0, db: MemoryDB = Depends(get_db)) -> dict:
    """List all memories with pagination.
    
    Args:
        limit: Maximum number of memories to return
        offset: Number of memories to skip
        db: Database name (query param)
        
    Returns:
        List of memories
    """
    
    results = db.search("", limit=limit + offset)
    return {
        "memories": results[offset:offset + limit],
        "count": len(results),
        "limit": limit,
        "offset": offset
    }


@app.get("/memories/{memory_id}")
async def get_memory(memory_id: int, db: MemoryDB = Depends(get_db)) -> dict:
    """Get a memory by ID.
    
    Args:
        memory_id: The ID of the memory
        db: Database name (query param)
        
    Returns:
        The memory data
    """
    
    memory = db.get_memory(memory_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return memory


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: int, db: MemoryDB = Depends(get_db)) -> dict:
    """Delete a memory by ID.
    
    Args:
        memory_id: The ID of the memory to delete
        db: Database name (query param)
        
    Returns:
        Success message
    """
    
    deleted = db.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {"deleted": memory_id, "message": "Memory deleted successfully"}


class UpdateMemoryRequest(BaseModel):
    """Request to update a memory."""
    properties: dict[str, str] | None = None
    keywords: list[str] | None = None


@app.put("/memories/{memory_id}")
async def update_memory(memory_id: int, request: UpdateMemoryRequest, db: MemoryDB = Depends(get_db)) -> dict:
    """Update a memory's properties and/or keywords.
    
    Args:
        memory_id: The ID of the memory to update
        request: Update request with properties and/or keywords
        db: Database name (query param)
        
    Returns:
        Updated memory data
    """
    
    updated = db.update_memory(memory_id, request.properties, request.keywords)
    if not updated:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return updated


@app.post("/embed")
async def get_embedding(request: EmbedRequest, db: MemoryDB = Depends(get_db)) -> dict:
    """Get embedding vector for text.
    
    Args:
        request: Text to embed
        db: Database name (query param)
        
    Returns:
        Embedding vector as list of floats
    """
    
    embedding = db.embedding.get(request.text)
    
    return {
        "text": request.text,
        "embedding": embedding.tolist(),
        "dimension": len(embedding)
    }


@app.get("/stats")
async def get_stats(db: MemoryDB = Depends(get_db)) -> dict:
    """Get memory statistics.
    
    Args:
        db: Database name (query param)
        
    Returns:
        Count of memories
    """
    
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
                "example": "p:status=active, p:rating>=4, p:opinion<0",
                "description": "Filter by property. Supports string equality and numeric comparisons: <, >, <=, >=, !="
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
