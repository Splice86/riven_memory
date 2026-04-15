"""Simplified memory database with vector embeddings."""

import sqlite3
import numpy as np
from datetime import datetime, timezone
from typing import Optional

# Try to import EmbeddingModel, fall back to simple version if not available
try:
    from embedding import EmbeddingModel
except ImportError:
    # Fallback embedding model when torch not available
    class EmbeddingModel:
        """Simple embedding model that returns zero vectors."""
        def __init__(self):
            self.dimension = 384
        
        def get(self, text: str) -> np.ndarray:
            return np.zeros(self.dimension, dtype=np.float32)

# Try to import tiktoken for token counting
try:
    import tiktoken
    tiktoken_available = True
except ImportError:
    tiktoken_available = False


DEFAULT_DB_PATH = "memory.db"


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



class MemoryDB:
    """SQLite-based memory storage with vector embeddings."""
    
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        self.db_path = db_path
        self.embedding = embedding_model or EmbeddingModel()
        init_db(db_path)
    
    def add_memory(
        self,
        content: str,
        keywords: list[str] | None = None,
        properties: dict[str, str] | None = None,
        embedding: np.ndarray | None = None,
        created_at: str | None = None,
        session: str | None = None
    ) -> int:
        """Add a memory with optional keywords and properties.
        
        Args:
            content: The memory text
            keywords: Optional keywords to tag the memory
            properties: Optional key-value pairs (e.g., {"role": "user"})
            embedding: Optional pre-computed embedding (generated from content if not provided)
            created_at: Optional ISO timestamp (e.g., "2025-01-01T10:00:00+00:00")
            session: Optional session ID to group memories (stored as property)
            
        Returns:
            The ID of the inserted memory
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = self.embedding.get(content)
        
        # Use provided timestamp or current time
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        
        # Compute token count
        token_count = _count_tokens(content)
        
        # Merge session into properties
        props = properties.copy() if properties else {}
        if session:
            props['session'] = session
        
        with sqlite3.connect(self.db_path) as conn:
            # Insert memory (session stored as property, not column)
            cursor = conn.execute(
                """INSERT INTO memories (content, embedding, created_at, last_updated, token_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (content, embedding.tobytes(), created_at, created_at, token_count)
            )
            memory_id = cursor.lastrowid
            
            # Handle keywords (lowercase, deduplicated)
            if keywords:
                unique_keywords = set(kw.lower().strip() for kw in keywords if kw.strip())
                
                for kw in unique_keywords:
                    # Get or create keyword
                    kw_row = conn.execute(
                        "SELECT id FROM keywords WHERE name = ?", (kw,)
                    ).fetchone()
                    
                    if kw_row is None:
                        # Insert new keyword
                        kw_embedding = self.embedding.get(kw)
                        cursor = conn.execute(
                            "INSERT INTO keywords (name, embedding) VALUES (?, ?)",
                            (kw, kw_embedding.tobytes())
                        )
                        kw_id = cursor.lastrowid
                    else:
                        kw_id = kw_row[0]
                    
                    # Link memory to keyword
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_keywords (memory_id, keyword_id) VALUES (?, ?)",
                        (memory_id, kw_id)
                    )
            
            # Handle properties (lowercase key names)
            if props:
                for key, value in props.items():
                    key_lower = key.lower().strip()
                    if key_lower:
                        conn.execute(
                            """INSERT OR REPLACE INTO properties (memory_id, key, value)
                               VALUES (?, ?, ?)""",
                            (memory_id, key_lower, value)
                        )
            
            conn.commit()
            
            return memory_id

    def add_link(self, source_id: int, target_id: int, link_type: str) -> None:
        """Add a link between two memories.
        
        The search system handles bidirectional traversal automatically,
        so only one link is stored.
        
        Args:
            source_id: ID of the source memory (the one doing the linking)
            target_id: ID of the target memory (the one being linked to)
            link_type: Type of link (e.g., "related_to", "summary_of", "child")
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO memory_links (source_id, target_id, link_type)
                   VALUES (?, ?, ?)""",
                (source_id, target_id, link_type)
            )
            conn.commit()

    def search(self, query_string: str, limit: int = 50) -> list[dict]:
        """Search memories using the query DSL.
        
        Args:
            query_string: Search query in DSL format
            limit: Maximum number of results
            
        Returns:
            List of matching memories with their data
        
        See search.py for DSL documentation.
        """
        from .search import MemorySearcher
        searcher = MemorySearcher(self.db_path, self.embedding)
        return searcher.search(query_string, limit)

    def get_memory(self, memory_id: int) -> dict | None:
        """Get a memory by ID.
        
        Args:
            memory_id: The ID of the memory
            
        Returns:
            Memory dict or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get memory
            row = conn.execute(
                "SELECT id, content, created_at, last_updated FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()
            
            if not row:
                return None
            
            # Get keywords
            keywords = [
                r[0] for r in conn.execute(
                    """SELECT k.name FROM keywords k
                       JOIN memory_keywords mk ON k.id = mk.keyword_id
                       WHERE mk.memory_id = ?""",
                    (memory_id,)
                ).fetchall()
            ]
            
            # Get properties
            properties = {
                r[0]: r[1] for r in conn.execute(
                    "SELECT key, value FROM properties WHERE memory_id = ?",
                    (memory_id,)
                ).fetchall()
            }
            
            return {
                "id": row["id"],
                "content": row["content"],
                "keywords": keywords,
                "properties": properties,
                "created_at": row["created_at"],
                "updated_at": row["last_updated"]
            }

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID.
        
        Args:
            memory_id: The ID of the memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            # Check if memory exists
            exists = conn.execute(
                "SELECT 1 FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()
            
            if not exists:
                return False
            
            # Delete in correct order (foreign keys)
            conn.execute("DELETE FROM memory_keywords WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM properties WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memory_links WHERE source_id = ? OR target_id = ?", (memory_id, memory_id))
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            
            return True
    
    def update_memory(self, memory_id: int, properties: dict = None, keywords: list = None) -> dict | None:
        """Update a memory's properties and/or keywords.
        
        Args:
            memory_id: The ID of the memory to update
            properties: Dict of properties to update
            keywords: List of keywords to set (replaces existing)
            
        Returns:
            Updated memory dict, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            # Check if memory exists
            memory = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()
            
            if not memory:
                return None
            
            # Update properties
            if properties:
                for key, value in properties.items():
                    # Upsert property
                    conn.execute(
                        """INSERT OR REPLACE INTO properties (memory_id, key, value) 
                           VALUES (?, ?, ?)""",
                        (memory_id, key.lower().strip(), value)
                    )
            
            # Update keywords
            if keywords is not None:
                # Remove existing keyword links
                conn.execute("DELETE FROM memory_keywords WHERE memory_id = ?", (memory_id,))
                
                # Add new keywords (same logic as add_memory)
                unique_keywords = set(kw.lower().strip() for kw in keywords if kw.strip())
                
                for kw in unique_keywords:
                    # Get or create keyword
                    kw_row = conn.execute(
                        "SELECT id FROM keywords WHERE name = ?", (kw,)
                    ).fetchone()
                    
                    if kw_row is None:
                        # Insert new keyword with embedding
                        kw_embedding = self.embedding.get(kw)
                        cursor = conn.execute(
                            "INSERT INTO keywords (name, embedding) VALUES (?, ?)",
                            (kw, kw_embedding.tobytes())
                        )
                        kw_id = cursor.lastrowid
                    else:
                        kw_id = kw_row[0]
                    
                    # Link memory to keyword
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_keywords (memory_id, keyword_id) VALUES (?, ?)",
                        (memory_id, kw_id)
                    )
            
            # Update last_updated timestamp
            conn.execute(
                "UPDATE memories SET last_updated = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), memory_id)
            )
            
            conn.commit()
            
            # Return updated memory
            return self.get_memory(memory_id)


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize the database schema.
    
    Args:
        db_path: Path to the SQLite database file
    """
    with sqlite3.connect(db_path) as conn:
        # Main memories table - session stored as property, not column
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                last_accessed TEXT,
                token_count INTEGER DEFAULT 0
            )
        """)
        
        # Keywords table (with embeddings for similarity search)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                embedding BLOB
            )
        """)
        
        # Memory keywords junction
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_keywords (
                memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                keyword_id INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
                PRIMARY KEY (memory_id, keyword_id)
            )
        """)
        
        # Memory properties (key-value store) - renamed from memory_properties
        conn.execute("""
            CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                UNIQUE(memory_id, key)
            )
        """)
        
        # Handle migration: rename old table if it exists (for existing databases)
        # Check if old table exists
        old_table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_properties'"
        ).fetchone()
        
        if old_table_exists:
            # Rename old table
            conn.execute("""
                ALTER TABLE memory_properties RENAME TO properties_old
            """)
            
            # Copy data from old table
            conn.execute("""
                INSERT INTO properties (memory_id, key, value)
                SELECT memory_id, key, value FROM properties_old
            """)
            
            # Drop old table
            conn.execute("DROP TABLE properties_old")
        
        # Add token_count column if it doesn't exist (for existing databases)
        # Check if column exists
        col_exists = conn.execute(
            "SELECT name FROM pragma_table_info('memories') WHERE name='token_count'"
        ).fetchone()
        
        if not col_exists:
            conn.execute("ALTER TABLE memories ADD COLUMN token_count INTEGER DEFAULT 0")
        
        # Memory links (directional relationships)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                target_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                link_type TEXT NOT NULL,
                UNIQUE(source_id, target_id, link_type)
            )
        """)
        
        # Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mk_memory ON memory_keywords(memory_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mk_keyword ON memory_keywords(keyword_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_keyword_name ON keywords(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mp_memory_key ON properties(memory_id, key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_links_type ON memory_links(link_type)")
        
        conn.commit()


if __name__ == "__main__":
    import tempfile
    import os
    
    # Mock embedding model that returns zeros
    class MockEmbeddingModel:
        def __init__(self):
            self.dimension = 384  # Common embedding dimension

        def get(self, text: str) -> np.ndarray:
            """Return a zero vector for testing."""
            return np.zeros(self.dimension, dtype=np.float32)

    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        print("Testing MemoryDB...")
        print("=" * 50)
        
        # Initialize with mock embedding
        db = MemoryDB(db_path, embedding_model=MockEmbeddingModel())
        
        # Test 1: Simple memory
        memory_id = db.add_memory("This is my first memory!")
        print(f"✓ Added memory {memory_id}: 'This is my first memory!'")
        
        # Test 2: Memory with keywords
        memory_id = db.add_memory(
            "Python is a great programming language",
            keywords=["Python", "programming", "python", "code"]  # should dedupe
        )
        print(f"✓ Added memory {memory_id} with keywords (deduplicated)")
        
        # Test 3: Memory with properties
        memory_id = db.add_memory(
            "User asked about the weather",
            properties={"role": "user", "source": "chat"}
        )
        print(f"✓ Added memory {memory_id} with properties")
        
        # Test 4: Memory with everything
        memory_id = db.add_memory(
            "Assistant provided a helpful response",
            keywords=["assistant", "help"],
            properties={"role": "assistant", "importance": "high"}
        )
        print(f"✓ Added memory {memory_id} with keywords AND properties")
        
        print("=" * 50)
        print("All tests passed! ✓")
        
    finally:
        # Cleanup
        os.unlink(db_path)
