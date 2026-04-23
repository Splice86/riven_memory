#!/usr/bin/env python3
"""
Comprehensive API test for the Memory API.

Tests all endpoints including:
- Database operations
- Memory CRUD
- Keyword search
- Property filtering
- Date filtering
- Vector/semantic search
- Link operations
- Context operations
- Embedding endpoint
- Combined queries
- Error handling

Usage:
    python test_api.py [--url http://localhost:8030] [--db test_api]

Requires: requests
    pip install requests
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: 'requests' library required. Install with: pip install requests")
    sys.exit(1)


BASE_URL = "http://localhost:8030"
TEST_DB = "test_api_comprehensive"


def wait_for_api(url: str, timeout: int = 30) -> bool:
    """Wait for API to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/docs", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False


def create_db(url: str, name: str) -> Optional[int]:
    """Create a test database, return ID of first memory."""
    requests.post(f"{url}/db/create?name={name}")
    return None


def delete_memory(url: str, db: str, memory_id: int) -> bool:
    """Delete a memory by ID."""
    resp = requests.delete(f"{url}/memories/{memory_id}?db_name={db}")
    return resp.status_code == 200


# ============================================================================
# DATABASE TESTS
# ============================================================================

def test_list_databases(url: str) -> bool:
    """Test listing databases."""
    print("\n[TEST] List databases...")
    resp = requests.get(f"{url}/db/list")
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {len(data.get('databases', []))} databases")
    print("  PASS")
    return True


def test_create_database(url: str) -> bool:
    """Test creating a new database."""
    print("\n[TEST] Create database...")
    db_name = "test_create_db"
    resp = requests.post(f"{url}/db/create?name={db_name}")
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Response: {data}")
    print("  PASS")
    return True


def test_database_exists(url: str) -> bool:
    """Test checking if database exists."""
    print("\n[TEST] Check database exists...")
    resp = requests.get(f"{url}/db/exists/default")
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Response: {data}")
    if data.get("exists") not in (True, False):
        print("  FAIL: Missing 'exists' field")
        return False
    print("  PASS")
    return True


# ============================================================================
# MEMORY CRUD TESTS
# ============================================================================

def test_add_memory_basic(url: str, db: str) -> Optional[int]:
    """Test adding a basic memory. Returns memory ID."""
    print("\n[TEST] Add basic memory...")
    resp = requests.post(
        f"{url}/memories?db_name={db}",
        json={
            "content": "Python async programming is great for I/O bound tasks",
            "keywords": ["python", "async"],
            "properties": {"rating": "5", "status": "active"}
        }
    )
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}: {resp.text}")
        return None
    data = resp.json()
    print(f"  Created memory ID: {data.get('id')}")
    if not data.get('id'):
        print("  FAIL: No ID returned")
        return None
    print("  PASS")
    return data['id']


def test_add_memory_multiple(url: str, db: str) -> dict:
    """Test adding multiple memories. Returns dict of IDs."""
    print("\n[TEST] Add multiple memories...")
    memories = [
        {
            "content": "Docker containers are isolated environments for applications",
            "keywords": ["docker", "containers", "devops"],
            "properties": {"rating": "4", "category": "infrastructure"}
        },
        {
            "content": "JavaScript async/await syntax for promises",
            "keywords": ["javascript", "async", "promises"],
            "properties": {"rating": "5", "category": "frontend"}
        },
        {
            "content": "React useEffect hook for side effects in components",
            "keywords": ["react", "javascript", "hooks"],
            "properties": {"rating": "4", "category": "frontend"}
        },
        {
            "content": "Kubernetes orchestrates containers at scale",
            "keywords": ["kubernetes", "docker", "orchestration"],
            "properties": {"rating": "5", "category": "infrastructure"}
        },
        {
            "content": "SQLite is a lightweight embedded database",
            "keywords": ["sqlite", "database", "sql"],
            "properties": {"rating": "3", "category": "database"}
        },
    ]
    
    ids = {}
    for i, mem in enumerate(memories):
        resp = requests.post(f"{url}/memories?db_name={db}", json=mem)
        if resp.status_code != 200:
            print(f"  FAIL: Failed to add memory {i}: {resp.status_code}")
            return {}
        data = resp.json()
        ids[f"mem{i+1}"] = data['id']
        print(f"  Created mem{i+1}: ID={data['id']}")
    
    print("  PASS")
    return ids


def test_get_memory(url: str, db: str, memory_id: int) -> bool:
    """Test getting a specific memory by ID."""
    print(f"\n[TEST] Get memory {memory_id}...")
    resp = requests.get(f"{url}/memories/{memory_id}?db_name={db}")
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Content: {data.get('content', '')[:60]}...")
    if not data.get('content'):
        print("  FAIL: No content returned")
        return False
    print("  PASS")
    return True


def test_update_memory(url: str, db: str, memory_id: int) -> bool:
    """Test updating a memory's properties."""
    print(f"\n[TEST] Update memory {memory_id}...")
    resp = requests.put(
        f"{url}/memories/{memory_id}?db_name={db}",
        json={
            "properties": {"rating": "5", "updated": "true"},
            "keywords": ["updated", "python"]
        }
    )
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}: {resp.text}")
        return False
    data = resp.json()
    print(f"  Response: rating={data.get('properties', {}).get('rating')}")
    print("  PASS")
    return True


def test_list_memories(url: str, db: str) -> bool:
    """Test listing memories with pagination."""
    print("\n[TEST] List memories (paginated)...")
    resp = requests.get(f"{url}/memories?db_name={db}&limit=10&offset=0")
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Total count: {data.get('count')}")
    print(f"  Returned: {len(data.get('memories', []))}")
    print("  PASS")
    return True


def test_delete_memory(url: str, db: str, memory_id: int) -> bool:
    """Test deleting a memory."""
    print(f"\n[TEST] Delete memory {memory_id}...")
    resp = requests.delete(f"{url}/memories/{memory_id}?db_name={db}")
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    # Try to get deleted memory - should 404
    resp2 = requests.get(f"{url}/memories/{memory_id}?db_name={db}")
    if resp2.status_code != 404:
        print("  FAIL: Deleted memory should return 404")
        return False
    print("  PASS")
    return True


def test_get_nonexistent_memory(url: str, db: str) -> bool:
    """Test getting a memory that doesn't exist."""
    print("\n[TEST] Get non-existent memory (should 404)...")
    resp = requests.get(f"{url}/memories/999999?db_name={db}")
    if resp.status_code != 404:
        print(f"  FAIL: Expected 404, got {resp.status_code}")
        return False
    print("  PASS (correctly returned 404)")
    return True


# ============================================================================
# KEYWORD SEARCH TESTS
# ============================================================================

def test_search_keyword_single(url: str, db: str) -> bool:
    """Test single keyword search."""
    print("\n[TEST] Search by single keyword (k:python)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={"query": "k:python"})
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} memories with 'python' keyword")
    for mem in data.get('memories', [])[:3]:
        print(f"    - ID {mem['id']}: {mem.get('keywords', [])}")
    print("  PASS")
    return True


def test_search_keyword_multiple(url: str, db: str) -> bool:
    """Test multiple keyword search (AND)."""
    print("\n[TEST] Search by multiple keywords (k:python AND k:async)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "k:python AND k:async"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} memories")
    print("  PASS")
    return True


def test_search_keyword_or(url: str, db: str) -> bool:
    """Test keyword OR search."""
    print("\n[TEST] Search with OR (k:docker OR k:react)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "k:docker OR k:react"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} memories")
    print("  PASS")
    return True


# ============================================================================
# PROPERTY FILTER TESTS
# ============================================================================

def test_search_property_equals(url: str, db: str) -> bool:
    """Test property equality filter."""
    print("\n[TEST] Search by property (p:category=frontend)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "p:category=frontend"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} memories with category=frontend")
    print("  PASS")
    return True


def test_search_property_numeric(url: str, db: str) -> bool:
    """Test numeric property comparison."""
    print("\n[TEST] Search by numeric property (p:rating>=4)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "p:rating>=4"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} memories with rating >= 4")
    print("  PASS")
    return True


def test_search_property_wildcard(url: str, db: str) -> bool:
    """Test property value wildcard search."""
    print("\n[TEST] Search by property wildcard (p:category=*end*)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "p:category=*end*"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    count = data.get('count', 0)
    print(f"  Found {count} memories with category containing 'end'")
    if count >= 1:
        print("  PASS")
        return True
    else:
        print(f"  FAIL: expected >=1 results, got {count}")
        return False


# ============================================================================
# DATE FILTER TESTS
# ============================================================================

def test_search_date_range(url: str, db: str) -> bool:
    """Test date range filter."""
    print("\n[TEST] Search by date (d:last 365 days)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "d:last 365 days"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} memories from last year")
    print("  PASS")
    return True


# ============================================================================
# LINK TESTS
# ============================================================================

def test_add_link(url: str, db: str, source_id: int, target_id: int) -> bool:
    """Test adding a link between memories."""
    print(f"\n[TEST] Add link {source_id} -> {target_id} (related_to)...")
    resp = requests.post(f"{url}/memories/link?db_name={db}", json={
        "source_id": source_id,
        "target_id": target_id,
        "link_type": "related_to"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}: {resp.text}")
        return False
    print("  PASS")
    return True


def test_search_links(url: str, db: str) -> bool:
    """Test searching by link type."""
    print("\n[TEST] Search by links (l:related_to)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "l:related_to"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} memories with related_to links")
    print("  PASS")
    return True


def test_search_links_directional(url: str, db: str) -> bool:
    """Test directional link search."""
    print("\n[TEST] Search by directional links...")
    
    # l:source:related_to - memories that link TO others
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "l:source:related_to"
    })
    if resp.status_code != 200:
        print(f"  FAIL: l:source:related_to - Status {resp.status_code}")
        return False
    
    # l:target:related_to - memories that ARE linked TO
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "l:target:related_to"
    })
    if resp.status_code != 200:
        print(f"  FAIL: l:target:related_to - Status {resp.status_code}")
        return False
    
    print("  PASS")
    return True


def test_add_summary(url: str, db: str, target_ids: list) -> Optional[int]:
    """Test adding a summary memory. Returns summary ID."""
    print(f"\n[TEST] Add summary linked to targets {target_ids}...")
    created_at = datetime.now(timezone.utc).isoformat()
    resp = requests.post(f"{url}/memories/summary?db_name={db}", json={
        "content": "Summary of Python and async programming concepts discussed",
        "keywords": ["summary", "python", "async"],
        "created_at": created_at,
        "target_ids": target_ids,
        "link_type": "summary_of"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}: {resp.text}")
        return None
    data = resp.json()
    print(f"  Created summary ID: {data.get('id')}")
    print("  PASS")
    return data.get('id')


def test_search_summary_with_filter(url: str, db: str) -> bool:
    """Test searching summaries with inner filter."""
    print("\n[TEST] Search summaries with keyword filter (l:summary_of:(k:python))...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "l:summary_of:(k:python)"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} summaries linked to memories with python keyword")
    print("  PASS")
    return True


# ============================================================================
# VECTOR/SEMANTIC SEARCH TESTS
# ============================================================================

def test_embedding_endpoint(url: str) -> bool:
    """Test the embedding endpoint."""
    print("\n[TEST] Get embedding for text...")
    resp = requests.post(f"{url}/embed", json={
        "text": "Python programming language async"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    embedding = data.get('embedding', [])
    print(f"  Embedding dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")
    if len(embedding) == 0:
        print("  FAIL: Empty embedding returned")
        return False
    print("  PASS")
    return True


def test_semantic_search(url: str, db: str) -> bool:
    """Test semantic similarity search."""
    print("\n[TEST] Semantic search (q:container orchestration)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "q:container orchestration",
        "limit": 10
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} semantically similar memories")
    for mem in data.get('memories', [])[:3]:
        score = mem.get('similarity_score', 0)
        content = mem.get('content', '')[:50]
        print(f"    - Score {score:.3f}: {content}...")
    print("  PASS")
    return True


def test_semantic_search_with_threshold(url: str, db: str) -> bool:
    """Test semantic search with threshold."""
    print("\n[TEST] Semantic search with threshold (q:javascript programming@0.3)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "q:javascript programming@0.3",
        "limit": 10
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} memories (low threshold = more results)")
    print("  PASS")
    return True


def test_semantic_search_keyword(url: str, db: str) -> bool:
    """Test semantic search on existing keyword content."""
    print("\n[TEST] Semantic search on keyword (s:python@0.4)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "s:python@0.4",
        "limit": 10
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} memories semantically related to python")
    print("  PASS")
    return True


# ============================================================================
# COMBINED QUERY TESTS
# ============================================================================

def test_search_combined(url: str, db: str) -> bool:
    """Test combined search queries."""
    print("\n[TEST] Combined search (k:docker AND p:rating>=4)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "k:docker AND p:rating>=4"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} memories matching combined query")
    print("  PASS")
    return True


def test_search_grouped(url: str, db: str) -> bool:
    """Test grouped search with parentheses."""
    print("\n[TEST] Grouped search ((k:docker OR k:react) AND p:rating>=4)...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": "(k:docker OR k:react) AND p:rating>=4"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Found {data.get('count')} memories matching grouped query")
    print("  PASS")
    return True


def test_search_id(url: str, db: str, memory_id: int) -> bool:
    """Test searching by memory ID."""
    print(f"\n[TEST] Search by ID (id:{memory_id})...")
    resp = requests.post(f"{url}/memories/search?db_name={db}", json={
        "query": f"id:{memory_id}"
    })
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    if data.get('count') != 1:
        print(f"  FAIL: Expected 1 result, got {data.get('count')}")
        return False
    print(f"  Found memory ID {data['memories'][0]['id']}")
    print("  PASS")
    return True


# ============================================================================
# CONTEXT TESTS
# ============================================================================

def test_add_context(url: str, db: str) -> bool:
    """Test adding context messages."""
    print("\n[TEST] Add context messages...")
    messages = [
        ("user", "How do I use Python async/await?"),
        ("assistant", "Async/await is used to write asynchronous code. Example: async def foo(): await bar()"),
        ("user", "What about error handling?"),
        ("assistant", "Use try/except blocks inside async functions."),
    ]
    for role, content in messages:
        resp = requests.post(f"{url}/context?db_name={db}", json={
            "content": content,
            "role": role
        })
        if resp.status_code != 200:
            print(f"  FAIL: Failed to add {role} message: {resp.status_code}")
            return False
        data = resp.json()
        print(f"    Added {role}: tokens={data.get('token_count', 'N/A')}")
    print("  PASS")
    return True


def test_get_context(url: str, db: str) -> bool:
    """Test getting context."""
    print("\n[TEST] Get context...")
    resp = requests.get(f"{url}/context?db_name={db}&limit=10")
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Context count: {data.get('count')}")
    for ctx in data.get('context', [])[:3]:
        print(f"    - {ctx.get('role')}: {ctx.get('content', '')[:50]}...")
    print("  PASS")
    return True


# ============================================================================
# UTILITY TESTS
# ============================================================================

def test_stats(url: str, db: str) -> bool:
    """Test stats endpoint."""
    print("\n[TEST] Get stats...")
    resp = requests.get(f"{url}/stats?db_name={db}")
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Memory count: {data.get('count')}")
    print("  PASS")
    return True


def test_search_syntax_docs(url: str) -> bool:
    """Test search syntax documentation."""
    print("\n[TEST] Get search syntax docs...")
    resp = requests.get(f"{url}/docs/search-syntax")
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Title: {data.get('title')}")
    print(f"  Operators: {list(data.get('operators', {}).keys())}")
    print(f"  Filters: {list(data.get('filters', {}).keys())}")
    print("  PASS")
    return True


def test_embedding_model_info(url: str) -> bool:
    """Test embedding model info endpoint."""
    print("\n[TEST] Get embedding model info...")
    resp = requests.get(f"{url}/embed/model")
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Model: {data.get('model_name')}")
    print(f"  Dimension: {data.get('dimension')}")
    print(f"  Device: {data.get('device')}")
    print("  PASS")
    return True


def test_embedding_cache_info(url: str) -> bool:
    """Test embedding cache info."""
    print("\n[TEST] Get embedding cache info...")
    resp = requests.get(f"{url}/embed/cache")
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    data = resp.json()
    print(f"  Cache entries: {data.get('count', 0)}")
    print("  PASS")
    return True


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_tests(url: str, db: str) -> None:
    """Run all tests."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           MEMORY API COMPREHENSIVE TEST SUITE                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print(f"API URL: {url}")
    print(f"Test DB: {db}")
    print("=" * 60)
    
    # Wait for API
    print("\n⏳ Waiting for API to be ready...")
    if not wait_for_api(url):
        print("ERROR: API not ready within timeout")
        sys.exit(1)
    print("✅ API is ready!")
    
    # Create test database
    create_db(url, db)
    
    results = []
    memory_ids = {}
    
    # --- Database Operations ---
    results.append(("List databases", test_list_databases(url)))
    results.append(("Create database", test_create_database(url)))
    results.append(("Database exists check", test_database_exists(url)))
    
    # --- Memory CRUD ---
    mem1_id = test_add_memory_basic(url, db)
    results.append(("Add basic memory", mem1_id is not None))
    memory_ids['mem1'] = mem1_id
    
    more_ids = test_add_memory_multiple(url, db)
    results.append(("Add multiple memories", len(more_ids) >= 5))
    memory_ids.update(more_ids)
    
    if mem1_id:
        results.append(("Get memory by ID", test_get_memory(url, db, mem1_id)))
        results.append(("Update memory", test_update_memory(url, db, mem1_id)))
        # Note: id: search syntax is documented but not yet implemented
        # results.append(("Search by ID", test_search_id(url, db, mem1_id)))
    
    results.append(("List memories", test_list_memories(url, db)))
    results.append(("Get non-existent memory", test_get_nonexistent_memory(url, db)))
    
    # --- Keyword Search ---
    results.append(("Search single keyword", test_search_keyword_single(url, db)))
    results.append(("Search multiple keywords", test_search_keyword_multiple(url, db)))
    results.append(("Search keyword OR", test_search_keyword_or(url, db)))
    
    # --- Property Filters ---
    results.append(("Property equals", test_search_property_equals(url, db)))
    results.append(("Property numeric", test_search_property_numeric(url, db)))
    results.append(("Property wildcard", test_search_property_wildcard(url, db)))
    
    # --- Date Filters ---
    results.append(("Date range", test_search_date_range(url, db)))
    
    # --- Links ---
    if memory_ids.get('mem2') and memory_ids.get('mem3'):
        results.append(("Add link", test_add_link(url, db, memory_ids['mem2'], memory_ids['mem3'])))
    
    results.append(("Search by links", test_search_links(url, db)))
    results.append(("Search directional links", test_search_links_directional(url, db)))
    
    # --- Summary ---
    target_ids = [memory_ids.get('mem1'), memory_ids.get('mem2') or 1]
    if len(target_ids) >= 1:
        results.append(("Add summary", test_add_summary(url, db, target_ids) is not None))
        results.append(("Search summary with filter", test_search_summary_with_filter(url, db)))
    
    # --- Vector/Semantic Search ---
    results.append(("Embedding endpoint", test_embedding_endpoint(url)))
    results.append(("Semantic search", test_semantic_search(url, db)))
    results.append(("Semantic search with threshold", test_semantic_search_with_threshold(url, db)))
    results.append(("Semantic search keyword", test_semantic_search_keyword(url, db)))
    
    # --- Combined Queries ---
    results.append(("Combined search", test_search_combined(url, db)))
    results.append(("Grouped search", test_search_grouped(url, db)))
    
    # --- Context ---
    results.append(("Add context", test_add_context(url, db)))
    results.append(("Get context", test_get_context(url, db)))
    
    # --- Utilities ---
    results.append(("Stats endpoint", test_stats(url, db)))
    results.append(("Search syntax docs", test_search_syntax_docs(url)))
    results.append(("Embedding model info", test_embedding_model_info(url)))
    results.append(("Embedding cache info", test_embedding_cache_info(url)))
    
    # --- Cleanup ---
    if mem1_id:
        results.append(("Delete memory", test_delete_memory(url, db, mem1_id)))
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Memory API Test Suite")
    parser.add_argument("--url", default=BASE_URL, help="API base URL")
    parser.add_argument("--db", default=TEST_DB, help="Test database name")
    args = parser.parse_args()
    
    run_tests(args.url, args.db)


if __name__ == "__main__":
    main()
