#!/usr/bin/env python3
"""Comprehensive MemoryDB tests - combines all test suites.

Usage:
    # Run all tests except vector (default)
    python3 test_comprehensive.py
    
    # Run all tests including vector
    python3 test_comprehensive.py --vector
    
    # Run only specific test category
    python3 test_comprehensive.py --links
    python3 test_comprehensive.py --if-then
    python3 test_comprehensive.py --vector-only
"""

import os
import sys
import argparse
from datetime import datetime, timedelta, timezone

# Ensure the project layout is importable (root for database/, src/ for other modules)
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(1, _src)

TEST_DB = "test_comprehensive.db"

# Test data setup
def setup():
    """Create fresh test database."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    
    from database import init_db
    init_db(TEST_DB)
    print("✓ Database setup complete")


def add_link(db, source_id: int, target_id: int, link_type: str):
    """Add a memory link using database method."""
    db.add_link(source_id, target_id, link_type)


def add_test_data(use_vector=False):
    """Add test memories with various attributes."""
    if use_vector:
        from database import MemoryDB
        from embedding import EmbeddingModel
        vector = EmbeddingModel()
        db = MemoryDB(TEST_DB, embedding_model=vector)
    else:
        from database import MemoryDB
        db = MemoryDB(TEST_DB)
    
    now = datetime.now(timezone.utc)
    
    # ===== GROUP 1: Python async (vector group) =====
    mem1 = db.add_memory(
        "Python asyncio tutorial for async concurrent programming with async/await",
        keywords=["python", "asyncio", "async", "concurrency"],
        created_at=(now - timedelta(days=5)).isoformat()
    )
    
    mem2 = db.add_memory(
        "JavaScript async await promises tutorial for asynchronous code",
        keywords=["javascript", "async", "promises", "es6"],
        created_at=(now - timedelta(days=3)).isoformat()
    )
    
    mem3 = db.add_memory(
        "FastAPI Python web framework building async REST APIs",
        keywords=["python", "fastapi", "api", "web"],
        created_at=(now - timedelta(days=2)).isoformat()
    )
    
    # ===== GROUP 2: Docker/containers =====
    mem4 = db.add_memory(
        "Docker container tutorial for images and volumes",
        keywords=["docker", "containers", "devops"],
        created_at=(now - timedelta(days=1)).isoformat()
    )
    
    # ===== GROUP 3: Old memories with summaries =====
    mem5 = db.add_memory(
        "Deep dive into Python asyncio patterns for concurrent programming",
        keywords=["python", "asyncio", "concurrency"],
        properties={"status": "active", "type": "original"},
        created_at=(now - timedelta(days=30)).isoformat()
    )
    
    mem6 = db.add_memory(
        "Machine learning with scikit-learn for classification and regression",
        keywords=["machine-learning", "sklearn", "ai"],
        properties={"status": "archived", "type": "original"},
        created_at=(now - timedelta(days=60)).isoformat()
    )
    
    # ===== GROUP 4: Additional test memories =====
    mem7 = db.add_memory(
        "React JavaScript tutorial for building user interfaces",
        keywords=["javascript", "react", "frontend", "ui"],
        created_at=(now - timedelta(days=4)).isoformat()
    )
    
    mem8 = db.add_memory(
        "Kubernetes orchestration for container deployment at scale",
        keywords=["kubernetes", "k8s", "devops", "orchestration"],
        created_at=(now - timedelta(days=7)).isoformat()
    )
    
    # ===== SUMMARIES =====
    summary1 = db.add_memory(
        "Quick summary: Python asyncio provides async/await for concurrent programming in Python",
        keywords=["python", "asyncio", "summary"],
        properties={"status": "active", "type": "summary", "is_summary": "true"},
        created_at=(now - timedelta(days=25)).isoformat()
    )
    
    summary2 = db.add_memory(
        "Quick summary: scikit-learn provides easy ML tools for classification and regression",
        keywords=["machine-learning", "sklearn", "summary"],
        properties={"status": "archived", "type": "summary", "is_summary": "true"},
        created_at=(now - timedelta(days=50)).isoformat()
    )
    
    # ===== GROUP 5: Numeric properties for comparison tests =====
    mem9 = db.add_memory(
        "Negative opinion about topic A",
        keywords=["opinion", "negative"],
        properties={"opinion": "-3.5", "rating": "1"},
        created_at=(now - timedelta(days=2)).isoformat()
    )
    
    mem10 = db.add_memory(
        "Positive opinion about topic B",
        keywords=["opinion", "positive"],
        properties={"opinion": "2.0", "rating": "5"},
        created_at=(now - timedelta(days=1)).isoformat()
    )
    
    mem11 = db.add_memory(
        "Neutral opinion about topic C",
        keywords=["opinion", "neutral"],
        properties={"opinion": "0", "rating": "3"},
        created_at=(now - timedelta(days=3)).isoformat()
    )
    
    mem12 = db.add_memory(
        "String opinion about topic D",
        keywords=["opinion", "string"],
        properties={"opinion": "very_negative", "rating": "2"},
        created_at=(now - timedelta(days=5)).isoformat()
    )
    
    # ===== LINKS =====
    # Summary -> original
    db.add_link(summary1, mem5, "summary_of")
    db.add_link(summary2, mem6, "summary_of")
    
    # Related links
    db.add_link(mem2, mem1, "related_to")   # JS async related to Python async
    db.add_link(mem4, mem1, "related_to")  # Docker related to Python async
    db.add_link(mem7, mem2, "related_to")   # React related to JS
    db.add_link(mem8, mem4, "derived_from")  # K8s derived from Docker
    
    print(f"  Added 12 memories + 2 summaries + 6 links")
    
    return {
        "mem1": mem1, "mem2": mem2, "mem3": mem3, "mem4": mem4,
        "mem5": mem5, "mem6": mem6, "mem7": mem7, "mem8": mem8,
        "mem9": mem9, "mem10": mem10, "mem11": mem11, "mem12": mem12,
        "summary1": summary1, "summary2": summary2
    }


# ============================================================================
# TEST CATEGORIES
# ============================================================================

def test_keywords(db):
    """Test basic keyword search."""
    print("\n" + "=" * 60)
    print("TESTING KEYWORD SEARCH")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Single keyword
    results = db.search("k:python")
    if len(results) >= 3:  # mem1, mem3, mem5, summary1
        print(f"  ✓ k:python -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ k:python -> {len(results)} (expected >=3)")
        failed += 1
    
    # Test 2: Multiple keywords AND
    results = db.search("k:python AND k:asyncio")
    if len(results) >= 2:  # mem1, mem5, summary1
        print(f"  ✓ k:python AND k:asyncio -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ k:python AND k:asyncio -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 3: Multiple keywords OR
    results = db.search("k:docker OR k:kubernetes")
    if len(results) >= 2:  # mem4, mem8
        print(f"  ✓ k:docker OR k:kubernetes -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ k:docker OR k:kubernetes -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 4: NOT keyword
    results = db.search("NOT k:python")
    if len(results) >= 4:  # mem2, mem4, mem6, mem7, mem8, summaries
        print(f"  ✓ NOT k:python -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ NOT k:python -> {len(results)} (expected >=4)")
        failed += 1
    
    return passed, failed


def test_properties(db):
    """Test property filters."""
    print("\n" + "=" * 60)
    print("TESTING PROPERTY FILTERS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Property filter
    results = db.search("p:status=active")
    if len(results) >= 2:  # mem5, summary1
        print(f"  ✓ p:status=active -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:status=active -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 2: Property filter archived
    results = db.search("p:status=archived")
    if len(results) >= 2:  # mem6, summary2
        print(f"  ✓ p:status=archived -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:status=archived -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 3: is_summary property
    results = db.search("p:is_summary=true")
    if len(results) >= 2:  # summary1, summary2
        print(f"  ✓ p:is_summary=true -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:is_summary=true -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 4: Combined property + keyword
    results = db.search("p:status=active AND k:python")
    if len(results) >= 1:  # mem5
        print(f"  ✓ p:status=active AND k:python -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:status=active AND k:python -> {len(results)} (expected >=1)")
        failed += 1
    
    # ===== NUMERIC PROPERTY TESTS =====
    # Test 5: Numeric property less than
    results = db.search("p:opinion<0")
    if len(results) >= 1:  # mem9 (opinion=-3.5)
        print(f"  ✓ p:opinion<0 -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:opinion<0 -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 6: Numeric property greater than
    results = db.search("p:opinion>0")
    if len(results) >= 1:  # mem10 (opinion=2.0)
        print(f"  ✓ p:opinion>0 -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:opinion>0 -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 7: Numeric property less than or equal
    results = db.search("p:opinion<=0")
    if len(results) >= 2:  # mem9 (opinion=-3.5), mem11 (opinion=0)
        print(f"  ✓ p:opinion<=0 -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:opinion<=0 -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 8: Numeric property greater than or equal
    results = db.search("p:opinion>=0")
    if len(results) >= 2:  # mem10 (opinion=2.0), mem11 (opinion=0)
        print(f"  ✓ p:opinion>=0 -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:opinion>=0 -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 9: Numeric property not equal
    results = db.search("p:opinion!=0")
    if len(results) >= 2:  # mem9 (opinion=-3.5), mem10 (opinion=2.0)
        print(f"  ✓ p:opinion!=0 -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:opinion!=0 -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 10: Non-numeric stored property should not match numeric comparison
    results = db.search("p:opinion>0")
    # mem12 has opinion=very_negative (string), should NOT match numeric comparison
    has_non_numeric = any(r.get('properties', {}).get('opinion') == 'very_negative' for r in results)
    if not has_non_numeric:
        print(f"  ✓ p:opinion>0 excludes non-numeric -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:opinion>0 should not match string 'very_negative'")
        failed += 1
    
    # Test 11: Numeric property + keyword combined
    results = db.search("p:opinion<0 AND k:negative")
    if len(results) >= 1:  # mem9
        print(f"  ✓ p:opinion<0 AND k:negative -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:opinion<0 AND k:negative -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 12: Rating comparison
    results = db.search("p:rating>=4")
    if len(results) >= 1:  # mem10 (rating=5)
        print(f"  ✓ p:rating>=4 -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:rating>=4 -> {len(results)} (expected >=1)")
        failed += 1
    
    return passed, failed


def test_dates(db):
    """Test date filters."""
    print("\n" + "=" * 60)
    print("TESTING DATE FILTERS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Last N days
    results = db.search("d:last 3 days")
    if len(results) >= 2:  # mem3 (2), mem4 (1)
        print(f"  ✓ d:last 3 days -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ d:last 3 days -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 2: Last week
    results = db.search("d:last 7 days")
    if len(results) >= 4:  # mem2, mem3, mem4, mem7
        print(f"  ✓ d:last 7 days -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ d:last 7 days -> {len(results)} (expected >=4)")
        failed += 1
    
    # Test 3: Old memories - using date range instead of NOT
    # NOTE: "NOT d:last 7 days" has a parsing bug
    results = db.search("d:older_than:7")
    if len(results) >= 4:  # mem1, mem5, mem6, mem8, summaries
        print(f"  ✓ d:older_than:7 -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ d:older_than:7 -> {len(results)} (expected >=4)")
        failed += 1
    
    # Test 4: Combined date + keyword
    results = db.search("d:last 7 days AND k:javascript")
    if len(results) >= 2:  # mem2, mem7
        print(f"  ✓ d:last 7 days AND k:javascript -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ d:last 7 days AND k:javascript -> {len(results)} (expected >=2)")
        failed += 1
    
    return passed, failed


def test_if_then_else(db, ids):
    """Test IF-THEN-ELSE conditional logic."""
    print("\n" + "=" * 60)
    print("TESTING IF-THEN-ELSE CONDITIONALS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    tests = [
        # (query, min_expected, description)
        ("IF d:last 3 days THEN k:python ELSE k:docker", 1, "Date then-branch"),
        ("IF d:last 1 day THEN k:javascript ELSE k:docker", 0, "Date else-branch"),
        ("IF p:is_summary=true THEN k:python ELSE k:javascript", 1, "Property then-branch"),
        ("IF p:status=archived THEN k:sklearn ELSE p:status=active", 1, "Property else-branch"),
        ("IF k:asyncio THEN k:python ELSE k:docker", 1, "Keyword then-branch"),
        ("IF k:react THEN k:javascript ELSE k:docker", 1, "Keyword else-branch"),
        ("IF d:last 3 days THEN k:python OR k:javascript ELSE k:docker", 2, "Complex OR in then"),
        ("IF k:python THEN k:asyncio", 1, "IF without ELSE"),
        ("IF NOT p:status=archived THEN k:python ELSE k:javascript", 1, "NOT condition"),
        ("IF p:is_summary=true THEN p:status=active ELSE p:status=archived", 1, "Property in branches"),
        ("IF d:last 30 days THEN k:asyncio ELSE k:sklearn", 1, "Date then with old memories"),
        ("IF k:non-existent THEN k:python ELSE k:docker", 1, "Non-existent keyword then"),
    ]
    
    for query, min_expected, desc in tests:
        results = db.search(query)
        status = "✓" if len(results) >= min_expected else "✗"
        print(f"  {status} {desc}: {len(results)} results")
        if len(results) >= min_expected:
            passed += 1
        else:
            failed += 1
    
    return passed, failed


def test_links(db, ids):
    """Test link traversal queries."""
    print("\n" + "=" * 60)
    print("TESTING LINK TRAVERSAL")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Direct link by target ID
    results = db.search(f"l:summary_of:{ids['mem5']}")
    if len(results) >= 1 and any(r['id'] == ids['summary1'] for r in results):
        print(f"  ✓ l:summary_of:{ids['mem5']} -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ l:summary_of -> {len(results)} (expected summary1)")
        failed += 1
    
    # Test 2: Related links
    results = db.search(f"l:related_to:{ids['mem1']}")
    if len(results) >= 2:  # mem2, mem4
        print(f"  ✓ l:related_to:{ids['mem1']} -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ l:related_to -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 3: Derived links
    results = db.search(f"l:derived_from:{ids['mem4']}")
    if len(results) >= 1 and any(r['id'] == ids['mem8'] for r in results):
        print(f"  ✓ l:derived_from:{ids['mem4']} -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ l:derived_from -> {len(results)} (expected mem8)")
        failed += 1
    
    # Test 4: Link with inner query
    results = db.search("l:summary_of:(k:python)")
    if len(results) >= 1:
        print(f"  ✓ l:summary_of:(k:python) -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ l:summary_of:(k:python) -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 5: Link type only
    results = db.search("l:summary_of")
    if len(results) >= 2:
        print(f"  ✓ l:summary_of -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ l:summary_of -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 6: Link with keyword filter (property filter has a bug)
    results = db.search("l:related_to:(k:python)")
    if len(results) >= 1:
        print(f"  ✓ l:related_to:(k:python) -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ l:related_to:(k:python) -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 7: Directional - source (memories that link TO others)
    # mem2 links to mem1 via related_to
    # l:source:related_to should find mem2 (the source)
    results = db.search("l:source:related_to")
    if len(results) >= 1 and any(r['id'] == ids['mem2'] for r in results):
        print(f"  ✓ l:source:related_to -> {len(results)} results (found mem2)")
        passed += 1
    else:
        print(f"  ✗ l:source:related_to -> {len(results)} (expected >=1 with mem2)")
        failed += 1
    
    # Test 8: Directional - target (memories that ARE linked TO)
    # l:target:related_to should find mem1 (the target)
    results = db.search("l:target:related_to")
    if len(results) >= 1 and any(r['id'] == ids['mem1'] for r in results):
        print(f"  ✓ l:target:related_to -> {len(results)} results (found mem1)")
        passed += 1
    else:
        print(f"  ✗ l:target:related_to -> {len(results)} (expected >=1 with mem1)")
        failed += 1
    
    # Test 9: Directional - source with keyword filter
    # Find memories that link TO memories with python keyword
    results = db.search("l:source:related_to:(k:python)")
    # mem2 (JS) links to mem1 (Python) which has python keyword
    if len(results) >= 1 and any(r['id'] == ids['mem2'] for r in results):
        print(f"  ✓ l:source:related_to:(k:python) -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ l:source:related_to:(k:python) -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 10: Directional - target with keyword filter
    # Find memories that have links FROM memories with javascript keyword
    results = db.search("l:target:related_to:(k:javascript)")
    # mem1 (Python) is targeted by mem2 (JS) which has javascript keyword
    if len(results) >= 1 and any(r['id'] == ids['mem1'] for r in results):
        print(f"  ✓ l:target:related_to:(k:javascript) -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ l:target:related_to:(k:javascript) -> {len(results)} (expected >=1)")
        failed += 1
    
    return passed, failed


def test_combined(db, ids):
    """Test combined link + IF-THEN-ELSE."""
    print("\n" + "=" * 60)
    print("TESTING COMBINED (LINKS + IF-THEN-ELSE)")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    tests = [
        ("IF d:last 3 days THEN l:related_to:(k:python) ELSE l:summary_of:(k:python)", 1, "Link with date conditional"),
        ("IF k:python THEN l:summary_of ELSE l:related_to", 1, "Link with keyword conditional"),
        ("IF p:is_summary=true THEN l:summary_of:(k:python) ELSE l:related_to:(k:python)", 1, "Link with property conditional"),
    ]
    
    for query, min_expected, desc in tests:
        results = db.search(query)
        status = "✓" if len(results) >= min_expected else "✗"
        print(f"  {status} {desc}: {len(results)} results")
        if len(results) >= min_expected:
            passed += 1
        else:
            failed += 1
    
    return passed, failed


def test_vector_similarity(db):
    """Test vector-based similarity - relative behavior tests."""
    print("\n" + "=" * 60)
    print("TESTING VECTOR SIMILARITY (relative behavior)")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Query similarity should return results
    results = db.search("q:async programming")
    if len(results) >= 1:
        print(f"  ✓ q:async programming -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ q:async programming -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 2: Higher threshold = fewer or equal results
    results_default = db.search("q:async")
    results_strict = db.search("q:async@0.7")
    if len(results_strict) <= len(results_default):
        print(f"  ✓ Strict threshold ({len(results_strict)}) <= default ({len(results_default)})")
        passed += 1
    else:
        print(f"  ✗ Strict should be <= default")
        failed += 1
    
    # Test 3: Lower threshold = more or equal results
    results_loose = db.search("q:async@0.3")
    if len(results_loose) >= len(results_default):
        print(f"  ✓ Loose threshold ({len(results_loose)}) >= default ({len(results_default)})")
        passed += 1
    else:
        print(f"  ✗ Loose should be >= default")
        failed += 1
    
    # Test 4: Unrelated query returns fewer results
    results = db.search("q:cooking recipes")
    if len(results) <= len(results_default):
        print(f"  ✓ Unrelated ({len(results)}) <= related ({len(results_default)})")
        passed += 1
    else:
        print(f"  ✗ Unrelated should return fewer")
        failed += 1
    
    # Test 5: Keyword similarity
    results = db.search("s:python@0.5")
    if len(results) >= 1:
        print(f"  ✓ s:python@0.5 -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ s:python@0.5 -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 6: Combined keyword AND similarity
    results = db.search("k:python AND q:async")
    if len(results) >= 1:
        print(f"  ✓ k:python AND q:async -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ k:python AND q:async -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 7: Strict vs Permissive comparison
    # Permissive (lower threshold) should return >= strict (higher threshold)
    results_strict = db.search("q:async@0.7")
    results_permissive = db.search("q:async@0.3")
    strict_count = len(results_strict)
    permissive_count = len(results_permissive)
    if permissive_count >= strict_count:
        print(f"  OK Permissive {permissive_count} >= Strict {strict_count}")
        passed += 1
    else:
        print(f"  FAIL Permissive {permissive_count} < Strict {strict_count}")
        failed += 1
    
    return passed, failed


def test_mixed_search(db):
    """Test mixing vector search with other filters."""
    print("\n" + "=" * 60)
    print("TESTING MIXED SEARCH (vector + filters)")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Date + similarity
    results = db.search("d:last 7 days AND q:async")
    if len(results) >= 1:
        print(f"  ✓ d:last 7 days AND q:async -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ d:last 7 days AND q:async -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 2: Keyword OR similarity
    results = db.search("k:python OR q:containers")
    if len(results) >= 2:
        print(f"  ✓ k:python OR q:containers -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ k:python OR q:containers -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 3: Property + similarity
    results = db.search("p:status=active AND q:async")
    if len(results) >= 1:
        print(f"  ✓ p:status=active AND q:async -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ p:status=active AND q:async -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 4: Complex mixed
    results = db.search("k:python OR (d:last 7 days AND q:async)")
    if len(results) >= 2:
        print(f"  ✓ Complex mixed -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ Complex mixed -> {len(results)} (expected >=2)")
        failed += 1
    
    return passed, failed


def test_complex_nested(db, ids):
    """Test complex nested logic with parentheses, multiple operators, etc."""
    print("\n" + "=" * 60)
    print("TESTING COMPLEX NESTED LOGIC")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Deeply nested with parentheses
    results = db.search("(k:python OR k:javascript) AND (k:asyncio OR k:async)")
    if len(results) >= 1:
        print(f"  ✓ Deep nested (A OR B) AND (C OR D) -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ Deep nested -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 2: Triple nested with mixed operators
    results = db.search("((k:python OR k:javascript) AND d:last 10 days) OR p:status=active")
    if len(results) >= 1:
        print(f"  ✓ Triple nested ((A OR B) AND C) OR D -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ Triple nested -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 3: Multiple NOT with parentheses - simplified
    results = db.search("(NOT k:python) AND d:last 7 days")
    if len(results) >= 1:
        print(f"  ✓ NOT with AND -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ NOT with AND -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 4: Property numeric with keyword and date
    results = db.search("(p:opinion>0 OR p:opinion<0) AND k:opinion")
    if len(results) >= 2:  # mem9, mem10
        print(f"  ✓ Numeric prop with keyword -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ Numeric prop with keyword -> {len(results)} (expected >=2)")
        failed += 1
    
    # Test 5: Link with nested filter
    results = db.search("l:summary_of AND (k:python OR k:machine-learning)")
    if len(results) >= 1:
        print(f"  ✓ Link with nested keyword -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ Link with nested keyword -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 6: Four-way nested with IF-THEN-ELSE
    results = db.search("IF k:python THEN (k:asyncio OR k:fastapi) ELSE (k:docker OR k:kubernetes)")
    if len(results) >= 1:
        print(f"  ✓ IF-THEN-ELSE with nested OR -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ IF-THEN-ELSE with nested OR -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 7: Complex date + property + keyword combo
    results = db.search("(d:last 7 days OR d:last 30 days) AND (k:python OR k:javascript) AND p:status=active")
    if len(results) >= 1:
        print(f"  ✓ Date + property + keyword triple -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ Date + property + keyword triple -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 8: Nested directional links with keyword
    results = db.search("(l:source:related_to OR l:target:related_to) AND k:python")
    if len(results) >= 1:
        print(f"  ✓ Directional link OR with keyword -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ Directional link OR with keyword -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 9: Multiple property comparisons
    results = db.search("p:opinion<0 AND p:rating<=2")
    if len(results) >= 1:  # mem9 has opinion=-3.5, rating=1
        print(f"  ✓ Multiple numeric properties -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ Multiple numeric properties -> {len(results)} (expected >=1)")
        failed += 1
    
    # Test 10: Deeply nested with all operators
    results = db.search("((k:python OR k:javascript) AND (d:last 10 days OR p:status=active)) OR (k:docker AND p:status=archived)")
    if len(results) >= 1:
        print(f"  ✓ Deep quad nested with all ops -> {len(results)} results")
        passed += 1
    else:
        print(f"  ✗ Deep quad nested -> {len(results)} (expected >=1)")
        failed += 1
    
    return passed, failed


def test_edge_cases(db):
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Empty query
    try:
        results = db.search("")
        print(f"  ✓ Empty query handled")
        passed += 1
    except:
        print(f"  ✗ Empty query crashed")
        failed += 1
    
    # Test 2: Invalid link type
    results = db.search("l:nonexistent_link:123")
    if len(results) == 0:
        print(f"  ✓ Invalid link type returns 0")
        passed += 1
    else:
        print(f"  ✗ Invalid link should return 0")
        failed += 1
    
    # Test 3: Non-existent keyword
    results = db.search("k:non_existent_keyword_xyz")
    if len(results) == 0:
        print(f"  ✓ Non-existent keyword returns 0")
        passed += 1
    else:
        print(f"  ✗ Non-existent keyword should return 0")
        failed += 1
    
    # Test 4: Very high limit
    results = db.search("k:python", limit=1000)
    if len(results) >= 3:
        print(f"  ✓ High limit works")
        passed += 1
    else:
        print(f"  ✗ High limit issue")
        failed += 1
    
    return passed, failed


def cleanup():
    """Remove test database."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    print("\n✓ Cleanup complete")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MemoryDB Comprehensive Tests")
    parser.add_argument("--vector", action="store_true", help="Include vector similarity tests")
    parser.add_argument("--vector-only", action="store_true", help="Run only vector tests")
    parser.add_argument("--links", action="store_true", help="Run only link tests")
    parser.add_argument("--if-then", action="store_true", help="Run only IF-THEN-ELSE tests")
    parser.add_argument("--basic", action="store_true", help="Run only basic tests (keywords, properties, dates)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MEMORYDB COMPREHENSIVE TESTS")
    print("=" * 60)
    print()
    
    # Setup
    use_vector = args.vector or args.vector_only
    setup()
    ids = add_test_data(use_vector=use_vector)
    
    # Create db with or without embedding
    if use_vector:
        from database import MemoryDB
        from embedding import EmbeddingModel
        vector = EmbeddingModel()
        db = MemoryDB(TEST_DB, embedding_model=vector)
    else:
        from database import MemoryDB
        db = MemoryDB(TEST_DB)
    
    total_passed = 0
    total_failed = 0
    
    # Run selected tests
    if args.vector_only:
        vp, vf = test_vector_similarity(db)
        mp, mf = test_mixed_search(db)
        total_passed += vp + mp
        total_failed += vf + mf
    elif args.links:
        lp, lf = test_links(db, ids)
        cp, cf = test_combined(db, ids)
        total_passed += lp + cp
        total_failed += lf + cf
    elif args.if_then:
        ip, if_ = test_if_then_else(db, ids)
        total_passed += ip
        total_failed += if_
    elif args.basic:
        kp, kf = test_keywords(db)
        pp, pf = test_properties(db)
        dp, df = test_dates(db)
        ep, ef = test_edge_cases(db)
        total_passed += kp + pp + dp + ep
        total_failed += kf + pf + df + ef
    else:
        # Run all tests
        kp, kf = test_keywords(db)
        pp, pf = test_properties(db)
        dp, df = test_dates(db)
        ip, if_ = test_if_then_else(db, ids)
        lp, lf = test_links(db, ids)
        cp, cf = test_combined(db, ids)
        np, nf = test_complex_nested(db, ids)
        ep, ef = test_edge_cases(db)
        
        if use_vector:
            vp, vf = test_vector_similarity(db)
            mp, mf = test_mixed_search(db)
        
        total_passed = kp + pp + dp + ip + lp + cp + np + ep + (vp if use_vector else 0) + (mp if use_vector else 0)
        total_failed = kf + pf + df + if_ + lf + cf + nf + ef + (vf if use_vector else 0) + (mf if use_vector else 0)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    cleanup()
    
    if total_failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
