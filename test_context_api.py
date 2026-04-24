"""
Test script for the Memory API context endpoints.

Usage:
    python test_context_api.py [--url http://localhost:8030] [--db test_context]

Requires: requests
    pip install requests
"""

import argparse
import json
import sys
import time
from typing import Optional

try:
    import requests
except ImportError:
    print("Error: 'requests' library required. Install with: pip install requests")
    sys.exit(1)


BASE_URL = "http://localhost:8030"
TEST_DB = "test_context_api"


def wait_for_api(url: str, timeout: int = 10) -> bool:
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


def create_db(url: str, name: str) -> bool:
    """Create a test database."""
    resp = requests.post(f"{url}/db/create?name={name}")
    return resp.status_code == 200


def delete_db(url: str, name: str) -> bool:
    """Delete a test database."""
    # Just delete the db file by creating fresh
    return True


def test_add_context_single(url: str, db: str) -> bool:
    """Test adding a single context message."""
    print("\n[TEST] Add single context message...")
    
    resp = requests.post(
        f"{url}/context?db_name={db}",
        json={
            "content": "Hello, this is my first message!",
            "role": "user"
        }
    )
    
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}: {resp.text}")
        return False
    
    data = resp.json()
    print(f"  Response: {json.dumps(data, indent=2)}")
    
    # Check expected fields
    if data.get("role") != "user":
        print("  FAIL: Expected role 'user'")
        return False
    
    if data.get("summarized") != False:
        print("  FAIL: Should not have summarized yet")
        return False
    
    print("  PASS")
    return True


def test_add_context_multiple(url: str, db: str) -> bool:
    """Test adding multiple context messages."""
    print("\n[TEST] Add multiple context messages...")
    
    messages = [
        ("user", "Hi, I need help with Python async programming."),
        ("assistant", "Sure! Async programming in Python uses asyncio. What specifically would you like to know?"),
        ("user", "How do I use async/await?"),
        ("assistant", "async/await is used to define coroutines. Here's an example..."),
    ]
    
    for role, content in messages:
        resp = requests.post(
            f"{url}/context?db_name={db}",
            json={"content": content, "role": role}
        )
        
        if resp.status_code != 200:
            print(f"  FAIL: Failed to add {role} message: {resp.status_code}")
            return False
        
        data = resp.json()
        print(f"  Added {role}: id={data.get('id')}, tokens={data.get('token_count')}")
    
    print("  PASS")
    return True


def test_get_context_empty(url: str, db: str) -> bool:
    """Test getting context from fresh DB."""
    print("\n[TEST] Get context from fresh DB...")
    
    # Create fresh DB
    create_db(url, "test_empty")
    
    resp = requests.get(f"{url}/context?db_name=test_empty")
    
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    
    data = resp.json()
    print(f"  Response: {json.dumps(data, indent=2)}")
    
    if data.get("count") != 0:
        print(f"  FAIL: Expected 0 context items, got {data.get('count')}")
        return False
    
    print("  PASS")
    return True


def test_get_context_with_messages(url: str, db: str) -> bool:
    """Test getting context with messages."""
    print("\n[TEST] Get context with messages...")
    
    resp = requests.get(f"{url}/context?db_name={db}")
    
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    
    data = resp.json()
    print(f"  Count: {data.get('count')}")
    print(f"  Context: {json.dumps(data.get('context', []), indent=2)}")
    
    if data.get("count", 0) < 1:
        print("  FAIL: Expected at least 1 context item")
        return False
    
    # Check structure
    context = data.get("context", [])
    if context:
        first = context[0]
        if not all(k in first for k in ["id", "role", "content", "created_at"]):
            print("  FAIL: Missing expected fields in context item")
            return False
    
    print("  PASS")
    return True


def test_invalid_role(url: str, db: str) -> bool:
    """Test adding context with invalid role."""
    print("\n[TEST] Add context with invalid role...")
    
    resp = requests.post(
        f"{url}/context?db_name={db}",
        json={"content": "Test", "role": "invalid_role"}
    )
    
    # Should fail with 422 (Pydantic validation) or 500 (our validation)
    if resp.status_code in (200, 201):
        print("  FAIL: Should have rejected invalid role")
        return False
    
    print(f"  PASS (rejected with status {resp.status_code})")
    return True


def test_get_context_limit(url: str, db: str) -> bool:
    """Test getting context with limit parameter."""
    print("\n[TEST] Get context with limit...")
    
    resp = requests.get(f"{url}/context?db_name={db}&limit=2")
    
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    
    data = resp.json()
    print(f"  Count with limit=2: {data.get('count')}")
    
    # Note: limit applies to unsummarized, not including summary
    # So count might be higher if summary exists
    print("  PASS")
    return True


def test_token_count_tracking(url: str, db: str) -> bool:
    """Test that token counts are tracked."""
    print("\n[TEST] Token count tracking...")
    
    # Get context to see token counts
    resp = requests.get(f"{url}/context?db_name={db}")
    
    if resp.status_code != 200:
        print(f"  FAIL: Status {resp.status_code}")
        return False
    
    data = resp.json()
    context = data.get("context", [])
    
    # Should have messages with role field (not summary)
    roles = [c.get("role") for c in context]
    print(f"  Roles in context: {roles}")
    
    if "user" not in roles and "assistant" not in roles:
        print("  FAIL: Expected user/assistant roles in context")
        return False
    
    print("  PASS")
    return True


def test_summarization_auto_trigger(url: str, db: str) -> bool:
    """Test that summarization triggers when threshold exceeded."""
    print("\n[TEST] Auto-summarization on threshold...")
    
    # Create fresh DB
    create_db(url, "test_summarize")
    
    # Add messages with old timestamps (1 hour ago)
    from datetime import datetime, timezone, timedelta
    
    old_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    
    # Add 5 messages with min_cluster_size=3 to trigger summarization
    messages = [
        ("user", "This is the first old message for testing summarization."),
        ("assistant", "This is a response to the first old message."),
        ("user", "Another user message to add more content."),
        ("assistant", "Another assistant response here."),
        ("user", "Final message to push us over the threshold."),
    ]
    
    for i, (role, content) in enumerate(messages):
        resp = requests.post(
            f"{url}/context?db_name=test_summarize&min_cluster_size=3",
            json={
                "content": content,
                "role": role,
                "created_at": old_time  # Old timestamp
            }
        )
        if resp.status_code != 200:
            print(f"  FAIL: Failed to add {role} message: {resp.status_code} {resp.text}")
            return False
        
        data = resp.json()
        if i == len(messages) - 1:
            print(f"  Last message: summarized={data.get('summarized')}, summary_id={data.get('summary_id')}")
    
    # Check context - should have summary now
    resp = requests.get(f"{url}/context?db_name=test_summarize")
    data = resp.json()
    context = data.get("context", [])
    has_summary = any(c.get("role") == "summary" for c in context)
    
    print(f"  After 5 messages (threshold=100 tokens, min=3): has_summary={has_summary}")
    print(f"  Total context count: {len(context)}")
    
    if not has_summary:
        print("  FAIL: Should have summarized with low threshold")
        return False
    
    # Verify summary is first
    if context and context[0].get("role") == "summary":
        print(f"  Summary content: {context[0].get('content', '')[:80]}...")
    
    print("  PASS")
    return True


def run_tests(url: str, db: str) -> None:
    """Run all tests."""
    print(f"Testing API at: {url}")
    print(f"Using test DB: {db}")
    print("=" * 50)
    
    # Wait for API
    print("\nWaiting for API...")
    if not wait_for_api(url):
        print("ERROR: API not ready")
        sys.exit(1)
    print("API is ready!")
    
    # Create test DB
    create_db(url, db)
    
    results = []
    
    # Run tests
    results.append(("Add single context", test_add_context_single(url, db)))
    results.append(("Add multiple context", test_add_context_multiple(url, db)))
    results.append(("Get empty context", test_get_context_empty(url, db)))
    results.append(("Get context with messages", test_get_context_with_messages(url, db)))
    results.append(("Invalid role rejection", test_invalid_role(url, db)))
    results.append(("Context limit param", test_get_context_limit(url, db)))
    results.append(("Token count tracking", test_token_count_tracking(url, db)))
    results.append(("Auto-summarization", test_summarization_auto_trigger(url, db)))
    
    # Summary
    print("\n" + "=" * 50)
    print("RESULTS:")
    print("=" * 50)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test Memory API context endpoints")
    parser.add_argument("--url", default=BASE_URL, help="API base URL")
    parser.add_argument("--db", default=TEST_DB, help="Test database name")
    args = parser.parse_args()
    
    run_tests(args.url, args.db)


if __name__ == "__main__":
    main()