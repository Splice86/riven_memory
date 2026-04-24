"""
Functional test for context summarization.

Creates memories, triggers summarization, and prints the resulting context
so you can evaluate the summary quality.

Usage:
    python test_summarization.py
"""

import os
import sys
import json
from datetime import datetime, timezone, timedelta

# Setup paths
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))

from database import MemoryDB, init_db
from context import Context
from config import reload


def clear_test_data(db: MemoryDB, session: str):
    """Clear existing test data for clean run."""
    print(f"\n{'='*60}")
    print(f"Clearing existing test data for session: {session}")
    print('='*60)
    
    # Delete context memories
    results = db.search(f"k:context AND p:session={session}", limit=10000)
    for mem in results:
        db.delete_memory(mem["id"])
    
    # Delete summaries
    results = db.search(f"k:summary AND p:session={session}", limit=10000)
    for mem in results:
        db.delete_memory(mem["id"])
    
    print(f"Cleared {len(results) * 2} memories")


def create_test_messages(ctx: Context, session: str, count: int = 10):
    """Create test messages that will trigger summarization."""
    print(f"\n{'='*60}")
    print(f"Creating {count} test messages")
    print('='*60)
    
    now = datetime.now(timezone.utc)
    
    # Make each message long enough to hit the cluster size limit quickly
    # With min_cluster_size=3, 3-4 messages should trigger summarization
    messages = [
        ("user", f"Hello! I'm working on a Python project and I need help with async programming. The main challenge is understanding asyncio.gather() vs asyncio.create_task(). I keep getting confused about when to use each one and how they handle exceptions differently."),
        ("assistant", "Great question! asyncio.gather() is used when you want to run multiple coroutines concurrently and wait for all of them to complete. It returns a list of results in the same order as the input tasks. asyncio.create_task() schedules a coroutine to run in the background and returns a Task object immediately. gather() is blocking - it waits for all tasks. create_task() is non-blocking for scheduling, but you can await the task later. For error handling, gather() has return_exceptions parameter."),
        ("user", "That's helpful! Can you show me an example of using asyncio.gather() with error handling? I want to make sure my application doesn't crash if one of the tasks fails."),
        ("assistant", "Here's an example with error handling: import asyncio async def safe_gather(*tasks): results = await asyncio.gather(*tasks, return_exceptions=True) return [r for r in results if not isinstance(r, Exception)] # Usage: tasks = [async_func1(), async_func2(), async_func3()] results = await safe_gather(*tasks) This pattern catches exceptions and filters them out so your code continues to work."),
        ("user", "I see. Now what about memory leaks? I noticed my application keeps growing in memory over time. I'm running a long-lived service that processes requests continuously."),
        ("assistant", "Memory leaks in async Python often come from: 1) Unfinished tasks still referenced in memory, 2) Event loops holding references to coroutines that were never awaited, 3) Large objects stored in closures that prevent garbage collection. Make sure to cancel tasks you no longer need with task.cancel() and use asyncio.TaskGroup() (Python 3.11+) for automatic cleanup. Also check for circular references that prevent garbage collection."),
        ("user", "Can you explain task.cancel() behavior? Does it stop immediately when I call it?"),
        ("assistant", "task.cancel() requests cancellation - it sets the CancelledError exception inside the task. The task will stop at its next await point, not immediately. You should await task if you need to confirm cancellation completed: task.cancel() try: await task except asyncio.CancelledError: pass  # Task was cancelled as expected You can also check task.cancelled() after awaiting to verify."),
        ("user", "What about using weakref to prevent circular references? My dataclasses seem to hold onto each other."),
        ("assistant", "weakref can help break circular references. Use weakref.ref() for object references that shouldn't prevent garbage collection: import weakref from dataclasses import dataclass @dataclass class Node: children: list parent = None  # Use weakref.proxy if parent is optional For parent references that might not always exist, use weakref.proxy or store the parent ID instead of a direct reference."),
        ("user", "This is getting complex. Can we switch topics? I also need to understand FastAPI dependency injection for database connections."),
        ("assistant", "FastAPI's dependency injection is great! You declare dependencies as functions with Depends(): from fastapi import Depends async def get_db(): db = Database() try: yield db finally: db.close() @app.get('/items') async def read_items(db: Database = Depends(get_db)): return db.query('SELECT * FROM items') Dependencies are evaluated once per request and shared across endpoints. The yield pattern ensures cleanup runs after the request."),
        ("user", "How do I test endpoints that use dependency injection? I want to mock the database without using a real one."),
        ("assistant", "Override dependencies in tests using app.dependency_overrides: def test_read_items(): def override_get_db(): return FakeDB()  # Your test double app.dependency_overrides[get_db] = override_get_db response = client.get('/items') app.dependency_overrides.clear() This is the cleanest way to test without real database connections. You can create FakeDB classes that return test data."),
        ("user", "I want to add authentication too. Should I use OAuth2 or JWT directly? This is for a simple internal API."),
        ("assistant", "For internal APIs, JWT is simpler and sufficient. Use python-jose for encoding/decoding: from jose import JWTError, jwt from datetime import datetime, timedelta SECRET_KEY = 'your-secret-key' ALGORITHM = 'HS256' def create_token(data: dict): to_encode = data.copy() expire = datetime.utcnow() + timedelta(minutes=30) to_encode.update({'exp': expire}) return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM) OAuth2 is more complex but better for public APIs with third-party logins."),
        ("user", "What's the best practice for storing the JWT secret in production? I don't want it in the codebase."),
        ("assistant", "Never hardcode secrets! Use environment variables: import os SECRET_KEY = os.environ.get('JWT_SECRET_KEY') if not SECRET_KEY: raise ValueError('JWT_SECRET_KEY not set') In Kubernetes, use Secrets. In Docker, use docker secret or env files that aren't committed to version control. Your CI/CD pipeline should inject secrets at deployment time."),
        ("user", "Last question for now - how do I handle CORS in FastAPI for my frontend?"),
        ("assistant", "FastAPI's CORSMiddleware handles it easily: from fastapi.middleware.cors import CORSMiddleware app.add_middleware( CORSMiddleware, allow_origins=['https://yourfrontend.com'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'], ) Be specific with origins in production! Don't use allow_origins=['*'] with credentials as browsers will block it. List only the domains that need access."),
    ]
    
    for i, (role, content) in enumerate(messages[:count]):
        # Add 1 second gap between messages to help with temporal clustering
        created_at = (now + timedelta(seconds=i)).isoformat()
        result = ctx.add(role, content, created_at=created_at, session=session)
        print(f"  [{i+1}] {role}: {content[:60]}...")
        if result.get("summarized"):
            print(f"       ⭐ SUMMARIZED! summary_id={result.get('summary_id')}, memories={result.get('memories_summarized')}")


def print_context(ctx: Context, session: str, limit: int = 50, max_summaries: int = 3):
    """Get and print the current context."""
    print(f"\n{'='*60}")
    print(f"CONTEXT (limit={limit}, max_summaries={max_summaries})")
    print('='*60)
    
    context = ctx.get(limit=limit, max_summaries=max_summaries, session=session)
    
    print(f"\nTotal items: {len(context)}\n")
    
    for i, item in enumerate(context):
        role = item.get("role", "unknown")
        content = item.get("content", "")
        created_at = item.get("created_at", "")
        summary_level = item.get("summary_level", "")
        
        prefix = "📋" if role == "summary" else "💬"
        level_str = f" [L{summary_level}]" if summary_level else ""
        
        print(f"\n{prefix} [{i+1}] {role.upper()}{level_str} | {created_at}")
        print("-" * 50)
        
        # Print full content, word-wrapped
        import textwrap
        wrapper = textwrap.TextWrapper(width=80, subsequent_indent="  ")
        print(wrapper.fill(content))


def show_summary_stats(db: MemoryDB, session: str):
    """Show statistics about summaries."""
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print('='*60)
    
    # Count summaries by level
    all_summaries = db.search(f"k:summary AND p:session={session}", limit=10000)
    active_summaries = [s for s in all_summaries if s.get("properties", {}).get("was_summarized") != "true"]
    
    by_level = {}
    for s in active_summaries:
        level = s.get("properties", {}).get("summary_level", "1")
        by_level[level] = by_level.get(level, 0) + 1
    
    print(f"\nActive summaries: {len(active_summaries)}")
    for level in sorted(by_level.keys()):
        print(f"  Level {level}: {by_level[level]}")
    
    # Show summary links
    print(f"\nTotal summaries in DB: {len(all_summaries)}")
    
    # Count unsummarized messages
    unsummarized = db.search(f"k:context AND p:session={session}", limit=10000)
    active = [m for m in unsummarized if m.get("properties", {}).get("was_summarized") != "true"]
    print(f"\nUnsummarized messages: {len(active)}")
    
    return active_summaries, active


def main():
    print("\n" + "="*60)
    print("SUMMARIZATION FUNCTIONAL TEST")
    print("="*60)
    
    # Setup
    reload()
    
    db_path = os.path.join(_root, "database", "test_summarize.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Initialize DB
    init_db(db_path)
    db = MemoryDB(db_path=db_path)
    
    # Create context with min_cluster_size=3 to trigger summarization
    ctx = Context(db, min_cluster_size=3)
    
    # Test session (unique per run)
    session = f"test_summarize_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\nUsing session: {session}")
    print(f"Context limits: min_cluster_size=3")
    
    # Clear any existing test data
    clear_test_data(db, session)
    
    # Create messages - should trigger summarization
    create_test_messages(ctx, session, count=10)
    
    # Show stats
    summaries, messages = show_summary_stats(db, session)
    
    # Print context for evaluation
    print_context(ctx, session, limit=50, max_summaries=3)
    
    # Print detailed summary content for quality evaluation
    print(f"\n{'='*60}")
    print("DETAILED SUMMARY CONTENT FOR QUALITY EVALUATION")
    print('='*60)
    
    for s in summaries:
        props = s.get("properties", {})
        level = props.get("summary_level", "1")
        summarized_count = props.get("summarized_count", "?")
        summarized_tokens = props.get("summarized_tokens", "?")
        was_summarized = props.get("was_summarized", "false")
        
        if was_summarized == "true":
            continue  # Skip summaries that were themselves summarized
        
        print(f"\n{'='*60}")
        print(f"SUMMARY (Level {level}) - summarized {summarized_count} messages, ~{summarized_tokens} tokens")
        print(f"ID: {s['id']}, Created: {s.get('created_at', 'unknown')}")
        print('='*60)
        content = s.get("content", "[no content]")
        print(content)  # Full content, not truncated
        print()
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print('='*60)
    print(f"\nDB location: {db_path}")
    print("You can inspect the database manually if needed.")
    
    # Write full summary to a file for detailed review
    summary_file = os.path.join(_root, "test_summary_output.txt")
    with open(summary_file, "w") as f:
        f.write("SUMMARIZATION TEST OUTPUT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Session: {session}\n")
        f.write(f"Context limits: min_cluster_size=3\n\n")
        
        for s in summaries:
            props = s.get("properties", {})
            level = props.get("summary_level", "1")
            summarized_count = props.get("summarized_count", "?")
            summarized_tokens = props.get("summarized_tokens", "?")
            was_summarized = props.get("was_summarized", "false")
            
            if was_summarized == "true":
                continue
            
            f.write(f"\n{'='*60}\n")
            f.write(f"SUMMARY (Level {level})\n")
            f.write(f"Summarized {summarized_count} messages, ~{summarized_tokens} tokens\n")
            f.write(f"ID: {s['id']}, Created: {s.get('created_at', 'unknown')}\n")
            f.write('='*60 + "\n\n")
            f.write(s.get("content", "[no content]"))
            f.write("\n\n")
    
    print(f"\nFull summary written to: {summary_file}")
    
    return db_path, session


if __name__ == "__main__":
    db_path, session = main()
