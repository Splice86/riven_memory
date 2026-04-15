# Riven Memory

Persistent vector memory API server with semantic search for Riven agents.

## What it does

- Stores memories with **keywords**, **structured properties**, and **vector embeddings**
- Full-text and semantic search via a powerful query DSL
- Automatic **context summarization** — groups messages by temporal clusters and summarizes stale ones
- In-memory context window per session, backed by persistent SQLite storage
- Fast **vector similarity search** using Microsoft's Harrier embeddings with SQLite caching

## Quick start

```bash
# Install dependencies
pip install -e .

# Configure secrets (optional if using env vars)
cp secrets_template.yaml secrets.yaml
$EDITOR secrets.yaml

# Or override with environment variables
export RV_LLM__URL=http://127.0.0.1:8010
export RV_LLM__API_KEY=sk-your-key

# Run the server
python server.py
# or
uvicorn __init__:app --reload --port 8030
# or (after pip install)
riven-memory
```

The server starts on **http://localhost:8030**. API docs at **http://localhost:8030/docs**.

## As a pip dependency

```bash
pip install riven-memory
```

## Config precedence

Config loads in layers (highest wins):

| Priority | Source | Notes |
|----------|--------|-------|
| 1 | `RV_*` env vars | Nested keys use `__` separator |
| 2 | `secrets.yaml` | User overrides (gitignored) |
| 3 | `config.yaml` | Committed defaults |
| 4 | `secrets_template.yaml` | Fallback baseline |

Example env var: `RV_LLM__URL=http://localhost:8000`

## API endpoints

### Memory operations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/memories` | Create a memory |
| `GET` | `/memories` | List memories (paginated) |
| `GET` | `/memories/{id}` | Get a specific memory |
| `PUT` | `/memories/{id}` | Update memory |
| `DELETE` | `/memories/{id}` | Delete a memory |
| `POST` | `/memories/search` | Search memories |

### Link operations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/memories/link` | Create a link between memories |
| `POST` | `/memories/summary` | Create a summary linked to multiple memories |

### Context operations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/context` | Add to active context |
| `GET` | `/context` | Retrieve context with summarization |

### Embedding operations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/embed` | Get embedding for text |
| `GET` | `/embed/model` | Get embedding model info |
| `GET` | `/embed/cache` | Get embedding cache stats |

### Database operations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/db/create` | Create a new database |
| `GET` | `/db/list` | List all databases |
| `GET` | `/db/exists/{name}` | Check if database exists |

### Utilities

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/stats` | Database statistics |
| `GET` | `/docs/search-syntax` | Query DSL documentation |

## Query DSL

### Filters

| Filter | Syntax | Description |
|--------|--------|-------------|
| Keyword | `k:<keyword>` | Exact keyword match (also just `<keyword>`) |
| Semantic | `s:<keyword>@<threshold>` | Semantic similarity to keyword |
| Semantic text | `q:<text>@<threshold>` | Semantic search on content |
| Property | `p:<key>=<value>` | Property equals |
| Property num | `p:<key>>=<value>` | Numeric comparison (supports `<`, `>`, `<=`, `>=`, `!=`) |
| Date | `d:last N days` or `d:YYYY-MM-DD` | Date filter |
| Link | `l:<type>` | Filter by link type |
| Link direction | `l:source:<type>` or `l:target:<type>` | Directional link filter |
| ID | `id:<memory_id>` | Find specific memory |

### Operators

| Operator | Syntax | Description |
|----------|--------|-------------|
| AND | `k:python AND k:async` | Both conditions required |
| OR | `k:python OR k:javascript` | Either condition |
| NOT | `NOT k:wontfix` | Negation |
| Grouping | `(k:a OR k:b) AND d:last 7 days` | Parentheses for precedence |

### Examples

```
# Boolean keyword search
k:python AND k:coding

# Semantic with threshold (lower = more results)
s:machine learning@0.85
q:context management@0.7

# Date filtering
d:last 30 days AND k:important

# Property filters
p:rating>=4 AND p:status=active
p:category=frontend

# Link filtering
l:related_to
l:summary_of:(k:python)
l:source:related_to
l:target:summary_of:(p:rating>=4)

# Grouped queries
(k:bug OR k:fix) AND NOT k:wontfix
(k:docker OR k:react) AND p:rating>=4
```

### Conditional queries

```
IF <condition> THEN <query> ELSE <query>
```

Example: `IF k:python THEN k:asyncio ELSE k:docker`

## Architecture

```
riven_memory/
├── __init__.py       # Package init, exports FastAPI app
├── server.py         # Entry point: uvicorn runner
│
└── src/
    ├── api.py        # FastAPI endpoints + request/response models
    ├── context.py    # Temporal clustering + LLM summarization
    ├── database.py   # SQLite + vector storage (SPLADE embeddings)
    ├── embedding.py  # Sentence transformer embeddings + cache
    ├── search.py     # Query DSL parser + search engine
    └── config.py     # Layered config loader

Config files:
├── config.yaml              # Committed defaults
├── secrets_template.yaml    # Baseline (copy to secrets.yaml)
└── secrets.yaml             # User overrides (gitignored)
```

## Testing

```bash
# Run comprehensive API tests (requires running server)
python test_api.py

# Run unit tests
python test_comprehensive.py

# Run context API tests
python test_context_api.py
```

## Dependencies

- **FastAPI** - Web framework
- **uvicorn** - ASGI server
- **SQLAlchemy** - ORM
- **sentence-transformers** - Embedding model (microsoft/harrier-oss-v1-270m)
- **torch** - ML framework
- **httpx** - Async HTTP client

Riven agents connect to it via HTTP (default: `http://127.0.0.1:8030`). Configure the URL via `RV_MEMORY_API__URL` or in the Riven config.
