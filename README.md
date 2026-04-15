# Riven Memory

Persistent vector memory API server with semantic search for Riven agents.

## What it does

- Stores memories with **keywords**, **structured properties**, and **vector embeddings**
- Full-text and semantic search via a query DSL (`k:`, `s:`, `q:`, `d:`, `p:`)
- Automatic **context summarization** — groups messages by temporal clusters and summarizes stale ones
- In-memory context window per session, backed by persistent storage

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
python -m riven_memory
# or
uvicorn riven_memory:app --reload --port 8030
```

The server starts on **http://localhost:8030**. API docs at **http://localhost:8030/docs**.

## As a pip dependency

```bash
pip install riven-memory
```

## Config precedence

| Priority | Source |
|----------|--------|
| 1 | `RV_*` env vars (`RV_SECTION__KEY`) |
| 2 | `secrets.yaml` (gitignored) |
| 3 | `secrets_template.yaml` (committed baseline) |
| 4 | `memory/config.yaml` (committed defaults) |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/memories` | Create a memory |
| `GET` | `/memories` | List memories |
| `GET` | `/memories/{id}` | Get a specific memory |
| `PUT` | `/memories/{id}` | Update keywords / properties |
| `DELETE` | `/memories/{id}` | Delete a memory |
| `POST` | `/memories/search` | Search (see query DSL below) |
| `PUT` | `/memories/link` | Link two memories |
| `POST` | `/context` | Add to active context |
| `GET` | `/context` | Retrieve context summary |
| `POST` | `/db/execute` | Raw SQL (debug) |
| `GET` | `/stats` | DB statistics |
| `GET` | `/health` | Health check |

## Query DSL

```
k:<keyword>   # Exact keyword match
s:<keyword>   # Semantic keyword similarity (with optional @threshold)
q:<text>      # Semantic text search
d:<date>      # Date filter ("today", "last 7 days", "YYYY-MM-DD")
p:<key=value> # Property filter
AND / OR / NOT  # Boolean operators
```

Examples:
```
k:python AND k:coding
s:machine learning@0.85
q:context management
d:last 30 days AND k:important
(k:bug OR k:fix) AND NOT k:wontfix
```

## Architecture

```
riven_memory/            <- the package
  api.py        - FastAPI endpoints
  database.py   - SQLite + vector storage
  context.py    - Temporal clustering + summarization
  embedding.py  - Sentence transformer embeddings
  search.py     - Query DSL parser + search engine
  config.yaml   - Committed defaults
  __init__.py
  __main__.py   - python -m riven_memory entry point
riven_memory_config.py  - layered config loader (shared by all modules)
```

Riven agents connect to it via HTTP (default: `http://127.0.0.1:8030`). Configure the URL via `RV_MEMORY_API__URL` or in the Riven config.
