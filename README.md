# Cafe Search API

A conversational job search API with semantic search, multi-turn refinement, and optional reranking.

## Architecture

- FastAPI + async SQLAlchemy
- PostgreSQL 15 + pgvector for cosine similarity search
- Redis for session state (fallback to PostgreSQL JSONB)
- Optional cross-encoder reranker (sentence-transformers)

## Quick Start

Choose one of the following options:

### Option 1: Local Development (Recommended for development)

**Prerequisites:**
- Python 3.10+
- PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector)
- Redis 7+
- OpenAI API key

**One-command startup:**

```bash
python run.py
```

This will:
1. Check dependencies (PostgreSQL, Redis)
2. Create virtual environment (if needed)
3. Install dependencies
4. Run database migrations
5. Start the FastAPI server with auto-reload

The API will be available at `http://localhost:8000` with docs at `http://localhost:8000/docs`.

### Option 2: Docker (Recommended for production-like environments)

```bash
# Start all services
docker-compose up --build

# Run migrations
docker-compose exec app alembic upgrade head

# Load data (optional)
docker-compose exec app python scripts/seed_jobs.py --path /path/to/jobs.jsonl
```

## Setup Details

### Local Setup Prerequisites

**macOS:**
```bash
# Install PostgreSQL with pgvector
brew install postgresql@15
brew install pgvector

# Install Redis
brew install redis

# Start services
brew services start postgresql@15
brew services start redis

# Create database
createdb cafesearch
```

**Ubuntu/Debian:**
```bash
# Install PostgreSQL with pgvector
sudo apt install postgresql-15 postgresql-15-pgvector

# Install Redis
sudo apt install redis-server

# Start services
sudo service postgresql start
sudo service redis-server start

# Create database
sudo -u postgres createdb cafesearch
```

**Environment Variables:**

Create a `.env` file:
```bash
OPENAI_API_KEY=your_key_here
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/cafesearch
REDIS_URL=redis://localhost:6379/0
```

## API

- `POST /search`
- `POST /refine/{session_id}`
- `POST /debug/intent`
- `GET /explain?job_id=...&query=...`
- `GET /health`

## Notes

- `org_type` supports: `startup`, `enterprise`, `non-profit`, `government`, `b-corp`.
- `jobs.jsonl` may be hex-encoded per line. The loader detects and decodes that format.
- Session TTL defaults to 1 hour; override with `SESSION_TTL_SECONDS`.
- Reranker is optional. Enable with `RERANKER_ENABLED=true`.

## Performance Targets

- Vector search p95 < 100ms (no rerank)
- Reranked p95 < 300ms

## Design Decisions

- **Three embeddings per job (explicit/inferred/company):** separates role intent from domain and culture signals for more stable routing.
- **Hybrid search (vector + SQL filters):** semantic relevance without losing hard constraints like location, remote, salary.
- **Session state with `semantic_query`:** preserves the original role intent across refinements.
- **Refinement as deltas:** LLM outputs filter changes only; filters are merged, not replaced.
- **Specialization validation:** refinement terms are validated against the corpus (title/description counts) before applying filters.
- **Lightweight intent routing:** keyword-based intent is <1ms and avoids LLM latency for routing.

## Trade-offs

- **String-based exclusions:** `NOT ILIKE` is simple and fast but misses semantic equivalents (e.g., “Staff” vs “Senior”).
- **Static role anchoring:** reliable for common roles but requires maintenance for new role categories.
- **No OR logic for multi-location/skills:** current filter model is AND-only for simplicity.
- **LLM refinement cost:** accurate parsing but dominates per-request cost at scale.
- **Postgres + pgvector scaling:** solid up to ~10M jobs; beyond that, a dedicated vector DB is recommended.

## Tuning Guide

- HNSW index params are set to `m=16`, `ef_construction=64`. For recall improvements, raise `ef_construction` during build.
- Adjust runtime `ef_search` at query time:

```sql
SET LOCAL hnsw.ef_search = 64;
```

- Add `ANALYZE jobs` after bulk loads.
- Increase Postgres `work_mem` for large filter + vector queries.

## Observability

- Structured logs include query, intent, filters, result count, and latency.
- `/health` returns DB connectivity and index definitions.
- `/explain` returns per-embedding similarity for a job/query.

## Tests

```bash
pytest
```

Integration tests are skipped unless `DATABASE_URL` is set.
