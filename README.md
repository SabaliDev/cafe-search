# Cafe Search API

A conversational job search API with semantic search, multi-turn refinement, and optional reranking.

## Architecture

- FastAPI + async SQLAlchemy
- PostgreSQL 15 + pgvector for cosine similarity search
- Redis for session state (fallback to PostgreSQL JSONB)
- Optional cross-encoder reranker (sentence-transformers)

## Setup

1. Start services:

```bash
docker-compose up --build
```

2. Run migrations:

```bash
alembic upgrade head
```

3. Load data:

```bash
python scripts/seed_jobs.py --path /path/to/jobs.jsonl
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
