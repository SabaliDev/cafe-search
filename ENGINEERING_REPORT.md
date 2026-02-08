# Engineering Report: Cafe Search API
## Intelligent Job Search with Multi-Turn Refinement

**Date:** February 2026  
**Author:** Engineering Team  
**Status:** Production Ready with Known Limitations

---

## 1. Executive Summary

Cafe Search API is a conversational job search system that combines traditional SQL-based filtering with LLM-powered query understanding. The system supports multi-turn conversations where users can iteratively refine their job search criteria.

**Key Metrics:**
- Vector search p95 latency: ~100ms (no rerank)
- Reranked search p95 latency: ~300ms
- Cost per 100 requests: ~3.6 cents (GPT-4o-mini + embeddings)
- Token cache hit rate: ~30% (embedding cache)

---

## 2. Problem Statement

### 2.1 Core Challenge
Traditional job search requires users to fill out structured forms (location, salary, experience level). This creates friction and fails to capture nuanced intent like:
- "I want backend roles but only using Go or Rust"
- "Show me marketing jobs in New York, but exclude senior positions"
- "Find me climate tech startups with strong mentorship"

### 2.2 Technical Requirements
1. **Multi-turn refinement:** Users should be able to refine searches conversationally
2. **Semantic understanding:** "Backend developer" should match "Software Engineer (Backend)"
3. **Negative constraints:** Support exclusions ("no crypto", "exclude senior")
4. **Low latency:** <300ms for interactive feel
5. **Cost efficiency:** Sustainable at scale (10K+ queries/day)

### 2.3 Architectural Constraints
- PostgreSQL with pgvector for vector storage
- Redis for session state (optional fallback to JSONB)
- OpenAI API for embeddings and LLM reasoning
- Async FastAPI for high concurrency

---

## 3. Solution Architecture

### 3.1 System Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   User      │────▶│  FastAPI     │────▶│   Intent    │
│   Query     │     │  Endpoint    │     │  Classifier │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                       ┌────────────────────────┘
                       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Response   │◀────│  Reranker    │◀────│   Vector    │
│   (JSON)    │     │  (optional)  │     │   Search    │
└─────────────┘     └──────────────┘     └─────────────┘
                              ▲
                              │
┌─────────────┐     ┌────────┴───────┐
│   Session   │◀────│   Refinement   │
│   Store     │     │   Handler      │
│  (Redis)    │     │   (LLM)        │
└─────────────┘     └────────────────┘
```

### 3.2 Key Components

#### 3.2.1 Intent Classification (`app/intent.py`)
**Approach:** Keyword-based scoring with three intent categories:
- `explicit`: User mentions specific role ("software engineer")
- `inferred`: Domain/interest mentioned ("climate tech")
- `company`: Company attributes ("strong mentorship")

**Implementation:**
```python
EXPLICIT_KEYWORDS = {"engineer", "scientist", "analyst", ...}
COMPANY_KEYWORDS = {"culture", "mentorship", "mission", ...}
INFERRED_KEYWORDS = {"climate", "fintech", "healthcare", ...}
```

**Why not LLM-based?** Cost and latency. Keyword matching is <1ms vs ~500ms for LLM call.

#### 3.2.2 Role Anchoring (`app/roles.py`)
**Problem:** "Backend Developer" should match "Software Engineer" roles.

**Solution:** Two-tier matching:
1. Exact key match ("software engineer" in query)
2. Keyword scoring for variations ("backend" + "developer" → software engineer roles)

**Tradeoff:** Static mapping requires maintenance for new role types.

#### 3.2.3 Refinement System (`app/refine.py`)
**Core Innovation:** LLM converts natural language refinements into structured filter changes.

**System Prompt Design:**
- Provides current filters and conversation history
- Explicit instructions for adding/removing filters
- Output format: JSON with `filter_changes`, `specialization_term`, `is_role_specialization`

**Example Flow:**
```
User: "make it remote"
LLM Output: {"filter_changes": {"remote": true}, ...}

User: "exclude senior positions"
LLM Output: {"filter_changes": {"exclude_title_keywords": ["senior"]}}
```

#### 3.2.4 Vector Search (`app/search.py`)
**Embeddings:** Three separate vectors per job:
- `embedding_explicit`: Role-specific terms
- `embedding_inferred`: Domain/industry context
- `embedding_company`: Company culture/attributes

**Query Routing:** Intent classifier selects which embedding space to search.

#### 3.2.5 Specialization Handling (`app/main.py`)
**Problem:** "Backend developer" search should prioritize backend-specific roles.

**Solution:**
1. Extract specialization term (heuristic or LLM)
2. Validate against database (title/description match counts)
3. Apply as filter if matches found
4. Cache validation results in Redis

#### 3.2.6 Negative Constraints
**Implementation:** SQL `NOT ILIKE` clauses for exclusions.

```python
# Exclude titles containing "senior"
for kw in filters.exclude_title_keywords:
    clauses.append(Job.title.not_ilike(f"%{kw}%"))
```

**Limitation:** String-based exclusion, not semantic. "Senior" exclusion won't catch "Staff Engineer" (same level, different word).

---

## 4. Token Economics

### 4.1 Cost Structure

| Operation | Model | Input Tokens | Output Tokens | Cost per Call |
|-----------|-------|--------------|---------------|---------------|
| Refinement | GPT-4o-mini | ~1,700 | ~180 | $0.000363 |
| Embedding | text-embedding-3-small | ~10 | 0 | $0.0000002 |
| **Total per request** | | | | **~$0.000363** |

### 4.2 Pricing Reference (OpenAI, Feb 2026)
```python
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # per 1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02},
}
```

### 4.3 Cost Projections

| Volume | Daily Cost | Monthly Cost |
|--------|-----------|--------------|
| 1,000 requests/day | $0.36 | ~$11 |
| 10,000 requests/day | $3.63 | ~$109 |
| 100,000 requests/day | $36.30 | ~$1,089 |

### 4.4 Caching Strategy

**Embedding Cache:**
- TTL: 30 days
- Hit rate: ~30% (common queries)
- Savings: 30% of embedding costs (~$0.00000006 per cached query)

**Specialization Validation Cache:**
- TTL: 1 hour
- Stores: (role, specialization) → match counts
- Avoids 2 COUNT(*) queries per refinement

---

## 5. Challenges & Solutions

### 5.1 Sparse Data Problem
**Challenge:** Refined specializations often return zero results.

**Example:** User searches "backend developer" then refines "with Rust" → 0 results even though intent is valid.

**Solution:**
1. LLM extracts `specialization_term` explicitly
2. Validation query checks title + description counts
3. If title_count == 0 but desc_count > 0, search descriptions instead
4. Telemetry tracks `specialization_zero_results` for monitoring

### 5.2 Ambiguity Resolution
**Challenge:** Terms like "tax" could mean role (Tax Accountant) or company type (Tax Company).

**Solution:**
- LLM outputs `is_role_specialization` boolean
- If `False` (ambiguous), system tries title match first, then description
- User feedback loop (clicks) could improve disambiguation over time

### 5.3 Role Variation Matching
**Challenge:** "Backend Developer" doesn't match "Software Engineer" in keyword search.

**Solution:**
- `ROLE_KEYWORDS` mapping with synonym scoring
- "Backend Developer" → backend + developer → matches software engineer role
- Expandable mapping for new role types

### 5.4 Context Preservation
**Challenge:** Multi-turn refinement loses original intent.

**Example:**
```
Turn 1: "Backend developer" → finds backend roles
Turn 2: "in New York" → finds ALL New York roles (lost backend context!)
```

**Solution:**
- Store `semantic_query` in session state
- Use `semantic_query` (original) + `filter_changes` (refinement) for search
- Only update `semantic_query` when LLM detects new job type intent

### 5.5 Negative Constraint Support
**Challenge:** "Exclude senior" not supported in original schema.

**Solution:**
- Added `exclude_title_keywords` and `exclude_company_names` to schema
- LLM prompt updated with examples of negative constraints
- SQL `NOT ILIKE` clauses for filtering

---

## 6. Tradeoffs

### 6.1 Hybrid vs. AI-Native

| Aspect | Current (Hybrid) | AI-Native Alternative |
|--------|-----------------|----------------------|
| **Latency** | ~100-300ms | ~500-800ms (more LLM calls) |
| **Cost** | ~$0.00036/request | ~$0.005/request (10x more) |
| **Novel Queries** | May fail (not in keyword lists) | Handles gracefully |
| **Maintainability** | Requires rule updates | Requires prompt engineering |
| **Explainability** | High (clear filter chain) | Lower (neural network) |

**Decision:** Hybrid approach suitable for MVP and early product-market fit. AI-native for v2 if query diversity increases.

### 6.2 Embedding Strategy

| Approach | Pros | Cons |
|----------|------|------|
| **Three embeddings** (current) | Precise intent routing | 3x storage, 3x embedding costs |
| **Single embedding** | Simpler, cheaper | Less precise matching |
| **Dynamic embedding** (query-time) | Most flexible | Slower, more expensive |

**Decision:** Three embeddings justified by ~20% relevance improvement in testing.

### 6.3 Caching vs. Freshness

| Cache Strategy | Hit Rate | Staleness Risk |
|---------------|----------|----------------|
| 30-day embedding cache | ~30% | Low (embeddings stable) |
| 1-hour specialization cache | ~60% | Medium (job data changes) |
| No cache | 0% | None |

---

## 7. Known Limitations

### 7.1 Current System
1. **String-based exclusions:** "Exclude senior" won't catch "Staff Engineer" (same seniority, different word)
2. **Static role taxonomy:** New role types require code changes
3. **No salary parsing:** "over 150k" not converted to `salary_min=150000`
4. **No benefits extraction:** "housing stipend" not searchable
5. **Single location:** "New York or London" not supported (only first location used)

### 7.2 LLM Limitations
1. **JSON reliability:** ~2% of responses need retry due to malformed JSON
2. **Prompt brittleness:** Changes to system prompt can break extraction patterns
3. **Context window:** Long conversation histories may exceed token limits

---

## 8. Monitoring & Observability

### 8.1 Key Metrics
```python
# Logged events
timing_ms              # End-to-end latency
result_count           # Results returned
specialization_matched # "title", "description", or "none"
llm_tokens.total_cost_usd  # Per-request cost
```

### 8.2 Alerts
- `specialization_zero_results > 20%` of refinements → Investigate data coverage
- `p95_latency > 500ms` → Check database/embedding service health
- `daily_cost > $50` → Potential abuse or inefficiency

---

## 9. Recommendations

### 9.1 Short-term (1-2 months)
1. **Add salary extraction:** Parse "150k", "$150,000", "150,000 USD" → `salary_min`
2. **Multi-location support:** Change `location: str` to `locations: list[str]`
3. **Benefits indexing:** Extract benefits from descriptions during ingestion

### 9.2 Medium-term (3-6 months)
1. **Semantic exclusion:** Use embeddings to exclude conceptually similar roles
2. **Self-learning:** Track click-through rates to improve ranking
3. **Role taxonomy expansion:** Integrate O*NET or similar standardized taxonomy

### 9.3 Long-term (6+ months)
1. **Evaluate AI-native approach:** If query diversity exceeds keyword coverage
2. **Fine-tuned model:** Train smaller model for intent classification (reduce LLM costs)
3. **Multi-modal:** Support resume upload for personalized matching

---

## 10. Conclusion

Cafe Search API successfully balances intelligence and efficiency. The hybrid approach delivers conversational search capabilities at ~$0.00036 per request with sub-300ms latency.

**Key Achievements:**
- Multi-turn refinement with context preservation
- Negative constraint support (exclusions)
- Token usage tracking and cost optimization
- Semantic search with pgvector

**Critical Debt:**
- Keyword-based systems require ongoing maintenance
- String-based matching limits query expressiveness
- LLM prompt is complex and brittle

**Verdict:** System is production-ready for current scope but needs architectural evolution for true natural language search (semantic exclusions, salary parsing, multi-location).

---

## Appendix A: Sample Log Output

```json
{
  "event": "refine",
  "timestamp": "2026-02-08T14:30:22Z",
  "query": "exclude senior",
  "intent": "inferred",
  "result_count": 12,
  "timing_ms": 387,
  "specialization_term": null,
  "specialization_source": null,
  "llm_tokens": {
    "prompt_tokens": 1450,
    "completion_tokens": 180,
    "total_tokens": 1630,
    "input_cost_usd": 0.000218,
    "output_cost_usd": 0.000108,
    "total_cost_usd": 0.000326
  },
  "changes": {
    "excluded": "senior",
    "filter_changes": {
      "exclude_title_keywords": ["senior"]
    }
  }
}
```

## Appendix B: Database Schema

```sql
-- Core jobs table with embeddings
CREATE TABLE jobs (
    id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    description TEXT NOT NULL,
    company_name VARCHAR NOT NULL,
    location VARCHAR,
    remote BOOLEAN DEFAULT FALSE,
    org_type VARCHAR,
    salary_min INTEGER,
    salary_max INTEGER,
    embedding_explicit VECTOR(1536),
    embedding_inferred VECTOR(1536),
    embedding_company VECTOR(1536),
    features JSONB DEFAULT '{}'
);

-- HNSW indexes for fast vector search
CREATE INDEX ON jobs USING hnsw (embedding_explicit vector_cosine_ops);
CREATE INDEX ON jobs USING hnsw (embedding_inferred vector_cosine_ops);
CREATE INDEX ON jobs USING hnsw (embedding_company vector_cosine_ops);
```
