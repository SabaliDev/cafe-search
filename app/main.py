from __future__ import annotations

import math
import time
from typing import Any
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import select, text, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session, check_db
from app.embeddings import embedding_service
from app.intent import classify_intent
from app.search import search_jobs
from app.schemas import (
    SearchRequest,
    SearchResponse,
    RefineRequest,
    RefineResponse,
    IntentDebugResponse,
    ExplainResponse,
    SearchFilters,
)
from app.session import get_session_store
from app.refine import apply_refinement
from app.models import Job
from app.rerank import reranker
from app.logger import setup_logging, log_event
from app.errors import embedding_unavailable
from app.roles import extract_role_anchor
from app.cache import specialization_cache

app = FastAPI(title="Cafe Search API", version="0.1.0")
setup_logging()

session_store = get_session_store()


def _normalize_rerank(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-score))


def _extract_specialization_term(query: str) -> str | None:
    """Extract specialization term from query, filtering out location/filter-only queries.
    
    Returns None if query appears to be a filter-only query (location, remote, salary, etc.)
    to avoid misclassifying filter refinements as specializations.
    """
    lowered = query.lower().strip()
    
    # Location indicators that suggest this is NOT a specialization
    # These must be checked as whole words to avoid false positives (e.g., "us" matching "use")
    location_indicators = {
        "in ", "near ", "around ", "close to ", "within ",
        " city", " state", " country", " region", " area",
        "usa", "us ", "united states", "america", "uk", "london", 
        "new york", "san francisco", "california", "texas", "florida",
        "europe", "asia", "remote", "onsite", "on-site", "hybrid",
        "from home", "wfh", "work from",
    }
    
    # If query contains location indicators at the start, it's likely a location query
    if any(lowered.startswith(indicator.strip()) for indicator in location_indicators):
        return None
    # Check for location indicators as whole words (with word boundaries)
    for indicator in location_indicators:
        ind_stripped = indicator.strip()
        # Check as whole word - either surrounded by spaces or at end
        if f" {ind_stripped} " in lowered or lowered.endswith(f" {ind_stripped}"):
            return None
    
    # Filter-only markers (not specializations)
    filter_only_markers = {
        "salary", "pay", "compensation", "k", "$", "usd", "per year", "annually",
        "non-profit", "nonprofit", "startup", "enterprise", "government", "b-corp", "bcorp",
        "benefits", "health insurance", "dental", "vision", "401k", "pto",
        "part-time", "full-time", "contract", "freelance", "internship",
        "senior", "junior", "lead", "principal", "staff", "entry level",
    }
    if any(marker in lowered for marker in filter_only_markers):
        return None
    
    # Tech stack keywords that ARE specializations (programming languages, frameworks)
    tech_keywords = {
        "python", "javascript", "typescript", "java", "go", "golang", "rust", 
        "c++", "c#", "ruby", "php", "swift", "kotlin", "scala", "r",
        "react", "angular", "vue", "svelte", "nextjs", "nodejs", "django",
        "flask", "spring", "rails", "laravel", "express",
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
        "postgres", "mysql", "mongodb", "redis", "elasticsearch",
        "machine learning", "ai", "data science", "analytics",
    }
    
    # Check for tech stack mentions first (these are valid specializations)
    found_tech = [t for t in lowered.replace(",", " ").replace("/", " ").split() 
                  if t in tech_keywords]
    if found_tech:
        return " ".join(found_tech[:3])
        
    # Strip common filler words and location prepositions
    filler_words = {
        "related", "role", "roles", "job", "jobs", "position", "positions", 
        "only", "just", "please", "show", "me", "with", "using", "that", "use",
        "something", "looking", "for", "the", "a", "an", "but",
        "in", "at", "on", "to", "from", "of", "by", "about", "like",
        "want", "wanting", "need", "needed", "should", "would", "could",
        "can", "will", "i", "we", "you", "they", "it", "this", "these",
        "have", "has", "had", "do", "does", "did", "be", "been",
    }
    
    # Replace common separators but keep "and"/"or" for tech combinations
    cleaned = lowered.replace("/", " ").replace(",", " ").replace("-", " ")
    tokens = [t for t in cleaned.split() if t not in filler_words and len(t) > 1]
    
    if not tokens:
        return None
        
    # Take up to 3 meaningful tokens
    return " ".join(tokens[:3])


def _build_filter_clauses(filters: SearchFilters) -> list:
    clauses = []
    if filters.remote is not None:
        clauses.append(Job.remote == filters.remote)
    if filters.org_types:
        clauses.append(Job.org_type.in_(filters.org_types))
    if filters.salary_min is not None:
        clauses.append(Job.salary_max.is_(None) | (Job.salary_max >= filters.salary_min))
    if filters.salary_max is not None:
        clauses.append(Job.salary_min.is_(None) | (Job.salary_min <= filters.salary_max))
    if filters.country:
        clauses.append(Job.features["country"].astext == filters.country)
    if filters.location:
        clauses.append(
            or_(
                Job.location.ilike(f"%{filters.location}%"),
                Job.title.ilike(f"%{filters.location}%")
            )
        )
    if filters.mission_keywords:
        clauses.append(Job.features["mission_keywords"].contains(filters.mission_keywords))
    if filters.company_industry:
        clauses.append(Job.features["company_industry"].contains(filters.company_industry))
    if filters.experience_level:
        level = filters.experience_level.lower()
        if level == "senior":
            clauses.append(or_(Job.title.ilike("%senior%"), Job.title.ilike("%lead%"), Job.title.ilike("%principal%")))
        elif level == "junior":
            clauses.append(or_(Job.title.ilike("%junior%"), Job.title.ilike("%entry%")))
    if filters.title_keywords:
        title_clauses = []
        for kw in filters.title_keywords:
            title_clauses.append(Job.title.ilike(f"%{kw}%"))
        clauses.append(or_(*title_clauses))
    if filters.description_keywords:
        desc_clauses = []
        for kw in filters.description_keywords:
            desc_clauses.append(Job.description.ilike(f"%{kw}%"))
        clauses.append(or_(*desc_clauses))
    # Negative constraints - exclusions
    if filters.exclude_title_keywords:
        for kw in filters.exclude_title_keywords:
            clauses.append(Job.title.not_ilike(f"%{kw}%"))
    if filters.exclude_company_names:
        for kw in filters.exclude_company_names:
            clauses.append(Job.company_name.not_ilike(f"%{kw}%"))
    return clauses


async def _validate_specialization(
    session: AsyncSession, term: str, filters: SearchFilters, role: str | None = None
) -> tuple[int, int]:
    """Validate specialization term against jobs, with caching.
    
    Returns (title_count, desc_count) indicating matches in titles vs descriptions.
    Results are cached in Redis to reduce latency on repeated validations.
    """
    # Check cache first
    cached = await specialization_cache.get(role, term)
    if cached is not None:
        return cached
    
    # Compute counts
    clauses = _build_filter_clauses(filters)
    title_count_stmt = select(func.count()).select_from(Job).where(
        and_(*clauses, Job.title.ilike(f"%{term}%"))
    )
    desc_count_stmt = select(func.count()).select_from(Job).where(
        and_(*clauses, Job.description.ilike(f"%{term}%"))
    )
    title_count = (await session.execute(title_count_stmt)).scalar_one()
    desc_count = (await session.execute(desc_count_stmt)).scalar_one()
    
    # Cache results
    await specialization_cache.set(role, term, int(title_count), int(desc_count))
    return int(title_count), int(desc_count)


@app.get("/health")
async def health(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    db_ok = await check_db()
    index_result = await session.execute(
        text(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'jobs'
            """
        )
    )
    indexes = [dict(row._mapping) for row in index_result.fetchall()]
    return {
        "ok": db_ok,
        "indexes": indexes,
        "redis_enabled": settings.redis_url is not None,
        "reranker_enabled": settings.reranker_enabled,
    }


async def _maybe_rerank(session: AsyncSession, query: str, results):
    if not settings.reranker_enabled:
        return results

    ids = [r.id for r in results]
    if not ids:
        return results

    result = await session.execute(
        select(Job.id, Job.title, Job.company_name, Job.description).where(Job.id.in_(ids))
    )
    text_map = {
        row.id: f"{row.title}\n{row.company_name}\n{row.description}" for row in result.all()
    }
    pairs = [(job_id, text_map.get(job_id, "")) for job_id in ids]
    scores = await reranker.rerank(session, query, pairs)
    if not scores:
        return results
    for r in results:
        if r.id in scores:
            r.score = _normalize_rerank(scores[r.id])
            r.explanation["reranker_score"] = r.score
    results.sort(key=lambda r: r.score, reverse=True)
    return results


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, session: AsyncSession = Depends(get_session)):
    start = time.monotonic()
    top_k = request.top_k
    search_k = max(top_k, 100) if settings.reranker_enabled else top_k
    role_anchor = extract_role_anchor(request.query)
    effective_filters = request.filters
    if role_anchor:
        if effective_filters is None:
            effective_filters = SearchFilters(inferred_titles=role_anchor)
        elif effective_filters.inferred_titles is None:
            effective_filters.inferred_titles = role_anchor
    try:
        results, analysis = await search_jobs(
            session=session,
            query=request.query,
            filters=effective_filters,
            top_k=search_k,
        )
    except Exception:
        return embedding_unavailable()

    results = await _maybe_rerank(session, request.query, results)
    results = results[:top_k]

    if request.session_id:
        session_id = request.session_id
        state = await session_store.get(session, session_id) or {}
    else:
        session_id = await session_store.create(session)
        state = {}

    state.update(
        {
            "role_anchor": role_anchor,
            "filters": (effective_filters.model_dump() if effective_filters else {}),
            "query_history": (state.get("query_history", []) + [request.query]),
            "previous_results_ids": [r.id for r in results],
            "active_embedding": analysis["intent"],
            "entities_history": (state.get("entities_history", []) + analysis["extracted_entities"]),
            "persistent_filters": state.get("persistent_filters", []),
            "semantic_query": request.query,  # Store the semantic query for refinements
        }
    )
    await session_store.save(session, session_id, state)

    timing_ms = int((time.monotonic() - start) * 1000)
    log_event(
        "search",
        {
            "query": request.query,
            "intent": analysis["intent"],
            "filters": analysis["applied_filters"],
            "result_count": len(results),
            "timing_ms": timing_ms,
        },
    )
    return SearchResponse(
        results=results,
        session_id=session_id,
        query_analysis=analysis,
        timing_ms=timing_ms,
    )


@app.post("/refine/{session_id}", response_model=RefineResponse)
async def refine(session_id: str, request: RefineRequest, session: AsyncSession = Depends(get_session)):
    start = time.monotonic()
    state = await session_store.get(session, session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    current_filters = state.get("filters", {})
    persistent_filters = state.get("persistent_filters", [])

    new_filters, changes, clear_previous, intent_result, new_job_type, llm_specialization, is_role_spec, token_usage = await apply_refinement(
        request.query, current_filters, persistent_filters, history=state.get("query_history", [])
    )

    active_embedding = state.get("active_embedding", "inferred")
    intent = active_embedding
    if intent_result.confidence >= 0.4 and intent_result.intent != active_embedding:
        intent = intent_result.intent
        changes["embedding_switched"] = intent

    if clear_previous:
        new_filters.exclude_job_ids = state.get("previous_results_ids", [])

    # Use accumulated semantic query for vector search, not the refinement delta
    semantic_query = state.get("semantic_query", request.query)
    # If LLM detected user wants a new job type, update semantic query
    if new_job_type:
        semantic_query = new_job_type

    role_anchor = state.get("role_anchor")
    if new_job_type:
        role_anchor = extract_role_anchor(new_job_type)
    if role_anchor and new_filters.inferred_titles is None:
        new_filters.inferred_titles = role_anchor

    # Use LLM-extracted specialization if available, fallback to heuristic
    specialization_term = llm_specialization or _extract_specialization_term(request.query)
    if specialization_term and not new_job_type:
        role_anchor = state.get("role_anchor")
        title_count, desc_count = await _validate_specialization(
            session, specialization_term, new_filters, role=role_anchor[0] if role_anchor else None
        )
        
        # Determine where to apply the specialization based on match counts and LLM confidence
        if title_count > 0 or (is_role_spec and title_count == 0):
            # Prefer title match when LLM is confident it's a role specialization
            existing = new_filters.inferred_titles or []
            new_filters.inferred_titles = list({*existing, specialization_term})
            changes["specialization"] = specialization_term
            changes["specialization_source"] = "llm" if llm_specialization else "heuristic"
            changes["specialization_matched"] = "title" if title_count > 0 else "none"
        elif desc_count > 0:
            existing = new_filters.description_keywords or []
            new_filters.description_keywords = list({*existing, specialization_term})
            changes["specialization"] = specialization_term
            changes["specialization_source"] = "llm" if llm_specialization else "heuristic"
            changes["specialization_matched"] = "description"
        else:
            # No matches - still record the attempt for telemetry
            changes["specialization"] = specialization_term
            changes["specialization_source"] = "llm" if llm_specialization else "heuristic"
            changes["specialization_matched"] = "none"
            changes["specialization_zero_results"] = True

    top_k = 20
    search_k = max(top_k, 100) if settings.reranker_enabled else top_k

    try:
        results, analysis = await search_jobs(
            session=session,
            query=semantic_query,
            filters=new_filters,
            top_k=search_k,
            force_intent=intent,
        )
    except Exception:
        return embedding_unavailable()

    results = await _maybe_rerank(session, request.query, results)
    results = results[:top_k]

    state.update(
        {
            "role_anchor": role_anchor,
            "filters": new_filters.model_dump(),
            "query_history": (state.get("query_history", []) + [request.query]),
            "previous_results_ids": [r.id for r in results],
            "active_embedding": intent,
            "entities_history": (state.get("entities_history", []) + intent_result.entities),
            "persistent_filters": persistent_filters,
            "semantic_query": semantic_query,  # Preserve/update semantic query
        }
    )
    await session_store.save(session, session_id, state)

    timing_ms = int((time.monotonic() - start) * 1000)
    log_event(
        "refine",
        {
            "query": request.query,
            "intent": analysis["intent"],
            "filters": analysis["applied_filters"],
            "result_count": len(results),
            "timing_ms": timing_ms,
            "specialization_term": specialization_term,
            "specialization_source": changes.get("specialization_source"),
            "specialization_matched": changes.get("specialization_matched"),
            "is_role_specialization": is_role_spec,
            "llm_tokens": token_usage.to_dict() if token_usage else None,
        },
    )
    return RefineResponse(
        results=results,
        session_id=session_id,
        query_analysis=analysis,
        timing_ms=timing_ms,
        changes=changes,
    )


@app.post("/debug/intent", response_model=IntentDebugResponse)
async def debug_intent(payload: dict[str, str]):
    query = payload.get("query", "")
    intent_result = classify_intent(query)
    return IntentDebugResponse(
        query=query,
        intent=intent_result.intent,
        confidence=intent_result.confidence,
        entities=intent_result.entities,
        breakdown=intent_result.breakdown,
    )


@app.get("/explain", response_model=ExplainResponse)
async def explain(job_id: str, query: str, session: AsyncSession = Depends(get_session)):
    try:
        embedding = await embedding_service.get_embedding(session, query)
    except Exception:
        return embedding_unavailable()

    stmt = select(
        Job.id,
        (1 - Job.embedding_explicit.cosine_distance(embedding)).label("explicit"),
        (1 - Job.embedding_inferred.cosine_distance(embedding)).label("inferred"),
        (1 - Job.embedding_company.cosine_distance(embedding)).label("company"),
        Job.features,
    ).where(Job.id == job_id)

    result = await session.execute(stmt)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return ExplainResponse(
        job_id=job_id,
        scores={
            "explicit": max(0.0, float(row.explicit)),
            "inferred": max(0.0, float(row.inferred)),
            "company": max(0.0, float(row.company)),
        },
        metadata_matches=row.features,
        reranker_score=None,
    )
