from __future__ import annotations

from typing import Any

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings import embedding_service
from app.intent import classify_intent
from app.models import Job, OrgType
from app.schemas import SearchFilters, JobResult


EMBEDDING_COLUMN = {
    "explicit": Job.embedding_explicit,
    "inferred": Job.embedding_inferred,
    "company": Job.embedding_company,
}


async def search_jobs(
    session: AsyncSession,
    query: str,
    filters: SearchFilters | None,
    top_k: int,
    force_intent: str | None = None,
) -> tuple[list[JobResult], dict[str, Any]]:
    intent_result = classify_intent(query)
    intent = force_intent or intent_result.intent
    embedding_col = EMBEDDING_COLUMN[intent]

    query_embedding = await embedding_service.get_embedding(session, query)
    distance = embedding_col.cosine_distance(query_embedding).label("distance")

    clauses = []
    applied_filters: dict[str, Any] = {}
    if filters:
        if filters.remote is not None:
            clauses.append(Job.remote == filters.remote)
            applied_filters["remote"] = filters.remote
        if filters.org_types:
            normalized = []
            for value in filters.org_types:
                value = value.lower().strip()
                if value == "nonprofit":
                    value = "non-profit"
                normalized.append(value)
            valid = [OrgType(v) for v in normalized if v in OrgType._value2member_map_]
            if valid:
                clauses.append(Job.org_type.in_(valid))
                applied_filters["org_types"] = [v.value for v in valid]
        if filters.salary_min is not None:
            clauses.append(Job.salary_max.is_(None) | (Job.salary_max >= filters.salary_min))
            applied_filters["salary_min"] = filters.salary_min
        if filters.salary_max is not None:
            clauses.append(Job.salary_min.is_(None) | (Job.salary_min <= filters.salary_max))
            applied_filters["salary_max"] = filters.salary_max
        if filters.location:
            loc = filters.location.strip()
            loc_lower = loc.lower()
            if loc_lower in {"usa", "u.s.a", "united states", "united states of america"}:
                loc = "US"

            if loc == "US":
                # Avoid false positives like "Colusa" when filtering for US
                us_patterns = [
                    "%, US%",
                    "% US,%",
                    "% US %",
                    "% US)%",
                ]
                clauses.append(
                    or_(
                        *[Job.location.ilike(p) for p in us_patterns],
                        *[Job.title.ilike(p) for p in us_patterns],
                    )
                )
            else:
                clauses.append(
                    or_(
                        Job.location.ilike(f"%{loc}%"),
                        Job.title.ilike(f"%{loc}%")
                    )
                )
            applied_filters["location"] = loc
        if filters.exclude_job_ids:
            clauses.append(Job.id.notin_(filters.exclude_job_ids))
            applied_filters["exclude_job_ids"] = filters.exclude_job_ids
        if filters.mission_keywords:
            clauses.append(Job.features["mission_keywords"].contains(filters.mission_keywords))
            applied_filters["mission_keywords"] = filters.mission_keywords
        if filters.company_industry:
            clauses.append(Job.features["company_industry"].contains(filters.company_industry))
            applied_filters["company_industry"] = filters.company_industry
        if filters.inferred_titles:
            title_filters = []
            for title in filters.inferred_titles:
                title_filters.append(Job.features["inferred_titles"].contains([title]))
                title_filters.append(Job.title.ilike(f"%{title}%"))
            clauses.append(or_(*title_filters))
            applied_filters["inferred_titles"] = filters.inferred_titles
        if filters.text_keywords:
            keyword_filters = []
            for keyword in filters.text_keywords:
                keyword_filters.append(Job.title.ilike(f"%{keyword}%"))
                keyword_filters.append(Job.description.ilike(f"%{keyword}%"))
            clauses.append(or_(*keyword_filters))
            applied_filters["text_keywords"] = filters.text_keywords
        if filters.title_keywords:
            title_clauses = []
            for kw in filters.title_keywords:
                title_clauses.append(Job.title.ilike(f"%{kw}%"))
            clauses.append(or_(*title_clauses))
            applied_filters["title_keywords"] = filters.title_keywords
        if filters.description_keywords:
            desc_clauses = []
            for kw in filters.description_keywords:
                desc_clauses.append(Job.description.ilike(f"%{kw}%"))
            clauses.append(or_(*desc_clauses))
            applied_filters["description_keywords"] = filters.description_keywords
        # Negative constraints - exclusions
        if filters.exclude_title_keywords:
            for kw in filters.exclude_title_keywords:
                clauses.append(Job.title.not_ilike(f"%{kw}%"))
            applied_filters["exclude_title_keywords"] = filters.exclude_title_keywords
        if filters.exclude_company_names:
            for kw in filters.exclude_company_names:
                clauses.append(Job.company_name.not_ilike(f"%{kw}%"))
            applied_filters["exclude_company_names"] = filters.exclude_company_names
        if filters.experience_level:
            level = filters.experience_level.lower()
            if level == "senior":
                clauses.append(or_(Job.title.ilike("%senior%"), Job.title.ilike("%lead%"), Job.title.ilike("%principal%")))
            elif level == "junior":
                clauses.append(or_(Job.title.ilike("%junior%"), Job.title.ilike("%entry%")))
            applied_filters["experience_level"] = filters.experience_level

    stmt = (
        select(
            Job,
            distance,
        )
        .where(and_(*clauses))
        .order_by(distance)
        .limit(top_k)
    )

    result = await session.execute(stmt)
    rows = result.all()

    results: list[JobResult] = []
    for job, dist in rows:
        score = max(0.0, 1.0 - float(dist))
        explanation = {
            "intent": intent,
            "matched_filters": applied_filters,
        }
        results.append(
            JobResult(
                id=job.id,
                title=job.title,
                company_name=job.company_name,
                location=job.location,
                remote=job.remote,
                org_type=job.org_type.value if job.org_type else None,
                salary_min=job.salary_min,
                salary_max=job.salary_max,
                posted_at=job.posted_at,
                apply_url=job.apply_url,
                score=score,
                explanation=explanation,
            )
        )

    analysis = {
        "intent": intent,
        "intent_confidence": intent_result.confidence,
        "extracted_entities": intent_result.entities,
        "applied_filters": applied_filters,
    }
    return results, analysis
