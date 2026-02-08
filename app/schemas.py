from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    remote: bool | None = None
    org_types: list[str] | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    location: str | None = None
    country: str | None = None
    experience_level: str | None = None
    exclude_job_ids: list[str] | None = None
    mission_keywords: list[str] | None = None
    company_industry: list[str] | None = None
    inferred_titles: list[str] | None = None
    text_keywords: list[str] | None = None
    title_keywords: list[str] | None = None
    description_keywords: list[str] | None = None
    exclude_title_keywords: list[str] | None = None
    exclude_company_names: list[str] | None = None
    title_keywords: list[str] | None = Field(
        default=None,
        description="Keywords to match specifically in job titles (e.g., 'tax', 'audit')"
    )
    description_keywords: list[str] | None = Field(
        default=None,
        description="Keywords to match specifically in job descriptions (e.g., 'payroll processing', 'GAAP')"
    )
    exclude_title_keywords: list[str] | None = Field(
        default=None,
        description="Keywords to exclude from job titles (e.g., 'senior', 'lead')"
    )
    exclude_company_names: list[str] | None = Field(
        default=None,
        description="Company names to exclude (e.g., 'crypto', 'gambling companies')"
    )


class SearchRequest(BaseModel):
    query: str
    session_id: str | None = None
    filters: SearchFilters | None = None
    top_k: int = Field(default=20, ge=1, le=100)


class JobResult(BaseModel):
    id: str
    title: str
    company_name: str
    location: str | None
    remote: bool
    org_type: str | None
    salary_min: int | None
    salary_max: int | None
    posted_at: datetime | None
    apply_url: str | None
    score: float
    explanation: dict[str, Any]


class QueryAnalysis(BaseModel):
    intent: str
    intent_confidence: float
    extracted_entities: list[str]
    applied_filters: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[JobResult]
    session_id: str
    query_analysis: QueryAnalysis
    timing_ms: int


class RefineRequest(BaseModel):
    query: str


class RefineResponse(SearchResponse):
    changes: dict[str, Any]


class IntentDebugResponse(BaseModel):
    query: str
    intent: str
    confidence: float
    entities: list[str]
    breakdown: dict[str, Any]


class ExplainResponse(BaseModel):
    job_id: str
    scores: dict[str, float]
    metadata_matches: dict[str, Any]
    reranker_score: float | None
