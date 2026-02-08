import enum
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
    Index,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class OrgType(str, enum.Enum):
    startup = "startup"
    enterprise = "enterprise"
    non_profit = "non-profit"
    government = "government"
    b_corp = "b-corp"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    company_name = Column(String, nullable=False)
    location = Column(String, nullable=True)
    remote = Column(Boolean, nullable=False, default=False)
    org_type = Column(Enum(OrgType, values_callable=lambda x: [e.value for e in x]), nullable=True)
    salary_min = Column(Integer, nullable=True)
    salary_max = Column(Integer, nullable=True)
    posted_at = Column(DateTime(timezone=True), nullable=True)
    apply_url = Column(String, nullable=True)

    embedding_explicit = Column(Vector(1536), nullable=False)
    embedding_inferred = Column(Vector(1536), nullable=False)
    embedding_company = Column(Vector(1536), nullable=False)

    features = Column(JSONB, nullable=False, default=dict)


class SessionState(Base):
    __tablename__ = "session_state"

    id = Column(String, primary_key=True)
    data = Column(JSONB, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class QueryEmbeddingCache(Base):
    __tablename__ = "query_embedding_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_hash = Column(String, nullable=False, unique=True)
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


class RerankCache(Base):
    __tablename__ = "rerank_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_hash = Column(String, nullable=False)
    job_id = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


Index("ix_jobs_company_name", Job.company_name)
Index("ix_jobs_remote", Job.remote)
Index("ix_jobs_posted_at", Job.posted_at)
Index("ix_jobs_org_type", Job.org_type)
Index("ix_jobs_salary_min", Job.salary_min)
Index("ix_jobs_salary_max", Job.salary_max)
Index("ix_rerank_cache_query_job", RerankCache.query_hash, RerankCache.job_id, unique=True)
