"""create jobs schema

Revision ID: 0001_create_jobs
Revises: 
Create Date: 2026-02-07
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


revision = "0001_create_jobs"
down_revision = None
branch_labels = None
depends_on = None


org_type_enum = postgresql.ENUM(
    "startup",
    "enterprise",
    "non-profit",
    "government",
    "b-corp",
    name="orgtype",
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    org_type_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("company_name", sa.String(), nullable=False),
        sa.Column("location", sa.String(), nullable=True),
        sa.Column("remote", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("org_type", postgresql.ENUM(name="orgtype", create_type=False), nullable=True),
        sa.Column("salary_min", sa.Integer(), nullable=True),
        sa.Column("salary_max", sa.Integer(), nullable=True),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("apply_url", sa.String(), nullable=True),
        sa.Column("embedding_explicit", Vector(1536), nullable=False),
        sa.Column("embedding_inferred", Vector(1536), nullable=False),
        sa.Column("embedding_company", Vector(1536), nullable=False),
        sa.Column("features", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )

    op.create_index("ix_jobs_company_name", "jobs", ["company_name"])
    op.create_index("ix_jobs_remote", "jobs", ["remote"])
    op.create_index("ix_jobs_posted_at", "jobs", ["posted_at"])
    op.create_index("ix_jobs_org_type", "jobs", ["org_type"])
    op.create_index("ix_jobs_salary_min", "jobs", ["salary_min"])
    op.create_index("ix_jobs_salary_max", "jobs", ["salary_max"])

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_jobs_embedding_explicit
        ON jobs USING hnsw (embedding_explicit vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_jobs_embedding_inferred
        ON jobs USING hnsw (embedding_inferred vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_jobs_embedding_company
        ON jobs USING hnsw (embedding_company vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )

    op.create_table(
        "session_state",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("data", postgresql.JSONB(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "query_embedding_cache",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("query_hash", sa.String(), nullable=False, unique=True),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "rerank_cache",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("query_hash", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_index(
        "ix_rerank_cache_query_job",
        "rerank_cache",
        ["query_hash", "job_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_rerank_cache_query_job", table_name="rerank_cache")
    op.drop_table("rerank_cache")
    op.drop_table("query_embedding_cache")
    op.drop_table("session_state")

    op.execute("DROP INDEX IF EXISTS ix_jobs_embedding_company")
    op.execute("DROP INDEX IF EXISTS ix_jobs_embedding_inferred")
    op.execute("DROP INDEX IF EXISTS ix_jobs_embedding_explicit")

    op.drop_index("ix_jobs_salary_max", table_name="jobs")
    op.drop_index("ix_jobs_salary_min", table_name="jobs")
    op.drop_index("ix_jobs_org_type", table_name="jobs")
    op.drop_index("ix_jobs_posted_at", table_name="jobs")
    op.drop_index("ix_jobs_remote", table_name="jobs")
    op.drop_index("ix_jobs_company_name", table_name="jobs")

    op.drop_table("jobs")

    org_type_enum.drop(op.get_bind(), checkfirst=True)
    op.execute("DROP EXTENSION IF EXISTS vector")
