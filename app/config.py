from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load .env from the root directory
load_dotenv(os.path.join(os.getcwd(), ".env"))


class Settings(BaseModel):
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@db:5432/cafesearch",
    )
    redis_url: str | None = os.getenv("REDIS_URL", "redis://redis:6379/0")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
    embedding_cache_ttl_days: int = int(os.getenv("EMBEDDING_CACHE_TTL_DAYS", "30"))
    rerank_cache_ttl_days: int = int(os.getenv("RERANK_CACHE_TTL_DAYS", "7"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
print(f"DEBUG: Loaded DATABASE_URL: {settings.database_url}")
