import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from openai import AsyncOpenAI
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import settings
from app.models import QueryEmbeddingCache
from app.logger import log_event


# OpenAI pricing per 1M tokens (as of 2025)
EMBEDDING_PRICING = {
    "text-embedding-3-small": 0.02,   # $0.02 per 1M tokens
    "text-embedding-3-large": 0.13,   # $0.13 per 1M tokens
    "text-embedding-ada-002": 0.10,   # $0.10 per 1M tokens
}


@dataclass
class EmbeddingTokenUsage:
    prompt_tokens: int
    total_tokens: int
    cost_usd: float
    
    @classmethod
    def from_response(cls, prompt_tokens: int, model: str):
        price_per_1m = EMBEDDING_PRICING.get(model, EMBEDDING_PRICING["text-embedding-3-small"])
        cost = (prompt_tokens / 1_000_000) * price_per_1m
        return cls(
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens,  # Embeddings only have input tokens
            cost_usd=round(cost, 8),
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


class EmbeddingService:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    @staticmethod
    def _hash_query(query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    async def get_embedding(self, session: AsyncSession, query: str) -> list[float]:
        query_hash = self._hash_query(query)
        ttl = datetime.now(timezone.utc) - timedelta(days=settings.embedding_cache_ttl_days)
        result = await session.execute(
            select(QueryEmbeddingCache).where(QueryEmbeddingCache.query_hash == query_hash)
        )
        cached = result.scalar_one_or_none()
        if cached and cached.created_at >= ttl:
            return list(cached.embedding)

        if not self._client:
            if cached:
                return list(cached.embedding)
            raise RuntimeError("OpenAI API key not configured")

        try:
            response = await self._client.embeddings.create(
                model=settings.openai_embedding_model,
                input=query,
            )
        except Exception:
            if cached:
                return list(cached.embedding)
            raise

        # Log token usage with cost
        if response.usage:
            token_usage = EmbeddingTokenUsage.from_response(
                prompt_tokens=response.usage.prompt_tokens,
                model=settings.openai_embedding_model,
            )
            log_event(
                "llm_token_usage",
                {
                    "model": settings.openai_embedding_model,
                    "operation": "embedding",
                    "prompt_tokens": token_usage.prompt_tokens,
                    "total_tokens": token_usage.total_tokens,
                    "cost_usd": token_usage.cost_usd,
                    "query_length": len(query),
                    "cached": False,
                },
            )

        embedding = response.data[0].embedding
        if cached:
            await session.execute(
                delete(QueryEmbeddingCache).where(QueryEmbeddingCache.query_hash == query_hash)
            )
        session.add(QueryEmbeddingCache(query_hash=query_hash, embedding=embedding))
        await session.commit()
        return embedding


embedding_service = EmbeddingService()
