import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import RerankCache

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional
    CrossEncoder = None


class Reranker:
    def __init__(self) -> None:
        self._model = None
        if settings.reranker_enabled and CrossEncoder:
            self._model = CrossEncoder(settings.reranker_model)

    @staticmethod
    def _hash_query(query: str) -> str:
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    async def rerank(
        self, session: AsyncSession, query: str, pairs: Sequence[tuple[str, str]]
    ) -> dict[str, float]:
        if not self._model:
            return {}

        query_hash = self._hash_query(query)
        ttl = datetime.utcnow() - timedelta(days=settings.rerank_cache_ttl_days)
        cached_scores: dict[str, float] = {}

        result = await session.execute(
            select(RerankCache).where(RerankCache.query_hash == query_hash)
        )
        for row in result.scalars().all():
            if row.created_at >= ttl:
                cached_scores[row.job_id] = row.score

        missing = [(job_id, text) for job_id, text in pairs if job_id not in cached_scores]
        if missing:
            texts = [(query, text) for _, text in missing]
            scores = await asyncio.get_event_loop().run_in_executor(
                None, self._model.predict, texts
            )
            for (job_id, _), score in zip(missing, scores):
                cached_scores[job_id] = float(score)
                session.add(
                    RerankCache(query_hash=query_hash, job_id=job_id, score=float(score))
                )
            await session.commit()

        return cached_scores


reranker = Reranker()
