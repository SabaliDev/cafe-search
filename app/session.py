import json
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import redis.asyncio as redis
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import SessionState


class SessionStore:
    async def create(self, session: AsyncSession) -> str:
        raise NotImplementedError

    async def get(self, session: AsyncSession, session_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def save(self, session: AsyncSession, session_id: str, data: dict[str, Any]) -> None:
        raise NotImplementedError

    async def touch(self, session: AsyncSession, session_id: str) -> None:
        raise NotImplementedError


class RedisSessionStore(SessionStore):
    def __init__(self, url: str) -> None:
        self._client = redis.from_url(url, decode_responses=True)

    async def create(self, session: AsyncSession) -> str:
        session_id = str(uuid4())
        await self._client.setex(session_id, settings.session_ttl_seconds, json.dumps(self._empty_state()))
        return session_id

    async def get(self, session: AsyncSession, session_id: str) -> dict[str, Any] | None:
        payload = await self._client.get(session_id)
        if not payload:
            return None
        return json.loads(payload)

    async def save(self, session: AsyncSession, session_id: str, data: dict[str, Any]) -> None:
        await self._client.setex(session_id, settings.session_ttl_seconds, json.dumps(data))

    async def touch(self, session: AsyncSession, session_id: str) -> None:
        await self._client.expire(session_id, settings.session_ttl_seconds)

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "filters": {},
            "query_history": [],
            "previous_results_ids": [],
            "active_embedding": "inferred",
            "entities_history": [],
            "persistent_filters": [],
        }


class DbSessionStore(SessionStore):
    async def create(self, session: AsyncSession) -> str:
        session_id = str(uuid4())
        expires_at = datetime.utcnow() + timedelta(seconds=settings.session_ttl_seconds)
        session.add(SessionState(id=session_id, data=self._empty_state(), expires_at=expires_at))
        await session.commit()
        return session_id

    async def get(self, session: AsyncSession, session_id: str) -> dict[str, Any] | None:
        await self._cleanup(session)
        result = await session.execute(
            select(SessionState).where(SessionState.id == session_id)
        )
        state = result.scalar_one_or_none()
        if not state:
            return None
        return state.data

    async def save(self, session: AsyncSession, session_id: str, data: dict[str, Any]) -> None:
        expires_at = datetime.utcnow() + timedelta(seconds=settings.session_ttl_seconds)
        result = await session.execute(
            select(SessionState).where(SessionState.id == session_id)
        )
        state = result.scalar_one_or_none()
        if state:
            state.data = data
            state.expires_at = expires_at
            state.updated_at = datetime.utcnow()
        else:
            session.add(SessionState(id=session_id, data=data, expires_at=expires_at))
        await session.commit()

    async def touch(self, session: AsyncSession, session_id: str) -> None:
        await self.save(session, session_id, (await self.get(session, session_id)) or self._empty_state())

    async def _cleanup(self, session: AsyncSession) -> None:
        await session.execute(delete(SessionState).where(SessionState.expires_at < datetime.utcnow()))
        await session.commit()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "filters": {},
            "query_history": [],
            "previous_results_ids": [],
            "active_embedding": "inferred",
            "entities_history": [],
            "persistent_filters": [],
        }


def get_session_store() -> SessionStore:
    if settings.redis_url:
        return RedisSessionStore(settings.redis_url)
    return DbSessionStore()
