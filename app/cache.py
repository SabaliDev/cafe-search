"""Caching utilities for specialization validation and other queries."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis
from app.config import settings


class SpecializationCache:
    """Redis-backed cache for specialization validation counts.
    
    Key format: spec:{role}:{specialization_hash}
    Value: {"title_count": int, "desc_count": int, "timestamp": str}
    TTL: 1 hour (3600 seconds)
    """
    
    DEFAULT_TTL = 3600  # 1 hour
    
    def __init__(self, url: str | None = None) -> None:
        self._url = url or settings.redis_url
        self._client: redis.Redis | None = None
        if self._url:
            try:
                self._client = redis.from_url(self._url, decode_responses=True)
            except Exception:
                self._client = None
    
    def _make_key(self, role: str | None, specialization: str) -> str:
        """Create cache key from role and specialization."""
        import hashlib
        role_part = role or "any"
        spec_hash = hashlib.sha256(specialization.lower().encode()).hexdigest()[:16]
        return f"spec:{role_part}:{spec_hash}"
    
    async def get(self, role: str | None, specialization: str) -> tuple[int, int] | None:
        """Get cached counts for a specialization.
        
        Returns:
            Tuple of (title_count, desc_count) or None if not cached.
        """
        if not self._client:
            return None
        try:
            key = self._make_key(role, specialization)
            data = await self._client.get(key)
            if data:
                parsed = json.loads(data)
                return (parsed["title_count"], parsed["desc_count"])
        except Exception:
            pass
        return None
    
    async def set(
        self, 
        role: str | None, 
        specialization: str, 
        title_count: int, 
        desc_count: int,
        ttl: int = DEFAULT_TTL
    ) -> None:
        """Cache counts for a specialization."""
        if not self._client:
            return
        try:
            key = self._make_key(role, specialization)
            value = json.dumps({
                "title_count": title_count,
                "desc_count": desc_count,
            })
            await self._client.setex(key, ttl, value)
        except Exception:
            pass
    
    async def invalidate(self, role: str | None, specialization: str) -> None:
        """Invalidate cache entry for a specialization."""
        if not self._client:
            return
        try:
            key = self._make_key(role, specialization)
            await self._client.delete(key)
        except Exception:
            pass


# Global cache instance
specialization_cache = SpecializationCache()
