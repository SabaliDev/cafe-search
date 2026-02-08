import os
import pytest
from httpx import AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_search_and_refine_flow():
    if not os.getenv("DATABASE_URL"):
        pytest.skip("DATABASE_URL not configured")

    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/search", json={"query": "data science jobs", "top_k": 5})
        assert resp.status_code in {200, 503}
        if resp.status_code != 200:
            return
        data = resp.json()
        session_id = data["session_id"]
        resp2 = await client.post(f"/refine/{session_id}", json={"query": "make it remote"})
        assert resp2.status_code in {200, 404, 503}
