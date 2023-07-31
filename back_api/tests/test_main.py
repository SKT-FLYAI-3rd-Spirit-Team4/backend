import pytest
from httpx import AsyncClient

from ..main import app


@pytest.mark.anyio
async def test_root():
    async with AsyncClient(app=app, base_url="http://localhost") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
