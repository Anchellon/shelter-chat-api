"""
Integration tests for the chat API.
Requires the MCP server and Postgres to be running.
Set env vars before running — see .env.example
"""
import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
async def client():
    from app.main import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_health(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


async def test_chat_empty_message_rejected(client: AsyncClient):
    response = await client.post("/api/v1/chat", json={"message": "   "})
    assert response.status_code == 400


async def test_chat_generates_conversation_id(client: AsyncClient):
    response = await client.post(
        "/api/v1/chat",
        json={"message": "I need emergency shelter tonight"},
    )
    assert response.status_code == 200
    assert "x-conversation-id" in response.headers


async def test_chat_respects_provided_conversation_id(client: AsyncClient):
    conv_id = "test-conv-123"
    response = await client.post(
        "/api/v1/chat",
        json={"conversation_id": conv_id, "message": "hello"},
    )
    assert response.status_code == 200
    assert response.headers["x-conversation-id"] == conv_id


async def test_chat_returns_sse_content_type(client: AsyncClient):
    response = await client.post(
        "/api/v1/chat",
        json={"message": "where can I get food?"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
