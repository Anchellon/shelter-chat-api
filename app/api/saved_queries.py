import logging
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.core.auth import require_user
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/saved-queries", tags=["saved-queries"])


class SaveQueryRequest(BaseModel):
    thread_id: str
    group_id: int
    title: str | None = None


@router.post("", status_code=201)
async def save_query(request: SaveQueryRequest, user_id: str = Depends(require_user)):
    from app.main import agent_graph

    state = await agent_graph.aget_state({"configurable": {"thread_id": request.thread_id}})

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Conversation not found")

    stored_user_id = (state.metadata or {}).get("user_id")
    if stored_user_id and stored_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    groups: list[dict] = state.values.get("groups", [])
    group = next((g for g in groups if g["group_id"] == request.group_id), None)
    if group is None:
        raise HTTPException(status_code=404, detail="Group not found in conversation")

    formatted: dict = state.values.get("formatted", {})
    group_result = formatted.get(str(request.group_id)) or formatted.get(request.group_id)
    if group_result is None:
        raise HTTPException(status_code=404, detail="No results found for this group")

    service_ids: list[int] = group_result.get("service_ids", [])
    rationale: str | None = group_result.get("rationale")

    title = request.title or f"{group.get('what', 'Search')} near {group.get('where', 'unknown location')}"

    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        row = await (
            await conn.execute(
                """
                INSERT INTO saved_queries (user_id, thread_id, group_id, title, group_data, rationale, service_ids)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (user_id, request.thread_id, request.group_id, title, psycopg.types.json.Jsonb(group), rationale, service_ids),
            )
        ).fetchone()
        await conn.commit()

    return {"id": str(row[0]), "title": title, "created_at": row[1].isoformat()}


@router.get("")
async def list_saved_queries(user_id: str = Depends(require_user)):
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        rows = await conn.execute(
            """
            SELECT id, thread_id, group_id, title, group_data, rationale, array_length(service_ids, 1), created_at
            FROM saved_queries
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 50
            """,
            (user_id,),
        )
        saved = [
            {
                "id": str(row[0]),
                "thread_id": row[1],
                "group_id": row[2],
                "title": row[3],
                "group": row[4],
                "rationale": row[5],
                "service_count": row[6] or 0,
                "created_at": row[7].isoformat(),
            }
            async for row in rows
        ]

    return {"saved_queries": saved}


@router.get("/{saved_query_id}")
async def get_saved_query(saved_query_id: UUID, user_id: str = Depends(require_user)):
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        row = await (
            await conn.execute(
                """
                SELECT id, thread_id, group_id, title, group_data, rationale, service_ids, created_at
                FROM saved_queries
                WHERE id = %s AND user_id = %s
                """,
                (str(saved_query_id), user_id),
            )
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Saved query not found")

    return {
        "id": str(row[0]),
        "thread_id": row[1],
        "group_id": row[2],
        "title": row[3],
        "group": row[4],
        "rationale": row[5],
        "service_ids": row[6],
        "created_at": row[7].isoformat(),
    }


@router.delete("/{saved_query_id}", status_code=204)
async def delete_saved_query(saved_query_id: UUID, user_id: str = Depends(require_user)):
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        result = await conn.execute(
            "DELETE FROM saved_queries WHERE id = %s AND user_id = %s",
            (str(saved_query_id), user_id),
        )
        await conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Saved query not found")
