import logging

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.auth import require_user
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/saved-services", tags=["saved-services"])


class SaveServiceRequest(BaseModel):
    service_id: int


@router.post("", status_code=201)
async def save_service(request: SaveServiceRequest, user_id: str = Depends(require_user)):
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        try:
            row = await (
                await conn.execute(
                    """
                    INSERT INTO saved_services (user_id, service_id)
                    VALUES (%s, %s)
                    RETURNING id, service_id, created_at
                    """,
                    (user_id, request.service_id),
                )
            ).fetchone()
            await conn.commit()
        except psycopg.errors.UniqueViolation:
            raise HTTPException(status_code=409, detail="Service already saved")

    return {
        "id": str(row[0]),
        "service_id": row[1],
        "created_at": row[2].isoformat(),
    }


@router.delete("/{service_id}", status_code=204)
async def unsave_service(service_id: int, user_id: str = Depends(require_user)):
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        result = await conn.execute(
            "DELETE FROM saved_services WHERE user_id = %s AND service_id = %s",
            (user_id, service_id),
        )
        await conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Saved service not found")


@router.get("")
async def list_saved_services(
    user_id: str = Depends(require_user),
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        rows = await (
            await conn.execute(
                """
                SELECT service_id, created_at
                FROM saved_services
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (user_id, limit + 1, offset),
            )
        ).fetchall()

    has_more = len(rows) > limit
    rows = rows[:limit]

    if not rows:
        return {"services": [], "has_more": False}

    from app.main import mcp_client

    if mcp_client is None:
        raise HTTPException(status_code=503, detail="MCP client not available")

    service_ids = [row[0] for row in rows]
    saved_at_map = {row[0]: row[1].isoformat() for row in rows}

    try:
        details = await mcp_client.invoke(
            "get_service_details_batch",
            {"service_ids": service_ids},
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    for svc in details:
        sid = svc.get("service_id")
        if sid in saved_at_map:
            svc["saved_at"] = saved_at_map[sid]

    return {"services": details, "has_more": has_more}
