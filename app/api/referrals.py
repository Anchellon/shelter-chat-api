import logging
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth import require_user
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/referrals", tags=["referrals"])


class UpdateReferralRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    saved: bool | None = None


class CreateReferralRequest(BaseModel):
    thread_id: str
    title: str | None = None
    groups: list[dict]
    formatted: dict[str, dict]  # {group_id: {rationale, service_ids}}


@router.post("", status_code=201)
async def create_referral(request: CreateReferralRequest, user_id: str = Depends(require_user)):
    # Merge each group with its formatted result into a flat object
    merged_groups = []
    for group in request.groups:
        gid = str(group["group_id"])
        fmt = request.formatted.get(gid) or request.formatted.get(int(gid), {})
        merged_groups.append({
            **group,
            "rationale": fmt.get("rationale"),
            "service_ids": fmt.get("service_ids", []),
        })

    title = request.title or f"{request.groups[0].get('what', 'Search')} near {request.groups[0].get('where', 'unknown')}"

    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        row = await (
            await conn.execute(
                """
                INSERT INTO referrals (user_id, thread_id, title, groups)
                VALUES (%s, %s, %s, %s)
                RETURNING id, title, saved, created_at
                """,
                (user_id, request.thread_id, title, psycopg.types.json.Jsonb(merged_groups)),
            )
        ).fetchone()
        await conn.commit()

    return {
        "id": str(row[0]),
        "title": row[1],
        "saved": row[2],
        "created_at": row[3].isoformat(),
    }


@router.patch("/{referral_id}")
async def update_referral(referral_id: UUID, request: UpdateReferralRequest, user_id: str = Depends(require_user)):
    fields = {k: v for k, v in request.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=422, detail="No fields to update")

    set_clause = ", ".join(f"{k} = %s" for k in fields)
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        result = await conn.execute(
            f"UPDATE referrals SET {set_clause} WHERE id = %s AND user_id = %s",
            (*fields.values(), str(referral_id), user_id),
        )
        await conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Referral not found")

    return {"id": str(referral_id), **fields}


@router.get("")
async def list_referrals(user_id: str = Depends(require_user)):
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        rows = await conn.execute(
            """
            SELECT id, thread_id, title, saved,
                   jsonb_agg(
                       jsonb_set(grp, '{service_count}', to_jsonb(jsonb_array_length(grp->'service_ids')))
                       - 'service_ids'
                   ) AS groups,
                   created_at
            FROM referrals,
                 jsonb_array_elements(groups) AS grp
            WHERE user_id = %s AND saved = TRUE
            GROUP BY id, thread_id, title, saved, created_at
            ORDER BY created_at DESC
            LIMIT 50
            """,
            (user_id,),
        )
        referrals = [
            {
                "id": str(row[0]),
                "thread_id": row[1],
                "title": row[2],
                "saved": row[3],
                "groups": row[4],
                "created_at": row[5].isoformat(),
            }
            async for row in rows
        ]

    return {"referrals": referrals}


@router.get("/{referral_id}")
async def get_referral(referral_id: UUID, user_id: str = Depends(require_user)):
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        row = await (
            await conn.execute(
                """
                SELECT id, thread_id, title, saved, groups, created_at
                FROM referrals
                WHERE id = %s AND user_id = %s
                """,
                (str(referral_id), user_id),
            )
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Referral not found")

    return {
        "id": str(row[0]),
        "thread_id": row[1],
        "title": row[2],
        "saved": row[3],
        "groups": row[4],
        "created_at": row[5].isoformat(),
    }



@router.delete("/{referral_id}", status_code=204)
async def delete_referral(referral_id: UUID, user_id: str = Depends(require_user)):
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        result = await conn.execute(
            "DELETE FROM referrals WHERE id = %s AND user_id = %s",
            (str(referral_id), user_id),
        )
        await conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Referral not found")
