import logging
from typing import Any

import psycopg

from app.core.config import settings

logger = logging.getLogger(__name__)


async def save_conversation_summary(
    thread_id: str,
    user_id: str,
    title: str,
) -> None:
    try:
        async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
            await conn.execute(
                """
                INSERT INTO conversation_summaries (thread_id, user_id, title)
                VALUES (%s, %s, %s)
                ON CONFLICT (thread_id) DO UPDATE SET updated_at = NOW()
                """,
                (thread_id, user_id, title),
            )
            await conn.commit()
    except Exception as e:
        logger.error(f"Failed to save conversation summary (thread={thread_id}): {e}")


async def create_referral(
    thread_id: str,
    user_id: str,
    groups: list[dict[str, Any]],
    formatted: dict[str, Any],
) -> str:
    """Insert a referral row and return its UUID string. Raises on failure."""
    merged_groups = []
    for group in groups:
        gid = str(group["group_id"])
        fmt = formatted.get(gid) or formatted.get(int(gid), {})
        merged_groups.append({
            **group,
            "rationale": fmt.get("rationale"),
            "service_ids": fmt.get("service_ids", []),
        })

    title = f"{groups[0].get('what', 'Search')} near {groups[0].get('where', 'unknown')}" if groups else "Search"

    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        row = await (
            await conn.execute(
                """
                INSERT INTO referrals (user_id, thread_id, title, groups)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (user_id, thread_id, title, psycopg.types.json.Jsonb(merged_groups)),
            )
        ).fetchone()
        await conn.commit()

    return str(row[0])
