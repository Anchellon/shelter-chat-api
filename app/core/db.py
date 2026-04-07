import logging

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
                ON CONFLICT (thread_id) DO NOTHING
                """,
                (thread_id, user_id, title[:80]),
            )
            await conn.commit()
    except Exception as e:
        logger.error(f"Failed to save conversation summary (thread={thread_id}): {e}")
