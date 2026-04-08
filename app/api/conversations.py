import logging

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from app.core.auth import require_user
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
async def list_conversations(user_id: str = Depends(require_user)):
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        rows = await conn.execute(
            """
            SELECT thread_id, title
            FROM conversation_summaries
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT 50
            """,
            (user_id,),
        )
        conversations = [
            {"id": row[0], "title": row[1]}
            async for row in rows
        ]

    return {"conversations": conversations}


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str, user_id: str = Depends(require_user)):
    from app.main import agent_graph

    state = await agent_graph.aget_state({"configurable": {"thread_id": conversation_id}})

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Validate ownership
    stored_user_id = (state.metadata or {}).get("user_id")
    if stored_user_id and stored_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Convert messages to frontend format
    messages = []
    for m in state.values.get("messages", []):
        if isinstance(m, HumanMessage) and m.content:
            messages.append({
                "id": m.id or f"msg_{len(messages)}",
                "role": "user",
                "type": "text",
                "content": m.content,
            })
        elif isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            messages.append({
                "id": m.id or f"msg_{len(messages)}",
                "role": "assistant",
                "type": "text",
                "content": m.content,
            })

    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        referral_rows = await conn.execute(
            """
            SELECT id, title, saved, groups, created_at
            FROM referrals
            WHERE thread_id = %s AND user_id = %s
            ORDER BY created_at ASC
            """,
            (conversation_id, user_id),
        )
        referrals = [
            {
                "id": str(r[0]),
                "title": r[1],
                "saved": r[2],
                "groups": r[3],
                "created_at": r[4].isoformat(),
            }
            async for r in referral_rows
        ]

    return {
        "id": conversation_id,
        "messages": messages,
        "groups": state.values.get("groups", []),
        "formatted": state.values.get("formatted", {}),
        "referrals": referrals,
    }
