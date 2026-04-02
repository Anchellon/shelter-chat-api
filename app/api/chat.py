import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.runner import stream_agent
from app.core.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    conversation_id: str | None = None   # if None, a new conversation is started
    message: str
    current_time: str | None = None      # e.g. "Monday 14:30" — sent by frontend, used as default when no "when" in query


async def _sse_generator(question: str, conversation_id: str, current_time: str, graph):
    msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    chunk_count = 0

    logger.info(f"SSE start — msg={msg_id}, conv={conversation_id}")
    yield f"data: {json.dumps({'type': 'text-start', 'id': msg_id})}\n\n"

    try:
        async for event in stream_agent(question, conversation_id, current_time, graph):
            if event["type"] == "text":
                chunk_count += 1
                yield f"data: {json.dumps({'type': 'text-delta', 'id': msg_id, 'delta': event['content']})}\n\n"

            elif event["type"] == "tool_start":
                # Visible status indicator — frontend renders this between text chunks
                yield f"data: {json.dumps({'type': 'tool-start', 'tool': event['tool'], 'status': event['status']})}\n\n"

            elif event["type"] == "tool_end":
                yield f"data: {json.dumps({'type': 'tool-end', 'tool': event['tool']})}\n\n"

            elif event["type"] == "groups_identified":
                yield f"data: {json.dumps({'type': 'groups_identified', 'groups': event['groups']})}\n\n"

            elif event["type"] == "format_complete":
                yield f"data: {json.dumps({'type': 'format_complete', 'formatted': event['formatted']})}\n\n"

            elif event["type"] == "intake_request":
                yield f"data: {json.dumps(event)}\n\n"
                return  # stream ends here — frontend resumes via POST /chat/resume

    except Exception as e:
        logger.error(f"Stream error (conv={conversation_id}): {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'errorText': str(e)})}\n\n"
        return

    logger.info(f"SSE end — msg={msg_id}, {chunk_count} chunks")
    yield f"data: {json.dumps({'type': 'text-end', 'id': msg_id})}\n\n"
    yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'stop'})}\n\n"


@router.post("")
async def chat(
    request: ChatRequest,
    _: str = Depends(require_api_key),
):
    # Import here to avoid circular import at module load time
    from app.main import agent_graph

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    from datetime import datetime
    conversation_id = request.conversation_id or str(uuid.uuid4())
    current_time = request.current_time or datetime.now().strftime("%A %H:%M")
    logger.info(f"POST /chat — conv={conversation_id}, msg='{request.message[:80]}'")

    return StreamingResponse(
        _sse_generator(request.message, conversation_id, current_time, agent_graph),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Conversation-Id": conversation_id,
        },
    )
