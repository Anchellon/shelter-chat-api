import json
import logging
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from app.agent.runner import stream_resume
from app.core.auth import require_user
from app.core.db import create_referral, save_conversation_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ResumeRequest(BaseModel):
    conversation_id: str
    action: str = "submit"      # "submit" | "cancel"
    answers: dict = {}          # {"what": "sfsg-shelter", "who": ["Adults"], "where": "Tenderloin"}


async def _sse_resume_generator(request: ResumeRequest, graph, config: dict):
    logger.info(f"SSE resume — conv={request.conversation_id}, action={request.action}")

    msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    has_text = False

    try:
        async for event in stream_resume(request, graph, config):
            if event["type"] == "text":
                if not has_text:
                    yield f"data: {json.dumps({'type': 'text-start', 'id': msg_id})}\n\n"
                    has_text = True
                yield f"data: {json.dumps({'type': 'text-delta', 'id': msg_id, 'delta': event['content']})}\n\n"
            elif event["type"] == "groups_identified":
                yield f"data: {json.dumps({'type': 'groups_identified', 'groups': event['groups']})}\n\n"
            elif event["type"] == "format_complete":
                formatted = event["formatted"]
                groups = event.get("groups", [])
                referral_id = await create_referral(
                    thread_id=request.conversation_id,
                    user_id=config["metadata"]["user_id"],
                    groups=groups,
                    formatted=formatted,
                )
                yield f"data: {json.dumps({'type': 'format_complete', 'formatted': formatted, 'groups': groups, 'referral_id': referral_id})}\n\n"
                title = f"{groups[0].get('what', 'Search')} near {groups[0].get('where', 'unknown')}" if groups else "Search"
                await save_conversation_summary(
                    thread_id=request.conversation_id,
                    user_id=config["metadata"]["user_id"],
                    title=title,
                )
            elif event["type"] == "intake_request":
                yield f"data: {json.dumps(event)}\n\n"
                return
            elif event["type"] == "tool_start":
                yield f"data: {json.dumps({'type': 'tool-start', 'tool': event['tool'], 'status': event['status']})}\n\n"
            elif event["type"] == "tool_end":
                yield f"data: {json.dumps({'type': 'tool-end', 'tool': event['tool']})}\n\n"
    except Exception as e:
        logger.error(f"Resume stream error (conv={request.conversation_id}): {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'errorText': str(e)})}\n\n"
        return

    if has_text:
        yield f"data: {json.dumps({'type': 'text-end', 'id': msg_id})}\n\n"
    yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'stop'})}\n\n"


@router.post("/resume")
async def resume(
    request: ResumeRequest,
    user_id: str = Depends(require_user),
):
    from app.main import agent_graph

    config = {
        "configurable": {"thread_id": request.conversation_id},
        "metadata": {"user_id": user_id},
    }

    return StreamingResponse(
        _sse_resume_generator(request, agent_graph, config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Conversation-Id": request.conversation_id,
        },
    )
