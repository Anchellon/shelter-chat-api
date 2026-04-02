import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from app.agent.runner import stream_resume
from app.core.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ResumeRequest(BaseModel):
    conversation_id: str
    action: str = "submit"      # "submit" | "cancel"
    answers: dict = {}          # {"what": "sfsg-shelter", "who": ["Adults"], "where": "Tenderloin"}


async def _sse_resume_generator(request: ResumeRequest, graph):
    from app.agent.runner import stream_resume
    logger.info(f"SSE resume — conv={request.conversation_id}, action={request.action}")

    try:
        async for event in stream_resume(request, graph):
            if event["type"] == "text":
                yield f"data: {json.dumps({'type': 'text-delta', 'delta': event['content']})}\n\n"
            elif event["type"] == "groups_identified":
                yield f"data: {json.dumps({'type': 'groups_identified', 'groups': event['groups']})}\n\n"
            elif event["type"] == "format_complete":
                yield f"data: {json.dumps({'type': 'format_complete', 'formatted': event['formatted']})}\n\n"
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

    yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'stop'})}\n\n"


@router.post("/resume")
async def resume(
    request: ResumeRequest,
    _: str = Depends(require_api_key),
):
    from app.main import agent_graph

    return StreamingResponse(
        _sse_resume_generator(request, agent_graph),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Conversation-Id": request.conversation_id,
        },
    )
