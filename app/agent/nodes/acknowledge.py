import logging

from langchain_core.messages import AIMessage

from app.agent.state import NavigatorState

logger = logging.getLogger(__name__)


async def acknowledge_node(state: NavigatorState) -> dict:
    pending = state.get("pending_action")
    if pending:
        content = "No problem — just let me know when you're ready."
    else:
        content = "Got it. Let me know if there's anything I can help with."
    logger.info("acknowledge_node: sending canned response")
    return {"messages": [AIMessage(content=content)]}
