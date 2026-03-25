import logging
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Human-readable status messages per tool + input
def _tool_status(tool_name: str, tool_input: dict) -> str:
    if tool_name == "search_services":
        query = tool_input.get("query", "")
        return f"🔍 Searching for {query}..." if query else "🔍 Searching services..."
    if tool_name == "get_service_details":
        service_id = tool_input.get("service_id", "")
        return f"📋 Getting details for service {service_id}..." if service_id else "📋 Getting service details..."
    return f"⚙️ Running {tool_name}..."


async def stream_agent(
    question: str,
    conversation_id: str,
    graph,
) -> AsyncGenerator[dict, None]:
    """
    Streams events from the LangGraph agent. Yields typed dicts:

      {"type": "text",       "content": "Here are some options..."}
      {"type": "tool_start", "tool": "search_services", "status": "🔍 Searching for shelter..."}
      {"type": "tool_end",   "tool": "search_services"}

    Conversation history is loaded automatically by the checkpointer via thread_id.
    """
    config = {"configurable": {"thread_id": conversation_id}}
    logger.info(f"stream_agent start — thread={conversation_id}, q='{question[:80]}'")

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # chunk.content can be a list (tool_use blocks) — guard before yielding
            if isinstance(chunk.content, str) and chunk.content:
                yield {"type": "text", "content": chunk.content}

        elif kind == "on_tool_start":
            tool_name = event.get("name", "unknown_tool")
            tool_input = event["data"].get("input") or {}
            logger.info(f"Tool start: {tool_name}({tool_input})")
            yield {
                "type": "tool_start",
                "tool": tool_name,
                "status": _tool_status(tool_name, tool_input),
            }

        elif kind == "on_tool_end":
            tool_name = event.get("name", "unknown_tool")
            logger.info(f"Tool end: {tool_name}")
            yield {"type": "tool_end", "tool": tool_name}
