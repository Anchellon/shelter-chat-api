import logging
from typing import TYPE_CHECKING, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from app.api.resume import ResumeRequest

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
    current_time: str,
    graph,
    config: dict,
) -> AsyncGenerator[dict, None]:
    """
    Streams events from the LangGraph agent. Yields typed dicts:

      {"type": "text",            "content": "..."}
      {"type": "tool_start",      "tool": "...", "status": "..."}
      {"type": "tool_end",        "tool": "..."}
      {"type": "groups_identified","groups": [...]}
      {"type": "intake_request",  "group_id": 1, "group_label": "...", "steps": [...]}
    """
    logger.info(f"stream_agent start — thread={conversation_id}, q='{question[:80]}'")

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=question)], "current_time": current_time, "groups": [], "results": {}, "formatted": {}},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_tool_start":
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

        elif kind == "on_chain_end" and event.get("name") == "guardrails":
            messages = event.get("data", {}).get("output", {}).get("messages", [])
            if messages and isinstance(messages[-1], AIMessage):
                logger.info("guardrails blocked — emitting refusal text")
                yield {"type": "text", "content": messages[-1].content}

        elif kind == "on_chain_end" and event.get("name") == "classify_groups":
            groups = event.get("data", {}).get("output", {}).get("groups", [])
            if groups:
                logger.info(f"groups_identified: {len(groups)} group(s) — held until format_complete")
            else:
                logger.info("classify_groups: 0 groups — emitting off-topic fallback")
                yield {
                    "type": "text",
                    "content": "I can only help find social services, shelters, food, health resources, and other support services in San Francisco. Please describe what you or someone you know is looking for.",
                }

        elif kind == "on_chain_end" and event.get("name") == "search_per_group":
            results = event.get("data", {}).get("output", {}).get("results", {})
            if results:
                logger.info(f"search_complete: {len(results)} group(s)")

        elif kind == "on_chain_end" and event.get("name") == "format_results":
            output = event.get("data", {}).get("output", {})
            formatted = output.get("formatted", {})
            groups = output.get("groups", [])
            messages = output.get("messages", [])
            intro = messages[0].content if messages and isinstance(messages[0].content, str) else ""
            if intro:
                yield {"type": "text", "content": intro}
            if formatted:
                logger.info(f"format_complete: {len(formatted)} group(s)")
                yield {"type": "format_complete", "formatted": formatted, "groups": groups}

    # After stream ends, check for pending interrupts (intake HITL)
    async for event in _drain_interrupts(graph, config):
        yield event


async def stream_resume(request, graph, config: dict) -> AsyncGenerator[dict, None]:
    """Resumes a graph paused at an interrupt with the user's intake answers."""
    resume_value = {"action": request.action, "answers": request.answers}
    logger.info(f"stream_resume — thread={request.conversation_id}, action={request.action}")

    async for event in graph.astream_events(Command(resume=resume_value), config=config, version="v2"):
        kind = event["event"]

        if kind == "on_tool_start":
            tool_name = event.get("name", "unknown_tool")
            tool_input = event["data"].get("input") or {}
            yield {"type": "tool_start", "tool": tool_name, "status": _tool_status(tool_name, tool_input)}

        elif kind == "on_tool_end":
            yield {"type": "tool_end", "tool": event.get("name", "unknown_tool")}

        elif kind == "on_chain_end" and event.get("name") == "classify_groups":
            pass  # held until format_complete

        elif kind == "on_chain_end" and event.get("name") == "search_per_group":
            pass  # held until format_complete

        elif kind == "on_chain_end" and event.get("name") == "format_results":
            output = event.get("data", {}).get("output", {})
            formatted = output.get("formatted", {})
            groups = output.get("groups", [])
            messages = output.get("messages", [])
            intro = messages[0].content if messages and isinstance(messages[0].content, str) else ""
            if intro:
                yield {"type": "text", "content": intro}
            if formatted:
                yield {"type": "format_complete", "formatted": formatted, "groups": groups}

    async for event in _drain_interrupts(graph, config):
        yield event


async def _drain_interrupts(graph, config) -> AsyncGenerator[dict, None]:
    """After a stream ends, emit any pending interrupt as intake_request."""
    try:
        state = await graph.aget_state(config)
        for task in state.tasks:
            for intr in getattr(task, "interrupts", []):
                data = intr.value if hasattr(intr, "value") else intr
                logger.info(f"intake_request interrupt: group_id={data.get('group_id')}")
                yield {"type": "intake_request", **data}
    except Exception as e:
        logger.warning(f"Could not check graph state for interrupts: {e}")
