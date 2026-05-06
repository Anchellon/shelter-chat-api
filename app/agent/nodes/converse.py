import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from app.agent.llm import get_llm
from app.agent.state import ClientContext, NavigatorState
from app.core.config import settings

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 3

_FOLLOW_UP_SYSTEM = """\
You are a social services navigator assistant. Answer the navigator's question based on the prior search results below.

Guidelines:
- Answer directly and concisely
- You can reason conditionally: "If shelter A has a waitlist, your best alternative would be..."
- You can rank or compare options based on what the navigator asks
- Be honest about real-time data you don't have (current waitlist status, live capacity) — but still give a useful recommendation based on what you know
- For broad questions spanning multiple searches ("summarize everything we found today"), read the full conversation history provided — not just the latest results
- If client context is set, factor it into your answer

Prior results:
{results_summary}

Client context: {client_context}\
"""

_QUERY_SYSTEM = """\
You are a social services navigator assistant with access to a directory of organizations and services in San Francisco.

Use the available tools to look up information about organizations and services:
- search_by_name: look up an organization by name — use this first for any named org (e.g. "Glide", "Compass Family")
- search_services: semantic search for services by description — use as fallback if search_by_name returns nothing
- get_service_details: fetch full details for a specific service by ID — use once you've identified the right org

Guidelines:
- For named org questions, always start with search_by_name
- Call get_service_details on the most relevant result to get hours, eligibility, contact info
- Cap tool calls at {max_iterations} iterations total
- If tools return nothing useful after {max_iterations} tries, say honestly that you couldn't find the organization
- Answer concisely — focus on exactly what the navigator asked
- Do not invent or guess information not returned by the tools
- If client context is set, mention how services align with the client's situation

Client context: {client_context}\
"""


def _format_results_summary(
    results: dict[str, list[dict]],
    formatted: dict[str, dict],
    groups: list[dict],
) -> str:
    if not results and not formatted:
        return "No prior search results."

    lines = []
    group_map = {str(g["group_id"]): g for g in groups} if groups else {}

    for group_id, fmt in formatted.items():
        group = group_map.get(group_id, {})
        label = f"{group.get('what', 'services')} for {group.get('who', 'client')}" if group else f"Group {group_id}"
        rationale = fmt.get("rationale", "")
        service_ids = fmt.get("service_ids", [])
        lines.append(f"**{label}** (group {group_id})")
        if rationale:
            lines.append(f"  Rationale: {rationale}")
        raw_services = results.get(group_id, [])
        shown = [s for s in raw_services if s.get("id") in service_ids][:5]
        for svc in shown:
            name = svc.get("name", "Unknown")
            org = svc.get("organization_name", "")
            lines.append(f"  - {name}" + (f" ({org})" if org else ""))

    return "\n".join(lines) if lines else "No prior search results."


def _context_summary(context: ClientContext | None) -> str:
    if not context:
        return "None"
    parts = [f"{k}: {v}" for k, v in context.items() if v]
    return ", ".join(parts) if parts else "None"


def _unwrap_tool_result(result) -> str:
    if isinstance(result, list) and result and isinstance(result[0], dict) and "text" in result[0]:
        return result[0]["text"]
    if isinstance(result, (dict, list)):
        return json.dumps(result)
    return str(result)


def build_converse_node(tools_by_name: dict):
    """
    Factory — returns converse_node with MCP tools in closure.
    Handles both follow_up (state-only) and query (mini-agent) intents.
    """

    async def _handle_follow_up(state: NavigatorState) -> dict:
        messages = state["messages"]
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )

        results = state.get("results") or {}
        formatted = state.get("formatted") or {}
        groups = state.get("groups") or []
        client_context = state.get("client_context")

        results_summary = _format_results_summary(results, formatted, groups)
        context_str = _context_summary(client_context)

        # For session-spanning questions, include recent conversation history
        history_lines = []
        for m in messages[-20:]:
            if isinstance(m, HumanMessage):
                history_lines.append(f"Navigator: {m.content}")
            elif isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
                history_lines.append(f"Assistant: {m.content[:200]}")
        history_str = "\n".join(history_lines[:-1])  # exclude the current message

        system = _FOLLOW_UP_SYSTEM.format(
            results_summary=results_summary,
            client_context=context_str,
        )

        prompt_messages = [SystemMessage(content=system)]
        if history_str:
            prompt_messages.append(HumanMessage(content=f"Conversation so far:\n{history_str}"))
        if last_human:
            prompt_messages.append(HumanMessage(content=last_human.content))

        if not results and not formatted:
            extra = interrupt("I don't have any search results to reference. What would you like to know?")
            prompt_messages.append(HumanMessage(content=str(extra)))

        llm = get_llm(settings.formatter_provider, settings.formatter_model)
        response = await llm.ainvoke(prompt_messages)
        logger.info("converse follow_up: answered from state")
        return {"messages": [AIMessage(content=response.content)]}

    async def _handle_query(state: NavigatorState) -> dict:
        messages = state["messages"]
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {"messages": [AIMessage(content="I didn't receive a question.")]}

        client_context = state.get("client_context")
        context_str = _context_summary(client_context)

        # Build tool list — search_by_name may not exist yet if MCP hasn't been updated
        query_tools = [
            t for name, t in tools_by_name.items()
            if name in ("search_by_name", "search_services", "get_service_details")
        ]

        llm = get_llm(settings.formatter_provider, settings.formatter_model)

        if not query_tools:
            # No tools available — fall back to honest response
            logger.warning("converse query: no query tools available")
            return {"messages": [AIMessage(content="I'm not able to look up organization details right now — the directory tools are unavailable.")]}

        llm_with_tools = llm.bind_tools(query_tools)

        system = _QUERY_SYSTEM.format(
            max_iterations=_MAX_TOOL_ITERATIONS,
            client_context=context_str,
        )

        tool_messages = [
            SystemMessage(content=system),
            HumanMessage(content=last_human.content),
        ]

        for iteration in range(_MAX_TOOL_ITERATIONS):
            response = await llm_with_tools.ainvoke(tool_messages)
            tool_messages.append(response)

            if not response.tool_calls:
                # LLM has finished — return the answer
                logger.info(f"converse query: answered after {iteration} tool calls")
                return {"messages": [AIMessage(content=response.content)]}

            # Execute all tool calls in this iteration
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool = tools_by_name.get(tool_name)
                if tool is None:
                    tool_result = f"Tool '{tool_name}' is not available."
                else:
                    try:
                        raw = await tool.ainvoke(tc["args"])
                        tool_result = _unwrap_tool_result(raw)
                    except Exception as e:
                        tool_result = f"Tool error: {e}"
                        logger.error(f"converse query tool error ({tool_name}): {e}")

                logger.info(f"converse query: tool {tool_name} → {tool_result[:100]}")
                tool_messages.append(
                    ToolMessage(content=tool_result, tool_call_id=tc["id"])
                )

        # Exceeded max iterations — ask LLM to summarize with what it has
        final_response = await get_llm(settings.formatter_provider, settings.formatter_model).ainvoke(tool_messages)
        logger.info("converse query: max iterations reached, returning best answer")
        return {"messages": [AIMessage(content=final_response.content)]}

    async def converse_node(state: NavigatorState) -> dict:
        intent = state.get("intent") or "follow_up"

        if intent == "query":
            return await _handle_query(state)
        return await _handle_follow_up(state)

    return converse_node
