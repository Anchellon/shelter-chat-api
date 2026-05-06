import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from app.agent.llm import get_llm
from app.agent.state import ClientContext, NavigatorState, effective_context
from app.core.config import settings

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 3

_FOLLOW_UP_SYSTEM = """\
You are a social services navigator assistant. Answer the navigator's question based on the prior context below.

Guidelines:
- Answer directly and concisely
- You can reason conditionally: "If shelter A has a waitlist, your best alternative would be..."
- You can rank or compare options based on what the navigator asks
- Be honest about real-time data you don't have (current waitlist status, live capacity) — but still give a useful recommendation based on what you know
- For broad questions spanning multiple searches ("summarize everything we found today"), read the full conversation history provided — not just the latest results
- Each group below is for a specific person in the case; their effective client profile is shown — factor it into your answer
- If the question is about a named org / topic from the prior query (e.g. "what locations are available?", "what about the Tenderloin one?"), answer from the "Prior org/topic query" section. When the navigator asks about locations, list each service grouped by its address/neighborhood — do not merge across locations.

Prior results:
{results_summary}

Prior org/topic query: {query_context}

Case context: {case_context}\
"""

_QUERY_SYSTEM = """\
You are a social services navigator assistant with access to a directory of organizations and services in San Francisco.

Use the available tools to look up information about organizations and services:
- search_by_name: look up an organization by name — use this first for any named org (e.g. "Glide", "Compass Family", "YMCA")
- search_services: semantic search for services by description — use as fallback if search_by_name returns nothing
- get_service_details: fetch full details for a specific service by ID — use once you've identified the right org

Guidelines:
- For named org questions, always start with search_by_name
- If the navigator asks for hours/eligibility of one specific service, call get_service_details on that service
- If the navigator asks broadly about an org with multiple locations (e.g. "what does the YMCA offer?"), DO NOT call get_service_details on every result. Present what search_by_name returned, **grouped by location/address** (one section per distinct address), then ask which location or program they want more details on.
- Never merge program details across different locations into one combined list — readers can't tell which programs run where.
- Cap tool calls at {max_iterations} iterations total
- If tools return nothing useful after {max_iterations} tries, say honestly that you couldn't find the organization
- Do not invent or guess information not returned by the tools
- If case context is set, mention how services align with the client's situation

Case context: {case_context}\
"""


def _format_results_summary(
    results: dict[str, list[dict]],
    formatted: dict[str, dict],
    groups: list[dict],
    case_context: ClientContext | None,
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
        eff = effective_context(case_context, group.get("client_context") if group else None)
        if eff:
            lines.append(f"  Person profile: {_context_summary(eff)}")
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


def _parse_services_from_tool_result(raw) -> list[dict]:
    """Extract service dicts from an MCP tool result, regardless of wrapper shape."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        try:
            parsed = json.loads(raw[0]["text"])
            if isinstance(parsed, list):
                return [s for s in parsed if isinstance(s, dict)]
        except Exception:
            return []
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, dict)]
    return []


def _parse_service_from_tool_result(raw) -> dict | None:
    """Extract a single service dict from an MCP tool result (e.g. get_service_details)."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        try:
            parsed = json.loads(raw[0]["text"])
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]
        except Exception:
            return None
    if isinstance(raw, dict):
        return raw
    return None


def _merge_service_into(by_id: dict, svc: dict) -> None:
    """Insert or merge a service dict into the id-keyed collection.

    Non-empty fields from `svc` overlay existing values so enriched data from
    get_service_details replaces skinny search-hit fields without losing the
    original ordering.
    """
    sid = svc.get("id")
    if sid is None:
        return
    existing = by_id.get(sid)
    if existing is None:
        by_id[sid] = dict(svc)
        return
    for k, v in svc.items():
        if v not in (None, "", [], {}):
            existing[k] = v


def _query_state_update(content, query_text: str, services: list[dict]) -> dict:
    """State update for converse query — always emits the AI message, plus the
    captured services and originating question when the query produced any."""
    update: dict = {"messages": [AIMessage(content=content)]}
    if services:
        update["last_query"] = query_text
        update["last_query_services"] = services
        logger.info(f"converse query: captured {len(services)} service(s) for follow-up")
    return update


def _format_query_context(query: str | None, services: list[dict]) -> str:
    if not services:
        return "None"
    lines = [f"Question: {query!r}" if query else "Question: (unrecorded)"]
    lines.append(f"{len(services)} service(s) returned:")
    for svc in services[:20]:
        name = svc.get("name", "Unknown")
        org = svc.get("organization_name", "")
        addr_parts = [p for p in (svc.get("address"), svc.get("city")) if p]
        location = ", ".join(addr_parts)
        sid = svc.get("id")
        line = f"- [id={sid}] {name}"
        if org:
            line += f" ({org})"
        if location:
            line += f" — {location}"
        lines.append(line)
    return "\n".join(lines)


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
        case_context = state.get("case_context")
        last_query = state.get("last_query")
        last_query_services = state.get("last_query_services") or []

        results_summary = _format_results_summary(results, formatted, groups, case_context)
        query_context = _format_query_context(last_query, last_query_services)
        context_str = _context_summary(case_context)

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
            query_context=query_context,
            case_context=context_str,
        )

        prompt_messages = [SystemMessage(content=system)]
        if history_str:
            prompt_messages.append(HumanMessage(content=f"Conversation so far:\n{history_str}"))
        if last_human:
            prompt_messages.append(HumanMessage(content=last_human.content))

        if not results and not formatted and not last_query_services:
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

        case_context = state.get("case_context")
        context_str = _context_summary(case_context)

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
            case_context=context_str,
        )

        tool_messages = [
            SystemMessage(content=system),
            HumanMessage(content=last_human.content),
        ]

        # Capture services returned by search/detail tools so follow-ups like
        # "what locations are available?" or "more about the Tenderloin one"
        # can reason about them from state. Search-hit fields are skinny;
        # get_service_details enriches them. Keyed by id so detail calls
        # overlay onto the same record without losing search ordering.
        captured_by_id: dict = {}

        for iteration in range(_MAX_TOOL_ITERATIONS):
            response = await llm_with_tools.ainvoke(tool_messages)
            tool_messages.append(response)

            if not response.tool_calls:
                # LLM has finished — return the answer
                logger.info(f"converse query: answered after {iteration} tool calls")
                return _query_state_update(response.content, last_human.content, list(captured_by_id.values()))

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
                        if tool_name in ("search_by_name", "search_services"):
                            for svc in _parse_services_from_tool_result(raw):
                                _merge_service_into(captured_by_id, svc)
                        elif tool_name == "get_service_details":
                            detail = _parse_service_from_tool_result(raw)
                            if detail:
                                _merge_service_into(captured_by_id, detail)
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
        return _query_state_update(final_response.content, last_human.content, list(captured_by_id.values()))

    async def converse_node(state: NavigatorState) -> dict:
        intent = state.get("intent") or "follow_up"

        if intent == "query":
            return await _handle_query(state)
        return await _handle_follow_up(state)

    return converse_node
