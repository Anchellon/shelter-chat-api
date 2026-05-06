import logging

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from app.agent.state import NavigatorState
from app.agent.nodes.acknowledge import acknowledge_node
from app.agent.nodes.clarify import clarify_node
from app.agent.nodes.classify_groups import classify_groups_node
from app.agent.nodes.converse import build_converse_node
from app.agent.nodes.format_results import build_format_results_node
from app.agent.nodes.geo_check import build_geo_check_node
from app.agent.nodes.help import help_node
from app.agent.nodes.intake import build_intake_node
from app.agent.nodes.refine_groups import refine_groups_node
from app.agent.nodes.resolve_intent import resolve_intent_node
from app.agent.nodes.search_per_group import build_search_per_group_node
from app.agent.nodes.update_client_context import update_client_context_node
from app.guardrails.node import guardrails_node

logger = logging.getLogger(__name__)


def build_graph(tools: list, checkpointer) -> StateGraph:
    tools_by_name = {t.name: t for t in tools}
    geo_check_node = build_geo_check_node(tools_by_name)
    intake_node = build_intake_node(tools_by_name)
    search_per_group_node = build_search_per_group_node(tools_by_name)
    format_results_node = build_format_results_node()
    converse_node = build_converse_node(tools_by_name)

    def after_guardrails(state: NavigatorState) -> str:
        messages = state["messages"]
        if messages and isinstance(messages[-1], AIMessage):
            return END
        return "resolve_intent"

    def after_resolve_intent(state: NavigatorState) -> str:
        intent = state.get("intent") or "new_search"
        return {
            "new_search": "classify_groups",
            "refine": "refine_groups",
            "follow_up": "converse",
            "query": "converse",
            "set_context": "update_client_context",
            "help": "help_node",
            "acknowledge": "acknowledge_node",
            "clarify": "clarify_node",
        }.get(intent, "classify_groups")

    def after_geo_check(state: NavigatorState) -> str:
        messages = state["messages"]
        if messages and isinstance(messages[-1], AIMessage):
            return END
        return "intake"

    def after_update_client_context(state: NavigatorState) -> str:
        queue = state.get("intent_queue") or []
        if not queue:
            return END
        return {
            "new_search": "classify_groups",
            "refine": "refine_groups",
            "follow_up": "converse",
            "query": "converse",
        }.get(queue[0], END)

    builder = StateGraph(NavigatorState)

    builder.add_node("guardrails", guardrails_node)
    builder.add_node("resolve_intent", resolve_intent_node)
    builder.add_node("classify_groups", classify_groups_node)
    builder.add_node("refine_groups", refine_groups_node)
    builder.add_node("geo_check", geo_check_node)
    builder.add_node("intake", intake_node)
    builder.add_node("search_per_group", search_per_group_node)
    builder.add_node("format_results", format_results_node)
    builder.add_node("converse", converse_node)
    builder.add_node("update_client_context", update_client_context_node)
    builder.add_node("help_node", help_node)
    builder.add_node("acknowledge_node", acknowledge_node)
    builder.add_node("clarify_node", clarify_node)

    builder.set_entry_point("guardrails")
    builder.add_conditional_edges("guardrails", after_guardrails, {END: END, "resolve_intent": "resolve_intent"})
    builder.add_conditional_edges("resolve_intent", after_resolve_intent, {
        "classify_groups": "classify_groups",
        "refine_groups": "refine_groups",
        "converse": "converse",
        "update_client_context": "update_client_context",
        "help_node": "help_node",
        "acknowledge_node": "acknowledge_node",
        "clarify_node": "clarify_node",
    })
    builder.add_edge("classify_groups", "geo_check")
    builder.add_edge("refine_groups", "geo_check")
    builder.add_conditional_edges("geo_check", after_geo_check, {END: END, "intake": "intake"})
    builder.add_edge("intake", "search_per_group")
    builder.add_edge("search_per_group", "format_results")
    builder.add_edge("format_results", END)
    builder.add_edge("converse", END)
    builder.add_conditional_edges("update_client_context", after_update_client_context, {
        END: END,
        "classify_groups": "classify_groups",
        "refine_groups": "refine_groups",
        "converse": "converse",
    })
    builder.add_edge("help_node", END)
    builder.add_edge("acknowledge_node", END)
    builder.add_edge("clarify_node", END)

    return builder.compile(checkpointer=checkpointer)
