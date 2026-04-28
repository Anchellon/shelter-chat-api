import logging

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from app.agent.state import NavigatorState
from app.agent.nodes.classify_groups import classify_groups_node
from app.agent.nodes.geo_check import build_geo_check_node
from app.agent.nodes.intake import build_intake_node
from app.agent.nodes.search_per_group import build_search_per_group_node
from app.agent.nodes.format_results import build_format_results_node
from app.guardrails.node import guardrails_node

logger = logging.getLogger(__name__)


def build_graph(tools: list, checkpointer) -> StateGraph:
    tools_by_name = {t.name: t for t in tools}
    geo_check_node = build_geo_check_node(tools_by_name)
    intake_node = build_intake_node(tools_by_name)
    search_per_group_node = build_search_per_group_node(tools_by_name)
    format_results_node = build_format_results_node()

    def after_guardrails(state: NavigatorState) -> str:
        messages = state["messages"]
        if messages and isinstance(messages[-1], AIMessage):
            return END
        return "classify_groups"

    def after_geo_check(state: NavigatorState) -> str:
        messages = state["messages"]
        if messages and isinstance(messages[-1], AIMessage):
            return END
        return "intake"

    builder = StateGraph(NavigatorState)
    builder.add_node("guardrails", guardrails_node)
    builder.add_node("classify_groups", classify_groups_node)
    builder.add_node("geo_check", geo_check_node)
    builder.add_node("intake", intake_node)
    builder.add_node("search_per_group", search_per_group_node)
    builder.add_node("format_results", format_results_node)

    builder.set_entry_point("guardrails")
    builder.add_conditional_edges("guardrails", after_guardrails, {END: END, "classify_groups": "classify_groups"})
    builder.add_edge("classify_groups", "geo_check")
    builder.add_conditional_edges("geo_check", after_geo_check, {END: END, "intake": "intake"})
    builder.add_edge("intake", "search_per_group")
    builder.add_edge("search_per_group", "format_results")
    builder.add_edge("format_results", END)

    return builder.compile(checkpointer=checkpointer)
