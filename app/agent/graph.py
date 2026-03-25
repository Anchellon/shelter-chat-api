import logging

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.core.config import settings
from app.guardrails.node import guardrails_node

logger = logging.getLogger(__name__)

# System prompt is explicit about tool use — critical for open source models
# which are more likely to answer from training data if not firmly instructed
SYSTEM_PROMPT = """You are a compassionate assistant helping people in San Francisco find social services. You ONLY answer questions about social services, shelters, food, health, and resources in San Francisco. If the user asks about anything outside of this scope — other cities, general knowledge, coding, weather, or anything unrelated — politely decline and redirect them to ask about SF social services.

You have two tools:
- search_services(query, limit): searches the live services database — ALWAYS call this first
- get_service_details(service_id): gets full details for a specific service

CRITICAL RULES:
- You MUST call search_services before answering ANY question about services, shelters, food, health, or resources
- NEVER answer from memory or training data — the database is the only source of truth
- If search returns no results, say so and suggest calling 211 (SF social services helpline)
- For crisis situations, always mention 988 (Suicide & Crisis Lifeline) or 911

When presenting results: include service name, what it offers, eligibility, location, hours, and how to apply.
Be warm, non-judgmental, and trauma-informed."""


def _has_tool_results(messages: list) -> bool:
    return any(getattr(m, "type", None) == "tool" for m in messages)


def build_graph(tools: list, checkpointer) -> StateGraph:
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=60,
    ).bind_tools(tools)

    async def agent_node(state: MessagesState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    def after_guardrails(state: MessagesState) -> str:
        """Route to END if guardrails produced a refusal, otherwise to agent."""
        messages = state["messages"]
        if messages and isinstance(messages[-1], AIMessage):
            return END
        return "agent"

    def after_agent(state: MessagesState) -> str:
        """
        Extended tools_condition with a safety net for open source models:
        if the model produced a text response without ever calling a tool,
        route to force_search to ensure the DB is always consulted.
        """
        last = state["messages"][-1]
        has_tool_calls = bool(getattr(last, "tool_calls", None))

        if has_tool_calls:
            return "tools"

        # Model skipped tool use entirely on first turn — force a search
        if not _has_tool_results(state["messages"]):
            logger.warning("Model answered without calling tools — forcing search")
            return "force_search"

        return END

    async def force_search_node(state: MessagesState) -> dict:
        """
        Injects an explicit instruction to search when the model skipped tool use.
        Appends a HumanMessage nudge so the model retries with tool calling.
        """
        messages = state["messages"]
        nudge = HumanMessage(
            content="Please use the search_services tool to find relevant services before answering."
        )
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = await llm.ainvoke(list(messages) + [nudge])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("guardrails", guardrails_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("force_search", force_search_node)

    builder.set_entry_point("guardrails")
    builder.add_conditional_edges("guardrails", after_guardrails, {END: END, "agent": "agent"})
    builder.add_conditional_edges("agent", after_agent, {"tools": "tools", "force_search": "force_search", END: END})
    builder.add_edge("tools", "agent")
    # After force_search, go back through normal tool condition
    builder.add_conditional_edges("force_search", tools_condition)

    return builder.compile(checkpointer=checkpointer)
