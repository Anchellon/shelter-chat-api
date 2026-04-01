"""
Smoke test for the intake HITL interrupt.
Requires: Ollama + MCP server both running.

Run from project root:
    python -m scripts.test_intake
"""
import asyncio
import json

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from app.agent.nodes.classify_groups import classify_groups_node
from app.agent.nodes.intake import build_intake_node
from app.agent.state import NavigatorState
from app.core.config import settings
from app.guardrails.node import guardrails_node


def build_test_graph(tools):
    tools_by_name = {t.name: t for t in tools}
    intake_node = build_intake_node(tools_by_name)

    def after_guardrails(state: NavigatorState) -> str:
        messages = state["messages"]
        if messages and isinstance(messages[-1], AIMessage):
            return END
        return "classify_groups"

    builder = StateGraph(NavigatorState)
    builder.add_node("guardrails", guardrails_node)
    builder.add_node("classify_groups", classify_groups_node)
    builder.add_node("intake", intake_node)
    builder.set_entry_point("guardrails")
    builder.add_conditional_edges("guardrails", after_guardrails, {END: END, "classify_groups": "classify_groups"})
    builder.add_edge("classify_groups", "intake")
    builder.add_edge("intake", END)
    return builder.compile(checkpointer=MemorySaver(), interrupt_before=[])


async def run():
    print("Connecting to MCP server...")
    client = MultiServerMCPClient({
        "shelter": {"url": settings.mcp_server_url, "transport": "streamable_http"}
    })
    tools = await asyncio.wait_for(client.get_tools(), timeout=10.0)
    print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}\n")

    graph = build_test_graph(tools)

    # This query has who=null → should trigger intake gap for "who"
    query = "Looking for emergency shelter in the Tenderloin"
    config = {"configurable": {"thread_id": "intake-test-1"}}

    print(f"QUERY: {query}")
    print("="*60)

    interrupt_data = None

    async for event in graph.astream_events(
        {
            "messages": [HumanMessage(content=query)],
            "current_time": "Monday 14:00",
            "groups": [],
            "results": {},
        },
        config=config,
        version="v2",
    ):
        name = event.get("name", "")
        if event["event"] == "on_chain_end" and name == "classify_groups":
            groups = event.get("data", {}).get("output", {}).get("groups", [])
            print(f"classify_groups → {json.dumps(groups, indent=2)}\n")

    # Check for pending interrupt
    state = await graph.aget_state(config)
    for task in state.tasks:
        for intr in getattr(task, "interrupts", []):
            interrupt_data = intr.value if hasattr(intr, "value") else intr

    if interrupt_data:
        print("INTERRUPT FIRED — intake_request:")
        print(json.dumps(interrupt_data, indent=2))
        print()

        # Simulate user answering: pick first option for each gap
        answers = {}
        for step in interrupt_data.get("steps", []):
            dim = step["dimension"]
            opts = step["options"]
            if step["type"] == "single_select":
                answers[dim] = opts[0] if opts else ""
                print(f"  Answering '{dim}' with: {answers[dim]}")
            elif step["type"] == "multi_select":
                # opts is a dict of dimension → list, pick first from first dimension
                first_dim = next(iter(opts))
                answers[dim] = [opts[first_dim][0]]
                print(f"  Answering '{dim}' with: {answers[dim]}")

        print("\nRESUMING with answers:", answers)
        print("="*60)

        async for event in graph.astream_events(
            Command(resume={"action": "submit", "answers": answers}),
            config=config,
            version="v2",
        ):
            name = event.get("name", "")
            if event["event"] == "on_chain_end" and name == "intake":
                groups = event.get("data", {}).get("output", {}).get("groups", [])
                print("intake complete → groups:")
                print(json.dumps(groups, indent=2))
    else:
        print("No interrupt — all gaps were filled by mapping (no HITL needed)")
        state = await graph.aget_state(config)
        print("Final groups:", state.values.get("groups", []))

    if hasattr(client, "aclose"):
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(run())
