"""
Quick smoke test for classify_groups_node.
Requires: Ollama running locally (no MCP, no Postgres needed).

Run from the project root:
    python -m scripts.test_classify_groups
"""
import asyncio
import json

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.agent.state import NavigatorState
from app.agent.nodes.classify_groups import classify_groups_node
from app.guardrails.node import guardrails_node


def build_test_graph():
    """Minimal graph: guardrails → classify_groups → END."""
    from langchain_core.messages import AIMessage

    def after_guardrails(state: NavigatorState) -> str:
        messages = state["messages"]
        if messages and isinstance(messages[-1], AIMessage):
            return END
        return "classify_groups"

    builder = StateGraph(NavigatorState)
    builder.add_node("guardrails", guardrails_node)
    builder.add_node("classify_groups", classify_groups_node)
    builder.set_entry_point("guardrails")
    builder.add_conditional_edges("guardrails", after_guardrails, {END: END, "classify_groups": "classify_groups"})
    builder.add_edge("classify_groups", END)
    return builder.compile(checkpointer=MemorySaver())


TEST_QUERIES = [
    # when: null (no time mentioned)
    "I need shelter for my elderly mother and food assistance for my teenage kids",
    # when: extracted
    "Looking for a soup kitchen open on Saturday morning near the Tenderloin",
    # when: extracted (relative)
    "My friend needs drug rehab tonight, he's in the Mission",
    # offer — should return []
    "I can offer free meals to anyone who needs them",
]


async def run():
    print("Building graph...")
    graph = build_test_graph()
    print("Graph ready. Starting queries...\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] QUERY: {query}")
        print("  → waiting for guardrails...")

        groups_found = []

        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": f"test-{i}"}},
            version="v2",
        ):
            name = event.get("name", "")
            if event["event"] == "on_chain_end" and name == "guardrails":
                print("  → guardrails done, classify_groups running...")
            if event["event"] == "on_chain_end" and name == "classify_groups":
                raw = event.get("data", {}).get("output", {})
                print(f"  → classify_groups raw state update: {raw}")
            if event["event"] == "on_chain_end" and name == "classify_groups":
                groups_found = event.get("data", {}).get("output", {}).get("groups", [])

        if groups_found:
            print(json.dumps(groups_found, indent=2))
        else:
            print("  (no groups extracted)")
        print()


if __name__ == "__main__":
    asyncio.run(run())
